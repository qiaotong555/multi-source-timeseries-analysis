# -*- coding: utf-8 -*-
"""预处理：时间轴归一、去重排序、单位换算、重采样、插值、标准化、去趋势。

每一步都写入 operations，保证清洗过程可追溯、可复算。不修改输入文件。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from .models import AnalysisPlan, CleanDataset, DatasetProfile, RawDataset

TIMESTAMP = "timestamp"

# 单位换算注册表：场景适配器通过 plan.parameters["unit_conversions"] 引用
CONVERSIONS: dict[str, Callable[[pd.Series], pd.Series]] = {
    "fahrenheit_to_celsius": lambda s: (s - 32.0) / 1.8,
    "celsius_to_fahrenheit": lambda s: s * 1.8 + 32.0,
    "kph_to_mps": lambda s: s / 3.6,
    "mps_to_kph": lambda s: s * 3.6,
    "inch_to_mm": lambda s: s * 25.4,
    "mm_to_inch": lambda s: s / 25.4,
}


def _apply_unit_conversions(frame: pd.DataFrame, conversions: list[dict]) -> list[dict]:
    applied = []
    for conv in conversions:
        col = conv.get("column")
        kind = conv.get("kind")
        if col not in frame.columns or kind not in CONVERSIONS:
            continue
        # 多实体数据必须把换算限定在问题所属的分组内，避免误改其他实体
        if conv.get("group_column") and conv["group_column"] in frame.columns:
            gmask = frame[conv["group_column"]].astype(str) == str(conv.get("group_value"))
        else:
            gmask = pd.Series(True, index=frame.index)
        # 区间按时间定位：输入可能乱序，行标签顺序不代表时间顺序
        mask = gmask
        if TIMESTAMP in frame.columns and (conv.get("start_time") or conv.get("end_time")):
            ts = pd.to_datetime(frame[TIMESTAMP])
            if conv.get("start_time"):
                mask = mask & (ts >= pd.Timestamp(conv["start_time"]))
            if conv.get("end_time"):
                mask = mask & (ts <= pd.Timestamp(conv["end_time"]))
        else:
            start = conv.get("start_index")
            end = conv.get("end_index")
            if start is not None:
                mask = mask & (frame.index >= start)
            if end is not None:
                mask = mask & (frame.index <= end)
        if not mask.any():
            continue
        before = float(pd.to_numeric(frame.loc[mask, col], errors="coerce").median())
        frame.loc[mask, col] = CONVERSIONS[kind](pd.to_numeric(frame.loc[mask, col], errors="coerce"))
        after = float(pd.to_numeric(frame.loc[mask, col], errors="coerce").median())
        applied.append({
            "column": col, "kind": kind, "rows": int(mask.sum()),
            "group_value": conv.get("group_value"),
            "start_time": conv.get("start_time"), "end_time": conv.get("end_time"),
            "median_before": round(before, 4), "median_after": round(after, 4),
            "unit_from": conv.get("from_unit", ""), "unit_to": conv.get("to_unit", ""),
        })
    return applied


def clean_dataset(
    raw: RawDataset,
    profile: DatasetProfile,
    parsed_time: Optional[pd.Series],
    plan: AnalysisPlan,
) -> CleanDataset:
    frame = raw.frame.copy()
    operations: list[dict] = []
    numeric_cols = [c for c in profile.numeric_columns if c in frame.columns]

    # 1) 时间轴归一（先补列，便于后续按时间排序与去重）
    if parsed_time is not None:
        frame[TIMESTAMP] = pd.to_datetime(parsed_time)
        n_bad = int(frame[TIMESTAMP].isna().sum())
        if n_bad:
            frame = frame[frame[TIMESTAMP].notna()].copy()
            operations.append({"op": "drop_unparsable_time", "rows": n_bad})
    else:
        frame[TIMESTAMP] = pd.RangeIndex(len(frame))
        operations.append({"op": "synthetic_index", "rows": len(frame)})

    # 2) 单位换算：必须在排序/插值之前，且此时索引仍为原始行标签
    conversions = _apply_unit_conversions(frame, plan.parameters.get("unit_conversions", []))
    if conversions:
        operations.append({"op": "unit_conversion", "detail": conversions})

    # 3) 时间索引 + 排序 + 去重
    group = plan.group_column if plan.group_column in frame.columns else None
    if group and frame[group].nunique(dropna=True) < 2:
        # 只有一个分组时无需透视，直接丢弃该列，避免列名被改写
        operations.append({"op": "drop_single_value_group", "column": group})
        frame = frame.drop(columns=[group])
        group = None
    if group:
        # 多实体共享时间戳：先按 (时间, 分组) 去重，再透视成宽表，保证时间索引唯一
        frame = frame.set_index([TIMESTAMP, group]).sort_index()
        dup = int(frame.index.duplicated().sum())
        if dup:
            agg = {c: ("mean" if c in numeric_cols else "first") for c in frame.columns}
            frame = frame.groupby(level=[0, 1], sort=True).agg(agg)
            operations.append({"op": "aggregate_duplicate_timestamps", "rows": dup,
                               "rule": "numeric=mean, other=first, key=(time, group)"})
        wide = frame[numeric_cols].unstack(level=1)
        if isinstance(wide, pd.Series):
            wide = wide.to_frame()
        new_cols = []
        for c in wide.columns:
            new_cols.append("@".join(str(x) for x in c) if isinstance(c, tuple) else f"{numeric_cols[0]}@{c}")
        wide.columns = new_cols
        groups = sorted({c.split("@")[-1] for c in new_cols})
        frame = wide.sort_index()
        remapped = [f"{c}@{g}" for c in plan.target_columns for g in groups
                    if f"{c}@{g}" in frame.columns]
        if remapped:
            plan.target_columns = remapped
        numeric_cols = list(frame.columns)
        operations.append({"op": "pivot_wide", "groups": groups,
                           "columns": len(frame.columns),
                           "target_columns": plan.target_columns})
    else:
        frame = frame.set_index(TIMESTAMP).sort_index()
        dup = int(frame.index.duplicated().sum())
        if dup:
            agg = {c: ("mean" if c in numeric_cols else "first") for c in frame.columns}
            frame = frame.groupby(level=0, sort=True).agg(agg)
            operations.append({"op": "aggregate_duplicate_timestamps", "rows": dup,
                               "rule": "numeric=mean, other=first"})
            # 数据里存在候选分组列但未指定分组字段时，跨实体求均值会混合不同对象
            if profile.group_columns:
                operations.append({"op": "warning_no_group_while_duplicated", "rows": dup,
                                   "candidate_group_columns": profile.group_columns,
                                   "hint": "重复时间戳已按均值合并，但数据中存在分组列，"
                                           "请显式指定 group_column 以避免跨实体混合"})
    operations.append({"op": "set_time_index",
                       "column": profile.time_axis.column if profile.time_axis else None,
                       "rows": len(frame)})

    # 4) 重采样
    rule = plan.resample_rule
    if rule:
        before = len(frame)
        frame = frame.resample(rule).mean(numeric_only=True)
        operations.append({"op": "resample", "rule": rule, "rows_before": before, "rows_after": len(frame)})
        numeric_cols = [c for c in numeric_cols if c in frame.columns]

    # 4b) 非有限值：inf/-inf 不是可观测值，按缺失处理并在 operations 留痕。
    # 不做静默丢弃：后续插值会填补，engine 会据此写 warning。
    nonfinite = 0
    for c in numeric_cols:
        if c not in frame.columns:
            continue
        s = pd.to_numeric(frame[c], errors="coerce")
        mask = s.notna() & ~np.isfinite(s)
        n = int(mask.sum())
        if n:
            frame.loc[mask, c] = np.nan
            nonfinite += n
    if nonfinite:
        operations.append({"op": "replace_nonfinite_with_nan", "cells": nonfinite})

    # 5) 插值
    imputed = 0
    imputed_index: list[str] = []
    if numeric_cols:
        na_before = int(frame[numeric_cols].isna().sum().sum())
        if na_before:
            imputed_index = [str(ts) for ts in frame.index[frame[numeric_cols].isna().any(axis=1)]]
            if isinstance(frame.index, pd.DatetimeIndex):
                frame[numeric_cols] = frame[numeric_cols].interpolate(
                    method="time", limit_direction="both"
                )
            else:
                frame[numeric_cols] = frame[numeric_cols].interpolate(
                    method="linear", limit_direction="both"
                )
            imputed = na_before - int(frame[numeric_cols].isna().sum().sum())
            operations.append({"op": "interpolate", "cells": int(imputed),
                               "method": "time" if isinstance(frame.index, pd.DatetimeIndex) else "linear"})

    # 6) 丢弃全空行
    before = len(frame)
    if numeric_cols:
        frame = frame.dropna(how="all", subset=numeric_cols)
    dropped = before - len(frame)
    if dropped:
        operations.append({"op": "drop_all_nan_rows", "rows": int(dropped)})

    # 7) 标准化 / 去趋势（可选）
    params = plan.parameters
    if params.get("standardize"):
        for c in numeric_cols:
            sd = float(frame[c].std(ddof=1))
            if sd and sd > 0:
                frame[c + "_z"] = (frame[c] - float(frame[c].mean())) / sd
        operations.append({"op": "standardize", "columns": numeric_cols})
    if params.get("detrend"):
        for c in list(numeric_cols):
            x = np.arange(len(frame), dtype=float)
            y = frame[c].to_numpy(dtype=float)
            ok = ~np.isnan(y)
            if ok.sum() > 3:
                coef = np.polyfit(x[ok], y[ok], 1)
                frame[c + "_detrended"] = y - np.polyval(coef, x)
        operations.append({"op": "detrend", "method": "linear_ols", "columns": numeric_cols})

    return CleanDataset(
        frame=frame,
        operations=operations,
        dropped_rows=int(dropped),
        imputed_cells=int(imputed),
        imputed_index=imputed_index,
        unit_conversions=conversions,
    )
