# -*- coding: utf-8 -*-
"""字段与采样摘要：识别时间轴、变量角色和采样特征。

只做识别，不做修改；识别结果供质量检查与清洗使用。
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from .models import DatasetProfile, RawDataset, SamplingInfo, TimeAxisInfo

TIME_NAME_PATTERN = re.compile(
    r"(time|date|datetime|timestamp|dt|时刻|时间|日期)", re.IGNORECASE
)
MIN_PARSE_RATIO = 0.80


def _try_parse(values: pd.Series) -> tuple[pd.Series, float]:
    """尝试把一列解析为时间，返回（解析结果，可解析比例）。"""
    non_null = values.dropna()
    if len(non_null) == 0:
        return pd.Series([pd.NaT] * len(values), index=values.index), 0.0
    parsed = pd.to_datetime(non_null, errors="coerce", utc=False)
    ratio = float(parsed.notna().sum()) / float(len(non_null))
    full = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    full.loc[non_null.index] = parsed
    return full, ratio


def _looks_like_epoch(values: pd.Series) -> bool:
    s = pd.to_numeric(values, errors="coerce").dropna()
    if len(s) == 0:
        return False
    med = float(s.median())
    return 1e8 < med < 4e10


def detect_time_axis(
    frame: pd.DataFrame, hint: Optional[str] = None
) -> tuple[Optional[str], Optional[pd.Series], float, bool]:
    """识别时间轴。返回（列名，解析后序列，可解析比例，是否按 epoch 解析）。"""
    if hint and hint in frame.columns:
        parsed, ratio = _try_parse(frame[hint])
        if ratio >= MIN_PARSE_RATIO:
            return hint, parsed, ratio, False
        if _looks_like_epoch(frame[hint]):
            conv = pd.to_datetime(pd.to_numeric(frame[hint], errors="coerce"), unit="s", errors="coerce")
            return hint, conv, float(conv.notna().mean()), True

    # 1) 列名匹配
    for col in frame.columns:
        if TIME_NAME_PATTERN.search(str(col)):
            parsed, ratio = _try_parse(frame[col])
            if ratio >= MIN_PARSE_RATIO:
                return col, parsed, ratio, False

    # 2) 任意可解析列
    best: tuple[Optional[str], Optional[pd.Series], float] = (None, None, 0.0)
    for col in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[col]):
            continue
        parsed, ratio = _try_parse(frame[col])
        if ratio > best[2]:
            best = (col, parsed, ratio)
    if best[2] >= MIN_PARSE_RATIO:
        return best[0], best[1], best[2], False

    # 3) epoch 数值列
    for col in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[col]) and _looks_like_epoch(frame[col]):
            conv = pd.to_datetime(pd.to_numeric(frame[col], errors="coerce"), unit="s", errors="coerce")
            return col, conv, float(conv.notna().mean()), True

    return None, None, 0.0, False


def _gaps_for(ts: pd.Series, group: Optional[str] = None) -> list[dict]:
    """在单个时间轴上找采样缺口。"""
    s = ts.dropna().drop_duplicates().sort_values()
    if len(s) < 3:
        return []
    diffs = s.diff().dropna()
    if diffs.empty:
        return []
    med = diffs.median()
    if med is None or pd.isna(med) or med.total_seconds() <= 0:
        return []
    med_sec = med.total_seconds()
    out, idx = [], s.index
    for pos in range(1, len(s)):
        d = (s.iloc[pos] - s.iloc[pos - 1]).total_seconds()
        if d > 3.0 * med_sec:
            missing = int(round(d / med_sec)) - 1
            item = {
                "start": str(s.iloc[pos - 1]),
                "end": str(s.iloc[pos]),
                "gap_seconds": float(d),
                "expected_missing": max(missing, 1),
                "index_before": int(idx[pos - 1]),
            }
            if group is not None:
                item["group"] = group
            out.append(item)
    return out


def _gap_report(parsed: pd.Series, frame: Optional[pd.DataFrame] = None,
                group_columns: Optional[list] = None) -> tuple[int, list[dict]]:
    """采样缺口检测。

    多实体数据必须按实体分别判断：不同实体共享时间轴时，某一实体缺失的时间点
    仍会由其他实体出现在时间戳并集里，按并集检测会漏掉真实缺口。
    """
    gaps: list[dict] = []
    group_col = None
    if frame is not None and group_columns:
        group_col = next((c for c in group_columns if c in frame.columns), None)
    if group_col:
        for gval, sub in frame.groupby(group_col):
            gaps.extend(_gaps_for(parsed.loc[sub.index], str(gval)))
    else:
        gaps.extend(_gaps_for(parsed))
    gaps.sort(key=lambda g: g["gap_seconds"], reverse=True)
    return len(gaps), gaps[:50]


def profile_dataset(raw: RawDataset) -> tuple[DatasetProfile, Optional[pd.Series]]:
    """生成字段与采样摘要；同时返回解析后的时间列（供后续排序与重采样）。"""
    frame = raw.frame
    col, parsed, ratio, is_epoch = detect_time_axis(frame, raw.spec.time_column)

    time_info = None
    sampling = SamplingInfo(interval_stable=False)
    if col is not None and parsed is not None:
        valid = parsed.dropna()
        is_monotonic = bool(valid.is_monotonic_increasing) if len(valid) else False
        # 采样间隔必须在「去重后的时间点」上计算：多实体共享时间戳会产生 0 间隔
        diffs = valid.drop_duplicates().sort_values().diff().dropna()
        med_sec = float(diffs.median().total_seconds()) if len(diffs) else None
        dup = int(valid.duplicated().sum())
        out_of_order = 0
        if len(valid) > 1:
            out_of_order = int((valid.diff().dt.total_seconds() < 0).sum())
        time_info = TimeAxisInfo(
            column=col,
            parse_ratio=round(ratio, 4),
            start=str(valid.min()) if len(valid) else None,
            end=str(valid.max()) if len(valid) else None,
            is_monotonic=is_monotonic,
            median_interval_seconds=med_sec,
            duplicated_count=dup,
            out_of_order_count=out_of_order,
        )
        try:
            freq = pd.infer_freq(valid.sort_values())
        except (ValueError, TypeError):
            freq = None
        if len(diffs) > 1 and med_sec:
            std_sec = float(diffs.dt.total_seconds().std())
            stable = std_sec <= 0.05 * med_sec
        else:
            stable = len(diffs) <= 1
        sampling = SamplingInfo(
            inferred_freq=freq,
            interval_stable=bool(stable),
            gap_count=0,
            gaps=[],
        )

    numeric = [
        c for c in frame.columns
        if c != col and pd.api.types.is_numeric_dtype(frame[c]) and frame[c].notna().sum() > 0
    ]
    categorical = [
        c for c in frame.columns
        if c != col and c not in numeric
    ]
    n = len(frame)
    upper = max(2, n // 20)
    group_cols = []
    for c in frame.columns:
        if c == col or c in numeric:
            continue
        k = int(frame[c].nunique(dropna=True))
        if 2 <= k <= min(50, upper):
            group_cols.append(c)
    for c in numeric:
        k = int(frame[c].nunique(dropna=True))
        if 2 <= k <= min(20, upper) and k < n * 0.02:
            group_cols.append(c)

    # 缺口检测需要分组信息，因此放在分组列识别之后
    if parsed is not None:
        gap_count, gaps = _gap_report(parsed, frame, group_cols)
        sampling.gap_count = gap_count
        sampling.gaps = gaps

    profile = DatasetProfile(
        time_axis=time_info,
        numeric_columns=numeric,
        categorical_columns=categorical,
        group_columns=group_cols,
        sampling=sampling,
        row_count=n,
        column_count=len(frame.columns),
    )
    return profile, parsed
