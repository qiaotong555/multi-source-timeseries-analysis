# -*- coding: utf-8 -*-
"""数据质量检查：缺失、重复、乱序、异常值、采样不连续、疑似单位变化。

只报告，不修改数据。每个问题都带 code、severity、位置与数量，可追溯到具体列与时间区间。
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from .models import DatasetProfile, QualityIssue, QualityReport, RawDataset

ROBUST_Z_THRESHOLD = 5.0
LEVEL_SHIFT_SIGMA = 3.0
UNIT_RATIO_LOW = 0.5
UNIT_RATIO_HIGH = 2.0


def robust_sigma(series: pd.Series) -> float:
    """基于 MAD 的稳健标准差估计，避免异常值本身污染尺度。"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return 0.0
    med = float(s.median())
    mad = float((s - med).abs().median())
    if mad > 0:
        return 1.4826 * mad
    return float(s.std(ddof=1)) if len(s) > 1 else 0.0


def robust_zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    med = float(s.median())
    sig = robust_sigma(s)
    if sig <= 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - med) / sig


def detect_spike_candidates(series: pd.Series, threshold: float = ROBUST_Z_THRESHOLD) -> pd.Series:
    """返回稳健 Z 值绝对值超过阈值的布尔掩码。"""
    z = robust_zscore(series)
    return z.abs() > threshold


def _refine_boundary(s: pd.Series, k: int, step: int) -> int:
    """在候选点邻域内寻找单点跃变最大的位置，作为真实切换点。"""
    vals = s.to_numpy(dtype=float)
    n = len(vals)
    if n < 3:
        return k
    d1 = np.abs(np.diff(vals))
    lo = max(1, k - max(2, 2 * step))
    hi = min(n - 1, k + max(2, 2 * step))
    if hi <= lo:
        return k
    seg = np.where(np.isnan(d1[lo - 1:hi]), -np.inf, d1[lo - 1:hi])
    j = int(np.argmax(seg))
    return lo + j


def detect_level_shifts(series: pd.Series, window: Optional[int] = None,
                        k_sigma: float = LEVEL_SHIFT_SIGMA * 2) -> list[dict]:
    """定位持续水平跳变，并区分「普通水平跳变」与「疑似单位变化」。

    做法：对每个候选分割点比较前后窗口的中位数得到位移序列 delta，再用 delta
    序列自身的稳健尺度（MAD）设定阈值。这样阈值来自「正常的季节/趋势位移量级」，
    而不是序列的水平波动量级——后者会让渐变型跳变永远达不到阈值。

    前后中位数比值显著偏离 1（默认 <0.5 或 >2）时判定为疑似单位或量纲变化。
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    if n < 60:
        return []
    w = window or max(12, n // 50)
    step = max(1, n // 400)
    cand = list(range(w, n - w, step))
    if len(cand) < 5:
        return []

    deltas = np.array([float(s.iloc[k:k + w].median()) - float(s.iloc[k - w:k].median())
                       for k in cand])
    abs_d = np.abs(deltas)
    med = float(np.median(abs_d))
    mad = float(np.median(np.abs(abs_d - med)))
    sig = 1.4826 * mad if mad > 0 else float(np.std(abs_d))
    if sig <= 0:
        return []
    thr = med + k_sigma * sig

    # 合并连续超过阈值的候选，每段只保留位移最大的位置
    picks, i = [], 0
    while i < len(abs_d):
        if abs_d[i] > thr:
            j = i
            while j < len(abs_d) and abs_d[j] > thr:
                j += 1
            seg = range(i, j)
            picks.append(int(max(seg, key=lambda t: abs_d[t])))
            i = j
        else:
            i += 1

    out, prev_end = [], None
    for t in picks:
        # 边界精化：候选点只在窗口量级上定位跳变，用单点最大跃变把边界收敛到真实切换点。
        # 边界偏差会让单位换算错误地作用到正常区间，进而制造大量伪异常点。
        k = _refine_boundary(s, cand[t], step)
        if prev_end is not None and k <= prev_end:
            continue
        b = float(s.iloc[max(0, k - w):k].median())
        a = float(s.iloc[k:k + w].median())
        if b == 0:
            continue
        ratio = a / b
        kind = "unit" if (ratio < UNIT_RATIO_LOW or ratio > UNIT_RATIO_HIGH) else "mean"
        out.append({"index": int(s.index[k]), "position": int(k),
                    "before_median": b, "after_median": a,
                    "ratio": ratio, "kind": kind})
        prev_end = k + w
    return out


def check_quality(
    raw: RawDataset, profile: DatasetProfile, parsed_time: Optional[pd.Series]
) -> QualityReport:
    frame = raw.frame
    issues: list[QualityIssue] = []
    missing_ratio: dict[str, float] = {}
    columns = profile.numeric_columns or [
        c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])
    ]

    # 1) 缺失
    for c in columns:
        r = float(frame[c].isna().mean())
        missing_ratio[c] = round(r, 6)
        if r > 0:
            sev = "error" if r > 0.5 else ("warn" if r > 0.05 else "info")
            issues.append(QualityIssue(
                code="MISSING_VALUES", severity=sev, column=c,
                count=int(frame[c].isna().sum()),
                detail=f"缺失率 {r:.2%}",
            ))

    # 2) 空数值列
    for c in columns:
        if frame[c].notna().sum() == 0:
            issues.append(QualityIssue(
                code="EMPTY_NUMERIC_COLUMN", severity="error", column=c,
                detail="整列无有效数值",
            ))

    # 3) 时间轴问题
    dup = out_of_order = 0
    if profile.time_axis is not None:
        ta = profile.time_axis
        if ta.duplicated_count:
            dup = ta.duplicated_count
            issues.append(QualityIssue(
                code="DUPLICATE_TIMESTAMPS", severity="warn", column=ta.column,
                count=dup, detail="存在重复时间戳，清洗阶段将按聚合规则合并",
            ))
        if ta.out_of_order_count:
            out_of_order = ta.out_of_order_count
            issues.append(QualityIssue(
                code="OUT_OF_ORDER", severity="warn", column=ta.column,
                count=out_of_order, detail="时间戳非单调递增，清洗阶段将排序",
            ))
        if ta.parse_ratio < 1.0:
            issues.append(QualityIssue(
                code="TIME_PARSE_PARTIAL", severity="warn" if ta.parse_ratio >= 0.8 else "error",
                column=ta.column, detail=f"时间列可解析比例 {ta.parse_ratio:.2%}",
            ))
    else:
        issues.append(QualityIssue(
            code="TIME_AXIS_NOT_FOUND", severity="error",
            detail="未识别到时间字段，时序分析无法执行",
        ))

    # 4) 采样不连续
    gap_count = profile.sampling.gap_count
    if gap_count:
        biggest = max(profile.sampling.gaps, key=lambda g: g["gap_seconds"])
        issues.append(QualityIssue(
            code="SAMPLING_GAP", severity="warn", count=gap_count,
            start=biggest["start"], end=biggest["end"],
            detail=f"最大缺口 {biggest['gap_seconds']:.0f} 秒，预计缺失 {biggest['expected_missing']} 个采样点",
        ))

    # 5) 异常值候选
    candidate = 0
    for c in columns:
        mask = detect_spike_candidates(frame[c])
        k = int(mask.sum())
        if k:
            candidate += k
            issues.append(QualityIssue(
                code="OUTLIER_CANDIDATE", severity="info", column=c, count=k,
                detail=f"稳健 |Z| > {ROBUST_Z_THRESHOLD} 的候选异常点",
            ))

    # 6) 疑似单位变化 / 水平跳变
    # 两个前提：一是必须先按时间排序（否则在乱序数据上无意义）；
    # 二是多实体数据必须按分组分别检测（否则不同实体的序列交错会淹没跳变信号）。
    group_col = profile.group_columns[0] if profile.group_columns else None
    if group_col and group_col in frame.columns:
        series_plan = [(c, str(g), sub[c]) for g, sub in frame.groupby(group_col) for c in columns]
    else:
        series_plan = [(c, None, frame[c]) for c in columns]

    for c, gval, series in series_plan:
        series_for_shift = series
        if parsed_time is not None and len(frame) > 1:
            order = pd.Series(parsed_time, index=frame.index).loc[series.index].sort_values().index
            series_for_shift = series.loc[order]
        for hit in detect_level_shifts(series_for_shift):
            tag = f"（分组 {gval}）" if gval else ""
            # 起始时间必须给出：输入可能乱序，行标签不能代表时间先后
            start_ts = None
            if parsed_time is not None:
                try:
                    start_ts = str(pd.Series(parsed_time, index=frame.index).loc[hit["index"]])
                except (KeyError, TypeError):
                    start_ts = None
            if hit["kind"] == "unit":
                issues.append(QualityIssue(
                    code="POSSIBLE_UNIT_CHANGE", severity="warn", column=c, group=gval,
                    row_label=hit["index"], start=start_ts,
                    count=1, detail=(f"第 {hit['position']} 个采样点{tag}前后中位数 "
                                     f"{hit['before_median']:.3f} → {hit['after_median']:.3f}"
                                     f"（比值 {hit['ratio']:.3f}），疑似单位或量纲变化"),
                ))
            else:
                issues.append(QualityIssue(
                    code="LEVEL_SHIFT", severity="info", column=c, group=gval,
                    row_label=hit["index"], start=start_ts, count=1,
                    detail=f"第 {hit['position']} 个采样点{tag}存在水平跳变（比值 {hit['ratio']:.3f}）",
                ))

    return QualityReport(
        issues=issues,
        missing_ratio=missing_ratio,
        duplicate_timestamps=dup,
        out_of_order=out_of_order,
        gap_count=gap_count,
        anomaly_candidate_count=candidate,
        # as_of 取数据自身截止时间而非墙钟时间：保证同输入重复运行完全一致
        as_of=(str(parsed_time.dropna().max()) if parsed_time is not None
               and parsed_time.notna().any() else ""),
    )
