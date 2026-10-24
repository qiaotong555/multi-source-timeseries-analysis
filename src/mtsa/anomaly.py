# -*- coding: utf-8 -*-
"""异常检测与变化点检测。

异常：以稳健残差（STL 残差或滚动中位数残差）的稳健 Z 值判定，避免异常值污染尺度。
变化点：基于最小二乘残差平方和的二分分割定位均值突变，并单独识别方差突变与单位跳变。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import AnalysisPlan, AnomalyPoint, ChangePoint, CleanDataset
from .quality import robust_sigma

MIN_SEGMENT = 12


def _seasonal_phase(index: pd.Index, period: int) -> np.ndarray:
    """季节相位必须按「真实时间」计算，不能按行位置取模。

    一旦序列存在整段采样缺失，行位置与周期相位就会错位，季节项无法对齐，
    残差里会残留完整周期波动，直接导致异常检测大规模误报。
    """
    if isinstance(index, pd.DatetimeIndex) and len(index) > 2:
        deltas = np.diff(index.values).astype("timedelta64[s]").astype(float)
        positive = deltas[deltas > 0]
        if len(positive):
            step = float(np.median(positive))
            if step > 0:
                elapsed = (index - index[0]).total_seconds().to_numpy() / step
                return np.round(elapsed).astype(np.int64) % period
    return np.arange(len(index)) % period


def _residual(series: pd.Series, period: int | None, window: int) -> pd.Series:
    """返回剔除趋势/季节后的残差。

    使用「滚动中位数趋势 + 相位中位数季节项」的稳健分解，而不用 STL：
    STL 的趋势/季节项由移动平均得到，单个极端值会被摊到相邻约一个周期的样本上，
    既制造大量伪异常点，又压低真异常点的残差。中位数分解对极端值稳健，不会拖尾。
    """
    if period and period >= 2 and len(series) >= 2 * period:
        p = int(period)
        trend = series.rolling(p, center=True, min_periods=max(3, p // 2)).median()
        detrended = series - trend
        seasonal = detrended.groupby(_seasonal_phase(series.index, p)).transform("median")
        resid = series - trend - seasonal
        # 序列两端窗口不完整，分解不可靠，必须排除，否则会系统性产生伪异常点
        resid.iloc[:p] = np.nan
        resid.iloc[-p:] = np.nan
        return resid
    trend = series.rolling(window, center=True, min_periods=max(3, window // 2)).median()
    resid = series - trend
    resid.iloc[:window] = np.nan
    resid.iloc[-window:] = np.nan
    return resid


def detect_anomalies(clean: CleanDataset, plan: AnalysisPlan) -> list[AnomalyPoint]:
    frame = clean.frame
    threshold = float(plan.parameters.get("anomaly_threshold", 5.0))
    period = plan.parameters.get("seasonal_period")
    window = int(plan.parameters.get("anomaly_window", 0)) or max(12, len(frame) // 50)
    # 插值点的值来自填补而非观测：长缺口上的线性插值无法还原季节形态，
    # 会把填补误差系统性识别为异常点，必须排除。
    imputed = set(clean.imputed_index or [])
    out: list[AnomalyPoint] = []

    for col in plan.target_columns:
        if col not in frame.columns:
            continue
        s = pd.to_numeric(frame[col], errors="coerce")
        if s.notna().sum() < 20:
            continue
        resid = _residual(s, int(period) if period else None, window)
        sig = robust_sigma(resid)
        if sig <= 0:
            continue
        z = (resid - float(resid.median())) / sig
        mask = z.abs() > threshold
        if not mask.any():
            continue
        method = f"robust_seasonal_trend_residual_robust_z(period={period})" \
            if (period and len(s) >= 2 * int(period)) \
            else f"rolling_median_residual_robust_z(window={window})"
        for idx in s.index[mask.fillna(False)]:
            val = s.loc[idx]
            if pd.isna(val) or str(idx) in imputed:
                continue
            out.append(AnomalyPoint(
                timestamp=str(idx),
                index=int(frame.index.get_loc(idx)),
                column=col,
                value=round(float(val), 6),
                score=round(float(z.loc[idx]), 4),
                method=f"{method}, threshold={threshold}",
            ))
    return out


def _best_split(y: np.ndarray, min_len: int) -> tuple[int | None, float]:
    """在给定序列上寻找使两段残差平方和最小的分割点。"""
    n = len(y)
    if n < 2 * min_len:
        return None, float("inf")
    cs = np.cumsum(y)
    css = np.cumsum(y ** 2)
    total = css[-1]
    best_k, best_sse = None, float("inf")
    for k in range(min_len, n - min_len + 1):
        s1, s2 = cs[k - 1], cs[-1] - cs[k - 1]
        q1, q2 = css[k - 1], total - css[k - 1]
        sse = (q1 - s1 ** 2 / k) + (q2 - s2 ** 2 / (n - k))
        if sse < best_sse:
            best_sse, best_k = sse, k
    return best_k, best_sse


def _segment(y: np.ndarray, min_len: int, max_depth: int, sse_full: float,
             ratio: float = 0.7) -> list[int]:
    """递归二分分割，只在残差平方和显著下降时继续。

    ratio 为接受判据：分割后残差平方和须降到全段的比例以下。默认 0.7 即要求
    下降超过 30%。暴露为参数是为了能做灵敏度分析——固定常量无法回答
    「判据放宽多少会多检出、多误报多少」。
    """
    if max_depth <= 0:
        return []
    k, sse = _best_split(y, min_len)
    if k is None or not np.isfinite(sse) or sse_full <= 0:
        return []
    if sse > ratio * sse_full:
        return []
    left = _segment(y[:k], min_len, max_depth - 1,
                    float(np.sum((y[:k] - y[:k].mean()) ** 2)), ratio)
    right = _segment(y[k:], min_len, max_depth - 1,
                     float(np.sum((y[k:] - y[k:].mean()) ** 2)), ratio)
    return sorted(left + [k] + [k + r for r in right])


def detect_change_points(clean: CleanDataset, plan: AnalysisPlan) -> list[ChangePoint]:
    frame = clean.frame
    max_depth = int(plan.parameters.get("change_point_depth", 2))
    ratio = float(plan.parameters.get("change_point_sse_ratio", 0.7))
    min_len = max(MIN_SEGMENT, len(frame) // 40)
    out: list[ChangePoint] = []

    for col in plan.target_columns:
        if col not in frame.columns:
            continue
        s = pd.to_numeric(frame[col], errors="coerce")
        if s.notna().sum() < 4 * min_len:
            continue
        s = s.interpolate(limit_direction="both")
        y = s.to_numpy(dtype=float)
        sse_full = float(np.sum((y - y.mean()) ** 2))
        for k in _segment(y, min_len, max_depth, sse_full, ratio):
            before = float(np.median(y[:k]))
            after = float(np.median(y[k:]))
            scale = robust_sigma(s)
            stat = abs(after - before) / scale if scale > 0 else float("nan")
            idx = frame.index[k]
            out.append(ChangePoint(
                timestamp=str(idx), index=int(k), column=col, kind="mean",
                statistic=round(float(stat), 4),
                before=round(before, 6), after=round(after, 6),
                confidence="high" if (np.isfinite(stat) and stat > 3) else "medium",
            ))
        # 方差突变：滚动标准差的跳跃
        w = max(12, len(y) // 40)
        roll = pd.Series(y).rolling(w, min_periods=w).std()
        d = roll.diff().abs()
        base = float(roll.median())
        if base > 0 and d.notna().any():
            j = int(d.idxmax())
            if float(d.loc[j]) > 2.0 * base:
                out.append(ChangePoint(
                    timestamp=str(frame.index[j]), index=j, column=col, kind="variance",
                    statistic=round(float(d.loc[j] / base), 4),
                    before=round(float(roll.loc[j - 1]) if j > 0 else float("nan"), 6),
                    after=round(float(roll.loc[j]), 6),
                    confidence="medium",
                ))
    return out

EXTREME_QUANTILE = 0.99
EXTREME_MIN_POINTS = 100
EXTREME_MAX_RUNS = 100
HIGH_ONLY_HINTS = ("rain", "precip", "wind", "gust", "降雨", "风速")


def _high_only(column: object) -> bool:
    """降雨与风速只关心高端；温度/湿度/气压两端都看。"""
    low = str(column).lower()
    return any(h in low for h in HIGH_ONLY_HINTS)


def _extreme_runs(mask: pd.Series, index: pd.Index) -> tuple[list[dict], int]:
    """把连续为真的点压成区间，最多保留 100 段；返回 (段列表, 截断前总段数)。

    上限不能太小：1% 的尾在小时数据上可有几十段，截得太狠会把后段事件漏掉，
    使「事件覆盖」被 payload 上限而不是检测能力决定（D2 首跑踩过）。
    """
    arr = mask.to_numpy()
    runs: list[dict] = []
    i, n = 0, len(arr)
    while i < n:
        if arr[i]:
            j = i
            while j + 1 < n and arr[j + 1]:
                j += 1
            runs.append({"start": str(index[i]), "end": str(index[j]), "count": j - i + 1})
            i = j + 1
        else:
            i += 1
    return runs[:EXTREME_MAX_RUNS], len(runs)


def detect_extremes(clean: CleanDataset, plan: AnalysisPlan) -> list[dict]:
    """绝对极端值：超过本列经验 p99（或低于 p1）的观测点。

    与 `detect_anomalies` 的「统计异常」是两件事：
    - 异常 = 偏离自身趋势/季节后的残差过大；
    - 极端 = 原始取值落在自身分布的两端。
    热浪在夏季是常态（不异常），但可以超过 p99（极端）。两者分开报告，互不替代。
    """
    frame = clean.frame
    q = float(plan.parameters.get("extreme_quantile") or EXTREME_QUANTILE)
    if not (0.5 < q < 1.0):
        q = EXTREME_QUANTILE
    imputed = {str(x) for x in (clean.imputed_index or [])}
    imputed_mask = pd.Series([str(i) in imputed for i in frame.index], index=frame.index)
    out: list[dict] = []

    for col in plan.target_columns:
        if col not in frame.columns:
            continue
        s = pd.to_numeric(frame[col], errors="coerce")
        observed = s.notna() & ~imputed_mask
        if int(observed.sum()) < EXTREME_MIN_POINTS:
            continue
        vals = s[observed]
        hi = float(vals.quantile(q))
        lo = float(vals.quantile(1.0 - q))
        if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
            continue

        tails = [("high", hi, s >= hi)]
        if not _high_only(col):
            tails.append(("low", lo, s <= lo))
        for tail, thr, mask in tails:
            mask = mask & observed
            count = int(mask.sum())
            if count == 0:
                continue
            runs, total = _extreme_runs(mask, frame.index)
            out.append({
                "column": col, "tail": tail, "quantile": q,
                "threshold": round(float(thr), 6), "count": count,
                "run_count": total, "runs_truncated": total > len(runs),
                "runs": runs,
                "method": "empirical_quantile_tail",
            })
    return out

