# -*- coding: utf-8 -*-
"""异常检测与变化点检测。

异常：以稳健残差（STL 残差或滚动中位数残差）的稳健 Z 值判定，避免异常值污染尺度。
变化点：基于最小二乘残差平方和的二分分割定位均值突变，再用局部信噪比门槛确认。
（方差突变通道已于 2026-11-03 停用，理由见 doc/A3方差通道.md。）
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .models import AnalysisPlan, AnomalyPoint, ChangePoint, CleanDataset
from .quality import robust_sigma

MIN_SEGMENT = 12
# 变化点的接受门槛：局部水平跳变 / 局部（去季节）散布。
# 3.0 与原先「statistic > 3 记 high confidence」用的是同一条线。
# 实测（v2 合成集）伪报 1.1σ、真跳变 24–28σ、18 hPa 网格注入 6–9σ，
# 门槛落在 2–4 之间结果完全相同，不是掐着数据选出来的。
MIN_SNR = 3.0
# 水平与尺度都在切点两侧的窗口里估；窗口至少覆盖这么多周期，否则中位数估不准水平
SNR_WINDOW_PERIODS = 3
# 有效点少于此值的列根本不进入异常检测。报告里写「已对 N 条序列执行检测」时
# 必须用**实际进入检测的列数**，不能拿目标列数顶替：列缺失或点太少都会让 N 虚高。
MIN_ANOMALY_POINTS = 20
# 段级（持续偏移）通道。点判据减一个周期的滚动中位数，会把 ≥半个周期的台阶当趋势吸收，
# 所以另开一条通道。参数标定见 doc/段级异常检测.md：
#   w=24、阈值 2.6 时，「≥3σ 且 ≥24h」的偏移召回 0.65–0.85，误报约 1 次/年；
#   2.5σ 以下或 12h 以下的峰值与噪声同量级，不做。
SEGMENT_THRESHOLD = 2.6
SEGMENT_BASELINE_PERIODS = 7   # 基线取 7 个周期（小时级 = 7 天）
SEGMENT_MIN_SPAN_PERIODS = 2   # 超阈区间至少跨 2 个周期才算「持续」，
                               # 否则只是贴着阈值的边缘抖动（实测 wind_speed 尤其多）
SEGMENT_MIN_MULTIPLE = 3       # 序列至少 3 个基线长度才判
SEGMENT_MAX_ZERO_SHARE = 0.5   # 零值占比 ≥50% 的列跳过（降水等）
# 变化点最小段长 = max(MIN_SEGMENT, n // CHANGE_POINT_MIN_DIVISOR)，
# 且序列有效点必须够 4 段才纳入。
CHANGE_POINT_MIN_DIVISOR = 40


def anomaly_window_len(n_rows: int, parameters: dict | None) -> int:
    """异常检测**实际**使用的滚动窗口。

    别用 `rolling_window` 报这个数——那是指标模块的滚动统计窗口（天气场景默认
    24 / 7），与异常检测无关。`anomaly_window` 未设置时按序列长度推出，
    17568 点得到 351，和 24 差一个量级。
    """
    w = int((parameters or {}).get("anomaly_window", 0) or 0)
    return w or max(12, n_rows // 50)


def examined_series(frame: pd.DataFrame, columns, min_points: int) -> int:
    """真正进入检测的列数：列不存在或有效点不足的会被跳过。"""
    n = 0
    for col in columns or ():
        if col not in frame.columns:
            continue
        if int(pd.to_numeric(frame[col], errors="coerce").notna().sum()) < min_points:
            continue
        n += 1
    return n


def change_point_min_len(n_rows: int) -> int:
    return max(MIN_SEGMENT, n_rows // CHANGE_POINT_MIN_DIVISOR)


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


def _step_delta(index: pd.Index) -> pd.Timedelta:
    """序列的典型采样间隔；索引不是时间时退化为 1 小时。"""
    if isinstance(index, pd.DatetimeIndex) and len(index) > 2:
        d = np.diff(index.values).astype("timedelta64[s]").astype(float)
        pos = d[d > 0]
        if len(pos):
            return pd.Timedelta(seconds=float(np.median(pos)))
    return pd.Timedelta(hours=1)


def _zero_inflated(s: pd.Series, share: float = SEGMENT_MAX_ZERO_SHARE) -> bool:
    """零值占比过半的列不做段级判定：窗口均值衡量的是「猜没下雨」，没有意义。"""
    v = s.dropna()
    if v.empty:
        return True
    return float((v == 0).mean()) >= share


def _segment_scan(s: pd.Series, period: int, threshold: float) -> list[dict]:
    """持续偏移扫描：7 个周期的基线 + 一个周期窗口均值的稳健 z。

    点判据减的是一个周期的中心滚动中位数，而它本身就是局部水平估计——持续超过
    半个周期的台阶会被吸收进趋势项，残差几乎为零（见 `doc/段级异常检测.md`）。
    基线拉长到 7 个周期后，台阶在窗口里占比远低于 50%，才留得下来。
    尺度 `scale` 用**滚动均值的 MAD 实测**，不用 σ/√w：残差有自相关，解析式会低估数倍。
    """
    w = int(period)
    baseline = w * SEGMENT_BASELINE_PERIODS
    n = int(s.notna().sum())
    if w < 2 or len(s) < SEGMENT_MIN_MULTIPLE * baseline or n < SEGMENT_MIN_MULTIPLE * w:
        return []
    base = s.rolling(baseline, center=True, min_periods=max(3, baseline // 2)).median()
    detrended = s - base
    seasonal = detrended.groupby(_seasonal_phase(s.index, w)).transform("median")
    resid = s - base - seasonal
    # 基线窗口不完整的端部不判：分解不可靠，会系统性产生伪段
    resid.iloc[:baseline] = np.nan
    resid.iloc[-baseline:] = np.nan
    mean = resid.rolling(w, min_periods=w).mean()
    scale = 1.4826 * float(np.nanmedian(np.abs(mean - np.nanmedian(mean))))
    if not np.isfinite(scale) or scale <= 0:
        return []
    score = mean / scale
    hit = (score > threshold).fillna(False).astype(bool)
    if not bool(hit.any()):
        return []

    step = _step_delta(score.index)
    min_hits = int(w * SEGMENT_MIN_SPAN_PERIODS)
    segs: list[dict] = []
    cur: list = []
    for ts in score.index[hit]:
        if cur and ts - cur[-1] > step:
            seg = _segment_from(cur, score, w, step, min_hits)
            if seg:
                segs.append(seg)
            cur = []
        cur.append(ts)
    if cur:
        seg = _segment_from(cur, score, w, step, min_hits)
        if seg:
            segs.append(seg)
    return segs


def _segment_from(ends: list, score: pd.Series, w: int, step: pd.Timedelta,
                  min_hits: int = 1) -> Optional[dict]:
    """把一个连续超阈的窗口链折算成区间与代表点（峰值处）。

    持续性按**连续超阈的窗口个数**算，不能按加宽后的区间长度算：
    窗口对齐会让区间前后各拖出 w−1 个点，12 小时的偏移也能凑出 50 多个点，
    那样「至少 2 个周期」就形同虚设（实测过一次，56 点的段只对应 12 小时注入）。
    """
    if len(ends) < max(1, min_hits):
        return None
    start = ends[0] - step * (w - 1)
    end = ends[-1]
    peak = max(ends, key=lambda t: float(score.loc[t]))
    return {
        "start": start, "end": end, "peak": peak,
        "score": float(score.loc[peak]),
        "points": int((end - start) / step) + 1,
        "hits": len(ends),
    }


def detect_anomalies(clean: CleanDataset, plan: AnalysisPlan) -> list[AnomalyPoint]:
    frame = clean.frame
    threshold = float(plan.parameters.get("anomaly_threshold", 5.0))
    period = plan.parameters.get("seasonal_period")
    window = anomaly_window_len(len(frame), plan.parameters)
    # 段级通道：默认**关闭**。2026-11-06 按方案 B（并入异常指标）实测后关闭，
    # 理由见 doc/段级异常检测.md 第八节：开持续性要求时召回与点检器持平，
    # 关掉则精确率从 0.7540 掉到 0.5966（标注集里没有「持续偏移」这一类，新检出全记 FP）。
    segment_on = bool(plan.parameters.get("segment_anomaly", False))
    seg_thr = float(plan.parameters.get("segment_threshold", SEGMENT_THRESHOLD))
    seg_w = int(plan.parameters.get("segment_window") or (int(period) if period else 0) or 0)
    # 插值点的值来自填补而非观测：长缺口上的线性插值无法还原季节形态，
    # 会把填补误差系统性识别为异常点，必须排除。
    imputed = set(clean.imputed_index or [])
    out: list[AnomalyPoint] = []

    for col in plan.target_columns:
        if col not in frame.columns:
            continue
        s = pd.to_numeric(frame[col], errors="coerce")
        if s.notna().sum() < MIN_ANOMALY_POINTS:
            continue
        col_points: list[AnomalyPoint] = []

        # ---- 点通道 ----
        resid = _residual(s, int(period) if period else None, window)
        sig = robust_sigma(resid)
        if sig > 0:
            z = (resid - float(resid.median())) / sig
            mask = z.abs() > threshold
            method = f"robust_seasonal_trend_residual_robust_z(period={period})" \
                if (period and len(s) >= 2 * int(period)) \
                else f"rolling_median_residual_robust_z(window={window})"
            for idx in s.index[mask.fillna(False)]:
                val = s.loc[idx]
                if pd.isna(val) or str(idx) in imputed:
                    continue
                col_points.append(AnomalyPoint(
                    timestamp=str(idx),
                    index=int(frame.index.get_loc(idx)),
                    column=col,
                    value=round(float(val), 6),
                    score=round(float(z.loc[idx]), 4),
                    method=f"{method}, threshold={threshold}",
                ))

        # ---- 段通道（持续偏移）----
        if segment_on and seg_w >= 2 and not _zero_inflated(s):
            seen = {p.timestamp for p in col_points}
            for seg in _segment_scan(s, seg_w, seg_thr):
                peak = seg["peak"]
                key = str(peak)
                # 峰值那一点若已由点通道报过，就不重复计：方案 B 只是并渠道，不是双计
                if key in seen or key in imputed:
                    continue
                val = s.loc[peak]
                if pd.isna(val):
                    continue
                col_points.append(AnomalyPoint(
                    timestamp=key,
                    index=int(frame.index.get_loc(peak)),
                    column=col,
                    value=round(float(val), 6),
                    score=round(float(seg["score"]), 4),
                    method=(f"sustained_offset(baseline={seg_w * SEGMENT_BASELINE_PERIODS}, "
                            f"window={seg_w}, threshold={seg_thr}), span={seg['points']} points, "
                            f"hits={seg['hits']}"),
                ))
                seen.add(key)

        col_points.sort(key=lambda p: p.index)
        out.extend(col_points)
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


def _segment(y: np.ndarray, min_len: int, max_depth: int,
             ratio: float = 0.7) -> list[tuple[int, int, int]]:
    """递归二分分割，只在残差平方和显著下降时继续；返回 [(k, 父段左, 父段右)]。

    ratio 为**候选**判据：分割后残差平方和须降到全段的比例以下。默认 0.7 即要求
    下降超过 30%。暴露为参数是为了能做灵敏度分析——固定常量无法回答
    「判据放宽多少会多检出、多误报多少」。

    它只负责产生候选，不负责决定「这是不是真的水平突变」：单看方差下降会把
    段内缓慢漂移的中点当成变化点（实测 v2 压力在 7300 点里漂 13 hPa，
    漂移中点能被判成解释 63% 方差的「变化点」，实际只跳了 2.4 hPa）。
    真实与否由 `_local_snr` 那一道门槛判。

    父段边界必须一并返回：后续的水平与尺度都要在**该点所属的段**内估计，
    用全序列的前后缀中位数会把别处的台阶算到这个点头上。
    """
    out: list[tuple[int, int, int]] = []

    def rec(lo: int, hi: int, depth: int) -> None:
        if depth <= 0 or hi - lo < 2 * min_len:
            return
        k_rel, sse = _best_split(y[lo:hi], min_len)
        if k_rel is None or not np.isfinite(sse):
            return
        sse_parent = float(np.sum((y[lo:hi] - y[lo:hi].mean()) ** 2))
        if sse_parent <= 0 or sse > ratio * sse_parent:
            return
        k = lo + k_rel
        out.append((k, lo, hi))
        rec(lo, k, depth - 1)
        rec(k, hi, depth - 1)

    rec(0, len(y), max_depth)
    return sorted(out)


def _deseason_dev(seg: np.ndarray, phase: np.ndarray | None, period: int) -> np.ndarray:
    """去掉水平与季节项后的偏差，用来估局部散布。

    为什么必须去季节：温度的日周期振幅可以有 6 °C，直接取散布会把「日变化」
    当成噪声，一个 6 °C 的真实台阶只算出 0.97σ（实测），判据就永远不认。
    周期项是可预测的结构，不是噪声。
    """
    dev = seg - float(np.median(seg))
    if period >= 2 and phase is not None and len(seg) >= 2 * period:
        out = dev.copy()
        for p in range(period):
            m = phase == p
            if int(m.sum()) >= 3:
                out[m] = out[m] - float(np.median(dev[m]))
        return out
    return dev


def _local_snr(y: np.ndarray, phase: np.ndarray | None, period: int,
               k: int, lo: int, hi: int, h: int) -> dict | None:
    """切点两侧的水平差 / 局部去季节散布。

    三个量都在**紧邻切点的窗口**里算，不用前缀/后缀：
    前缀/后缀中位数会被段内漂移稀释（18 hPa 的台阶只量出 9.97），
    也会把同列其它台阶的影响算进来。
    """
    a, b = max(lo, k - h), min(hi, k + h)
    left, right = y[a:k], y[k:b]
    if len(left) < 4 or len(right) < 4:
        return None
    before = float(np.median(left))
    after = float(np.median(right))
    jump = abs(after - before)
    dev = np.concatenate([_deseason_dev(left, None if phase is None else phase[a:k], period),
                          _deseason_dev(right, None if phase is None else phase[k:b], period)])
    scale = 1.4826 * float(np.median(np.abs(dev)))
    if not np.isfinite(scale) or scale <= 0:
        # 局部散布为 0（窗口内近乎常数，典型是降水列的大片零）时**不能判显著**。
        # 曾经写成 jump>0 就给 inf，于是「两侧中位数差 0.1 mm」被当成最高置信度的
        # 变化点报出来（实测 4 个降水列各报 2–3 个）。没有噪声模型就没有证据，
        # 这种情况下返回 0 而不是 inf。
        snr = 0.0
    else:
        snr = jump / scale
    return {"before": before, "after": after, "jump": jump, "scale": scale, "snr": snr}


def detect_change_points(clean: CleanDataset, plan: AnalysisPlan) -> list[ChangePoint]:
    frame = clean.frame
    max_depth = int(plan.parameters.get("change_point_depth", 2))
    ratio = float(plan.parameters.get("change_point_sse_ratio", 0.7))
    period = int(plan.parameters.get("seasonal_period") or 0)
    min_len = change_point_min_len(len(frame))
    out: list[ChangePoint] = []

    for col in plan.target_columns:
        if col not in frame.columns:
            continue
        s = pd.to_numeric(frame[col], errors="coerce")
        if s.notna().sum() < 4 * min_len:
            continue
        s = s.interpolate(limit_direction="both")
        y = s.to_numpy(dtype=float)
        # 相位按真实时间算（有整段缺失时行位置与周期会错位）
        phase = _seasonal_phase(frame.index, period) if period >= 2 else None
        h = min(max(2 * min_len, SNR_WINDOW_PERIODS * max(period, 1), 24),
                max(8, len(y) // 4))
        for k, lo, hi in _segment(y, min_len, max_depth, ratio):
            loc = _local_snr(y, phase, period, k, lo, hi, h)
            if loc is None:
                continue
            # 候选只是「方差解释得动」；这里问的是「跳变是否大于局部噪声」
            if not (loc["snr"] >= MIN_SNR):
                continue
            out.append(ChangePoint(
                timestamp=str(frame.index[k]), index=int(k), column=col, kind="mean",
                statistic=round(float(loc["snr"]), 4),
                before=round(loc["before"], 6), after=round(loc["after"], 6),
                confidence="high" if loc["snr"] >= 2 * MIN_SNR else "medium",
                window=int(h),
            ))
        # 方差突变通道已于 2026-11-03 停用，理由是被实测证明没有分辨力（见 doc/A3方差通道.md）：
        # 1) 量纲错：判据是「滚动标准差的单步跳变 > 2 × 滚动标准差的中位数」。
        #    σ→2σ 时跳变 ≈ σ、基线也 ≈ σ，比值恒 ≈1.0，**真变化永远够不到 2.0**；
        #    它只在单个极端事件支配窗口时偶然越过（降水就是这种情况）。
        # 2) 零膨胀 + 重尾下降解：v2 降水为 iid，却出现 7.0 与 11.1 倍的标准差比；
        #    置换检验给出 p=0.499 / 0.005，而真正注入的 σ×3 变化 p=0.035——噪声比信号更「显著」。
        # 3) 去季节残差上改判「相邻窗标准差比」也不成立：未注入时四列各报一个
        #    σ 比 2.4–3.4 的假点（位置随机），而 σ×2 的真变化只有 2.1，埋在噪声上限之下。
        # 结论：在本项目的样本量（每小时 1 年 / 窗口 219 点）下，缩放估计的抽样波动
        # 大于要检测的变化。宁可不报，也不报不可验证的点。
    return out


EXTREME_QUANTILE = 0.99
EXTREME_MIN_POINTS = 100
EXTREME_MAX_RUNS = 2000
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


def _observed_mask(clean: CleanDataset, frame: pd.DataFrame, col: str) -> pd.Series:
    """该列真正的观测点掩码（排除插补值）。

    逐列插补位置优先：行级并集会让「气压缺一个点」把同刻的阵风极值一起排除掉。
    没有逐列信息时（旧调用方/单测）退回行级，行为与 D2 一致。
    """
    s = pd.to_numeric(frame[col], errors="coerce")
    per_col = getattr(clean, "imputed_columns", None) or {}
    if per_col:
        bad = {str(x) for x in (per_col.get(col) or [])}
        col_mask = pd.Series([str(i) in bad for i in frame.index], index=frame.index)
    else:
        imputed = {str(x) for x in (clean.imputed_index or [])}
        col_mask = pd.Series([str(i) in imputed for i in frame.index], index=frame.index)
    return s.notna() & ~col_mask


def extreme_examined(clean: CleanDataset, plan: AnalysisPlan) -> tuple[int, int]:
    """(实际参与极值判定的列数, 目标列数)。

    判据与 `detect_extremes` 的跳过条件一致：列缺失、或**观测点**（排除插补）
    少于 `EXTREME_MIN_POINTS` 的列不参与。报告里说「已检查 N 条序列」时用这个数。
    """
    frame = clean.frame
    total = len(plan.target_columns or [])
    n = 0
    for col in plan.target_columns or ():
        if col not in frame.columns:
            continue
        if int(_observed_mask(clean, frame, col).sum()) < EXTREME_MIN_POINTS:
            continue
        n += 1
    return n, total


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
    out: list[dict] = []

    for col in plan.target_columns:
        if col not in frame.columns:
            continue
        s = pd.to_numeric(frame[col], errors="coerce")
        observed = _observed_mask(clean, frame, col)
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

