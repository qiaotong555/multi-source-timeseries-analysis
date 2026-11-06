# -*- coding: utf-8 -*-
"""可行性实验：段级异常（持续偏移）能不能检出来，代价多大。

问题
----
现行检测器是**逐点**的：残差 z 超过阈值就报点。它的残差 = 原值 − 24 小时中心滚动中位数
− 相位中位数季节项。24 小时的中心滚动中位数本身就是**局部水平估计**——一个持续超过
约 12 小时的台阶会被它吸收进「趋势」项，残差里几乎没有残留。所以「单点不超阈、
整段持续偏移」这类漂移/驻留异常检不出（`doc/评估设计.md` 3.5 第 3 条）。

段级判据
--------
把基线换成**更长窗口**（默认 7 天 = 168 点）：7 天的中位数不会被 12–48 小时的台阶搬动
（占比远低于 50%），台阶就能留在残差里。再对长度为 w 的窗口取均值，得到

    score = mean(resid[t-w+1 .. t]) / S_w

S_w 用**实测**得到（干净数据上滚动均值的 1.4826·MAD），不用 σ/√w ——
残差有自相关，解析式会低估 S_w。这一点是 A3（方差通道）翻车的原因：
用 iid 假设算尺度，零膨胀重尾下噪声就能给出 12–13 倍标准差比。

实验设计
--------
- 数据：`weather_eval_v1.csv` 的 STA_A 气温（2 年小时级，17409 点）。
- 切分：前一半 dev（选阈值），后一半 eval（报召回）。时间序，不随机。
- 注入：在 eval 段随机位置叠加长度 ∈ {12,24,48} 小时、幅度 ∈ {1.0,1.5,2.0,2.5}·σ_r
  的台阶（σ_r 是段级残差的标准差）。
- 判据一（现行）：`detect_anomalies`，阈值 7.0，注入块内出现任一检出点即算召回。
- 判据二（候选）：段分数超阈值；阈值在 dev 上按「每 30 天误报 ≤ 预算」选定。
- 误报预算取 1 / 2 / 4 次每 30 天，分别报召回。

用法：
    python scripts/exp_segment_anomaly.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from mtsa.anomaly import detect_anomalies  # noqa: E402
from mtsa.quality import robust_sigma  # noqa: E402
from mtsa.models import AnalysisPlan, CleanDataset  # noqa: E402

FIX = os.path.join(ROOT, "fixtures")
COLUMN = "temperature"
STATION = "STA_A"
BASELINE = 168          # 7 天，段级基线窗口
WINDOWS = (12, 24, 48)  # 段检测窗口
LENGTHS = (12, 24, 48)  # 注入长度（小时）
AMPS = (2.5, 3.0, 4.0, 5.0)
REPS = 40
POINT_THRESHOLD = 7.0
BUDGETS = (1, 2, 4)     # 每 30 天允许的误报次数
SEED = 20261106


def load_series() -> pd.Series:
    df = pd.read_csv(os.path.join(FIX, "weather_eval_v1.csv"), parse_dates=["timestamp"])
    s = df[df["station"] == STATION].set_index("timestamp")[COLUMN].sort_index()
    # 该夹具里有 25 个重复时刻（同值重复行，DST 迁移残留）。去重保留首条，
    # 否则 reindex 会报 duplicate labels；重复值本身相同，不影响结论。
    s = s[~s.index.duplicated(keep="first")]
    # 补齐到完整小时网格，缺口用 NaN 表示（不插值，避免制造虚假台阶）
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    return s.reindex(full)


def segment_residual(s: pd.Series) -> tuple[pd.Series, float]:
    """7 天基线 + 相位中位数季节项 → 残差，返回（残差, 稳健标准差）。"""
    base = s.rolling(BASELINE, center=True, min_periods=BASELINE // 2).median()
    detrended = s - base
    phase = detrended.index.hour.to_numpy()
    seasonal = detrended.groupby(phase).transform("median")
    resid = s - base - seasonal
    sig = robust_sigma(resid)
    return resid, float(sig)


def window_score(resid: pd.Series, w: int, sig: float) -> pd.Series:
    """窗口均值的稳健 z 分数；尺度用干净数据上滚动均值的 MAD 实测。"""
    mean = resid.rolling(w, min_periods=w).mean()
    scale = 1.4826 * float(np.nanmedian(np.abs(mean - np.nanmedian(mean))))
    return mean / (scale if scale > 0 else sig)


def window_scale(resid: pd.Series, w: int) -> float:
    """滚动均值的实测稳健标准差（自相关已隐含在里面，不用 σ/√w）。"""
    mean = resid.rolling(w, min_periods=w).mean()
    return 1.4826 * float(np.nanmedian(np.abs(mean - np.nanmedian(mean))))


def alarms(score: pd.Series, w: int, thr: float) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """窗口分数超阈 → 合并成连续区间（窗口右端对齐，向前扩 w-1 点）。"""
    hit = score > thr
    hit = hit.fillna(False)
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    idx = list(score.index[hit])
    for end in idx:
        start = end - pd.Timedelta(hours=w - 1)
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
    return out


def alarms_per_30d(alarm_list, index: pd.DatetimeIndex) -> float:
    days = (index[-1] - index[0]).total_seconds() / 86400
    return len(alarm_list) * 30.0 / days if days > 0 else 0.0


def overlaps(start, end, s, e) -> bool:
    lo = max(start, s)
    hi = min(end, e)
    if hi <= lo:
        return False
    return (hi - lo) >= 0.5 * (e - s)


def make_plan() -> AnalysisPlan:
    return AnalysisPlan(
        scene_id="weather", scene_confidence=1.0, question="段级异常可行性实验",
        target_columns=[COLUMN], analyses=["anomaly"],
        parameters={"seasonal_period": 24, "anomaly_threshold": POINT_THRESHOLD},
        not_supported=[], plan_source="rule",
    )


def point_detected(stamps: set[str], s: pd.Timestamp, e: pd.Timestamp) -> bool:
    cur = s
    while cur < e:
        if str(cur) in stamps:
            return True
        cur += pd.Timedelta(hours=1)
    return False


def main() -> int:
    s = load_series()
    resid, sig = segment_residual(s)
    valid = resid.dropna().index
    half = valid[len(valid) // 2]
    dev_idx = valid[valid < half]
    eval_idx = valid[valid >= half]
    print("序列 %d 点（%s ~ %s），残差 σ_r = %.3f" % (
        len(s), s.index[0].date(), s.index[-1].date(), sig))
    print("dev %d 点，eval %d 点（按时间切，不随机）" % (len(dev_idx), len(eval_idx)))
    print()

    # ---------- 一、候选阈值的标定（dev） ----------
    print("=== 阈值标定（dev 上按误报预算选：取满足预算的**最小**阈值）===")
    print("| 窗口 w | S_w（滚动均值实测 σ） | 预算/30天 | 阈值 | dev 实际误报/30天 | 单次最长区间 |")
    print("|---|---|---|---|---|---|")
    chosen: dict[tuple[int, int], float] = {}
    grid = np.round(np.arange(2.0, 12.01, 0.25), 2)
    for w in WINDOWS:
        sc_dev = window_score(resid, w, sig).loc[dev_idx]
        s_w = window_scale(resid, w)
        for budget in BUDGETS:
            pick = None
            for thr in grid:  # 升序：第一个满足预算的即最小可行阈值
                n = len(alarms(sc_dev, w, thr))
                rate = n * 30.0 / ((dev_idx[-1] - dev_idx[0]).total_seconds() / 86400)
                if rate <= budget:
                    pick = (thr, n, rate)
                    break
            if pick is None:
                print("| %d | %.3f | %d | 网格内无解 | — | — |" % (w, s_w, budget))
                continue
            chosen[(w, budget)] = pick[0]
            al = alarms(sc_dev, w, pick[0])
            longest = max((b - a for a, b in al), default=pd.Timedelta(0))
            print("| %d | %.3f | %d | %.2f | %.2f | %s |" % (
                w, s_w, budget, pick[0], pick[2], str(longest)))
    print()

    # ---------- 二、注入实验（先记峰值分数，再离线扫阈值） ----------
    rng = np.random.default_rng(SEED)
    span = (eval_idx[-1] - eval_idx[0])
    lo_off = pd.Timedelta(hours=BASELINE)
    usable = [t for t in eval_idx if lo_off <= t - eval_idx[0] <= span - lo_off]
    print("=== 注入实验（eval 段，每格 %d 次；记录块内峰值分数）===" % REPS)

    plan = make_plan()
    peaks: dict[tuple[float, int, int], list[float]] = {}
    point_hits: dict[tuple[float, int], int] = {}
    for amp in AMPS:
        for length in LENGTHS:
            point_hits[(amp, length)] = 0
            for w in WINDOWS:
                peaks[(amp, length, w)] = []
            for _ in range(REPS):
                start = usable[int(rng.integers(0, len(usable)))]
                if start + pd.Timedelta(hours=length) > eval_idx[-1]:
                    start = eval_idx[-1] - pd.Timedelta(hours=length + 1)
                end = start + pd.Timedelta(hours=length - 1)

                inj = s.copy()
                inj.loc[start:end] = inj.loc[start:end] + amp * sig

                frame = pd.DataFrame({COLUMN: inj})
                pts = detect_anomalies(CleanDataset(frame=frame), plan)
                if point_detected({str(p.timestamp) for p in pts}, start, end):
                    point_hits[(amp, length)] += 1

                r2, _ = segment_residual(inj)
                for w in WINDOWS:
                    sc = window_score(r2, w, sig)
                    win = sc.loc[start:end].dropna()
                    peaks[(amp, length, w)].append(float(win.max()) if len(win) else 0.0)
            print("  完成 %.1fσ / %d h" % (amp, length))

    # ---------- 三、峰值分数与 ROC ----------
    print()
    print("=== 块内峰值分数（中位数，越大越容易与噪声区分）===")
    head = "| 幅度 | 长度 |" + "".join(" w=%d 峰值 |" % w for w in WINDOWS) + " 点检器召回@7σ |"
    print(head)
    print("|---|---|" + "---|" * (len(WINDOWS) + 1))
    for amp in AMPS:
        for length in LENGTHS:
            row = "| %.1fσ | %d h |" % (amp, length)
            for w in WINDOWS:
                row += " %.2f |" % float(np.median(peaks[(amp, length, w)]))
            row += " %.2f |" % (point_hits[(amp, length)] / REPS)
            print(row)

    print()
    print("=== ROC（w=24，阈值扫 eval 段未注入数据上的误报）===")
    print("| 阈值 | 误报/30天 | 最长误报区间 | 3σ/24h 召回 | 4σ/24h 召回 | 3σ/48h 召回 |")
    print("|---|---|---|---|---|---|")
    sc_clean = window_score(resid, 24, sig).loc[eval_idx]
    days = (eval_idx[-1] - eval_idx[0]).total_seconds() / 86400
    for thr in np.round(np.arange(1.8, 3.41, 0.2), 2):
        al = alarms(sc_clean, 24, thr)
        longest = max((b - a for a, b in al), default=pd.Timedelta(0))
        r1 = np.mean([p > thr for p in peaks[(3.0, 24, 24)]])
        r2 = np.mean([p > thr for p in peaks[(4.0, 24, 24)]])
        r3 = np.mean([p > thr for p in peaks[(3.0, 48, 24)]])
        print("| %.2f | %.2f | %s | %.2f | %.2f | %.2f |" % (
            thr, len(al) * 30.0 / days, str(longest), r1, r2, r3))

    # ---------- 四、生产配置下的实测（含持续性要求）----------
    # 上面的 ROC 只用了阈值，没有含「区间至少跨 2 个周期」这条要求。
    # 生产配置两者都有，必须按生产配置重新量，否则会拿一个不存在的召回对外说。
    from mtsa import anomaly as A
    print()
    print("=== 生产配置实测（threshold=%.1f，区间 ≥%d 个周期）===" % (
        A.SEGMENT_THRESHOLD, A.SEGMENT_MIN_SPAN_PERIODS))
    print("| 幅度 | 长度 | 段级召回 | 点检器召回@7σ |")
    print("|---|---|---|---|")
    for amp in AMPS:
        for length in (24, 48, 72):
            hit_seg = hit_pt = 0
            for _ in range(REPS):
                start = usable[int(rng.integers(0, len(usable)))]
                if start + pd.Timedelta(hours=length) > eval_idx[-1]:
                    start = eval_idx[-1] - pd.Timedelta(hours=length + 1)
                end = start + pd.Timedelta(hours=length - 1)
                inj = s.copy()
                inj.loc[start:end] = inj.loc[start:end] + amp * sig
                segs = A._segment_scan(inj, 24, A.SEGMENT_THRESHOLD)
                if any(overlaps(x["start"], x["end"], start, end) for x in segs):
                    hit_seg += 1
                frame = pd.DataFrame({COLUMN: inj})
                pts = detect_anomalies(CleanDataset(frame=frame), plan)
                if point_detected({str(p.timestamp) for p in pts}, start, end):
                    hit_pt += 1
            print("| %.1fσ | %d h | %.2f | %.2f |" % (amp, length, hit_seg / REPS, hit_pt / REPS))

    seg_clean = A._segment_scan(s.loc[eval_idx[0]:eval_idx[-1]], 24, A.SEGMENT_THRESHOLD)
    print()
    print("eval 段未注入数据上的段级误报：%d 次 / %.1f 天 = %.2f 次/30 天" % (
        len(seg_clean), days, len(seg_clean) * 30.0 / days))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
