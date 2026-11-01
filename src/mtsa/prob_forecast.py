# -*- coding: utf-8 -*-
"""零膨胀列的事件概率预报：P(y_{t+h} > θ)、Brier 分、可靠性曲线。

## 为什么需要它

90% 区间覆盖率在零膨胀列上失去判别力。实测（`doc/预测场景化.md`）：

- R13 亚的斯亚贝巴 rainfall 整体覆盖率 **0.9948**，看着很好；
- 但测试段 384 个点里只有 **2 个**非零观测 —— 覆盖率几乎完全由「预测 0、实际也是 0」贡献；
- 同一列换成事件概率来看，气候基线给出的概率本身就该在 0 附近，区间宽窄无关紧要。

结论：零膨胀列要评的不是「区间兜没兜住」，而是「概率报得准不准」。
本模块给出三件事的定量回答：Brier 分（越低越好）、相对气候基线的技能分 BSS、
以及可靠性曲线（预测 30% 的那些时刻，实际发生率是不是 30%）。

## 模型

三个，全部只依赖训练段（`y[:t]`），不含未来信息：

1. `climatology`（**基线**）：按周期类别（小时数据按 hour-of-day）的经验事件频率，
   向全局频率做拉普拉斯平滑。它代表「不看任何近期信息」的水平。
2. `persistence`：加入「近一个周期内是否出现过事件」这个状态，
   按 horizon 步长分别估计 P(事件 | 状态)。
3. `blend`：前两者的等权平均。**这是主模型**，选它而不是「三选优」，
   是因为在评估集上挑最好的模型会把 BSS 抬高（选择性偏差）。

BSS = 1 − Brier(blend) / Brier(climatology)。BSS ≤ 0 表示没有技能，如实报告。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 降水事件阈值（mm/h）。气象上 0.1mm 是「有降水」的常用门槛。
EVENT_THRESHOLD = 0.1
RELIABILITY_BINS = 10
# 向全局频率收缩的强度：类别样本少时重点收缩，避免用 2 个样本报概率
SMOOTH_ALPHA = 5.0
MIN_STATE_POINTS = 20
# 短状态窗口（点数）。降水的时间相关以小时计，用一整个周期（24h）做状态在多数站点
# 上几乎恒为「湿」，判别力很弱；3 个点才分得开「刚下过」与「一直干」。
SHORT_STATE = 3
DEFAULT_ORIGINS = 40
TEST_FRACTION = 0.4


def _origins(n: int, horizon: int, k: int = DEFAULT_ORIGINS) -> list[int]:
    """与 `forecast._origins` 同口径：最后 30% 内等距取 origin，每个后面留满 horizon。"""
    end = n - horizon
    start = max(int(n * 0.7), 2 * horizon)
    if end <= start:
        return []
    k = max(3, min(k, end - start + 1))
    return sorted({int(x) for x in np.linspace(start, end, k)})


def _class_labels(index, period: int, n: int) -> np.ndarray:
    """周期类别：小时数据用 hour-of-day，日数据用 weekday，其他统一归一类。"""
    if not isinstance(index, pd.DatetimeIndex):
        return np.zeros(n, dtype=int)
    if period == 24:
        return index.hour.to_numpy()
    if period == 7:
        return index.dayofweek.to_numpy()
    return np.zeros(n, dtype=int)


def _smoothed_rates(wet: np.ndarray, labels: np.ndarray, n_class: int,
                    glob: float) -> np.ndarray:
    """按类别的经验频率，向全局频率平滑。"""
    out = np.full(n_class, glob, dtype=float)
    for c in range(n_class):
        m = labels == c
        cnt = int(m.sum())
        if cnt:
            out[c] = (float(wet[m].sum()) + SMOOTH_ALPHA * glob) / (cnt + SMOOTH_ALPHA)
    return out


def _recent_wet(wet: np.ndarray, window: int) -> np.ndarray:
    """位置 i 的「近 window 个点内是否出现过事件」。"""
    win = max(2, int(window))
    roll = pd.Series(wet).rolling(win, min_periods=max(2, win // 2)).max()
    return roll.fillna(0.0).to_numpy()


def _probabilities(wet: np.ndarray, cls: np.ndarray, recent_long: np.ndarray,
                   recent_short: np.ndarray, period: int, t: int,
                   horizon: int) -> tuple:
    """只用 `wet[:t]` 给出各模型在 t+0..t+horizon-1 上的概率。

    两个持续性状态：
    - `long` 用一整个周期（小时数据即 24h）。多数站点上这个状态几乎恒为「湿」，
      判别力很弱——它主要在强季节性气候里有用。
    - `short` 用 `SHORT_STATE` 个点（默认 3h）。降水的时间相关以小时计，
      这个状态才分得开「刚下过」和「一直干」。
    主模型固定为 climatology 与 **short** 的等权平均，理由是降水自相关尺度短；
    这不依赖评估结果，`long` 只作为对照报告。
    """
    train_wet = wet[:t]
    glob = float(train_wet.mean()) if t else 0.0
    n_class = int(cls.max()) + 1 if len(cls) else 1
    rates = _smoothed_rates(train_wet, cls[:t], n_class, glob)
    clim = rates[cls[t:t + horizon]]

    lo = max(period, 2)
    hi = t - horizon
    pers_long = np.full(horizon, glob, dtype=float)
    pers_short = np.full(horizon, glob, dtype=float)
    if hi > lo + MIN_STATE_POINTS:
        idx = np.arange(lo, hi)
        for recent, out in ((recent_long, pers_long), (recent_short, pers_short)):
            state_now = bool(recent[t - 1] > 0) if t >= 1 else False
            states = recent[idx] > 0
            m = states if state_now else ~states
            n_m = int(m.sum())
            if n_m < MIN_STATE_POINTS:
                continue
            for j in range(horizon):
                lab = train_wet[idx + j]
                out[j] = (float(lab[m].sum()) + SMOOTH_ALPHA * glob) / (n_m + SMOOTH_ALPHA)
    blend = np.clip(0.5 * clim + 0.5 * pers_short, 0.0, 1.0)
    return clim, pers_long, pers_short, blend


def _brier(p: np.ndarray, o: np.ndarray) -> float | None:
    if p.size == 0:
        return None
    return float(np.mean((p - o) ** 2))


# BSS 分档：单看「大于 0」会让人误以为有可用能力。0.02 的技能实际没有实用价值。
SKILL_BANDS = ((0.15, "moderate"), (0.05, "weak"), (0.0, "very_weak"))
# 事件点数低于该值时不给技能等级。1 个事件点也能算出 0.35 的 BSS，
# 那只是「测试期恰好没下雨、模型也基本没报」的产物，标成等级会误导。
MIN_EVENTS_FOR_SKILL = 100


def skill_level(bss: float | None, event_points: int | None = None) -> str:
    """把 BSS 翻译成人能读的等级。事件点不足时直接标 `insufficient_events`。"""
    if event_points is not None and event_points < MIN_EVENTS_FOR_SKILL:
        return "insufficient_events"
    if bss is None:
        return "unknown"
    if bss <= 0:
        return "none"
    for edge, name in SKILL_BANDS:
        if bss > edge:
            return name
    return "very_weak"


def _reliability(p: np.ndarray, o: np.ndarray, bins: int = RELIABILITY_BINS) -> dict:
    """可靠性：按预测概率分箱，比较箱内平均预测概率与实测频率。"""
    if p.size == 0:
        return {"bins": [], "ece": None, "bins_used": 0}
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    ece = 0.0
    used = 0
    for b in range(bins):
        m = which == b
        n_b = int(m.sum())
        if n_b == 0:
            rows.append({"bin": [round(float(edges[b]), 2), round(float(edges[b + 1]), 2)],
                         "n": 0, "mean_p": None, "freq": None})
            continue
        mean_p = float(p[m].mean())
        freq = float(o[m].mean())
        rows.append({"bin": [round(float(edges[b]), 2), round(float(edges[b + 1]), 2)],
                     "n": n_b, "mean_p": round(mean_p, 4), "freq": round(freq, 4)})
        ece += n_b * abs(mean_p - freq)
        used += 1
    return {"bins": rows, "ece": round(ece / p.size, 4), "bins_used": used}


def evaluate_events(y, index, period: int, horizon: int,
                    threshold: float = EVENT_THRESHOLD,
                    test_fraction: float = TEST_FRACTION) -> dict:
    """对单列做事件概率预报的 walk-forward 评估。

    事件定义：`y[t+h] > threshold`。测试段与区间覆盖率同口径（最后 40% 的 origin）。
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    wet = (y > threshold).astype(float)
    event_rate = float(wet.mean()) if n else 0.0
    base = {
        "threshold": threshold, "event_rate": round(event_rate, 4),
        "points": 0, "origins": 0,
    }
    if event_rate == 0.0:
        return {**base, "status": "no_events",
                "note": "整列为常数或全部低于阈值 %.3g，无法做事件预报" % threshold}
    if event_rate > 0.5:
        return {**base, "status": "threshold_too_low",
                "note": "事件率 %.2f%% 过高，阈值 %.3g 不是罕见事件，改用连续量评估更合适"
                        % (event_rate * 100, threshold)}

    origins = _origins(n, horizon)
    if len(origins) < 4:
        return {**base, "status": "insufficient_data",
                "note": "序列太短，最后 30%% 内放不下 4 个 %d 步回测窗口" % horizon}

    start = max(1, int(np.floor(len(origins) * (1.0 - test_fraction))))
    cls = _class_labels(index, period, n)
    recent_long = _recent_wet(wet, period)
    recent_short = _recent_wet(wet, SHORT_STATE)

    p_clim, p_long, p_short, p_blend, obs = [], [], [], [], []
    for t in origins[start:]:
        c, pl, ps, b = _probabilities(wet, cls, recent_long, recent_short,
                                      period, t, horizon)
        o = wet[t:t + horizon]
        if len(o) < horizon:
            continue
        p_clim.append(c)
        p_long.append(pl)
        p_short.append(ps)
        p_blend.append(b)
        obs.append(o)

    if not obs:
        return {**base, "status": "insufficient_data", "note": "测试段没有可评估的 origin"}

    O = np.concatenate(obs)
    P_C = np.concatenate(p_clim)
    P_L = np.concatenate(p_long)
    P_S = np.concatenate(p_short)
    P_B = np.concatenate(p_blend)

    b_clim = _brier(P_C, O)
    b_long = _brier(P_L, O)
    b_short = _brier(P_S, O)
    b_blend = _brier(P_B, O)
    bss = None
    if b_clim is not None and b_clim > 0 and b_blend is not None:
        bss = round(1.0 - b_blend / b_clim, 4)

    rel = _reliability(P_B, O)
    event_points = int(O.sum())
    return {
        **base,
        "status": "ok",
        "origins": len(obs),
        "points": int(O.size),
        "observed_rate": round(float(O.mean()), 4),
        "event_points": event_points,
        "brier": {
            "climatology": None if b_clim is None else round(b_clim, 6),
            "persistence_long": None if b_long is None else round(b_long, 6),
            "persistence_short": None if b_short is None else round(b_short, 6),
            "blend": None if b_blend is None else round(b_blend, 6),
        },
        "bss": bss,
        "skill": skill_level(bss, event_points),
        "bss_vs_persistence_short": (
            None if (b_short is None or b_short <= 0 or b_blend is None)
            else round(1.0 - b_blend / b_short, 4)),
        "reliability": rel["bins"],
        "ece": rel["ece"],
        "bins_used": rel["bins_used"],
        "note": ("主模型是 climatology 与 persistence_short(近 %d 点) 的等权平均；"
                 "BSS 相对 climatology。BSS≤0 表示没有技能，不构成可用的概率预报。"
                 "事件点数 event_points 太小时 BSS 不可解释——「基本不报」就能赢过基线。"
                 % SHORT_STATE),
    }
