# -*- coding: utf-8 -*-
"""零膨胀列的事件概率预报：Brier / BSS / 可靠性。

接缝：
- `prob_forecast.evaluate_events` 的状态判定（无事件、事件率过高、样本不足）；
- 概率的时间隔离（改动 origin 之后的观测，该 origin 的概率必须不变）；
- 合成阵雨序列上的技能为正（有持续性结构时模型才该学到东西）；
- `skill_level` 分档与 BSS 符号一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mtsa import prob_forecast as pf


def _shower_series(n: int = 3000, seed: int = 1) -> np.ndarray:
    """阵雨型序列：事件成段出现（3–8 小时），段间为 0。

    这种结构下「近 3 小时是否下过」有信息量，persistence 短状态才可能有用；
    若换成独立同分布的伯努利事件，任何持续性模型都不该赢。
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    i = 0
    while i < n:
        if rng.random() < 0.02:
            dur = int(rng.integers(3, 9))
            y[i:i + dur] = rng.gamma(2.0, 1.5, len(y[i:i + dur]))
            i += dur
        else:
            i += 1
    return y


def _wet_markov(n: int = 6000, seed: int = 1, p_enter: float = 0.15,
                p_leave: float = 0.22) -> np.ndarray:
    """湿润率约 40% 的两状态 Markov 序列，保证测试段有足够事件点（≥100）。"""
    rng = np.random.default_rng(seed)
    state = False
    wet = np.zeros(n, dtype=bool)
    for i in range(n):
        state = (rng.random() > p_leave) if state else (rng.random() < p_enter)
        wet[i] = state
    return np.where(wet, rng.gamma(2.0, 1.5, n), 0.0)


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="h")


def test_no_events_is_reported_not_scored():
    y = np.zeros(1000)
    r = pf.evaluate_events(y, _idx(1000), 24, 24)
    assert r["status"] == "no_events"
    assert "bss" not in r


def test_threshold_too_low_is_reported():
    y = np.ones(1000)
    r = pf.evaluate_events(y, _idx(1000), 24, 24)
    assert r["status"] == "threshold_too_low"


def test_short_series_is_reported():
    y = np.zeros(80)
    y[5:9] = 2.0
    r = pf.evaluate_events(y, _idx(80), 24, 24)
    assert r["status"] == "insufficient_data"


def test_persistence_short_beats_climatology_at_horizon_1():
    """一步预报上，短状态持续性必须比气候基线准。

    这是模型逻辑的核心：3 小时窗口的状态若连一步都赢不了基线，
    那后面所有 BSS 都是噪声。
    """
    y = _wet_markov(6000, seed=1, p_enter=0.10, p_leave=0.15)
    r = pf.evaluate_events(y, _idx(6000), 24, 1)
    assert r["status"] == "ok"
    assert r["brier"]["persistence_short"] < r["brier"]["climatology"]


def test_skill_decays_with_horizon():
    """持续性信息随 horizon 衰减：一步的技能必须高于一天。"""
    y = _wet_markov(6000, seed=1, p_enter=0.10, p_leave=0.15)
    one = pf.evaluate_events(y, _idx(6000), 24, 1)
    day = pf.evaluate_events(y, _idx(6000), 24, 24)
    assert one["bss"] > day["bss"]


def test_wet_markov_evaluation_is_well_formed():
    y = _wet_markov(6000, seed=1, p_enter=0.10, p_leave=0.15)
    r = pf.evaluate_events(y, _idx(6000), 24, 24)
    assert r["status"] == "ok"
    assert r["event_points"] >= pf.MIN_EVENTS_FOR_SKILL, r["event_points"]
    assert r["bss"] is not None and isinstance(r["bss"], float)
    assert r["brier"]["blend"] is not None and r["brier"]["blend"] >= 0
    # 可靠性分箱的样本数之和必须等于评估点数，且箱内概率均值在 [0,1]
    assert sum(b["n"] for b in r["reliability"]) == r["points"]
    for b in r["reliability"]:
        if b["n"]:
            assert 0.0 <= b["mean_p"] <= 1.0
            assert 0.0 <= b["freq"] <= 1.0


def test_sparse_events_get_no_skill_grade():
    """事件点不足时不给等级——稀疏事件下的 BSS 会被「基本不报」抬高。"""
    y = _shower_series(3000, seed=1)
    r = pf.evaluate_events(y, _idx(3000), 24, 24)
    assert r["status"] == "ok"
    assert r["event_points"] < pf.MIN_EVENTS_FOR_SKILL
    assert r["skill"] == "insufficient_events"


def test_probabilities_ignore_future():
    """概率只用 origin 及以前的数据：改掉未来，该 origin 的三个模型都不许变。"""
    y = _shower_series(2000, seed=5)
    n = len(y)
    wet = (y > pf.EVENT_THRESHOLD).astype(float)
    cls = np.zeros(n, dtype=int)
    t = 1500
    long1 = pf._recent_wet(wet, 24)
    short1 = pf._recent_wet(wet, pf.SHORT_STATE)
    before = pf._probabilities(wet, cls, long1, short1, 24, t, 24)

    y2 = y.copy()
    y2[t:] = 9.9                       # 把「未来」整段换掉
    wet2 = (y2 > pf.EVENT_THRESHOLD).astype(float)
    after = pf._probabilities(wet2, cls, pf._recent_wet(wet2, 24),
                              pf._recent_wet(wet2, pf.SHORT_STATE), 24, t, 24)
    for a, b in zip(before, after):
        assert np.array_equal(a, b)


def test_skill_level_bands():
    assert pf.skill_level(None) == "unknown"
    assert pf.skill_level(0.0) == "none"
    assert pf.skill_level(-0.1) == "none"
    assert pf.skill_level(0.01) == "very_weak"
    assert pf.skill_level(0.08) == "weak"
    assert pf.skill_level(0.2) == "moderate"
    # 事件点不足时不许给等级：1 个点也能算出 0.35 的 BSS
    assert pf.skill_level(0.35, 1) == "insufficient_events"
    assert pf.skill_level(0.35, pf.MIN_EVENTS_FOR_SKILL) == "moderate"
