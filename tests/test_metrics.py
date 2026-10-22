# -*- coding: utf-8 -*-
"""指标复算测试：引擎输出必须与独立实现一致（相对误差 <= 1e-6）。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from mtsa import metrics, profiler, preprocess
from mtsa.io_loaders import build_spec
from mtsa.models import AnalysisPlan, RawDataset
from mtsa.scenarios.stock import StockAdapter, max_drawdown
from .helpers import make_stock_frame, make_weather_frame

TOL = 1e-6


def _clean(df: pd.DataFrame, plan: AnalysisPlan):
    raw = RawDataset(frame=df, spec=build_spec("mem.csv"), file_hash="0" * 40,
                     row_count=len(df), column_dtypes={c: str(t) for c, t in df.dtypes.items()})
    profile, parsed = profiler.profile_dataset(raw)
    return preprocess.clean_dataset(raw, profile, parsed, plan)


def _val(table, metric_id):
    row = next(r for r in table.rows if r.metric_id == metric_id)
    return row.value


def test_desc_stats_match_numpy(make_plan):
    df = make_weather_frame(n=800)
    plan = make_plan("weather", ["temperature"])
    clean = _clean(df, plan)
    t = metrics.compute_metrics(clean, plan)
    s = clean.frame["temperature"]
    assert abs(_val(t, "ts.desc.mean:temperature") - float(s.mean())) <= TOL * abs(float(s.mean()))
    assert abs(_val(t, "ts.desc.std:temperature") - float(s.std(ddof=1))) <= TOL * abs(float(s.std(ddof=1)))
    assert abs(_val(t, "ts.desc.median:temperature") - float(s.median())) <= TOL


def test_annualized_vol_formula(make_plan):
    df = make_weather_frame(n=1500)
    plan = make_plan("weather", ["temperature"])
    clean = _clean(df, plan)
    t = metrics.compute_metrics(clean, plan)
    rets = clean.frame["temperature"].diff()
    ppy = metrics.periods_per_year(clean.frame.index)
    expect = float(rets.std(ddof=1)) * math.sqrt(ppy)
    # 指标表里对原序列计算 std 后年化，两者口径一致
    got = _val(t, "ts.vol.annualized:temperature")
    expect_series = float(clean.frame["temperature"].std(ddof=1)) * math.sqrt(ppy)
    assert abs(got - expect_series) <= TOL * abs(expect_series)
    assert ppy > 8000  # 小时级数据每年采样点数应接近 8766


def test_pearson_matches_numpy(make_plan):
    df = make_weather_frame(n=800)
    plan = make_plan("weather", ["temperature", "humidity"])
    clean = _clean(df, plan)
    t = metrics.compute_metrics(clean, plan)
    a = clean.frame["temperature"].to_numpy()
    b = clean.frame["humidity"].to_numpy()
    expect = float(np.corrcoef(a, b)[0, 1])
    got = _val(t, "ts.corr.pearson:temperature~humidity")
    assert abs(got - expect) <= TOL


def test_max_drawdown_recompute():
    df = make_stock_frame(n=400)
    plan = AnalysisPlan(scene_id="stock", scene_confidence=1.0, question="t",
                        target_columns=["close"], analyses=[], parameters={
                            "close_column": "close", "volume_column": "volume",
                            "benchmark_column": "benchmark", "rolling_window": 20,
                            "n_folds": 5, "rf": 0.02})
    clean = _clean(df, plan)
    extra = StockAdapter().extra_metrics(clean, plan)
    px = clean.frame["close"].to_numpy(dtype=float)
    # 引擎口径：rets[0]=0，nav = cumprod(1+rets) = px / px[0]
    expect_mdd, _, _ = max_drawdown(px / px[0])
    got = next(r.value for r in extra.rows if r.metric_id == "stk.risk.max_drawdown")
    assert abs(got - expect_mdd) <= TOL * max(abs(expect_mdd), 1e-9)


def test_cumulative_return_recompute():
    df = make_stock_frame(n=300)
    plan = AnalysisPlan(scene_id="stock", scene_confidence=1.0, question="t",
                        target_columns=["close"], analyses=[], parameters={
                            "close_column": "close", "volume_column": "volume",
                            "benchmark_column": "benchmark", "rolling_window": 20,
                            "n_folds": 5, "rf": 0.02})
    clean = _clean(df, plan)
    extra = StockAdapter().extra_metrics(clean, plan)
    px = pd.to_numeric(clean.frame["close"])
    expect = float(px.iloc[-1] / px.iloc[0] - 1.0)
    got = next(r.value for r in extra.rows if r.metric_id == "stk.ret.cumulative")
    assert abs(got - expect) <= TOL * max(abs(expect), 1e-9)


def test_lag_correlation_recovers_known_lead():
    """a 领先 b 3 期，即 b[t] = a[t-3]；按约定 lag>0 表示 a 领先 b，故最优 lag 应为 +3。"""
    n = 400
    rng = np.random.default_rng(5)
    a = rng.normal(0, 1, n)
    b = np.r_[rng.normal(0, 1, 3), a[:-3]]        # b[t] = a[t-3]
    lag, c = metrics.lag_correlation(pd.Series(a), pd.Series(b), max_lag=6)
    assert lag == 3
    assert c > 0.99


def test_lag_correlation_sign_flips_when_b_leads():
    """b 领先 a 3 期（b[t] = a[t+3]）时，最优 lag 应为 -3。"""
    n = 400
    rng = np.random.default_rng(15)
    a = rng.normal(0, 1, n)
    b = np.r_[a[3:], rng.normal(0, 1, 3)]
    lag, c = metrics.lag_correlation(pd.Series(a), pd.Series(b), max_lag=6)
    assert lag == -3
    assert c > 0.99
