# -*- coding: utf-8 -*-
"""统计检验测试：平稳性、自相关、正态性、趋势、随机性与卡方。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mtsa import stats_tests
from mtsa.models import AnalysisPlan


def _plan(analyses, columns):
    return AnalysisPlan(scene_id="weather", scene_confidence=1.0, question="t",
                        target_columns=columns, analyses=analyses, parameters={})


def _clean_frame(series: pd.Series) -> pd.DataFrame:
    # 必须用 to_numpy()：直接传 Series 会与新 index 做标签对齐，导致整列 NaN
    return pd.DataFrame({"v": series.to_numpy()},
                        index=pd.date_range("2024-01-01", periods=len(series), freq="h"))


def _run(series, analyses):
    from mtsa.models import CleanDataset
    return stats_tests.run_stat_tests(CleanDataset(frame=_clean_frame(series)), _plan(analyses, ["v"]))


def _by_name(rows, name):
    return next(r for r in rows if r.name == name)


def test_adf_detects_random_walk_as_nonstationary():
    rng = np.random.default_rng(0)
    rw = np.cumsum(rng.normal(0, 1, 400))
    rows = _run(pd.Series(rw), ["stationarity"])
    adf = _by_name(rows, "ADF 单位根检验")
    assert adf.p_value > 0.05, "随机游走应不能拒绝单位根原假设"


def test_adf_detects_white_noise_as_stationary():
    rng = np.random.default_rng(1)
    wn = rng.normal(0, 1, 400)
    rows = _run(pd.Series(wn), ["stationarity"])
    adf = _by_name(rows, "ADF 单位根检验")
    assert adf.p_value < 0.05


def test_ljungbox_detects_autocorrelation():
    rng = np.random.default_rng(2)
    e = rng.normal(0, 1, 400)
    ar = np.zeros(400)
    for i in range(1, 400):
        ar[i] = 0.8 * ar[i - 1] + e[i]
    rows = _run(pd.Series(ar), ["autocorr"])
    lb = next(r for r in rows if r.name.startswith("Ljung-Box"))
    assert lb.p_value < 0.05


def test_jarque_bera_detects_non_normal():
    rng = np.random.default_rng(3)
    exp_samples = rng.exponential(2.0, 500)
    rows = _run(pd.Series(exp_samples), ["normality"])
    jb = _by_name(rows, "Jarque-Bera 正态性检验")
    assert jb.p_value < 0.05


def test_mann_kendall_detects_upward_trend():
    rng = np.random.default_rng(4)
    x = 0.01 * np.arange(400) + rng.normal(0, 0.5, 400)
    rows = _run(pd.Series(x), ["trend"])
    mk = next(r for r in rows if r.name.startswith("Mann-Kendall"))
    assert mk.p_value < 0.05 and mk.statistic > 0


def test_runs_test_on_alternating_series():
    alt = np.tile([-1.0, 1.0], 200)
    z, p = stats_tests.runs_test(alt)
    assert np.isfinite(z) and p < 0.05


def test_runs_test_on_random_series():
    rng = np.random.default_rng(6)
    z, p = stats_tests.runs_test(rng.normal(0, 1, 400))
    assert np.isfinite(z) and p > 0.05


def test_chi2_uniform_detects_skew():
    rng = np.random.default_rng(7)
    stat, p, dof = stats_tests.chi_square_uniformity(rng.exponential(1.0, 600))
    assert p < 0.05
    stat2, p2, _ = stats_tests.chi_square_uniformity(rng.uniform(0, 1, 600))
    assert p2 > 0.05


def test_short_sample_skips_instead_of_guessing():
    rows = _run(pd.Series(np.arange(5, dtype=float)), ["stationarity", "normality"])
    assert rows and rows[0].name == "样本不足"
    assert all(r.p_value is None for r in rows)
