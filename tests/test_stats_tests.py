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


# ---------------- Ljung-Box 的 FFT 实现 ----------------
# 这组测试锁住「换实现不改结论」：数值必须与 statsmodels 一致，
# 异常输入必须回退，且大样本下不能退化为 O(n·lag) 的慢路径。

def test_ljungbox_fft_matches_statsmodels_numerically():
    from statsmodels.stats.diagnostic import acorr_ljungbox

    rng = np.random.default_rng(20260927)
    cases = {
        "white_noise": rng.normal(0, 1, 500),
        "random_walk": np.cumsum(rng.normal(0, 1, 500)),
        "ar1": None,
    }
    ar = np.zeros(500)
    e = rng.normal(0, 1, 500)
    for i in range(1, 500):
        ar[i] = 0.9 * ar[i - 1] + e[i]
    cases["ar1"] = ar

    for label, x in cases.items():
        for lags in (1, 5, 10):
            q1, p1 = stats_tests.ljungbox_fft(x, lags)
            lb = acorr_ljungbox(x, lags=[lags], return_df=True)
            q2 = float(lb["lb_stat"].iloc[0])
            p2 = float(lb["lb_pvalue"].iloc[0])
            assert abs(q1 - q2) / max(abs(q2), 1e-12) < 1e-9, f"{label} lag={lags} Q 不一致"
            assert abs(p1 - p2) < 1e-12, f"{label} lag={lags} p 不一致"


def test_ljungbox_fft_rejects_short_series():
    q, p = stats_tests.ljungbox_fft(np.arange(5, dtype=float), 10)
    assert np.isnan(q) and np.isnan(p)


def test_ljungbox_fft_rejects_constant_series():
    """常数列方差为零，自相关无定义，必须返回 nan 而不是 0 或崩溃。"""
    q, p = stats_tests.ljungbox_fft(np.ones(100), 5)
    assert np.isnan(q) and np.isnan(p)


def test_ljungbox_falls_back_when_fft_unusable(monkeypatch):
    """FFT 路径算不出结果时要回退 statsmodels，不能因为优化而丢结论。"""
    monkeypatch.setattr(stats_tests, "ljungbox_fft",
                        lambda x, lags: (float("nan"), float("nan")))
    vals = np.random.default_rng(1).normal(0, 1, 300)
    q, p = stats_tests._ljungbox(vals, 10)
    assert np.isfinite(q) and np.isfinite(p)


def test_run_stat_tests_ljungbox_same_with_and_without_fft(monkeypatch):
    """整条检验链上，FFT 与 statsmodels 给出的 Ljung-Box 结果必须一致。"""
    rng = np.random.default_rng(9)
    series = pd.Series(np.cumsum(rng.normal(0, 1, 800)))

    rows_fft = _run(series, ["autocorr"])
    original = stats_tests._ljungbox
    monkeypatch.setattr(stats_tests, "_ljungbox",
                        lambda vals, lags, use_fft=True: original(vals, lags, use_fft=False))
    rows_sm = _run(series, ["autocorr"])

    a = next(r for r in rows_fft if r.name.startswith("Ljung-Box"))
    b = next(r for r in rows_sm if r.name.startswith("Ljung-Box"))
    assert abs(a.statistic - b.statistic) / max(abs(b.statistic), 1e-12) < 1e-9
    assert abs(a.p_value - b.p_value) < 1e-12
    assert a.reject_null == b.reject_null


def test_ljungbox_fft_stays_fast_on_large_series():
    """10 万点应在毫秒级；阈值放宽到 1 秒，只用于拦截退化为 O(n·lag) 的实现。"""
    import time

    x = np.cumsum(np.random.default_rng(3).normal(0, 1, 100_000))
    started = time.perf_counter()
    q, _ = stats_tests.ljungbox_fft(x, 10)
    elapsed = time.perf_counter() - started
    assert np.isfinite(q)
    assert elapsed < 1.0, f"FFT 实现耗时 {elapsed:.3f}s，疑似退化为慢路径"
