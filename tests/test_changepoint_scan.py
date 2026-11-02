# -*- coding: utf-8 -*-
"""变化点网格扫描测试：检出、漏检、伪报、定位误差与判据参数。"""
from __future__ import annotations

import pytest

from evals import changepoint_scan as CP
from evals import split_eval as SE
from .helpers import make_weather_frame


@pytest.fixture(scope="module")
def series(tmp_path_factory):
    """一份小型单站点数据，跑一次清洗后复用（扫描只改一列的后缀偏移）。"""
    tmp = tmp_path_factory.mktemp("cp")
    path = tmp / "w.csv"
    make_weather_frame(n=2000, stations=1).to_csv(path, index=False)
    clean, plan = SE.prepare_clean(str(path), "weather", "分析温度的趋势与季节性")
    col = next(c for c in clean.frame.columns if c.startswith("pressure"))
    return clean, plan, col


def _cells(res):
    return {(c["delta"], c["position"]): c for c in res["cells"]}


def test_detects_large_shift_and_ignores_zero(series):
    clean, plan, col = series
    res = CP.scan_change_points(clean, plan, column=col,
                                positions=[0.5], deltas=[0.0, 20.0])
    cells = _cells(res)
    assert cells[(20.0, 0.5)]["detected"] is True
    assert cells[(0.0, 0.5)]["detected"] is False
    assert res["false_positive_rate_at_zero"] == 0.0


def test_small_shift_is_not_detected(series):
    clean, plan, col = series
    res = CP.scan_change_points(clean, plan, column=col,
                                positions=[0.5], deltas=[0.5])
    assert res["cells"][0]["detected"] is False
    assert res["detection_rate"] == 0.0


def test_localisation_error_within_tolerance(series):
    clean, plan, col = series
    res = CP.scan_change_points(clean, plan, column=col,
                                positions=[0.5], deltas=[20.0])
    cell = res["cells"][0]
    assert cell["detected"] is True
    assert cell["error"] is not None and cell["error"] <= CP.POSITION_TOLERANCE
    assert res["error_stats"]["p90"] <= CP.POSITION_TOLERANCE


def test_relaxing_sse_ratio_cannot_push_noise_through(series):
    """放宽 SSE 比例**不能**把噪声报成变化点（2026-11-02 起的新契约）。

    该夹具压力列的局部散布 σ≈0.72，1.0 单位的台阶只有 1.45σ。旧实现下
    sse_ratio=0.98 会把它报出来，并被当成「灵敏度」写进本条测试；
    量过之后才发现那是伪报——同一套判据在 v2 上把 1.1σ 的漂移中点
    报成 high confidence，精确率只有 0.33。
    """
    clean, plan, col = series
    for ratio in (0.7, 0.98):
        res = CP.scan_change_points(clean, plan, column=col,
                                    positions=[0.5], deltas=[1.0], sse_ratio=ratio)
        assert res["cells"][0]["detected"] is False, ratio


def test_sse_ratio_is_still_wired_below_the_gate(series, monkeypatch):
    """门槛之下，SSE 比例仍然决定候选——证明旋钮没被拆掉，只是不再能越过显著性。

    把 MIN_SNR 调到 1.0 后，1.45σ 的跳变重新成了候选：严格 0.7 不给、
    宽松 0.98 给。这说明 gate 是叠加在候选生成之上的独立一层，
    而不是把 sse_ratio 悄悄换掉。
    """
    from mtsa import anomaly

    clean, plan, col = series
    monkeypatch.setattr(anomaly, "MIN_SNR", 1.0)
    strict = CP.scan_change_points(clean, plan, column=col,
                                   positions=[0.5], deltas=[1.0], sse_ratio=0.7)
    loose = CP.scan_change_points(clean, plan, column=col,
                                  positions=[0.5], deltas=[1.0], sse_ratio=0.98)
    assert strict["cells"][0]["detected"] is False
    assert loose["cells"][0]["detected"] is True
    assert strict["sse_ratio"] == 0.7 and loose["sse_ratio"] == 0.98


def test_rates_are_aggregated_by_delta_and_position(series):
    clean, plan, col = series
    res = CP.scan_change_points(clean, plan, column=col,
                                positions=[0.4, 0.6], deltas=[0.0, 20.0])
    assert res["grid_size"] == 4
    assert res["detection_rate_by_delta"]["20.0"] == 1.0
    assert res["detection_rate_by_delta"]["0.0"] == 0.0
    assert set(res["detection_rate_by_position"]) == {"0.4", "0.6"}


def test_missing_column_raises(series):
    clean, plan, _ = series
    with pytest.raises(KeyError):
        CP.scan_change_points(clean, plan, column="no_such_column",
                              positions=[0.5], deltas=[1.0])


def test_grid_table_has_row_per_position(series):
    clean, plan, col = series
    res = CP.scan_change_points(clean, plan, column=col,
                                positions=[0.4, 0.6], deltas=[0.0, 20.0])
    table = CP.format_grid(res)
    assert table.count("\n") == 3          # 表头 + 分隔行 + 两行数据
    assert "0.40" in table and "0.60" in table
