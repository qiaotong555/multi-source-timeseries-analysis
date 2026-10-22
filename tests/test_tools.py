# -*- coding: utf-8 -*-
"""分析层工具化测试。

验收重点不是「能跑」，而是：
1. 工具产出与直接调用底层函数**完全一致**（否则工具化改变了行为）
2. 幂等，且不修改上下文（否则 agent 循环会互相污染）
3. 摘要不含原始数据数组（否则「模型不接触原始数值」这条底线形同虚设）
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from mtsa import anomaly as anomaly_mod
from mtsa import freq as freq_mod
from mtsa import metrics as metrics_mod
from mtsa import stats_tests as stats_mod
from mtsa import tools
from mtsa.models import CleanDataset
from mtsa.scenarios import stock as _stock  # noqa: F401  注册适配器
from mtsa.scenarios import weather as _weather  # noqa: F401
from mtsa.scenarios.base import get_adapter
from .helpers import make_plan, make_stock_frame, make_weather_frame


def _weather_ctx(n: int = 1500) -> tools.ToolContext:
    raw = make_weather_frame(n=n, stations=1)
    raw.index = pd.to_datetime(raw["timestamp"])
    wide = raw.drop(columns=["timestamp", "station"])
    clean = CleanDataset(frame=wide)
    cols = [c for c in wide.columns]
    plan = make_plan("weather", cols)
    return tools.ToolContext(clean=clean, plan=plan, adapter=get_adapter("weather"))


def _stock_ctx(n: int = 300) -> tools.ToolContext:
    raw = make_stock_frame(n=n)
    raw.index = pd.to_datetime(raw["date"])
    wide = raw.drop(columns=["date", "symbol"])
    clean = CleanDataset(frame=wide)
    plan = make_plan("stock", list(wide.columns), close_column="close")
    return tools.ToolContext(clean=clean, plan=plan, adapter=get_adapter("stock"))


def test_all_six_tools_registered():
    assert set(tools.tool_names()) == {
        "compute_metrics", "analyze_spectrum", "run_stat_tests",
        "detect_anomalies", "detect_change_points", "check_lookahead"}


def test_describe_tools_has_required_fields():
    desc = tools.describe_tools()
    assert len(desc) == 6
    for d in desc:
        assert d["name"] and d["description"] and d["when_to_use"]


def test_metrics_matches_direct_call():
    ctx = _weather_ctx()
    got = tools.run_tool("compute_metrics", ctx)
    assert got.outcome == "ok"
    expect_generic = metrics_mod.compute_metrics(ctx.clean, ctx.plan)
    expect_extra = ctx.adapter.extra_metrics(ctx.clean, ctx.plan)
    assert got.data["generic"].model_dump() == expect_generic.model_dump()
    assert got.data["scenario"].model_dump() == expect_extra.model_dump()


def test_spectrum_matches_direct_call():
    ctx = _weather_ctx()
    got = tools.run_tool("analyze_spectrum", ctx)
    expect = freq_mod.analyze_spectrum(ctx.clean, ctx.plan)
    assert [s.model_dump() for s in got.data] == [s.model_dump() for s in expect]


def test_stat_tests_matches_direct_call():
    ctx = _weather_ctx()
    got = tools.run_tool("run_stat_tests", ctx)
    expect = stats_mod.run_stat_tests(ctx.clean, ctx.plan)
    assert [r.model_dump() for r in got.data] == [r.model_dump() for r in expect]


def test_anomalies_matches_direct_call():
    ctx = _weather_ctx()
    got = tools.run_tool("detect_anomalies", ctx)
    expect = anomaly_mod.detect_anomalies(ctx.clean, ctx.plan)
    assert [a.model_dump() for a in got.data] == [a.model_dump() for a in expect]


def test_change_points_matches_direct_call():
    ctx = _weather_ctx()
    got = tools.run_tool("detect_change_points", ctx)
    expect = anomaly_mod.detect_change_points(ctx.clean, ctx.plan)
    assert [c.model_dump() for c in got.data] == [c.model_dump() for c in expect]


def test_tools_are_idempotent():
    """同一上下文重复调用，结果必须相同——否则 agent 重放轨迹会得到不同结果。"""
    for name in ("compute_metrics", "run_stat_tests", "detect_anomalies",
                 "detect_change_points", "analyze_spectrum"):
        ctx = _weather_ctx()
        first = tools.run_tool(name, ctx)
        second = tools.run_tool(name, ctx)
        assert first.summary == second.summary, name


def test_tools_do_not_mutate_context():
    ctx = _weather_ctx()
    before = ctx.clean.frame.copy(deep=True)
    plan_before = copy.deepcopy(ctx.plan.model_dump())
    for name in tools.tool_names():
        tools.run_tool(name, ctx)
    pd.testing.assert_frame_equal(ctx.clean.frame, before)
    assert ctx.plan.model_dump() == plan_before


def test_summary_has_no_raw_arrays():
    """摘要中不得出现与样本等长的数值序列。"""
    ctx = _weather_ctx()
    for name in tools.tool_names():
        res = tools.run_tool(name, ctx)
        for key, val in tools.iter_summary_values(res.summary):
            if isinstance(val, np.ndarray):
                raise AssertionError(f"{name}.{key} 是 ndarray，不应进入摘要")
            if isinstance(val, (list, tuple)):
                assert len(val) <= 20, f"{name}.{key} 过长（{len(val)} 项），疑似原始序列"


def test_spectrum_summary_excludes_fft_arrays():
    """SpectrumResult 的 freqs/power 是原始数组，绝不能进摘要。"""
    ctx = _weather_ctx()
    res = tools.run_tool("analyze_spectrum", ctx)
    assert "freqs" not in str(res.summary)
    assert "power" not in str(res.summary)


def test_lookahead_skipped_outside_stock_scene():
    ctx = _weather_ctx()
    res = tools.run_tool("check_lookahead", ctx)
    assert res.outcome == "skipped"
    assert res.reason and "stock" in res.reason


def test_lookahead_runs_on_stock_scene():
    ctx = _stock_ctx()
    res = tools.run_tool("check_lookahead", ctx)
    assert res.outcome == "ok"
    assert "issue_count" in res.summary


def test_unknown_tool_returns_failed_not_raise():
    ctx = _weather_ctx()
    res = tools.run_tool("correlation", ctx)
    assert res.outcome == "failed"
    assert res.reason == "unknown_tool"
    assert "correlation" in res.error


def test_exception_converted_to_failed_result(monkeypatch):
    """工具内部抛异常时，须转为 failed 结果，由调用方决定如何处理。"""
    def boom(*_a, **_k):
        raise RuntimeError("模拟内部错误")

    monkeypatch.setattr(metrics_mod, "compute_metrics", boom)
    ctx = _weather_ctx()
    res = tools.run_tool("compute_metrics", ctx)
    assert res.outcome == "failed"
    assert "RuntimeError" in res.error


def test_tool_result_summary_dict_is_json_safe():
    import json

    ctx = _weather_ctx()
    for name in tools.tool_names():
        res = tools.run_tool(name, ctx)
        json.dumps(res.to_summary_dict(), ensure_ascii=False)   # 不抛即通过


def test_elapsed_ms_recorded():
    ctx = _weather_ctx()
    res = tools.run_tool("detect_anomalies", ctx)
    assert res.elapsed_ms >= 0


def test_scenario_metrics_present_for_stock():
    ctx = _stock_ctx()
    res = tools.run_tool("compute_metrics", ctx)
    assert res.summary["scenario_metrics"] > 0
