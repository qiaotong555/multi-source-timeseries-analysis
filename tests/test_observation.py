# -*- coding: utf-8 -*-
"""T2.3 观察摘要器测试：结构、覆盖、压缩与预算。"""
from __future__ import annotations

import pandas as pd

from mtsa.agent_runtime import AgentStep, AgentTrajectory, WEATHER_AGENT_TOOLS
from mtsa.models import CleanDataset
from mtsa.observation import (MAX_TOTAL_TOKENS, build_observation, coverage_summary,
                              estimate_tokens, tool_states)
from mtsa.scenarios import weather as _weather  # noqa: F401
from mtsa.scenarios.base import get_adapter
from mtsa.tools import ToolContext, ToolResult
from .helpers import make_plan, make_weather_frame


def _ctx() -> ToolContext:
    raw = make_weather_frame(n=400, stations=1)
    raw.index = pd.to_datetime(raw["timestamp"])
    wide = raw.drop(columns=["timestamp", "station"])
    plan = make_plan("weather", list(wide.columns))
    return ToolContext(clean=CleanDataset(frame=wide), plan=plan,
                       adapter=get_adapter("weather"))


def _traj(n_steps: int = 2, executed=("compute_metrics", "detect_anomalies")) -> AgentTrajectory:
    traj = AgentTrajectory(question="q", allowed_tools=list(WEATHER_AGENT_TOOLS))
    for i in range(n_steps):
        name = executed[i % len(executed)]
        traj.steps.append(AgentStep(
            index=i, action=name, reason="理由", outcome="ok",
            summary={"count": i + 1, "result_count": i + 2,
                     "peaks": [{"column": f"c{j}", "period": 1.5} for j in range(20)]},
            elapsed_ms=10 + i, started_at="2026-10-23T00:00:00",
        ))
    for name in executed:
        traj.results[name] = ToolResult(tool=name, outcome="ok", summary={"count": 1})
    return traj


def test_contains_scene_quality_plan_and_coverage():
    ctx = _ctx()
    text = build_observation(ctx, "分析温度", _traj(), WEATHER_AGENT_TOOLS, max_steps=8)
    assert "数据概况" in text
    assert "weather" in text
    assert "计划分析项覆盖" in text
    assert "已执行" in text
    assert "尚未执行" in text


def test_no_raw_arrays_enter_context():
    """1000 个原始点必须被压成计数，而不是整段进上下文。"""
    ctx = _ctx()
    traj = _traj(1, executed=("analyze_spectrum",))
    traj.steps[0].summary = {"freqs": [round(0.001 * i, 6) for i in range(1000)]}
    text = build_observation(ctx, "q", traj, WEATHER_AGENT_TOOLS, max_steps=8)
    assert "…共 1000 项" in text
    assert len(text) < 3000


def test_token_budget_for_long_trajectory():
    ctx = _ctx()
    traj = _traj(8, executed=("compute_metrics", "analyze_spectrum", "run_stat_tests",
                              "detect_anomalies", "detect_change_points"))
    for s in traj.steps:
        s.summary = {f"k{i}": [f"value-{i}-{j}" for j in range(200)] for i in range(8)}
    text = build_observation(ctx, "q", traj, WEATHER_AGENT_TOOLS, max_steps=8)
    assert estimate_tokens(text) <= MAX_TOTAL_TOKENS


def test_earlier_steps_are_compressed():
    ctx = _ctx()
    traj = _traj(5)
    text = build_observation(ctx, "q", traj, WEATHER_AGENT_TOOLS, max_steps=8)
    assert "较早步骤（已压缩）" in text
    assert "最近步骤（含结果摘要）" in text
    assert text.count("[5]") == 1


def test_coverage_counts_executed_and_planned_analyses():
    ctx = _ctx()
    traj = _traj(1, executed=("detect_anomalies",))
    cov = coverage_summary(ctx.plan, traj, WEATHER_AGENT_TOOLS)
    assert cov["tool_coverage"] == 0.2
    assert cov["executed_tools"] == ["detect_anomalies"]
    assert "anomaly" in cov["covered_analyses"]
    assert "trend" in cov["missing_analyses"]
    assert 0.0 < cov["analysis_coverage"] < 1.0


def test_last_reject_reason_is_visible():
    ctx = _ctx()
    traj = _traj(1)
    traj.rejects.append({"kind": "invalid_action", "detail": "工具 x 不在白名单"})
    text = build_observation(ctx, "q", traj, WEATHER_AGENT_TOOLS, max_steps=8)
    assert "工具 x 不在白名单" in text


def test_observation_is_deterministic():
    ctx = _ctx()
    traj = _traj(3)
    a = build_observation(ctx, "q", traj, WEATHER_AGENT_TOOLS, max_steps=8)
    b = build_observation(ctx, "q", traj, WEATHER_AGENT_TOOLS, max_steps=8)
    assert a == b



def test_tool_states_distinguish_failed_skipped_and_not_run():
    traj = AgentTrajectory(question="q")
    traj.results["a"] = ToolResult(tool="a", outcome="ok")
    traj.results["b"] = ToolResult(tool="b", outcome="failed")
    traj.results["c"] = ToolResult(tool="c", outcome="skipped")
    states = tool_states(traj, ["a", "b", "c", "d"])
    assert states == {"a": "ok", "b": "failed", "c": "skipped", "d": "not_run"}
