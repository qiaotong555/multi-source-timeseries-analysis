# -*- coding: utf-8 -*-
"""Agent 运行时测试（全离线，不调用真实模型）。

验收重点：流程长度必须由决策序列决定，而不是由代码固定；
四种终止条件各自可达；非法动作被拒绝且不留痕到结果里。
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from mtsa import tools
from mtsa.agent_runtime import (AgentRunner, DEFAULT_MAX_STEPS, MAX_CONSECUTIVE_REJECTS,
                                MAX_DECISION_ERRORS, WEATHER_AGENT_TOOLS,
                                build_agent_runner)
from mtsa.llm_client import LLMError
from mtsa.models import CleanDataset
from mtsa.scenarios import stock as _stock  # noqa: F401
from mtsa.scenarios import weather as _weather  # noqa: F401
from mtsa.scenarios.base import get_adapter
from .helpers import make_plan, make_weather_frame


class FakeClient:
    """按预设序列返回决策；序列耗尽即为「多调了模型」，测试应立即失败。"""

    model = "fake-model"

    def __init__(self, decisions):
        self._decisions = list(decisions)
        self.calls: list[str] = []

    def chat_json(self, system: str, user: str) -> dict:
        self.calls.append(user)
        if not self._decisions:
            raise AssertionError("模型调用次数超出预期")
        item = self._decisions.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _ctx(n: int = 1200) -> tools.ToolContext:
    raw = make_weather_frame(n=n, stations=1)
    raw.index = pd.to_datetime(raw["timestamp"])
    wide = raw.drop(columns=["timestamp", "station"])
    plan = make_plan("weather", list(wide.columns))
    return tools.ToolContext(clean=CleanDataset(frame=wide), plan=plan,
                             adapter=get_adapter("weather"))


def _act(name: str, reason: str = "因为需要") -> dict:
    return {"action": name, "args": {}, "reason": reason}


def test_finish_right_after_one_tool():
    client = FakeClient([_act("compute_metrics"), {"action": "finish", "args": {}, "reason": "已足够"}])
    traj = AgentRunner(_ctx(), client, "看图", scope_mode="free").run()
    assert traj.terminate_reason == "model_finished"
    assert traj.step_count == 1
    assert traj.tools_run == ["compute_metrics"]


def test_trajectory_length_follows_decisions():
    """同一 runner 配置下，决策不同则步数不同——这是「流程由模型决定」的证据。"""
    short = AgentRunner(_ctx(), FakeClient([
        _act("detect_anomalies"), {"action": "finish", "args": {}, "reason": "ok"}]),
        "q", scope_mode="free").run()
    long = AgentRunner(_ctx(), FakeClient([
        _act("compute_metrics"), _act("run_stat_tests"), _act("detect_anomalies"),
        _act("detect_change_points"), {"action": "finish", "args": {}, "reason": "ok"}]),
        "q", scope_mode="free").run()
    assert short.step_count == 1
    assert long.step_count == 4
    assert short.step_count != long.step_count


def test_max_steps_terminates():
    client = FakeClient([_act("compute_metrics"), _act("analyze_spectrum"),
                         _act("run_stat_tests")])
    traj = AgentRunner(_ctx(), client, "q", scope_mode="free", max_steps=2).run()
    assert traj.terminate_reason == "max_steps"
    assert traj.step_count == 2


def test_repeated_tool_is_rejected_then_idle_loop_stops():
    client = FakeClient([_act("compute_metrics")] + [_act("compute_metrics")] * 5)
    traj = AgentRunner(_ctx(), client, "q", scope_mode="free").run()
    assert traj.terminate_reason == "idle_loop"
    assert traj.step_count == 1                      # 只执行了一次，其余全被拒
    assert len(traj.rejects) == MAX_CONSECUTIVE_REJECTS


def test_unknown_tool_rejected():
    client = FakeClient([_act("correlation"),
                         {"action": "finish", "args": {}, "reason": "改主意"}])
    traj = AgentRunner(_ctx(), client, "q", scope_mode="free").run()
    assert traj.terminate_reason == "model_finished"
    assert traj.step_count == 0
    assert traj.rejects[0]["kind"] == "invalid_action"


def test_args_are_rejected_not_silently_ignored():
    """模型传了参数却被静默忽略，会让它误以为参数生效——必须拒绝并告知。"""
    client = FakeClient([{"action": "run_stat_tests", "args": {"columns": ["temperature"]},
                          "reason": "只看温度"},
                         {"action": "finish", "args": {}, "reason": "结束"}])
    traj = AgentRunner(_ctx(), client, "q", scope_mode="free").run()
    assert traj.step_count == 0
    assert "不接受参数" in traj.rejects[0]["detail"]


def test_empty_reason_is_decision_error():
    client = FakeClient([{"action": "compute_metrics", "args": {}, "reason": ""},
                         {"action": "compute_metrics", "args": {}, "reason": ""}])
    traj = AgentRunner(_ctx(), client, "q", scope_mode="free").run()
    assert traj.terminate_reason == "decision_error"
    assert len(traj.rejects) == MAX_DECISION_ERRORS


def test_malformed_output_is_decision_error():
    client = FakeClient([{"args": {}}, {"nope": 1}])
    traj = AgentRunner(_ctx(), client, "q", scope_mode="free").run()
    assert traj.terminate_reason == "decision_error"


def test_llm_error_counts_as_decision_error():
    client = FakeClient([LLMError("网络超时"), LLMError("网络超时")])
    traj = AgentRunner(_ctx(), client, "q", scope_mode="free").run()
    assert traj.terminate_reason == "decision_error"
    assert "网络超时" in traj.rejects[0]["detail"]


def test_rejects_reset_after_progress():
    """有实质进展后清零拒绝计数，避免偶发无效动作累计成误判空转。"""
    client = FakeClient([
        _act("nope1"), _act("nope2"),                 # 两次无效
        _act("compute_metrics"),                       # 有进展，清零
        _act("nope3"), _act("nope4"),                 # 再两次无效
        {"action": "finish", "args": {}, "reason": "完"},
    ])
    traj = AgentRunner(_ctx(), client, "q", scope_mode="free").run()
    assert traj.step_count == 1
    assert traj.terminate_reason == "model_finished"


def test_skipped_tool_counts_as_executed():
    """非股票场景的 check_lookahead 返回 skipped，仍算已执行，不应被重复选择。"""
    client = FakeClient([_act("check_lookahead"),
                         {"action": "finish", "args": {}, "reason": "已看过"}])
    traj = AgentRunner(_ctx(), client, "q", scope_mode="free").run()
    assert traj.step_count == 1
    assert traj.steps[0].outcome == "skipped"


def test_user_prompt_contains_previous_results():
    """第二步的上下文里必须能看到第一步的结果摘要，否则不构成观察—决策闭环。"""
    client = FakeClient([_act("detect_anomalies"), {"action": "finish", "args": {}, "reason": "ok"}])
    AgentRunner(_ctx(), client, "找异常", scope_mode="free").run()
    second = client.calls[1]
    assert "detect_anomalies" in second
    assert "count" in second or "by_column" in second
    assert "尚未执行" in second


def test_trajectory_serializable():
    client = FakeClient([_act("compute_metrics"), {"action": "finish", "args": {}, "reason": "ok"}])
    traj = AgentRunner(_ctx(), client, "q", scope_mode="free").run()
    blob = json.dumps(traj.to_dict(), ensure_ascii=False)
    assert "compute_metrics" in blob
    assert traj.to_dict()["terminate_reason"] == "model_finished"


def test_runner_requires_client():
    with pytest.raises(ValueError):
        AgentRunner(_ctx(), None, "q")


def test_build_runner_returns_none_without_config():
    assert build_agent_runner(_ctx(), "q", env={}) is None


def test_default_max_steps_covers_all_tools():
    assert DEFAULT_MAX_STEPS >= len(tools.tool_names())


# ---------------- 天气场景白名单（T2.2） ----------------

def test_weather_whitelist_excludes_stock_tool():
    """股票专属工具必须在天气 Agent 里被拒，而不是执行后返回 skipped。"""
    client = FakeClient([_act("check_lookahead"),
                         {"action": "finish", "args": {}, "reason": "结束"}])
    traj = AgentRunner(_ctx(), client, "q", scope_mode="free", allowed_tools=WEATHER_AGENT_TOOLS).run()
    assert traj.step_count == 0
    assert "白名单" in traj.rejects[0]["detail"]
    assert "check_lookahead" in traj.rejects[0]["detail"]


def test_prompt_only_lists_allowed_tools():
    client = FakeClient([{"action": "finish", "args": {}, "reason": "结束"}])
    AgentRunner(_ctx(), client, "q", scope_mode="free", allowed_tools=WEATHER_AGENT_TOOLS).run()
    system_or_user = client.calls[0]
    assert "check_lookahead" not in system_or_user
    assert "compute_metrics" in system_or_user


def test_trajectory_records_whitelist_and_step_time():
    client = FakeClient([_act("compute_metrics"),
                         {"action": "finish", "args": {}, "reason": "ok"}])
    traj = AgentRunner(_ctx(), client, "q", scope_mode="free", allowed_tools=WEATHER_AGENT_TOOLS).run()
    d = traj.to_dict()
    assert d["allowed_tools"] == list(WEATHER_AGENT_TOOLS)
    assert d["steps"][0]["started_at"], "每一步必须记录开始时间，供 events/重放使用"


def test_unknown_whitelist_entry_raises():
    with pytest.raises(ValueError):
        AgentRunner(_ctx(), FakeClient([]), "q", allowed_tools=["compute_metrics", "nope"])


def test_empty_whitelist_raises():
    with pytest.raises(ValueError):
        AgentRunner(_ctx(), FakeClient([]), "q", allowed_tools=[])


def test_build_runner_passes_whitelist():
    env = {"MTSA_LLM_BASE_URL": "http://example.invalid/v1", "MTSA_LLM_API_KEY": "k",
           "MTSA_LLM_MODEL": "fake"}
    runner = build_agent_runner(_ctx(), "q", env=env, allowed_tools=WEATHER_AGENT_TOOLS)
    assert runner is not None
    assert runner.allowed_tools == WEATHER_AGENT_TOOLS


def test_tool_failure_does_not_break_loop(monkeypatch):
    """单个工具失败要变成 failed 结果回灌模型，而不是让整次运行崩掉。"""
    from mtsa.tools import ToolResult
    real = tools.run_tool

    def fake_run(name, ctx):
        if name == "detect_anomalies":
            return ToolResult(tool=name, outcome="failed", error="boom")
        return real(name, ctx)

    monkeypatch.setattr(tools, "run_tool", fake_run)
    client = FakeClient([_act("detect_anomalies"), _act("compute_metrics"),
                         {"action": "finish", "args": {}, "reason": "换一个工具已完成"}])
    traj = AgentRunner(_ctx(), client, "q", scope_mode="free", allowed_tools=WEATHER_AGENT_TOOLS).run()
    assert traj.steps[0].outcome == "failed"
    assert traj.steps[0].error == "boom"
    assert traj.tools_run == ["compute_metrics"]
    assert traj.terminate_reason == "model_finished"


# ---------------- T2.4 安全边界 / T2.7 失败恢复 ----------------

def test_failed_tool_can_retry_once_and_succeed(monkeypatch):
    from mtsa.tools import ToolResult
    real = tools.run_tool
    state = {"n": 0}

    def fake_run(name, ctx):
        if name == "detect_anomalies":
            state["n"] += 1
            if state["n"] == 1:
                return ToolResult(tool=name, outcome="failed", error="transient")
        return real(name, ctx)

    monkeypatch.setattr(tools, "run_tool", fake_run)
    client = FakeClient([_act("detect_anomalies"), _act("detect_anomalies"),
                         {"action": "finish", "args": {}, "reason": "重试成功"}])
    traj = AgentRunner(_ctx(), client, "q", scope_mode="free", allowed_tools=WEATHER_AGENT_TOOLS).run()
    assert [s.outcome for s in traj.steps] == ["failed", "ok"]
    assert traj.tools_run == ["detect_anomalies"]           # 去重
    assert traj.guard_stats["retries"]["detect_anomalies"] == 1


def test_locked_parameter_is_rejected_in_runtime():
    client = FakeClient([
        {"action": "run_stat_tests", "args": {"anomaly_threshold": 5.0},
         "reason": "想调阈值"},
        {"action": "finish", "args": {}, "reason": "结束"},
    ])
    traj = AgentRunner(_ctx(), client, "q", scope_mode="free", allowed_tools=WEATHER_AGENT_TOOLS).run()
    assert traj.step_count == 0
    assert traj.rejects[0]["reason_kind"] == "locked_parameter"
    assert "标定参数" in traj.rejects[0]["detail"]


def test_guard_stats_and_events_are_serialized():
    client = FakeClient([_act("not_a_tool"),
                         {"action": "finish", "args": {}, "reason": "结束"}])
    traj = AgentRunner(_ctx(), client, "q", scope_mode="free", allowed_tools=WEATHER_AGENT_TOOLS).run()
    d = traj.to_dict()
    assert d["guard_stats"]["violations"] == 1
    assert d["guard_stats"]["by_kind"]["tool_not_allowed"] == 1
    assert d["guard_events"][0]["kind"] == "tool_not_allowed"
    assert d["guard_stats"]["policy"]["locked_parameters"]
