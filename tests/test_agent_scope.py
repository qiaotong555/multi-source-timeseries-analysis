# -*- coding: utf-8 -*-
"""T2.11 范围下限：把「至少要做什么」变成确定性规则。

验收重点不是「模型更聪明」，而是**同一问题两次运行的步数不再漂移**：
下限由关键词决定，达标前不得结束，下限之外只允许追加有限步数。
"""
from __future__ import annotations

import pandas as pd
import pytest

from mtsa import tools
from mtsa.agent_runtime import (AgentRunner, SCOPE_FREE, SCOPE_GUIDED,
                                WEATHER_AGENT_TOOLS)
from mtsa.models import CleanDataset
from mtsa.observation import build_observation
from mtsa.scenarios import stock as _stock  # noqa: F401
from mtsa.scenarios import weather as _weather  # noqa: F401
from mtsa.scenarios.base import get_adapter
from mtsa.scope import infer_floor
from tests.helpers import make_plan, make_weather_frame


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


def _ctx(n: int = 600) -> tools.ToolContext:
    raw = make_weather_frame(n=n, stations=1)
    raw.index = pd.to_datetime(raw["timestamp"])
    wide = raw.drop(columns=["timestamp", "station"])
    plan = make_plan("weather", list(wide.columns))
    return tools.ToolContext(clean=CleanDataset(frame=wide), plan=plan,
                             adapter=get_adapter("weather"))


def _act(name: str, reason: str = "因为需要") -> dict:
    return {"action": name, "args": {}, "reason": reason}


def _finish(reason: str = "已完成") -> dict:
    return {"action": "finish", "args": {}, "reason": reason}


# ---------------- 下限推断（纯函数） ----------------

def test_floor_always_includes_base_metrics():
    floor = infer_floor("帮我看看这批数据", WEATHER_AGENT_TOOLS)
    assert floor.tools == ("compute_metrics",)
    assert floor.source == "rule_keywords"


def test_floor_stationarity_keywords():
    floor = infer_floor("评估这条序列是否平稳，并说明自相关结构", WEATHER_AGENT_TOOLS)
    assert floor.tools == ("compute_metrics", "run_stat_tests")
    assert "平稳" in floor.matched["run_stat_tests"]


def test_floor_anomaly_and_changepoint():
    floor = infer_floor("检查异常点和突变更点", WEATHER_AGENT_TOOLS)
    assert set(floor.tools) == {"compute_metrics", "detect_anomalies",
                                "detect_change_points"}


def test_floor_order_follows_whitelist_not_keyword_order():
    floor = infer_floor("先看突变，再看是否平稳，最后找异常", WEATHER_AGENT_TOOLS)
    assert floor.tools == ("compute_metrics", "run_stat_tests",
                           "detect_anomalies", "detect_change_points")


def test_floor_ignores_words_that_only_look_like_keywords():
    """「极端事件的频率」说的是事件次数，不是频域；不能因此加频谱工具。"""
    floor = infer_floor("统计极端高温事件的频率与强度", WEATHER_AGENT_TOOLS)
    assert floor.tools == ("compute_metrics",)


def test_floor_respects_whitelist():
    floor = infer_floor("检查异常", ["compute_metrics"])
    assert floor.tools == ("compute_metrics",)


def test_floor_is_deterministic_across_calls():
    q = "温度是否平稳，有没有异常和突变"
    first = infer_floor(q, WEATHER_AGENT_TOOLS)
    second = infer_floor(q, WEATHER_AGENT_TOOLS)
    assert first.tools == second.tools
    assert first.matched == second.matched


# ---------------- guided 运行时 ----------------

def test_finish_before_floor_is_rejected_then_completes():
    client = FakeClient([_finish("先结束"), _act("compute_metrics"),
                         _act("run_stat_tests"), _finish()])
    traj = AgentRunner(_ctx(), client, "评估这条序列是否平稳").run()
    assert traj.terminate_reason == "model_finished"
    assert traj.step_count == 2
    assert traj.rejects[0]["kind"] == "scope_incomplete"
    assert "run_stat_tests" in traj.rejects[0]["detail"]
    assert traj.scope["met"] is True
    assert traj.scope["tools"] == ["compute_metrics", "run_stat_tests"]


def test_one_extra_step_allowed_second_rejected():
    client = FakeClient([_act("compute_metrics"), _act("analyze_spectrum"),
                         _act("run_stat_tests"), _finish()])
    traj = AgentRunner(_ctx(), client, "帮我看看这批数据").run()
    assert traj.step_count == 2
    assert traj.tools_run == ["compute_metrics", "analyze_spectrum"]
    assert traj.rejects[0]["kind"] == "scope_exceeded"
    assert traj.guard_stats["by_kind"]["extra_step_budget"] == 1
    assert traj.scope["extra_used"] == 1
    # 预算用尽即结束：不再问模型，避免 3 次无意义的拒绝往返
    assert traj.terminate_reason == "budget_exhausted"
    assert len(client.calls) == 3


def test_repeated_finish_becomes_scope_unmet():
    client = FakeClient([_finish()] * 3)
    traj = AgentRunner(_ctx(), client, "评估这条序列是否平稳").run()
    assert traj.terminate_reason == "scope_unmet"
    assert traj.step_count == 0
    assert traj.scope["met"] is False


def test_failed_floor_tool_after_retry_allows_finish(monkeypatch):
    """下限工具彻底失败时放行结束：否则会把「做不到」伪装成「没做完」。"""
    from mtsa.tools import ToolResult
    real = tools.run_tool

    def fake_run(name, ctx):
        if name == "detect_change_points":
            return ToolResult(tool=name, outcome="failed", error="boom")
        return real(name, ctx)

    monkeypatch.setattr(tools, "run_tool", fake_run)
    client = FakeClient([_act("compute_metrics"), _act("detect_change_points"),
                         _act("detect_change_points"), _finish()])
    traj = AgentRunner(_ctx(), client, "找出均值突变位置").run()
    assert traj.terminate_reason == "model_finished"
    assert traj.scope["met"] is False
    assert traj.scope["missing"] == ["detect_change_points"]


def test_free_mode_has_no_floor():
    client = FakeClient([_finish()])
    traj = AgentRunner(_ctx(), client, "评估这条序列是否平稳",
                       scope_mode=SCOPE_FREE).run()
    assert traj.terminate_reason == "model_finished"
    assert traj.step_count == 0
    assert traj.scope == {"mode": SCOPE_FREE, "enforced": False}


def test_invalid_scope_mode_raises():
    with pytest.raises(ValueError):
        AgentRunner(_ctx(), FakeClient([]), "q", scope_mode="loose")


def test_trajectory_serializes_scope():
    client = FakeClient([_act("compute_metrics"), _finish()])
    traj = AgentRunner(_ctx(), client, "看看数据").run()
    d = traj.to_dict()
    assert d["scope_mode"] == SCOPE_GUIDED
    assert d["scope"]["tools"] == ["compute_metrics"]
    assert d["scope"]["met"] is True


def test_observation_shows_floor_and_missing():
    """不调用模型：直接看第一步之前的观察文本，确认下限被写进上下文。"""
    runner = AgentRunner(_ctx(), FakeClient([]), "评估这条序列是否平稳")
    text = build_observation(runner.ctx, "评估这条序列是否平稳", runner.traj,
                             runner.allowed_tools, runner.max_steps,
                             floor=runner.floor,
                             max_extra_steps=runner.max_extra_steps)
    assert "范围下限" in text
    assert "未覆盖：compute_metrics、run_stat_tests" in text
    assert "下限之外已追加：0/1" in text
