# -*- coding: utf-8 -*-
"""T2.13 token 预算：到限就停，且停得明白。

验收重点：
1. 超限必须给出 `token_budget`，不能伪装成 `model_finished`；
2. 不设限时行为不变；
3. 服务端不给 usage 时标注「未生效」，不估算、不猜测；
4. 预算只约束模型调用，不丢已选中的工具结果。
"""
from __future__ import annotations

import pandas as pd

from mtsa import tools
from mtsa.agent_runtime import AgentRunner
from mtsa.models import CleanDataset
from mtsa.scenarios import stock as _stock  # noqa: F401
from mtsa.scenarios import weather as _weather  # noqa: F401
from mtsa.scenarios.base import get_adapter
from tests.helpers import make_plan, make_weather_frame


class UsageClient:
    """每次调用累加固定 token，模拟服务端返回 usage。"""

    model = "fake-usage"

    def __init__(self, decisions, per_call: int = 100):
        self._decisions = list(decisions)
        self.per_call = per_call
        self.calls = 0
        self.usage_totals = {"prompt_tokens": 0, "completion_tokens": 0,
                             "total_tokens": 0, "cached_tokens": 0}

    def chat_json(self, system: str, user: str) -> dict:
        if not self._decisions:
            raise AssertionError("模型调用次数超出预期")
        item = self._decisions.pop(0)
        self.calls += 1
        self.usage_totals["total_tokens"] += self.per_call
        return item

    def usage_summary(self):
        if not self.calls:
            return None
        return {"calls": self.calls, **self.usage_totals}


class NoUsageClient:
    """服务端不返回 usage 的客户端（现有 FakeClient 就是这种）。"""

    model = "fake-no-usage"

    def __init__(self, decisions):
        self._decisions = list(decisions)
        self.calls = 0

    def chat_json(self, system: str, user: str) -> dict:
        if not self._decisions:
            raise AssertionError("模型调用次数超出预期")
        self.calls += 1
        return self._decisions.pop(0)


def _ctx(n: int = 400) -> tools.ToolContext:
    raw = make_weather_frame(n=n, stations=1)
    raw.index = pd.to_datetime(raw["timestamp"])
    wide = raw.drop(columns=["timestamp", "station"])
    plan = make_plan("weather", list(wide.columns))
    return tools.ToolContext(clean=CleanDataset(frame=wide), plan=plan,
                             adapter=get_adapter("weather"))


def _act(name: str, reason: str = "需要") -> dict:
    return {"action": name, "args": {}, "reason": reason}


def test_budget_stops_before_next_model_call():
    client = UsageClient([_act("compute_metrics"), _act("run_stat_tests")], per_call=100)
    traj = AgentRunner(_ctx(), client, "评估这条序列是否平稳", max_tokens=150).run()
    assert traj.terminate_reason == "token_budget"
    assert client.calls == 2                       # 刚好两次调用后停，没有第三次
    assert traj.step_count == 2                    # 超限前选中的工具仍执行完
    assert traj.budget["used"] == 200
    assert traj.budget["limit"] == 150
    assert traj.budget["exhausted"] is True
    assert traj.budget["remaining"] == 0
    assert traj.budget["unavailable"] is False


def test_budget_stops_after_one_step():
    client = UsageClient([_act("compute_metrics")], per_call=100)
    traj = AgentRunner(_ctx(), client, "看看数据", max_tokens=50).run()
    assert traj.terminate_reason == "token_budget"
    assert client.calls == 1
    assert traj.step_count == 1
    assert traj.budget["remaining"] == 0


def test_zero_limit_keeps_previous_behavior():
    client = UsageClient([_act("compute_metrics"),
                          {"action": "finish", "args": {}, "reason": "够了"}], per_call=100)
    traj = AgentRunner(_ctx(), client, "看看数据", max_tokens=0).run()
    assert traj.terminate_reason == "model_finished"
    assert traj.budget["enforced"] is False
    assert traj.budget["limit"] == 0
    assert traj.budget["remaining"] is None
    assert traj.budget["used"] == 200


def test_missing_usage_is_marked_unavailable_and_does_not_stop():
    client = NoUsageClient([_act("compute_metrics"),
                            {"action": "finish", "args": {}, "reason": "够了"}])
    traj = AgentRunner(_ctx(), client, "看看数据", max_tokens=10).run()
    assert traj.terminate_reason == "model_finished"
    assert traj.budget["enforced"] is True
    assert traj.budget["unavailable"] is True
    assert traj.budget["used"] is None
    assert traj.budget["exhausted"] is False


def test_budget_is_serialized():
    client = UsageClient([_act("compute_metrics")], per_call=100)
    d = AgentRunner(_ctx(), client, "看看数据", max_tokens=50).run().to_dict()
    assert d["budget"] == {"limit": 50, "used": 100, "remaining": 0,
                           "enforced": True, "exhausted": True,
                           "unavailable": False}


def test_finish_on_last_call_is_not_reported_as_budget_cut():
    """模型在超限的那次调用里自己结束：这是正常结束，不该说被预算截断。"""
    client = UsageClient([{"action": "finish", "args": {}, "reason": "结束了"}], per_call=100)
    # guided 会先拒绝「下限未覆盖」的结束；这里要测的是「结束与预算同一次调用」
    traj = AgentRunner(_ctx(), client, "看看数据", max_tokens=10,
                       scope_mode="free").run()
    assert traj.terminate_reason == "model_finished"
    assert traj.budget["exhausted"] is False
    assert traj.budget["used"] == 100
