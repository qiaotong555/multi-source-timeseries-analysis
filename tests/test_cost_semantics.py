# -*- coding: utf-8 -*-
"""E2 成本语义：价目表严格加载、金额估算、预算前置检查。

验收重点：
1. 没有价目表就明确失败，不静默按 0 计价；
2. 金额标「估算」并带出处，且不进入任何统计结论；
3. 预算在下一次模型调用之前拦下，而不是花完再报超限；
4. 服务端不给 usage 时不估算、不猜测。
"""
from __future__ import annotations

import json
import os

import pandas as pd
import pytest

import mtsa.engine as engine
from mtsa import cost, tools
from mtsa.agent_runtime import AgentRunner
from mtsa.models import AnalysisRequest, CleanDataset
from mtsa.scenarios import stock as _stock  # noqa: F401
from mtsa.scenarios import weather as _weather  # noqa: F401
from mtsa.scenarios.base import get_adapter
from tests.helpers import make_plan, make_weather_frame


# ---------------- 价目表 ----------------

def _table_doc(**over):
    doc = {"currency": "CNY", "unit_tokens": 1000000, "input": 2.0, "output": 8.0,
           "cached_input": 0.5, "effective_date": "2026-10-01",
           "source": "测试用价目表"}
    doc.update(over)
    return doc


def _write(tmp_path, doc, name="prices.json"):
    p = tmp_path / name
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return str(p)


def test_missing_price_table_fails_with_actionable_message(tmp_path):
    missing = str(tmp_path / "nope.json")
    with pytest.raises(cost.PriceTableError) as err:
        cost.load_price_table(missing)
    msg = str(err.value)
    assert missing in msg
    assert "不存在" in msg
    assert "不做金额换算" in msg, "必须写清「没有价格就不换算」，而不是让人误以为免费"


def test_missing_path_is_not_treated_as_zero_price():
    with pytest.raises(cost.PriceTableError):
        cost.load_price_table("")


def test_unit_tokens_must_be_explicit(tmp_path):
    doc = _table_doc()
    doc.pop("unit_tokens")
    with pytest.raises(cost.PriceTableError) as err:
        cost.load_price_table(_write(tmp_path, doc))
    assert "unit_tokens" in str(err.value)


def test_source_is_required_for_reviewability(tmp_path):
    doc = _table_doc()
    doc.pop("source")
    with pytest.raises(cost.PriceTableError) as err:
        cost.load_price_table(_write(tmp_path, doc))
    assert "source" in str(err.value)


def test_negative_price_is_rejected(tmp_path):
    with pytest.raises(cost.PriceTableError):
        cost.load_price_table(_write(tmp_path, _table_doc(output=-1)))


def test_env_path_is_read_and_empty_means_unset():
    assert cost.price_table_path({"MTSA_PRICE_TABLE": " /tmp/p.json "}) == "/tmp/p.json"
    assert cost.price_table_path({}) is None
    assert cost.price_table_path({"MTSA_PRICE_TABLE": "  "}) is None


# ---------------- 金额估算 ----------------

def test_estimate_splits_input_cached_and_output(tmp_path):
    table = cost.load_price_table(_write(tmp_path, _table_doc()))
    usage = {"calls": 2, "prompt_tokens": 1_000_000, "completion_tokens": 200_000,
             "total_tokens": 1_200_000, "cached_tokens": 400_000}
    out = cost.estimate_cost(usage, table)
    # 未命中缓存 60 万 × 2.0/1M = 1.2；缓存 40 万 × 0.5/1M = 0.2；输出 20 万 × 8.0/1M = 1.6
    assert out["breakdown"] == {"input": 1.2, "cached_input": 0.2, "output": 1.6}
    assert out["total"] == 3.0
    assert out["estimated"] is True
    assert out["currency"] == "CNY"
    assert out["price"]["path"].endswith("prices.json")


def test_estimate_refuses_when_prompt_or_completion_missing(tmp_path):
    table = cost.load_price_table(_write(tmp_path, _table_doc()))
    with pytest.raises(cost.CostError):
        cost.estimate_cost({"total_tokens": 1000}, table)


def test_estimate_marks_cached_fallback(tmp_path):
    doc = _table_doc()
    doc.pop("cached_input")
    table = cost.load_price_table(_write(tmp_path, doc))
    out = cost.estimate_cost({"prompt_tokens": 100, "completion_tokens": 100,
                              "cached_tokens": 100}, table)
    assert out["cached_price_fallback"] is True
    assert out["note"].startswith("按 token 用量")


def test_cached_tokens_cannot_exceed_prompt(tmp_path):
    table = cost.load_price_table(_write(tmp_path, _table_doc()))
    out = cost.estimate_cost({"prompt_tokens": 10, "completion_tokens": 0,
                              "cached_tokens": 999}, table)
    assert out["tokens"]["cached"] == 10
    assert out["breakdown"]["input"] == 0.0


# ---------------- 预算前置检查 ----------------

class UsageClient:
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
        prompt = int(self.per_call * 0.7)
        self.usage_totals["prompt_tokens"] += prompt
        self.usage_totals["completion_tokens"] += self.per_call - prompt
        self.usage_totals["total_tokens"] += self.per_call
        return item

    def usage_summary(self):
        if not self.calls:
            return None
        return {"calls": self.calls, **self.usage_totals}


def _ctx(n: int = 400) -> tools.ToolContext:
    raw = make_weather_frame(n=n, stations=1)
    raw.index = pd.to_datetime(raw["timestamp"])
    wide = raw.drop(columns=["timestamp", "station"])
    plan = make_plan("weather", list(wide.columns))
    return tools.ToolContext(clean=CleanDataset(frame=wide), plan=plan,
                             adapter=get_adapter("weather"))


def _act(name: str, reason: str = "需要") -> dict:
    return {"action": name, "args": {}, "reason": reason}


def test_precheck_stops_before_a_call_that_would_exceed():
    """已用 100 / 上限 150，单次约 100：第二次调用前就该停，不能先花再报。"""
    client = UsageClient([_act("compute_metrics"), _act("run_stat_tests")], per_call=100)
    traj = AgentRunner(_ctx(), client, "评估这条序列是否平稳", max_tokens=150).run()
    assert traj.terminate_reason == "token_budget"
    assert client.calls == 1, "第二次调用根本不该发生"
    assert traj.budget["used"] == 100
    assert traj.budget["precheck"]["blocked"] is True
    assert traj.budget["precheck"]["reserve"] == 100
    assert traj.budget["precheck"]["basis"] == "observed_avg"


def test_first_call_cannot_be_prechecked_and_says_so():
    client = UsageClient([_act("compute_metrics")], per_call=100)
    traj = AgentRunner(_ctx(), client, "看看数据", max_tokens=50).run()
    assert client.calls == 1
    assert traj.budget["precheck"]["blocked"] is False
    assert traj.budget["precheck"]["basis"] == "unavailable"


def test_zero_limit_disables_precheck():
    client = UsageClient([_act("compute_metrics"),
                          {"action": "finish", "args": {}, "reason": "够了"}], per_call=100)
    traj = AgentRunner(_ctx(), client, "看看数据", max_tokens=0).run()
    assert traj.budget["precheck"] == {"enabled": False, "reserve": None,
                                       "basis": "unavailable", "blocked": False}


# ---------------- 端到端：金额只影响展示 ----------------

def _write_csv(tmp_path, rows: int = 400) -> str:
    path = tmp_path / "wx.csv"
    make_weather_frame(n=rows, stations=1).to_csv(path, index=False)
    return str(path)


def test_pipeline_result_is_identical_with_or_without_price_table(tmp_path):
    """金额是展示层信息：配置价目表不得改变任何指标/结论。"""
    csv = _write_csv(tmp_path)
    base = AnalysisRequest(request_id="r1", input_path=csv, scene_id="weather",
                           question="分析温度的趋势与异常", seed=7,
                           output_dir=str(tmp_path / "a"))
    with_price = base.model_copy(update={"price_table": _write(tmp_path, _table_doc()),
                                         "output_dir": str(tmp_path / "b")})
    a = engine.run_analysis(base, verbose=False)
    b = engine.run_analysis(with_price, verbose=False)
    assert a.status == "succeeded", a.error
    assert b.status == "succeeded", b.error
    assert a.data["metrics"] == b.data["metrics"]
    assert a.conclusions == b.conclusions
    assert [e.model_dump() for e in a.evidence] == [e.model_dump() for e in b.evidence]
    assert "cost" not in (b.data.get("agent") or {})


def test_agent_cost_is_estimated_and_written_to_report(tmp_path, monkeypatch):
    csv = _write_csv(tmp_path)
    decisions = [_act("compute_metrics"), {"action": "finish", "args": {}, "reason": "够了"}]

    def fake_runner(ctx, question, env=None, max_steps=8, allowed_tools=None,
                    scope_mode="guided", max_extra_steps=1, max_tokens=0):
        return AgentRunner(ctx, UsageClient(decisions), question,
                           max_steps=max_steps, allowed_tools=allowed_tools,
                           scope_mode=scope_mode, max_extra_steps=max_extra_steps,
                           max_tokens=max_tokens)

    monkeypatch.setattr(engine, "build_agent_runner", fake_runner)
    req = AnalysisRequest(request_id="r2", input_path=csv, scene_id="weather",
                          question="看看指标", seed=7, mode="agent", agent_scope="free",
                          price_table=_write(tmp_path, _table_doc()),
                          output_dir=str(tmp_path / "out"))
    res = engine.run_analysis(req, verbose=False)
    assert res.status == "succeeded", res.error
    got = res.data["agent"]["cost"]
    assert got["estimated"] is True
    assert got["total"] > 0
    assert got["currency"] == "CNY"
    report_path = [a.path for a in res.artifacts if a.path.endswith(".md")][0]
    text = open(report_path, encoding="utf-8").read()
    assert "成本（估算）" in text
    assert "不等于供应商账单" in text


def test_bad_price_table_fails_before_any_model_call(tmp_path, monkeypatch):
    csv = _write_csv(tmp_path)
    called = {"n": 0}

    def fake_runner(*a, **kw):                 # pragma: no cover - 不应被调用
        called["n"] += 1
        raise AssertionError("价目表非法时不应构造 AgentRunner")

    monkeypatch.setattr(engine, "build_agent_runner", fake_runner)
    req = AnalysisRequest(request_id="r3", input_path=csv, scene_id="weather",
                          question="看看指标", seed=7, mode="agent",
                          price_table=str(tmp_path / "missing.json"),
                          output_dir=str(tmp_path / "out3"))
    res = engine.run_analysis(req, verbose=False)
    assert res.status == "failed"
    assert res.error.code == "PRICE_TABLE_INVALID"
    assert called["n"] == 0
