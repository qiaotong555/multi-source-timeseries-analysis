# -*- coding: utf-8 -*-
"""B2 补强：金额预算「跑完再标记」+ 价目表与模型绑定。

与 test_cost_semantics.py 的分工：那边测价目表加载与 token 预算硬停，
这边测金额预算的软语义（超支不中止）与随包价目表本身是否正确。
"""
from __future__ import annotations

import glob
import os

import pytest

import mtsa.engine as engine
from mtsa import cost
from mtsa.agent_runtime import AgentRunner
from mtsa.models import AnalysisRequest
from tests.helpers import make_weather_frame

PRICES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prices")


# ---------------- 随包价目表 ----------------

def test_shipped_price_tables_load_and_hold_documented_numbers():
    """表里的数字必须与供应商页一致；写错一位就是钱算错。"""
    expected = {
        "deepseek-flash.peak.json": ("deepseek-flash", "peak", 2.0, 0.04, 8.0),
        "deepseek-flash.offpeak.json": ("deepseek-flash", "off_peak", 1.0, 0.02, 4.0),
        "deepseek-v4-pro.peak.json": ("deepseek-v4-pro", "peak", 9.0, 0.30, 27.0),
        "deepseek-v4-pro.offpeak.json": ("deepseek-v4-pro", "off_peak", 4.5, 0.15, 13.5),
    }
    files = sorted(os.path.basename(x) for x in glob.glob(os.path.join(PRICES, "*.json")))
    assert files == sorted(expected)
    for name, (model, tier, inp, cached, out) in expected.items():
        table = cost.load_price_table(os.path.join(PRICES, name))
        assert table.currency == "CNY"
        assert table.unit_tokens == 1_000_000
        assert (table.model, table.tier) == (model, tier)
        assert (table.input_per_unit, table.cached_input_per_unit,
                table.output_per_unit) == (inp, cached, out)
        assert table.source.startswith("https://")
        assert table.effective_date


def test_one_million_tokens_costs_the_listed_price():
    """1M 输入 + 1M 输出按 flash 高峰价应为 2 + 8 = 10 元。"""
    table = cost.load_price_table(os.path.join(PRICES, "deepseek-flash.peak.json"))
    out = cost.estimate_cost({"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
                             table)
    assert out["total"] == 10.0


# ---------------- 预算：只标记，不中止 ----------------

def test_check_budget_flags_overrun_without_raising():
    budget = cost.check_budget({"total": 10.0, "currency": "CNY"}, 8.0)
    assert budget["checked"] is True
    assert budget["exceeded"] is True
    assert budget["over_by"] == 2.0
    assert budget["remaining"] == 0.0
    assert budget["policy"] == "mark_only"


def test_check_budget_without_amount_is_not_checked():
    budget = cost.check_budget({"total": None, "currency": "CNY"}, 8.0)
    assert budget["checked"] is False
    assert budget["reason"] == "no_amount"


def test_env_max_cost_parsing():
    assert cost.max_cost_from_env({"MTSA_MAX_COST": "1.5"}) == 1.5
    assert cost.max_cost_from_env({}) is None
    with pytest.raises(cost.CostError):
        cost.max_cost_from_env({"MTSA_MAX_COST": "abc"})
    with pytest.raises(cost.CostError):
        cost.max_cost_from_env({"MTSA_MAX_COST": "0"})


# ---------------- 端到端 ----------------

class UsageClient:
    model = "deepseek-flash"

    def __init__(self, decisions, per_call: int = 100_000):
        self._decisions = list(decisions)
        self.per_call = per_call
        self.calls = 0
        self.usage_totals = {"prompt_tokens": 0, "completion_tokens": 0,
                             "total_tokens": 0, "cached_tokens": 0}

    def chat_json(self, system: str, user: str) -> dict:
        item = self._decisions.pop(0)
        self.calls += 1
        prompt = int(self.per_call * 0.7)
        self.usage_totals["prompt_tokens"] += prompt
        self.usage_totals["completion_tokens"] += self.per_call - prompt
        self.usage_totals["total_tokens"] += self.per_call
        return item

    def usage_summary(self):
        return {"calls": self.calls, **self.usage_totals} if self.calls else None


def _act(name: str) -> dict:
    return {"action": name, "args": {}, "reason": "需要"}


def _patch_runner(monkeypatch, model: str = "deepseek-flash"):
    def fake_runner(ctx, question, env=None, max_steps=8, allowed_tools=None,
                    scope_mode="guided", max_extra_steps=1, max_tokens=0):
        client = UsageClient([_act("compute_metrics"),
                              {"action": "finish", "args": {}, "reason": "够了"}])
        client.model = model
        return AgentRunner(ctx, client, question, max_steps=max_steps,
                           allowed_tools=allowed_tools, scope_mode=scope_mode,
                           max_extra_steps=max_extra_steps, max_tokens=max_tokens)
    monkeypatch.setattr(engine, "build_agent_runner", fake_runner)


def _run_agent(tmp_path, monkeypatch, *, max_cost=None, table="deepseek-flash.peak.json",
               model="deepseek-flash"):
    _patch_runner(monkeypatch, model=model)
    csv = tmp_path / "wx.csv"
    make_weather_frame(n=400, stations=1).to_csv(csv, index=False)
    req = AnalysisRequest(request_id="b2", input_path=str(csv), scene_id="weather",
                          question="看看指标", seed=7, mode="agent", agent_scope="free",
                          price_table=os.path.join(PRICES, table),
                          max_cost=max_cost,
                          output_dir=str(tmp_path / "out"))
    return engine.run_analysis(req, verbose=False)


def _report_text(res):
    path = [a.path for a in res.artifacts if a.path.endswith(".md")][0]
    return open(path, encoding="utf-8").read()


def test_over_budget_is_marked_and_run_still_succeeds(tmp_path, monkeypatch):
    res = _run_agent(tmp_path, monkeypatch, max_cost=0.0001)
    assert res.status == "succeeded", res.error
    got = res.data["agent"]["cost"]
    assert got["budget"]["exceeded"] is True
    assert got["budget"]["policy"] == "mark_only"
    assert any("超出预算" in w for w in res.warnings)
    text = _report_text(res)
    assert "超出预算" in text and "只标记，未中止" in text


def test_within_budget_has_no_marker(tmp_path, monkeypatch):
    res = _run_agent(tmp_path, monkeypatch, max_cost=1000)
    assert res.status == "succeeded", res.error
    assert res.data["agent"]["cost"]["budget"]["exceeded"] is False
    assert not any("超出预算" in w for w in res.warnings)
    assert "超出预算" not in _report_text(res)


def test_model_mismatch_is_flagged(tmp_path, monkeypatch):
    res = _run_agent(tmp_path, monkeypatch, model="some-other-model")
    assert res.status == "succeeded", res.error
    assert res.data["agent"]["cost"]["model_match"] is False
    assert any("不一致" in w for w in res.warnings)


def test_max_cost_without_price_table_fails_before_running(tmp_path, monkeypatch):
    called = {"n": 0}

    def fake_runner(*a, **kw):                 # pragma: no cover - 不应被调用
        called["n"] += 1
        raise AssertionError("没有价目表时不应启动模型")

    monkeypatch.setattr(engine, "build_agent_runner", fake_runner)
    csv = tmp_path / "wx.csv"
    make_weather_frame(n=200, stations=1).to_csv(csv, index=False)
    req = AnalysisRequest(request_id="b2x", input_path=str(csv), scene_id="weather",
                          question="看看指标", mode="agent", max_cost=5.0,
                          output_dir=str(tmp_path / "out2"))
    res = engine.run_analysis(req, verbose=False)
    assert res.status == "failed"
    assert res.error.code == "PRICE_TABLE_INVALID"
    assert called["n"] == 0


def test_pipeline_ignores_max_cost_and_says_so(tmp_path):
    csv = tmp_path / "wx.csv"
    make_weather_frame(n=200, stations=1).to_csv(csv, index=False)
    req = AnalysisRequest(request_id="b2p", input_path=str(csv), scene_id="weather",
                          question="分析温度的趋势", seed=7,
                          price_table=os.path.join(PRICES, "deepseek-flash.peak.json"),
                          max_cost=1.0, output_dir=str(tmp_path / "out3"))
    res = engine.run_analysis(req, verbose=False)
    assert res.status == "succeeded", res.error
    assert any("金额预算" in w and "不适用" in w for w in res.warnings)
