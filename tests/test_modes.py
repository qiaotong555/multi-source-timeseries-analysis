# -*- coding: utf-8 -*-
"""T2.6 双模式：pipeline / agent / auto 的解析与回退。"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

import mtsa.engine as engine
from mtsa.agent_runtime import AgentRunner
from mtsa.cli import _resolve_mode
from mtsa.models import AnalysisRequest

FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")
WEATHER_CSV = os.path.join(FIX, "weather_v1.csv")
STOCK_CSV = os.path.join(FIX, "stock_v1.csv")


class FakeClient:
    model = "fake-model"

    def __init__(self):
        self.n = 0

    def chat_json(self, system, user):
        self.n += 1
        if self.n == 1:
            return {"action": "compute_metrics", "args": {}, "reason": "先算指标"}
        return {"action": "finish", "args": {}, "reason": "测试结束"}


class DummyClient:
    model = "dummy"


def _fake_build(ctx, question, env=None, max_steps=8, allowed_tools=None, scope_mode="guided", max_extra_steps=1, max_tokens=0):
    return AgentRunner(ctx, FakeClient(), question, max_steps=max_steps,
                       allowed_tools=allowed_tools,
                           scope_mode=scope_mode, max_extra_steps=max_extra_steps,
                           max_tokens=max_tokens)


def _ensure_fixtures():
    if not (os.path.exists(WEATHER_CSV) and os.path.exists(STOCK_CSV)):
        from fixtures.gen_fixtures import main as gen_main
        gen_main()


def test_resolve_mode_priority_and_env_values():
    assert _resolve_mode(None, {}) == "pipeline"
    assert _resolve_mode(None, {"MTSA_AGENT": "off"}) == "pipeline"
    assert _resolve_mode(None, {"MTSA_AGENT": "pipeline"}) == "pipeline"
    assert _resolve_mode(None, {"MTSA_AGENT": "on"}) == "agent"
    assert _resolve_mode(None, {"MTSA_AGENT": "agent"}) == "agent"
    assert _resolve_mode(None, {"MTSA_AGENT": "auto"}) == "auto"
    # 显式 CLI 优先于环境变量
    assert _resolve_mode("pipeline", {"MTSA_AGENT": "on"}) == "pipeline"
    assert _resolve_mode("agent", {"MTSA_AGENT": "off"}) == "agent"


def test_auto_without_model_falls_back_to_pipeline(tmp_path):
    _ensure_fixtures()
    with patch("mtsa.llm_client.client_from_env", return_value=None):
        req = AnalysisRequest(request_id="req_auto_nomodel", input_path=WEATHER_CSV,
                              scene_id="weather", question="分析温度",
                              output_dir=str(tmp_path / "auto_nomodel"), mode="auto")
        res = engine.run_analysis(req, verbose=False)
    assert res.status == "succeeded", res.error
    assert res.datasets[0]["mode"] == "pipeline"
    assert res.datasets[0]["mode_requested"] == "auto"
    assert "agent" not in res.data
    assert any("回退 pipeline" in w for w in res.warnings)


def test_auto_with_model_uses_agent(tmp_path):
    _ensure_fixtures()
    with patch("mtsa.llm_client.client_from_env", return_value=DummyClient()), \
         patch.object(engine, "build_agent_runner", _fake_build):
        # auto → agent 只验证「路由确实切到 agent」，脚本里没有工具，
        # 因此用 free 让「直接结束」保持合法。
        req = AnalysisRequest(request_id="req_auto_agent", input_path=WEATHER_CSV,
                              scene_id="weather", question="分析温度",
                              output_dir=str(tmp_path / "auto_agent"), mode="auto",
                              agent_scope="free")
        res = engine.run_analysis(req, verbose=False)
    assert res.status == "succeeded", res.error
    assert res.datasets[0]["mode"] == "agent"
    assert "agent" in res.data


def test_auto_on_stock_falls_back_to_pipeline(tmp_path):
    """模型可用但场景不支持 Agent 时，auto 不应报错，而应回退 pipeline。"""
    _ensure_fixtures()
    with patch("mtsa.llm_client.client_from_env", return_value=DummyClient()):
        req = AnalysisRequest(request_id="req_auto_stock", input_path=STOCK_CSV,
                              scene_id="stock", question="计算收益与回撤",
                              output_dir=str(tmp_path / "auto_stock"), mode="auto")
        res = engine.run_analysis(req, verbose=False)
    assert res.status == "succeeded", res.error
    assert res.datasets[0]["mode"] == "pipeline"
    assert "agent" not in res.data
    assert any("不是天气" in w for w in res.warnings)
