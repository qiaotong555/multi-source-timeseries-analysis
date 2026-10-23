# -*- coding: utf-8 -*-
"""T2.2 集成测试：Agent 接入 engine，且本轮只支持天气场景。

全部使用 FakeClient，不访问网络。测试重点是「接入是否真的发生」：
- 正式入口能跑通 Agent 闭环；
- 轨迹写进 result.json，报告里有轨迹章节；
- 股票场景被明确拒绝；
- 没有模型配置、没有产出时都明确失败，不生成空报告冒充成功。
"""
from __future__ import annotations

import json
import os

import pytest

import mtsa.engine as engine
from mtsa.agent_runtime import AgentRunner
from mtsa.models import AnalysisRequest

FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")
WEATHER_CSV = os.path.join(FIX, "weather_v1.csv")
STOCK_CSV = os.path.join(FIX, "stock_v1.csv")


class FakeClient:
    model = "fake-model"

    def __init__(self, decisions):
        self._decisions = list(decisions)
        self.calls: list[str] = []

    def chat_json(self, system: str, user: str) -> dict:
        self.calls.append(user)
        if not self._decisions:
            raise AssertionError("模型调用次数超出预期")
        return self._decisions.pop(0)


def _act(name: str, reason: str = "需要这类分析") -> dict:
    return {"action": name, "args": {}, "reason": reason}


def _finish() -> dict:
    return {"action": "finish", "args": {}, "reason": "已覆盖问题"}


def _patch_runner(monkeypatch, decisions):
    def fake(ctx, question, env=None, max_steps=8, allowed_tools=None, scope_mode="guided", max_extra_steps=1):
        return AgentRunner(ctx, FakeClient(decisions), question,
                           max_steps=max_steps, allowed_tools=allowed_tools,
                           scope_mode=scope_mode, max_extra_steps=max_extra_steps)
    monkeypatch.setattr(engine, "build_agent_runner", fake)


def _ensure_fixtures():
    if not (os.path.exists(WEATHER_CSV) and os.path.exists(STOCK_CSV)):
        from fixtures.gen_fixtures import main as gen_main
        gen_main()


def test_weather_agent_runs_through_engine(monkeypatch, tmp_path):
    _ensure_fixtures()
    _patch_runner(monkeypatch, [
        _act("compute_metrics"),
        _act("analyze_spectrum"),
        _act("detect_anomalies"),
        _finish(),
    ])
    out = str(tmp_path / "agent_wx")
    req = AnalysisRequest(request_id="req_agent_wx", input_path=WEATHER_CSV,
                          scene_id="weather", question="分析温度的趋势、季节性和异常",
                          output_dir=out, mode="agent")
    res = engine.run_analysis(req, verbose=False)

    assert res.status == "succeeded", res.error
    agent = res.data["agent"]
    assert agent["mode"] == "agent"
    assert agent["allowed_tools"] == [
        "compute_metrics", "analyze_spectrum", "run_stat_tests",
        "detect_anomalies", "detect_change_points",
    ]
    assert agent["tools_run"] == ["compute_metrics", "analyze_spectrum", "detect_anomalies"]
    assert agent["terminate_reason"] == "model_finished"
    assert agent["step_count"] == 3
    assert all(s["reason"] for s in agent["steps"])

    # 结果确实来自 Agent 选中的工具，而不是又跑了一遍固定流水线
    assert res.data["metrics"], "compute_metrics 的产出应进入 result.data.metrics"
    assert res.data["spectrum"], "analyze_spectrum 的产出应进入 result.data.spectrum"
    assert res.data["anomalies"], "detect_anomalies 的产出应进入 result.data.anomalies"
    assert res.data["change_points"] == [], "未选变化点，不应凭空产生变化点"
    assert res.data["tests"] == [], "未选统计检验，不应凭空产生检验结果"

    nodes = [e.node for e in res.events]
    for name in ("agent_analysis", "compute_metrics", "analyze_spectrum", "detect_anomalies"):
        assert name in nodes

    assert any("Agent 未执行以下分析工具" in w for w in res.warnings)

    result_json = os.path.join(out, "result.json")
    assert os.path.exists(result_json)
    saved = json.load(open(result_json, encoding="utf-8"))
    assert saved["data"]["agent"]["step_count"] == 3

    report_path = next(a.path for a in res.artifacts if a.kind == "report_md")
    report = open(report_path, encoding="utf-8").read()
    assert "### 3.1 Agent 执行轨迹" in report
    assert "fake-model" in report
    assert "未执行工具" in report


def test_stock_scene_is_rejected_in_agent_mode(monkeypatch, tmp_path):
    _ensure_fixtures()
    _patch_runner(monkeypatch, [_finish()])
    req = AnalysisRequest(request_id="req_agent_stock", input_path=STOCK_CSV,
                          scene_id="stock", question="分析收益与回撤",
                          output_dir=str(tmp_path / "agent_stock"), mode="agent")
    res = engine.run_analysis(req, verbose=False)
    assert res.status == "failed"
    assert res.error.code == "AGENT_SCENE_UNSUPPORTED"
    assert res.error.details["scene"] == "stock"


def test_agent_mode_without_client_fails_explicitly(monkeypatch, tmp_path):
    _ensure_fixtures()
    monkeypatch.setattr(engine, "build_agent_runner",
                        lambda *a, **k: None)
    req = AnalysisRequest(request_id="req_no_client", input_path=WEATHER_CSV,
                          scene_id="weather", question="分析温度",
                          output_dir=str(tmp_path / "no_client"), mode="agent")
    res = engine.run_analysis(req, verbose=False)
    assert res.status == "failed"
    assert res.error.code == "AGENT_UNAVAILABLE"
    assert "MTSA_LLM_BASE_URL" in res.error.details["required_env"]


def test_agent_finishing_without_tools_is_not_success(monkeypatch, tmp_path):
    _ensure_fixtures()
    _patch_runner(monkeypatch, [_finish()])
    out = str(tmp_path / "empty_agent")
    # 本用例验证「模型一个工具都不跑就结束」的失败路径；guided 会拒绝这种结束，
    # 因此显式用 free 复现原始语义。
    req = AnalysisRequest(request_id="req_empty", input_path=WEATHER_CSV,
                          scene_id="weather", question="分析温度",
                          output_dir=out, mode="agent", agent_scope="free")
    res = engine.run_analysis(req, verbose=False)
    assert res.status == "failed"
    assert res.error.code == "AGENT_NO_RESULT"
    # 失败也要留下轨迹，便于排查模型为什么直接结束
    saved = json.load(open(os.path.join(out, "result.json"), encoding="utf-8"))
    assert saved["data"]["agent"]["terminate_reason"] == "model_finished"
    assert saved["data"]["agent"]["step_count"] == 0


def test_cli_accepts_mode_and_max_steps():
    from mtsa.cli import _build_parser
    args = _build_parser().parse_args([
        "run", "--input", "x.csv", "--question", "q",
        "--mode", "agent", "--agent-max-steps", "5",
    ])
    assert args.mode == "agent"
    assert args.agent_max_steps == 5
    from mtsa.cli import _resolve_mode
    default = _build_parser().parse_args(["run", "--input", "x.csv", "--question", "q"])
    assert default.mode is None                     # 缺省由 _resolve_mode 决定
    assert _resolve_mode(default.mode, {}) == "pipeline"


def test_failed_tool_is_marked_in_result_and_report(monkeypatch, tmp_path):
    """工具失败不能让流程静默；result 与报告都要能看出「试过但失败」。"""
    _ensure_fixtures()
    # guided 下限要求 detect_anomalies 成功；它一直失败时要重试一次，
    # 重试用尽后运行时才允许结束（否则会空转到 max_steps）。
    _patch_runner(monkeypatch, [_act("compute_metrics"), _act("detect_anomalies"),
                                _act("detect_anomalies"), _finish()])

    import mtsa.tools as tools_mod
    from mtsa.tools import ToolResult
    real_run = tools_mod.run_tool

    def fake_run(name, ctx):
        if name == "detect_anomalies":
            return ToolResult(tool=name, outcome="failed", error="boom")
        return real_run(name, ctx)

    monkeypatch.setattr(tools_mod, "run_tool", fake_run)
    out = str(tmp_path / "failed_tool")
    req = AnalysisRequest(request_id="req_failed_tool", input_path=WEATHER_CSV,
                          scene_id="weather", question="分析温度异常",
                          output_dir=out, mode="agent")
    res = engine.run_analysis(req, verbose=False)

    assert res.status == "succeeded", res.error
    assert res.data["agent"]["tool_states"]["detect_anomalies"] == "failed"
    assert res.data["agent"]["tool_states"]["analyze_spectrum"] == "not_run"
    assert any("执行失败" in w for w in res.warnings)
    report_path = next(a.path for a in res.artifacts if a.kind == "report_md")
    report = open(report_path, encoding="utf-8").read()
    assert "失败工具" in report and "detect_anomalies" in report
