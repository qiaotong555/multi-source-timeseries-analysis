# -*- coding: utf-8 -*-
"""T2.8 Agent 评估指标：纯函数口径测试（离线、不调用模型）。"""
from __future__ import annotations

import pytest

from evals import agent_eval


def _doc(executed=(), steps=(), states=None, guard=None, tokens=None,
         terminate="model_finished", status="succeeded", replay=None):
    """构造一份最小 result.json 形状。"""
    states = states or {t: "ok" for t in executed}
    agent = {
        "step_count": len(steps),
        "terminate_reason": terminate,
        "allowed_tools": list(agent_eval.WEATHER_AGENT_TOOLS),
        "steps": list(steps),
        "tool_states": states,
        "guard_stats": guard or {"violations": 0, "by_kind": {}},
        "rejects": [],
        "coverage": {
            "allowed_tools": list(agent_eval.WEATHER_AGENT_TOOLS),
            "executed_tools": list(executed),
            "tool_coverage": round(len(executed) / 5, 4),
            "planned_analyses": list(agent_eval.RULE_PLAN_ITEMS),
            "covered_analyses": sorted(agent_eval.covered_analyses(list(executed))),
            "analysis_coverage": 0.0,   # 单次覆盖率不是本测试重点
        },
        "elapsed_ms": 1200,
    }
    if tokens:
        agent["tokens"] = tokens
    if replay is not None:
        agent["replay"] = replay
    return {"status": status, "run_id": "r1",
            "datasets": [{"run_id": "r1", "mode": "agent"}],
            "data": {"agent": agent}}


def test_full_execution_covers_rule_plan():
    row = agent_eval.metrics_from_result(
        _doc(agent_eval.WEATHER_AGENT_TOOLS), expect=["trend", "spectrum"])
    assert row["tool_coverage"] == 1.0
    assert row["rule_plan_coverage"] == 1.0
    assert row["rule_plan_missing"] == []
    assert row["need_coverage"] == 1.0


def test_narrow_execution_scores_low_on_rule_plan_but_can_cover_need():
    """只跑 compute_metrics：规则计划只有 5/12，但「趋势」这个需求已覆盖。"""
    row = agent_eval.metrics_from_result(_doc(["compute_metrics"]), expect=["trend"])
    assert row["tool_coverage"] == pytest.approx(0.2)
    assert row["rule_plan_coverage"] == pytest.approx(5 / 12, abs=1e-4)
    assert row["need_coverage"] == 1.0
    assert "spectrum" in row["rule_plan_missing"]


def test_need_coverage_ignores_probe_cases():
    row = agent_eval.metrics_from_result(_doc(["compute_metrics"]), expect=[])
    assert row["need_coverage"] is None
    assert row["need_missing"] == []


def test_tool_failure_rate_and_guard_counts():
    steps = [
        {"action": "compute_metrics", "outcome": "ok"},
        {"action": "analyze_spectrum", "outcome": "failed"},
    ]
    doc = _doc(executed=["compute_metrics"], steps=steps,
               states={"compute_metrics": "ok", "analyze_spectrum": "failed"},
               guard={"violations": 2, "by_kind": {"unknown_tool": 2}})
    row = agent_eval.metrics_from_result(doc)
    assert row["tool_failure_rate"] == 0.5
    assert row["attempted_steps"] == 2
    assert row["failed_steps"] == 1
    assert row["guard_violations"] == 2
    assert row["guard_by_kind"] == {"unknown_tool": 2}
    # 失败的工具有结果对象但不算覆盖
    assert row["executed_tools"] == ["compute_metrics"]


def test_replay_flag_is_carried():
    doc = _doc(["compute_metrics"], replay={"match": True, "source": "x.json"})
    assert agent_eval.metrics_from_result(doc)["replay_ok"] is True
    doc2 = _doc(["compute_metrics"])
    assert agent_eval.metrics_from_result(doc2)["replay_ok"] is None


def test_summary_reports_token_cost_only_when_present():
    a = agent_eval.metrics_from_result(
        _doc(["compute_metrics"], tokens={"calls": 3, "prompt_tokens": 100,
                                         "completion_tokens": 20, "total_tokens": 120}))
    b = agent_eval.metrics_from_result(_doc(["compute_metrics"]))
    s = agent_eval.summarize([a, b])
    assert s["tokens_available_runs"] == 1
    assert s["tokens_total"] == 120
    assert s["tokens_avg_per_run"] == 120.0

    s2 = agent_eval.summarize([b])
    assert s2["tokens_total"] is None
    assert s2["tokens_avg_per_run"] is None


def test_summary_counts_termination_and_replay():
    a = agent_eval.metrics_from_result(
        _doc(["compute_metrics"], terminate="model_finished",
             replay={"match": True, "source": "a.json"}), expect=["trend"])
    b = agent_eval.metrics_from_result(
        _doc(["compute_metrics"], terminate="max_steps",
             replay={"match": False, "source": "b.json"}), expect=["trend"])
    s = agent_eval.summarize([a, b])
    assert s["terminate_distribution"] == {"model_finished": 1, "max_steps": 1}
    assert s["avg_steps"] == 0.0            # 两个用例都没有步骤记录
    assert s["replayable_rate"] == 0.5
    assert s["replay_pool_size"] == 2
    assert s["mean_need_coverage"] == 1.0


def test_summary_without_replay_marks_check_skipped():
    row = agent_eval.metrics_from_result(_doc(["compute_metrics"]), expect=["trend"])
    s = agent_eval.summarize([row])
    assert s["replayable_rate"] is None
    assert any("未测重放" in x for x in s["checks_skipped"])


def test_failed_run_lowers_success_rate():
    ok = agent_eval.metrics_from_result(_doc(["compute_metrics"]), expect=["trend"])
    bad = {"case": "x", "status": "failed", "steps": 0, "tool_coverage": 0.0,
           "rule_plan_coverage": 0.0, "need_coverage": 0.0, "tool_failure_rate": 0.0,
           "guard_violations": 0, "rejects": 0, "elapsed_ms": None,
           "terminate_reason": "decision_error", "replay_ok": None, "tokens": None,
           "analysis_coverage": None}
    s = agent_eval.summarize([ok, bad])
    assert s["success_rate"] == 0.5
    assert s["passed"] is False
    assert s["terminate_distribution"].get("decision_error") == 1


def test_cases_cover_thirteen_weather_questions():
    """9 个原用例 + 4 个扩容用例（长问题/缺失数据/否定表述/开放探针）。"""
    assert len(agent_eval.CASES) == 13
    probes = [c for c in agent_eval.CASES if c.get("probe")]
    assert len(probes) == 2
    ids = [c["id"] for c in agent_eval.CASES]
    assert len(ids) == len(set(ids)), "用例 id 不得重复"
    for c in agent_eval.CASES:
        assert set(c.get("expect") or []) <= set(agent_eval.RULE_PLAN_ITEMS)

def test_mock_case_end_to_end_replay(tmp_path):
    """端到端锁住：评估里的重放请求必须带 mode=agent，否则引擎会拒绝。"""
    case = {"id": "wx_trend", "question": "分析温度的趋势、季节性与极端事件",
            "expect": ["trend", "seasonal", "extreme"]}
    row = agent_eval.run_case(case, str(tmp_path),
                              client=agent_eval.ScriptedClient(["compute_metrics"]),
                              do_replay=True, verbose=False)
    assert row["status"] == "succeeded", row.get("error")
    assert row["need_coverage"] == 1.0
    assert row["rule_plan_coverage"] == pytest.approx(5 / 12, abs=1e-4)
    assert row["replay_ok"] is True
