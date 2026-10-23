# -*- coding: utf-8 -*-
"""T2.9 对照实验：指标提取与汇总口径（离线）。"""
from __future__ import annotations

import pytest

from evals import mode_compare


def _doc(conclusions=3, evidence=3, analyses=None):
    analyses = analyses or list(mode_compare.RULE_PLAN_ITEMS)
    return {
        "status": "succeeded",
        "conclusions": [{"id": i} for i in range(conclusions)],
        "evidence": [{"id": i} for i in range(evidence)],
        "data": {
            "plan": {"analyses": analyses},
            "metrics": [{}] * 10,
            "scenario_metrics": [{}] * 2,
            "tests": [{}] * 4,
            "spectrum": [{}] * 2,
            "anomalies": [{}] * 1,
            "change_points": [{}] * 1,
        },
    }


def test_pipeline_metrics_counts_and_coverage():
    m = mode_compare.pipeline_metrics(_doc(5, 4), expect=["trend", "spectrum"])
    assert m["conclusions"] == 5 and m["evidence"] == 4
    assert m["metrics"] == 10 and m["tests"] == 4
    assert m["steps"] == mode_compare.PIPELINE_ANALYSIS_UNITS
    assert m["rule_plan_coverage"] == 1.0
    assert m["need_coverage"] == 1.0
    assert m["tokens"] is None


def test_pipeline_metrics_respects_plan_slice():
    """计划只有 6 项时，规则计划覆盖率就是 6/12，不能写成 1.0。"""
    m = mode_compare.pipeline_metrics(_doc(analyses=list(mode_compare.RULE_PLAN_ITEMS[:6])),
                                      expect=[])
    assert m["rule_plan_coverage"] == 0.5
    assert m["need_coverage"] is None


def _row(case, mode, conclusions, evidence, need, wall, steps=5, tokens=None):
    return {"case": case, "mode": mode, "status": "succeeded",
            "conclusions": conclusions, "evidence": evidence, "need": ["trend"],
            "need_coverage": need, "rule_plan_coverage": 1.0 if mode == "pipeline" else 0.4,
            "wall_ms": wall, "steps": steps, "tokens": tokens}


def test_compare_reports_deltas_and_passes_when_need_kept():
    rows = [
        _row("c1", "pipeline", 10, 10, 1.0, 1000),
        _row("c1", "agent", 4, 4, 1.0, 800, steps=2,
             tokens={"total_tokens": 500}),
    ]
    r = mode_compare.compare(rows, source="mock", model="m")
    assert r["n_cases"] == 1
    assert r["pipeline"]["mean_conclusions"] == 10
    assert r["agent"]["mean_conclusions"] == 4
    assert r["deltas"]["mean_conclusions"] == -6
    assert r["agent"]["tokens_total"] == 500
    assert r["pipeline"]["tokens_total"] == 0
    assert r["passed"] is True


def test_compare_fails_when_agent_loses_need_coverage():
    rows = [
        _row("c1", "pipeline", 10, 10, 1.0, 1000),
        _row("c1", "agent", 1, 1, 0.5, 500, steps=1),
    ]
    r = mode_compare.compare(rows, source="mock", model="m")
    assert r["checks"]["agent 需求覆盖不低于 pipeline"] is False
    assert r["passed"] is False


def test_compare_ignores_unpaired_cases():
    rows = [_row("c1", "pipeline", 10, 10, 1.0, 1000)]
    r = mode_compare.compare(rows, source="mock", model="m")
    assert r["n_cases"] == 0
    assert r["passed"] is False


def test_compare_handles_missing_tokens():
    rows = [
        _row("c1", "pipeline", 10, 10, 1.0, 1000),
        _row("c1", "agent", 5, 5, 1.0, 500, steps=2, tokens=None),
    ]
    r = mode_compare.compare(rows, source="mock", model="m")
    assert r["agent"]["tokens_total"] is None
    assert r["agent"]["tokens_avg_per_run"] is None


def test_mock_case_runs_both_modes(tmp_path):
    """端到端锁住：两种模式都能从正式入口跑通，且结果文件可读。"""
    case = {"id": "wx_spectrum", "question": "温度序列的周期性与主要频率成分是什么",
            "expect": ["spectrum"]}
    p = mode_compare.run_one(case, "pipeline", str(tmp_path / "both"))
    a = mode_compare.run_one(case, "agent", str(tmp_path / "both"),
                             client_factory=lambda c: mode_compare.ScriptedClient(
                                 ["compute_metrics", "analyze_spectrum"]))
    assert p["status"] == "succeeded", p.get("error")
    assert a["status"] == "succeeded", a.get("error")
    assert p["rule_plan_coverage"] == 1.0
    assert a["need_coverage"] == 1.0
    # guided 下限含 compute_metrics，脚本又追加了 analyze_spectrum
    assert a["steps"] == 2