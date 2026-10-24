# -*- coding: utf-8 -*-
"""一键 demo 的纯函数与边界测试（不跑真实分析）。"""
from __future__ import annotations

from scripts import demo


def _doc():
    return {
        "status": "succeeded", "run_id": "r1",
        "data": {
            "profile": {"row_count": 100, "column_count": 5,
                        "time_axis": {"start": "2024-01-01 00:00:00",
                                      "end": "2024-01-05 00:00:00",
                                      "median_interval_seconds": 3600.0}},
            "quality": {"issues": [{"code": "MISSING_VALUES"},
                                   {"code": "MISSING_VALUES"},
                                   {"code": "SAMPLING_GAP"}]},
            "anomalies": [{"column": "temperature@STA_A"},
                          {"column": "temperature@STA_A"},
                          {"column": "humidity@STA_B"}],
            "change_points": [{"column": "pressure@STA_A",
                               "timestamp": "2024-02-01 00:00:00",
                               "kind": "mean", "statistic": 8.0}],
            "agent": {"step_count": 4, "terminate_reason": "budget_exhausted",
                      "tools_run": ["compute_metrics"], "max_steps": 8,
                      "max_tokens": 8000, "tokens": {"total_tokens": 123},
                      "coverage": {"missing_tools": ["run_stat_tests"]},
                      "elapsed_ms": 10},
        },
        "conclusions": [
            {"category": "趋势", "text": "temperature@STA_A：方向上升", "evidence_ids": ["e1"]},
            {"category": "季节性", "text": "temperature@STA_A：峰值月 7 月", "evidence_ids": ["e2"]},
            {"category": "异常检测", "text": "检出 2 个异常点", "evidence_ids": ["e1", "e2"]},
            {"category": "异常检测", "text": "无证据的结论", "evidence_ids": ["missing"]},
        ],
        "evidence": [{"evidence_id": "e1"}, {"evidence_id": "e2"}],
        "warnings": [], "error": None,
    }


def _summary(agent=True):
    d = demo.summarize_result(_doc(), "agent" if agent else "pipeline", 12,
                              "artifacts/demo/agent/result.json",
                              "artifacts/demo/agent/report.md")
    return d


def _page(runs):
    return {"generated_at": "2026-10-24T00:00:00", "input": "fixtures/weather_v1.csv",
            "question": "q", "scene": "weather", "planner": "rule", "model": None,
            "command": "python scripts/demo.py", "runs": runs}


def test_summarize_counts_coverage_and_agent_fields():
    s = _summary()
    assert s["status"] == "succeeded" and s["run_id"] == "r1"
    assert (s["rows"], s["columns"]) == (100, 5)
    assert s["quality_issues"] == 3
    assert s["quality_codes"] == {"MISSING_VALUES": 2, "SAMPLING_GAP": 1}
    assert (s["conclusions"], s["evidence"]) == (4, 2)
    assert s["evidence_coverage"] == 0.75, "缺证据的结论不能算覆盖"
    assert (s["anomalies"], s["anomaly_series"]) == (3, 2)
    assert s["change_points"] == 1
    assert s["change_point_series"] == ["pressure@STA_A"]
    assert s["agent"]["steps"] == 4
    assert s["agent"]["missing_tools"] == ["run_stat_tests"]
    assert s["agent"]["tokens"]["total_tokens"] == 123
    assert len(s["conclusion_samples"]["异常检测"]) == 2
    assert s["conclusion_samples"]["变化点"] == []


def test_pipeline_summary_has_no_agent_fields():
    s = _summary(agent=False)
    assert s["agent"] is None
    md = demo.render_md(_page([s]))
    assert "| Agent 步数 | — |" in md
    assert "| Token | — |" in md


def test_render_two_modes_and_table():
    md = demo.render_md(_page([_summary(False), _summary(True)]))
    assert "| 项 | pipeline | agent |" in md
    assert "0.75" in md
    assert "4 步（budget_exhausted）" in md
    assert "未执行 run_stat_tests" in md
    assert "artifacts/demo/agent/result.json" in md


def test_render_marks_skipped_agent():
    p = _summary(False)
    md = demo.render_md(_page([p, {"mode": "agent", "skipped": True,
                                   "reason": "缺少 MTSA_LLM_API_KEY"}]))
    assert "| 状态 | succeeded | 跳过 |" in md
    assert "跳过：缺少 MTSA_LLM_API_KEY" in md


def test_main_missing_input_returns_3(tmp_path):
    assert demo.main(["--input", str(tmp_path / "nope.csv"),
                      "--mode", "pipeline"]) == 3


def test_main_agent_without_model_returns_2(monkeypatch):
    for k in demo.MODEL_KEYS:
        monkeypatch.delenv(k, raising=False)
    assert demo.main(["--input", "fixtures/weather_v1.csv",
                      "--mode", "agent"]) == 2
