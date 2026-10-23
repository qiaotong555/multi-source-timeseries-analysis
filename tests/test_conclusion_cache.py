# -*- coding: utf-8 -*-
"""评估收口缓存读取的纯函数测试。"""
from __future__ import annotations

import json

from evals import conclusion_cache


def _mode():
    return {
        "anomaly": {"scored_runs": 2, "not_run_runs": 1, "truth_per_run": 3,
                    "not_run_truth_instances": 3,
                    "point": {"precision": 0.1, "recall": 1.0, "f1": 0.18},
                    "event": {"precision": 0.2, "recall": 1.0, "f1": 0.33}},
        "changepoint": {"scored_runs": 1, "point": {"precision": 0.5, "recall": 1.0, "f1": 0.66}},
        "trend": {"accuracy": 0.5},
        "seasonal": {"accuracy": 1.0},
        "conclusion_need": {"coverage": 0.8},
        "quality": {},
    }


def _write(tmp_path, name, data):
    (tmp_path / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_load_conclusion_eval_and_flatten(tmp_path, monkeypatch):
    monkeypatch.setattr(conclusion_cache, "EVAL_DIR", str(tmp_path))
    _write(tmp_path, "conclusion_eval_results.json", {
        "generated_at": "x", "result_root": "artifacts/x", "source": "existing_results",
        "summary": {"by_mode": {"pipeline": _mode(), "agent": _mode()},
                    "agent_vs_pipeline": {"available": True,
                                          "anomaly": {"pipeline_tp_instances": 5,
                                                      "agent_tp_instances": 3,
                                                      "agent_not_run_runs": 1,
                                                      "missed_instances_due_to_not_run": 3,
                                                      "agent_distinct_missed": 0}}},
    })
    ce = conclusion_cache.load_conclusion_eval()
    assert ce["available"] is True
    assert ce["agent"]["anomaly_point"]["f1"] == 0.18
    assert ce["agent_vs_pipeline"]["anomaly"]["agent_distinct_missed"] == 0
    flat = conclusion_cache.flatten(ce, {}, {}, {}, {})
    assert flat["结论级异常点F1(agent)"] == 0.18


def test_missing_cache_is_not_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(conclusion_cache, "EVAL_DIR", str(tmp_path))
    ce = conclusion_cache.load_conclusion_eval()
    assert ce["available"] is False
    assert "不存在" in ce["reason"]


def test_load_conclusion_diff(tmp_path, monkeypatch):
    monkeypatch.setattr(conclusion_cache, "EVAL_DIR", str(tmp_path))
    _write(tmp_path, "conclusion_diff_results.json", {
        "generated_at": "x",
        "summary": {"pipeline_total": 10, "agent_total": 8, "missing_total": 2,
                    "mean_missing_per_case": 0.5, "unattributed": 0,
                    "unattributed_rate": 0.0, "attribution_complete": True},
        "category_matrix": {"周期": {"pipeline": 5, "agent": 3, "missing": 2,
                                     "tool_not_run": 2, "unattributed": 0}},
    })
    cd = conclusion_cache.load_conclusion_diff()
    assert cd["available"] is True
    assert cd["summary"]["missing_total"] == 2
    assert cd["category_matrix"]["周期"]["tool_not_run"] == 2


def test_load_short_repro_and_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(conclusion_cache, "EVAL_DIR", str(tmp_path))
    _write(tmp_path, "short_repro_results.json",
           {"identical": True, "diffs": [], "conclusions_count": [14, 14]})
    _write(tmp_path, "agent_eval_short_results.json", {
        "source": "real", "model": "deepseek-chat", "input_path": "fixtures/weather_short_v1.csv",
        "summary": {"n_runs": 13, "n_succeeded": 13, "mean_need_coverage": 1.0,
                    "replayable_rate": 1.0, "tokens_total": 50067},
    })
    sr = conclusion_cache.load_short_repro()
    sa = conclusion_cache.load_short_agent_eval()
    assert sr["identical"] is True and sr["diffs"] == []
    assert sa["available"] is True and sa["judged"] is True
    assert sa["summary"]["tokens_total"] == 50067


def test_mock_short_agent_is_not_judged(tmp_path, monkeypatch):
    monkeypatch.setattr(conclusion_cache, "EVAL_DIR", str(tmp_path))
    _write(tmp_path, "agent_eval_short_results.json", {"source": "mock", "summary": {}})
    sa = conclusion_cache.load_short_agent_eval()
    assert sa["available"] is True and sa["judged"] is False