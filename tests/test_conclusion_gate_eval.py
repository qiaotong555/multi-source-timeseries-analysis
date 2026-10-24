# -*- coding: utf-8 -*-
"""方案 2 门禁的纯函数测试：事件口径、未执行护栏、Wilson 区间、报告渲染。"""
from __future__ import annotations

from evals import conclusion_gate_eval as G


def _labels():
    return [
        {"column": "temperature", "station": "STA_A", "kind": "anomaly",
         "event_id": "t_A_c0", "timestamp": "2024-01-01 00:00:00"},
        {"column": "temperature", "station": "STA_A", "kind": "anomaly",
         "event_id": "t_A_c0", "timestamp": "2024-01-01 01:00:00"},
        {"column": "temperature", "station": "STA_A", "kind": "anomaly",
         "event_id": "t_A_c0", "timestamp": "2024-01-01 02:00:00"},
        {"column": "pressure", "station": "STA_B", "kind": "anomaly",
         "event_id": "p_B_p0", "timestamp": "2024-02-01 10:00:00"},
        {"column": "pressure", "station": "STA_B", "kind": "level_shift_start",
         "timestamp": "2024-03-01 00:00:00"},
    ]


def _row(case, mode, state, detected):
    return {"case": case, "mode": mode,
            "anomaly": {"state": state, "point": None, "event": None},
            "_detected_events": detected}


def test_truth_events_group_by_event_id():
    events = G._truth_events(_labels())
    assert len(events) == 2, "3 个持续段点 + 1 个孤立点应合并为 2 个事件"
    by_id = {e["event_id"]: e for e in events}
    assert by_id["t_A_c0"]["count"] == 3
    assert by_id["t_A_c0"]["series"] == "temperature@STA_A"
    assert by_id["t_A_c0"]["start"].startswith("2024-01-01T00:00")
    assert by_id["t_A_c0"]["end"].startswith("2024-01-01T02:00")


def test_detected_events_merge_within_three_hours():
    doc = {"data": {"anomalies": [
        {"column": "temperature@STA_A", "timestamp": "2024-01-01 00:00:00"},
        {"column": "temperature@STA_A", "timestamp": "2024-01-01 01:00:00"},
        {"column": "temperature@STA_A", "timestamp": "2024-01-01 04:00:00"},
        {"column": "temperature@STA_A", "timestamp": "2024-01-01 09:00:00"},
    ]}}
    events = G._detected_events(doc)
    assert len(events) == 2, "3 小时内合并，超过 3 小时切分"
    assert events[0]["count"] == 3
    assert events[1]["start"].startswith("2024-01-01 09:00")


def test_event_metrics_counts_not_run_in_recall_denominator():
    truth = [{"series": "temperature@STA_A", "start": "2024-01-01T00:00:00",
              "end": "2024-01-01T02:00:00"}]
    labels = [{"column": "temperature", "station": "STA_A", "kind": "anomaly",
               "event_id": "e1", "timestamp": "2024-01-01 00:00:00"},
              {"column": "temperature", "station": "STA_A", "kind": "anomaly",
               "event_id": "e1", "timestamp": "2024-01-01 02:00:00"}]
    assert len(G._truth_events(labels)) == len(truth)
    rows = [
        _row("a", "pipeline", "scored",
             [{"series": "temperature@STA_A", "start": "2024-01-01T00:00:00",
               "end": "2024-01-01T02:00:00", "count": 3}]),
        _row("b", "pipeline", "not_run", []),
    ]
    m = G._event_metrics(rows, labels)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 0)
    assert m["not_run_runs"] == 1
    assert m["not_run_truth_instances"] == 1
    assert m["recall"] == 0.5, "未执行的用例必须计入召回分母"
    assert m["f1"] == 0.6667


def test_event_metrics_does_not_match_across_series():
    labels = [{"column": "temperature", "station": "STA_A", "kind": "anomaly",
               "event_id": "e1", "timestamp": "2024-01-01 00:00:00"}]
    rows = [_row("a", "pipeline", "scored",
                 [{"series": "temperature@STA_B", "start": "2024-01-01T00:00:00",
                   "end": "2024-01-01T00:00:00", "count": 1}])]
    m = G._event_metrics(rows, labels)
    assert (m["tp"], m["fp"], m["fn"]) == (0, 1, 1)
    assert m["precision"] == 0.0 and m["recall"] == 0.0


def test_wilson_edges():
    assert G._wilson(0, 0) is None
    full = G._wilson(4, 4)
    assert full["value"] == 1.0 and full["hi"] == 1.0
    assert 0.45 < full["lo"] < 0.55
    zero = G._wilson(0, 10)
    assert zero["lo"] == 0.0 and zero["value"] == 0.0


def test_check_treats_none_and_boundary():
    assert G._check("x", 0.5, 0.5)["ok"] is True
    assert G._check("x", 0.4999, 0.5)["ok"] is False
    assert G._check("x", None, 0.5)["ok"] is False


def _fake_doc():
    metric = {"tp": 1, "fp": 0, "fn": 0, "detected": 1, "truth": 1,
              "precision": 1.0, "recall": 1.0, "f1": 1.0}
    ci = {"k": 1, "n": 1, "value": 1.0, "lo": 0.2, "hi": 1.0}
    mode = {
        "runs": 1, "succeeded": 1,
        "anomaly": {"point": metric, "event": metric,
                    "event_recall_adjusted": 1.0, "event_f1_adjusted": 1.0,
                    "event_ci": {"precision": ci, "recall": ci}},
        "changepoint": {"point": metric, "f1_adjusted": 1.0},
        "trend": {"accuracy": 1.0, "accuracy_ci": ci, "scored_rate": 1.0},
        "seasonal": {"accuracy": 1.0, "accuracy_ci": ci, "scored_rate": 1.0},
        "conclusion_need": {"coverage": 1.0, "coverage_ci": ci},
        "success_rate": 1.0, "success_rate_ci": ci,
    }
    return {
        "generated_at": "2026-10-24T00:00:00",
        "dataset": "d.csv", "dataset_sha1": "a", "labels": "l.csv",
        "labels_sha1": "b", "truth": "t.json", "truth_sha1": "c",
        "dataset_spec": {"rows": 10, "columns": ["temperature"],
                         "stations": ["STA_A"], "start": "2024-01-01",
                         "end": "2024-01-02",
                         "label_kinds": {"anomaly": 1}, "anomaly_events": 1},
        "modes": ["pipeline"],
        "summary": {"by_mode": {"pipeline": mode}},
        "evidence_coverage": {"pipeline": 1.0},
        "evidence_coverage_ci": {"pipeline": ci},
        "runs": [{"case": "c1", "mode": "pipeline", "status": "succeeded",
                  "steps": None, "terminate_reason": None,
                  "conclusion_count": 3, "evidence_coverage": 1.0,
                  "need_coverage": 1.0, "anomaly_point": metric,
                  "anomaly_event": metric, "changepoint_point": metric}],
        "checks": {"checks": [], "passed": False,
                   "reason": "pipeline-only 模式不构成两模式门禁"},
        "passed": False,
    }


def test_render_md_supports_pipeline_only():
    md = G.render_md(_fake_doc())
    assert "## 分模式指标" in md
    assert "| 指标 | pipeline |" in md
    assert "pipeline-only 模式不构成两模式门禁" in md
    assert "| c1 | pipeline |" in md


def test_v2_fixture_shape_and_labels():
    import pandas as pd
    df = pd.read_csv(G.DATA, encoding="utf-8-sig")
    assert len(df) == 17520
    assert set(df["station"].unique()) == {"STA_A", "STA_B"}
    labels = G.load_labels(G.LABELS)
    kinds = {}
    for x in labels:
        kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
    assert kinds == {"anomaly": 100, "level_shift_start": 2, "missing_gap_start": 1}
    assert len(G._truth_events(labels)) == 30
    truth = G.load_truth(G.TRUTH)
    assert truth["seed"] == 20261024
    assert "pressure@STA_A" in truth["note"]

def test_pipeline_only_defaults_do_not_touch_official_outputs():
    """诊断运行不得覆盖正式 24 项证据文件。"""
    out_json, out_md = G._output_paths(True, None, None)
    assert out_json != G.OUT_JSON
    assert out_md != G.OUT_MD
    assert "pipeline_only" in out_json
    assert "pipeline_only" in out_md


def test_full_gate_defaults_still_write_official_outputs():
    out_json, out_md = G._output_paths(False, None, None)
    assert out_json == G.OUT_JSON
    assert out_md == G.OUT_MD


def test_explicit_paths_win_in_both_modes():
    for flag in (True, False):
        assert G._output_paths(flag, "x.json", "x.md") == ("x.json", "x.md")
