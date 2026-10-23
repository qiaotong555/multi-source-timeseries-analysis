# -*- coding: utf-8 -*-
"""A/B 结论级评估的纯函数测试：不跑引擎、不调模型、不依赖 artifacts。"""
from __future__ import annotations

import json
from pathlib import Path

from evals.conclusion_common import (
    discover_result_dirs, match_events, match_points, merge_events,
    month_distance, parse_seasonal, parse_trend, point_metrics,
)
from evals.conclusion_diff import attribute_case
from evals.conclusion_eval import evaluate_run, summarize


def _doc(tools=None, conclusions=None, anomalies=None, change_points=None):
    return {
        "status": "succeeded",
        "data": {
            "agent": {"coverage": {"executed_tools": tools or []}},
            "anomalies": anomalies or [],
            "change_points": change_points or [],
            "quality": {"issues": []},
            "profile": {"time_axis": {"median_interval_seconds": 3600}},
            "metrics": [],
            "spectrum": [],
        },
        "conclusions": conclusions or [],
    }


def test_parse_trend_up_and_down():
    up = parse_trend("temperature@STA_A：Mann-Kendall tau=0.05，p=0.01，存在显著单调趋势，方向上升；月度均值 OLS 斜率 +0.12 单位/月。")
    assert up["parsed"] is True and up["direction"] == "up"
    assert up["ols_slope"] == 0.12
    down = parse_trend("temperature@STA_B：Mann-Kendall tau=-0.07，p=0.01，存在显著单调趋势，方向下降；月度均值 OLS 斜率 -0.14 单位/月。")
    assert down["direction"] == "down" and down["ols_slope"] == -0.14


def test_parse_trend_none_is_not_guessed():
    r = parse_trend("temperature@STA_A：Mann-Kendall tau=0.01，p=0.4，未发现显著单调趋势。")
    assert r["parsed"] is True and r["direction"] == "none"


def test_parse_seasonal():
    r = parse_seasonal("temperature@STA_B：月均振幅 22.623，峰值月 6 月，谷值月 12 月。")
    assert r["parsed"] is True
    assert r["amplitude"] == 22.623
    assert r["peak_month"] == 6 and r["trough_month"] == 12


def test_parse_seasonal_incomplete_stays_unlabeled():
    r = parse_seasonal("temperature@STA_B：月均振幅 22.623。")
    assert r["parsed"] is False


def test_match_points_exact_and_tolerance():
    det = {("temperature@STA_A", "2024-02-17 16:00:00")}
    truth = {("temperature@STA_A", "2024-02-17 16:00:00")}
    m = match_points(det, truth, 0)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 0)
    near = {("temperature@STA_A", "2024-02-17 17:00:00")}
    assert match_points(near, truth, 0)["tp"] == 0
    assert match_points(near, truth, 3600)["tp"] == 1


def test_event_merge_and_match():
    rows = [
        {"series": "temperature@STA_A", "timestamp": "2024-01-01 00:00:00"},
        {"series": "temperature@STA_A", "timestamp": "2024-01-01 01:00:00"},
        {"series": "temperature@STA_A", "timestamp": "2024-01-01 06:00:00"},
    ]
    ev = merge_events(rows, gap_seconds=3 * 3600)
    assert len(ev) == 2
    truth = [{"series": "temperature@STA_A", "start": "2024-01-01 01:00:00",
              "end": "2024-01-01 01:00:00"}]
    m = match_events(ev, truth)
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 0


def test_point_metrics():
    det = {("a", "t1"), ("a", "t2")}
    truth = {("a", "t1"), ("a", "t3")}
    m = point_metrics(det, truth)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_month_distance_wraps():
    assert month_distance(1, 12) == 1
    assert month_distance(6, 7) == 1


def test_truth_file_matches_generator_values():
    p = Path(__file__).resolve().parents[1] / "fixtures" / "truth_weather_v1.json"
    t = json.loads(p.read_text(encoding="utf-8"))
    a = t["series"]["temperature@STA_A"]
    b = t["series"]["temperature@STA_B"]
    assert a["trend_per_year"] == 0.8 and a["peak_month"] == 7
    assert b["peak_month"] == 6 and b["trough_month"] == 12
    assert a["annual_amplitude_peak_to_trough"] == 24.0


def test_discover_result_dirs(tmp_path):
    (tmp_path / "wx_a_agent").mkdir()
    (tmp_path / "wx_a_agent" / "result.json").write_text("{}", encoding="utf-8")
    (tmp_path / "wx_b" / "pipeline").mkdir(parents=True)
    (tmp_path / "wx_b" / "pipeline" / "result.json").write_text("{}", encoding="utf-8")
    got = discover_result_dirs(str(tmp_path))
    assert {(x["case"], x["mode"]) for x in got} == {("wx_a", "agent"), ("wx_b", "pipeline")}


def _trend(text):
    return {"category": "趋势", "text": text}


def _seasonal(text):
    return {"category": "季节性", "text": text}


def test_evaluate_run_not_run_is_not_zero_score():
    labels = [{"column": "temperature", "timestamp": "2024-02-17 16:00:00",
               "kind": "anomaly", "station": "STA_A"}]
    truth = {"series": {"temperature@STA_A": {
        "trend_direction": "up", "annual_amplitude_peak_to_trough": 24.0,
        "peak_month": 7, "trough_month": 1}}}
    doc = _doc(tools=["compute_metrics"], conclusions=[
        _trend("temperature@STA_A：Mann-Kendall tau=0.1，p=0.01，存在显著单调趋势，方向上升；月度均值 OLS 斜率 +0.1 单位/月。"),
        _seasonal("temperature@STA_A：月均振幅 23.5，峰值月 7 月，谷值月 1 月。"),
    ])
    row = evaluate_run(doc, "wx_x", "agent", "q", ["anomaly"], labels, truth)
    assert row["anomaly"]["state"] == "not_run"
    assert row["anomaly"]["point"] is None
    assert row["trend"]["details"][0]["correct"] is True
    assert row["seasonal"]["details"][0]["correct"] is True


def test_attribute_tool_not_run():
    p = _doc(conclusions=[_trend("temperature@STA_A：Mann-Kendall tau=0.1，p=0.01，存在显著单调趋势，方向上升。")])
    p["conclusions"].append({"category": "周期", "text": "x"})
    a = _doc(tools=["compute_metrics"], conclusions=p["conclusions"][:1])
    row = attribute_case("wx_x", p, a)
    cat = {x["category"]: x for x in row["categories"]}
    assert cat["周期"]["missing"] == 1
    assert cat["周期"]["reason"] == "tool_not_run"
    assert row["unattributed"] == 0


def test_attribute_no_detection():
    p = _doc(conclusions=[{"category": "异常检测", "text": "x"}])
    a = _doc(tools=["detect_anomalies"], conclusions=[])
    row = attribute_case("wx_x", p, a)
    cat = {x["category"]: x for x in row["categories"]}
    assert cat["异常检测"]["reason"] == "no_detection"
    assert row["unattributed"] == 0


def test_attribute_plan_or_parameter_diff():
    p = _doc(conclusions=[_trend("x"), _trend("y")])
    a = _doc(tools=["compute_metrics"], conclusions=[_trend("x")])
    row = attribute_case("wx_x", p, a)
    cat = {x["category"]: x for x in row["categories"]}
    assert cat["趋势"]["missing"] == 1
    assert cat["趋势"]["reason"] == "plan_or_parameter_diff"


def test_summarize_excludes_not_run_from_prf():
    labels = [{"column": "temperature", "timestamp": "2024-02-17 16:00:00",
               "kind": "anomaly", "station": "STA_A"}]
    truth = {"series": {"temperature@STA_A": {
        "trend_direction": "up", "annual_amplitude_peak_to_trough": 24.0,
        "peak_month": 7, "trough_month": 1}}}
    doc = _doc(tools=["compute_metrics"], conclusions=[
        _trend("temperature@STA_A：Mann-Kendall tau=0.1，p=0.01，存在显著单调趋势，方向上升。"),
        _seasonal("temperature@STA_A：月均振幅 23.5，峰值月 7 月，谷值月 1 月。"),
    ])
    row = evaluate_run(doc, "wx_x", "agent", "q", [], labels, truth)
    row["_detected_anomalies"] = set()
    row["_detected_change_points"] = set()
    row["_path"] = "x"
    s = summarize([row], {("temperature@STA_A", "2024-02-17 16:00:00")}, set(), 7200)
    a = s["by_mode"]["agent"]["anomaly"]
    assert a["scored_runs"] == 0 and a["not_run_runs"] == 1
    assert a["not_run_truth_instances"] == 1
    assert s["by_mode"]["agent"]["trend"]["correct"] == 1