# -*- coding: utf-8 -*-
"""端到端冒烟与一致性测试：两个场景跑通、产物齐全、结论有证据、重复运行一致。"""
from __future__ import annotations

import os

import pytest

from mtsa.engine import run_analysis
from mtsa.models import AnalysisRequest

FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")
WEATHER_CSV = os.path.join(FIX, "weather_v1.csv")
STOCK_CSV = os.path.join(FIX, "stock_v1.csv")


def _ensure_fixtures():
    if not (os.path.exists(WEATHER_CSV) and os.path.exists(STOCK_CSV)):
        from fixtures.gen_fixtures import main as gen_main
        gen_main()


@pytest.fixture(scope="module")
def weather_result(tmp_path_factory):
    _ensure_fixtures()
    out = str(tmp_path_factory.mktemp("wx"))
    req = AnalysisRequest(request_id="req_wx", input_path=WEATHER_CSV, scene_id="weather",
                          question="分析温度的趋势、季节性与极端事件", seed=20260927,
                          output_dir=out)
    return run_analysis(req, verbose=False)


@pytest.fixture(scope="module")
def stock_result(tmp_path_factory):
    _ensure_fixtures()
    out = str(tmp_path_factory.mktemp("stk"))
    req = AnalysisRequest(request_id="req_stk", input_path=STOCK_CSV, scene_id="stock",
                          question="计算收益风险指标并做 walk-forward 回测", seed=20260927,
                          output_dir=out)
    return run_analysis(req, verbose=False)


def test_weather_pipeline_succeeds(weather_result):
    assert weather_result.status == "succeeded", weather_result.error
    assert weather_result.data["plan"]["scene_id"] == "weather"


def test_stock_pipeline_succeeds(stock_result):
    assert stock_result.status == "succeeded", stock_result.error
    assert stock_result.data["plan"]["scene_id"] == "stock"


def test_events_recorded_for_every_node(weather_result):
    nodes = [e.node for e in weather_result.events]
    for required in ("load_dataset", "profile_dataset", "check_quality", "plan",
                     "clean_dataset", "compute_metrics", "detect_anomalies", "render_report"):
        assert required in nodes, f"缺少节点事件：{required}"
    assert all(e.status == "succeeded" for e in weather_result.events)


def test_artifacts_exist_on_disk(weather_result):
    for a in weather_result.artifacts:
        assert os.path.exists(a.path), f"产物不存在：{a.path}"
    kinds = {a.kind for a in weather_result.artifacts}
    assert "figure_interactive" in kinds
    assert "report_md" in kinds
    assert "script" in kinds


def test_every_conclusion_has_evidence(weather_result, stock_result):
    for res in (weather_result, stock_result):
        ids = {e.evidence_id for e in res.evidence}
        assert res.conclusions
        for c in res.conclusions:
            assert c.evidence_ids, f"结论 {c.conclusion_id} 无证据"
            assert set(c.evidence_ids) <= ids


def test_quality_report_captures_injected_issues(weather_result):
    codes = {i["code"] for i in weather_result.data["quality"]["issues"]}
    # fixture 注入：缺失段、重复时间戳、乱序、不规则采样、单位变化
    assert "MISSING_VALUES" in codes
    assert "DUPLICATE_TIMESTAMPS" in codes
    assert "OUT_OF_ORDER" in codes
    assert "SAMPLING_GAP" in codes


def test_unit_conversion_applied_in_weather(weather_result):
    conv = weather_result.data["clean"]["unit_conversions"]
    assert conv, "未识别并修正华氏度区间"
    assert conv[0]["kind"] == "fahrenheit_to_celsius"


def test_stock_adjust_inconsistency_flagged(stock_result):
    rows = {r["metric_id"]: r["value"] for r in stock_result.data["scenario_metrics"]}
    assert "stk.adj.max_rel_diff" in rows
    assert rows["stk.adj.max_rel_diff"] > 1e-6


def test_result_json_written(weather_result):
    paths = [a.path for a in weather_result.artifacts if a.artifact_id == "art_result_json"]
    assert paths and os.path.exists(paths[0])


def test_determinism_same_input_twice(tmp_path):
    _ensure_fixtures()
    kwargs = dict(request_id="req_det", input_path=STOCK_CSV, scene_id="stock",
                  question="回测", seed=20260927)
    r1 = run_analysis(AnalysisRequest(output_dir=str(tmp_path / "a"), **kwargs), verbose=False)
    r2 = run_analysis(AnalysisRequest(output_dir=str(tmp_path / "b"), **kwargs), verbose=False)
    assert r1.status == r2.status == "succeeded"
    assert r1.data == r2.data
    assert [c.model_dump() for c in r1.conclusions] == [c.model_dump() for c in r2.conclusions]
    assert [e.model_dump() for e in r1.evidence] == [e.model_dump() for e in r2.evidence]
