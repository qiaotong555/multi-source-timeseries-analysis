# -*- coding: utf-8 -*-
"""B 真实数据验证：数据注册、标注协议、匹配口径与汇总口径的单测。

这些测试只读 fixtures，不运行分析引擎、不调用模型，保证离线可跑。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from evals.real_data_eval import (
    data_facts, jaccard, load_labels, load_registry, match_quality,
    macro, worst, summarize_dataset,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "fixtures" / "real_weather"


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- 数据注册 ----------

def test_registry_ids_are_sorted_and_unique():
    """D3 起 registry 从 10 份增到 17 份；R01–R10 的编号与内容必须保持不变。"""
    reg = load_registry()
    ids = [m["id"] for m in reg]
    assert len(reg) >= 10
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
    assert set("R%02d" % i for i in range(1, 11)) <= set(ids)


def test_every_dataset_has_files_and_matching_sha1():
    for m in load_registry():
        d = BASE / Path(m["data_path"]).parent.name
        for name in ("data.csv", "metadata.json", "sha1.txt", "labels.csv"):
            assert (d / name).exists(), "%s 缺 %s" % (m["id"], name)
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        sha = (d / "sha1.txt").read_text(encoding="utf-8").split()[0]
        assert meta["data_sha1"] == sha
        assert len(sha) == 40


def test_registry_covers_multiple_sources_and_frequencies():
    reg = load_registry()
    assert len({m["source"] for m in reg}) >= 3
    assert len({m["frequency"] for m in reg}) >= 2


def test_converter_and_registry_agree_on_datasets():
    """脚本里的数据集定义必须与 registry 一一对应，否则「重跑能复原」是假的。"""
    mod = _load_script("fetch_real_weather.py")
    ids = [d["id"] for d in mod.DATASETS]
    assert len(set(ids)) == len(ids)
    assert ids == ["R%02d" % i for i in range(1, len(ids) + 1)]
    assert set(ids) == {m["id"] for m in load_registry()}
    url = mod.openmeteo_url(mod.DATASETS[0], 52.52, 13.41)
    assert "temperature_2m" in url and "timezone=UTC" in url
    # 旧 10 份的请求口径不能被后续改动带偏（URL 变了 SHA1 就对不上）
    assert "wind_gusts" not in url


# ---------- 标注协议 ----------

def test_labels_are_deterministic():
    mod = _load_script("build_real_labels.py")
    for m in load_registry():
        if m["id"] not in ("R06", "R08"):
            continue
        a, _ = mod.build_dataset(m)
        b, _ = mod.build_dataset(m)
        assert a == b


def test_r06_labels_point_at_real_missing_runs():
    m = next(x for x in load_registry() if x["id"] == "R06")
    labels = load_labels(m["data_path"])
    gaps = [x for x in labels if x["kind"] == "missing_gap"]
    assert len(gaps) >= 5
    df = pd.read_csv(ROOT / m["data_path"], encoding="utf-8-sig")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for g in gaps:
        seg = df[(df["timestamp"] >= g["start"]) & (df["timestamp"] <= g["end"])]
        assert seg["wind_speed"].isna().all()
        assert int(g["count"]) == len(seg)


def test_r08_sampling_gap_label_matches_data():
    m = next(x for x in load_registry() if x["id"] == "R08")
    labels = [x for x in load_labels(m["data_path"]) if x["kind"] == "sampling_gap"]
    assert len(labels) == 1
    assert labels[0]["count"] == "4"
    assert labels[0]["start"].startswith("2024-04-23")


def test_label_files_have_only_known_kinds():
    known = {"missing_gap", "sampling_gap", "duplicate_timestamp", "out_of_order",
             "constant_run", "sentinel", "unit_change", "verified_anomaly"}
    for m in load_registry():
        for x in load_labels(m["data_path"]):
            assert x["kind"] in known
            assert set(x) >= {"kind", "column", "station", "start", "end", "source", "note"}


# ---------- 匹配口径 ----------

def _labels(*items):
    return [dict(kind=k, column=c, station="S", start=t, end=t, count="3", source="x", note="")
            for k, c, t in items]


def test_missing_gap_is_reported_but_not_localized():
    out = match_quality(_labels(("missing_gap", "wind_speed", "2020-01-01 00:00:00")),
                        [{"code": "MISSING_VALUES", "column": "wind_speed", "count": 9}])
    assert out["total"] == 1 and out["detected"] == 1
    assert out["rows"][0]["status"] == "reported_unlocalized"
    assert out["localized"] == 0 and out["localized_recall"] == 0.0


def test_sampling_gap_with_same_start_is_localized():
    out = match_quality(_labels(("sampling_gap", "timestamp", "2024-04-23 06:00:00")),
                        [{"code": "SAMPLING_GAP", "column": None,
                          "start": "2024-04-23 06:00:00", "end": "2024-04-23 11:00:00"}])
    assert out["rows"][0]["status"] == "detected_localized"
    assert out["localized_recall"] == 1.0


def test_wrong_column_is_not_detected():
    out = match_quality(_labels(("missing_gap", "humidity", "2020-01-01 00:00:00")),
                        [{"code": "MISSING_VALUES", "column": "wind_speed", "count": 9}])
    assert out["rows"][0]["status"] == "not_detected"
    assert out["recall"] == 0.0


def test_label_without_detector_is_counted_as_miss():
    out = match_quality(_labels(("constant_run", "humidity", "2020-01-01 00:00:00")), [])
    assert out["rows"][0]["reason"] == "系统无对应检测器"
    assert out["recall"] == 0.0


def test_empty_labels_give_none_recall():
    out = match_quality([], [{"code": "MISSING_VALUES"}])
    assert out["total"] == 0 and out["recall"] is None


def test_wilson_interval_is_reported():
    out = match_quality(_labels(("missing_gap", "a", "2020-01-01 00:00:00")),
                        [{"code": "MISSING_VALUES", "column": "a"}])
    ci = out["recall_ci"]
    assert ci["k"] == 1 and ci["n"] == 1 and ci["lo"] < 1.0 <= ci["hi"] + 1e-9


# ---------- 一致性 ----------

def test_jaccard_edge_cases():
    assert jaccard(set(), set()) == 1.0
    assert jaccard({("a", "1")}, {("a", "1")}) == 1.0
    assert jaccard({("a", "1")}, {("a", "2")}) == 0.0
    assert jaccard({("a", "1"), ("a", "2")}, {("a", "2")}) == 0.5


# ---------- 汇总口径 ----------

def _fake_runs(dataset, mode, ok=4, total=4, anomalies=2, cp=1, ev=1.0, need=1.0):
    runs = []
    for i in range(total):
        status = "succeeded" if i < ok else "failed"
        runs.append({
            "dataset": dataset, "question": "q%d" % (i + 1), "mode": mode,
            "status": status, "result_exists": status == "succeeded",
            "report_exists": status == "succeeded", "report_has_quality_section": True,
            "conclusion_count": 3, "conclusion_categories": {"异常检测": 1},
            "evidence_coverage": ev, "need_coverage": need,
            "quality_issues": 1, "quality_codes": ["MISSING_VALUES"],
            "quality_issue_rows": [], "anomalies": anomalies, "change_points": cp,
            "anomaly_candidate_count": 3, "executed_tools": ["detect_anomalies"],
            "steps": 4, "terminate_reason": "model_finished", "tokens": 1000,
            "missing_tools": [], "unit_conversions": [], "error": {}, "dir": "x",
            "_anomaly_keys": {("a", "1")}, "_cp_keys": {("b", "2")},
        })
    return runs


def _meta(i="R01"):
    return {"id": i, "name": "x", "source": "s", "note": "", "data_path": "x",
            "frequency": "hourly"}


def test_summarize_computes_success_rate_and_means():
    runs = _fake_runs("R01", "pipeline", ok=3, total=4)
    facts = {"stations": ["A"], "per_station_duplicate_rows": 0,
             "dropped_all_null_columns": [], "missing_ratio": {"a": 0.0}}
    out = summarize_dataset(_meta(), runs, [], facts)
    assert out["by_mode"]["pipeline"]["success_rate"] == 0.75
    assert out["by_mode"]["pipeline"]["evidence_coverage"] == 1.0
    assert out["by_mode"]["pipeline"]["anomalies"] == 2.0
    assert out["by_mode"]["agent"]["runs"] == 0


def test_summarize_flags_multistation_duplicate_false_positive():
    runs = _fake_runs("R05", "pipeline", total=1, ok=1)
    runs[0]["quality_issue_rows"] = [{"code": "DUPLICATE_TIMESTAMPS", "column": "timestamp"}]
    facts = {"stations": ["A", "B"], "per_station_duplicate_rows": 0,
             "dropped_all_null_columns": [], "missing_ratio": {}}
    out = summarize_dataset(_meta("R05"), runs, [], facts)
    assert out["duplicate_timestamp_false_positive"] is True


def test_macro_ignores_none_and_worst_breaks_ties():
    per = [
        {"id": "A", "m": {"v": 1.0}, "w": {"x": 0.5}},
        {"id": "B", "m": {"v": 0.0}, "w": {"x": 0.5}},
        {"id": "C", "m": {"v": None}, "w": {"x": None}},
    ]
    assert macro(per, ["m", "v"]) == 0.5
    w = worst(per, ["w", "x"])
    assert w["value"] == 0.5 and w["datasets"] == ["A", "B"]


def test_data_facts_detects_dropped_columns_and_no_false_duplicates():
    reg = {m["id"]: m for m in load_registry()}
    f7 = data_facts(reg["R07"], [])
    assert "wind_speed" in f7["dropped_all_null_columns"]
    assert f7["per_station_duplicate_rows"] == 0
    f5 = data_facts(reg["R05"], [])
    assert len(f5["stations"]) == 3
    assert f5["per_station_duplicate_rows"] == 0