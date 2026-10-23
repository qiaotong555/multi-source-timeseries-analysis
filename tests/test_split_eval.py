# -*- coding: utf-8 -*-
"""评估集划分测试：时间切分、阈值选择、真值比对、参数覆盖。"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from evals import split_eval as SE
from mtsa.engine import run_analysis
from mtsa.models import AnalysisRequest
from .helpers import make_weather_frame


# ---------------- 切分 ----------------

def test_split_partitions_without_overlap():
    df = make_weather_frame(n=100, stations=2)      # 200 行，100 个唯一时间点
    dev, info_dev = SE.split_by_time(df, "timestamp", 0.0, 0.6)
    hold, info_hold = SE.split_by_time(df, "timestamp", 0.6, 1.0)

    assert len(dev) + len(hold) == len(df)
    assert set(dev["timestamp"]) & set(hold["timestamp"]) == set()
    assert info_dev["unique_timepoints"] == 60
    assert info_hold["unique_timepoints"] == 40
    assert info_dev["rows"] == len(dev) and info_hold["rows"] == len(hold)


def test_split_keeps_same_timestamp_together():
    """同一时刻的多实体记录必须落在同一侧，否则实体会被劈开。"""
    df = make_weather_frame(n=100, stations=2)
    dev, _ = SE.split_by_time(df, "timestamp", 0.0, 0.6)
    assert (dev.groupby("timestamp").size() == 2).all()


def test_split_covers_full_range():
    df = make_weather_frame(n=100, stations=1)
    whole, _ = SE.split_by_time(df, "timestamp", 0.0, 1.0)
    assert len(whole) == len(df)


def test_split_labels_follow_same_boundaries():
    df = make_weather_frame(n=100, stations=1)
    _, info = SE.split_by_time(df, "timestamp", 0.0, 0.6)
    labels = pd.DataFrame([
        {"column": "temperature", "station": "STA_0", "kind": "anomaly",
         "timestamp": str(df["timestamp"].iloc[10])},
        {"column": "temperature", "station": "STA_0", "kind": "anomaly",
         "timestamp": str(df["timestamp"].iloc[90])},
    ])
    dev_labels = SE.split_labels(labels, info["start"], info["end"])
    assert len(dev_labels) == 1
    assert str(dev_labels.iloc[0]["timestamp"]) == str(df["timestamp"].iloc[10])


# ---------------- 真值比对 ----------------

def test_anomaly_pr_counts_tp_fp_fn():
    labels = pd.DataFrame([
        {"column": "temperature", "station": "STA_0", "kind": "anomaly",
         "timestamp": "2024-01-01 01:00:00"},
        {"column": "humidity", "station": "STA_0", "kind": "anomaly",
         "timestamp": "2024-01-01 02:00:00"},
    ])
    anomalies = [
        {"column": "temperature@STA_0", "timestamp": "2024-01-01 01:00:00"},  # TP
        {"column": "temperature@STA_0", "timestamp": "2024-01-01 03:00:00"},  # FP
        {"column": "rainfall@STA_0", "timestamp": "2024-01-01 04:00:00"},     # 未注入列，不计
    ]
    pr = SE.anomaly_pr(anomalies, labels)
    assert pr["tp"] == 1 and pr["fp"] == 1 and pr["fn"] == 1
    assert pr["precision"] == 0.5 and pr["recall"] == 0.5


def test_anomaly_pr_ignores_columns_without_truth():
    labels = pd.DataFrame([
        {"column": "temperature", "station": "STA_0", "kind": "anomaly",
         "timestamp": "2024-01-01 01:00:00"},
    ])
    anomalies = [{"column": "rainfall@STA_0", "timestamp": "2024-01-01 05:00:00"}]
    pr = SE.anomaly_pr(anomalies, labels)
    assert pr["fp"] == 0          # 不把合理检出算作误报
    assert pr["fn"] == 1


# ---------------- 阈值选择 ----------------

def _row(thr, p, r):
    return {"threshold": thr, "precision": p, "recall": r,
            "f1": SE._f1(p, r), "tp": 0, "fp": 0, "fn": 0, "detected_count": 0}


def test_select_threshold_prefers_recall_under_precision_constraint():
    scan = [_row(5.0, 0.9, 0.5), _row(6.0, 0.8, 0.8), _row(7.0, 0.6, 0.9)]
    choice = SE.select_threshold(scan, min_precision=0.70)
    assert choice["threshold"] == 6.0
    assert "召回最高" in choice["rule"]


def test_select_threshold_breaks_tie_toward_larger_threshold():
    scan = [_row(5.0, 0.8, 0.8), _row(6.0, 0.8, 0.8)]
    assert SE.select_threshold(scan, 0.70)["threshold"] == 6.0


def test_select_threshold_falls_back_to_f1():
    scan = [_row(5.0, 0.50, 0.50), _row(6.0, 0.40, 0.40)]
    choice = SE.select_threshold(scan, min_precision=0.70)
    assert choice["threshold"] == 5.0
    assert "退化为 F1" in choice["rule"]


def test_select_threshold_rejects_empty_scan():
    with pytest.raises(ValueError):
        SE.select_threshold([])


# ---------------- 汇总判定 ----------------

def test_summarize_skips_folds_with_insufficient_truth():
    folds = [
        {"fold": "f1", "status": "succeeded", "holdout_precision": 0.80, "holdout_recall": 0.90,
         "holdout_event_recall": 0.85, "holdout_recall_by_type": {"point": 0.9, "collective": 0.8},
         "degradation_precision": 0.05, "degradation_recall": 0.05, "selected_threshold": 8.0,
         "holdout_truth_count": 6, "insufficient_truth": False},
        {"fold": "f2", "status": "succeeded", "holdout_precision": 0.20, "holdout_recall": 1.00,
         "holdout_event_recall": 0.10, "holdout_recall_by_type": {"point": 0.1, "collective": 0.1},
         "degradation_precision": 0.60, "degradation_recall": 0.00, "selected_threshold": 8.0,
         "holdout_truth_count": 2, "insufficient_truth": True},
        {"fold": "f3", "status": "succeeded", "holdout_precision": 0.90, "holdout_recall": 0.95,
         "holdout_event_recall": 0.90, "holdout_recall_by_type": {"point": 1.0, "collective": 0.8},
         "degradation_precision": 0.03, "degradation_recall": 0.02, "selected_threshold": 8.0,
         "holdout_truth_count": 7, "insufficient_truth": False},
    ]
    out = SE.summarize(folds)
    assert out["summary"]["mean_holdout_precision"] == 0.85
    assert out["summary"]["skipped_folds"] == ["f2"]
    assert out["summary"]["valid_folds"] == ["f1", "f3"]
    assert out["summary"]["mean_collective_recall"] == 0.8
    assert out["summary"]["passed"] is True


def test_summarize_inconclusive_when_single_valid_fold():
    """只有一折有效时不得判为通过：点估计不可用于验收。"""
    folds = [{"fold": "f1", "status": "succeeded", "holdout_precision": 1.0, "holdout_recall": 1.0,
              "degradation_precision": 0.0, "degradation_recall": 0.0, "selected_threshold": 10.0,
              "holdout_truth_count": 5, "insufficient_truth": False}]
    out = SE.summarize(folds)
    assert out["summary"]["inconclusive"] is True
    assert out["summary"]["passed"] is False
    assert "证据不足" in out["summary"]["reason"]


def test_summarize_flags_threshold_at_grid_edge():
    folds = [
        {"fold": "f1", "status": "succeeded", "holdout_precision": 0.8, "holdout_recall": 0.9,
         "degradation_precision": 0.05, "degradation_recall": 0.05,
         "selected_threshold": SE.THRESHOLD_GRID[-1],
         "holdout_truth_count": 6, "insufficient_truth": False},
        {"fold": "f2", "status": "succeeded", "holdout_precision": 0.8, "holdout_recall": 0.9,
         "degradation_precision": 0.05, "degradation_recall": 0.05,
         "selected_threshold": SE.THRESHOLD_GRID[-1],
         "holdout_truth_count": 6, "insufficient_truth": False},
    ]
    out = SE.summarize(folds)
    assert out["summary"]["threshold_stable"] is True
    assert out["summary"]["threshold_at_grid_edge"] is True


def test_summarize_only_penalises_degradation():
    """holdout 优于 dev 不算不稳定：差距取有方向的退化量。"""
    folds = [
        {"fold": "f1", "status": "succeeded", "holdout_precision": 1.0, "holdout_recall": 0.9,
         "holdout_event_recall": 0.9, "holdout_recall_by_type": {"point": 1.0, "collective": 0.8},
         "degradation_precision": -0.25, "degradation_recall": -0.10, "selected_threshold": 7.0,
         "holdout_truth_count": 12, "insufficient_truth": False},
        {"fold": "f2", "status": "succeeded", "holdout_precision": 0.95, "holdout_recall": 0.85,
         "holdout_event_recall": 0.85, "holdout_recall_by_type": {"point": 0.9, "collective": 0.8},
         "degradation_precision": -0.15, "degradation_recall": 0.05, "selected_threshold": 7.0,
         "holdout_truth_count": 12, "insufficient_truth": False},
    ]
    out = SE.summarize(folds)
    assert out["summary"]["max_degradation_precision"] < 0
    assert out["summary"]["passed"] is True


def test_wilson_lower_reflects_sample_size():
    assert SE.wilson_lower(4, 4) < 0.60          # 全中但样本极少，下界很低
    assert SE.wilson_lower(40, 40) > 0.90        # 样本充分，下界可信
    assert SE.wilson_lower(0, 5) == 0.0
    assert SE.wilson_lower(0, 0) == 0.0


# ---------------- 事件级评估 ----------------

_BASE = pd.Timestamp("2024-01-01")


def _ts(hour: int) -> str:
    return str(_BASE + pd.Timedelta(hours=hour))


def _span_labels(start_hour, hours, etype="collective", eid="e1", col="temperature", st="STA_0"):
    return [{"column": col, "station": st, "kind": "anomaly",
             "timestamp": _ts(start_hour + h),
             "anomaly_type": etype, "event_id": eid} for h in range(hours)]


def _span_detected(start_hour, hours, col="temperature@STA_0"):
    return [{"column": col, "timestamp": _ts(start_hour + h)} for h in range(hours)]


def test_event_pr_merges_continuous_span_into_one_event():
    """一段持续的 8 个点属于一个事件：点级会给 8 个 TP，事件级只给 1 个。"""
    labels = pd.DataFrame(_span_labels(0, 8))
    anomalies = _span_detected(0, 8)

    point = SE.anomaly_pr(anomalies, labels)
    event = SE.event_pr(anomalies, labels)

    assert point["tp"] == 8
    assert event["tp"] == 1 and event["fp"] == 0 and event["fn"] == 0
    assert event["truth_events"] == 1 and event["detected_events"] == 1
    assert event["recall_by_type"]["collective"] == 1.0


def test_event_pr_separates_distant_events():
    labels = pd.DataFrame(_span_labels(0, 3, eid="e1") + _span_labels(10, 3, eid="e2"))
    anomalies = _span_detected(0, 3) + _span_detected(10, 3)
    r = SE.event_pr(anomalies, labels)
    assert r["truth_events"] == 2 and r["detected_events"] == 2
    assert r["tp"] == 2 and r["fp"] == 0


def test_event_pr_counts_split_detection_as_extra_event():
    """真值一段，但检出被缺口劈成两段：仍算命中，多出的一段计假阳性。"""
    labels = pd.DataFrame(_span_labels(0, 8))
    anomalies = _span_detected(0, 3) + _span_detected(6, 2)   # 中间断开 3 小时
    r = SE.event_pr(anomalies, labels)
    assert r["tp"] == 1 and r["fp"] == 1


def test_event_pr_reports_missed_collective():
    labels = pd.DataFrame(_span_labels(0, 8, etype="collective", eid="c1"))
    r = SE.event_pr([], labels)
    assert r["recall_by_type"]["collective"] == 0.0
    assert r["fn"] == 1 and r["truth_events"] == 1


def test_event_pr_recall_by_type_is_separated():
    labels = pd.DataFrame(_span_labels(0, 1, etype="point", eid="p1")
                          + _span_labels(20, 8, etype="collective", eid="c1"))
    anomalies = _span_detected(0, 1)          # 只检出了点异常
    r = SE.event_pr(anomalies, labels)
    assert r["recall_by_type"]["point"] == 1.0
    assert r["recall_by_type"]["collective"] == 0.0


def test_summarize_fails_on_large_gap():
    folds = [{"fold": "f1", "status": "succeeded", "holdout_precision": 0.50, "holdout_recall": 0.90,
              "degradation_precision": 0.35, "degradation_recall": 0.00,
              "selected_threshold": 8.0, "holdout_truth_count": 8,
              "insufficient_truth": False},
             {"fold": "f2", "status": "succeeded", "holdout_precision": 0.60, "holdout_recall": 0.90,
              "degradation_precision": 0.30, "degradation_recall": 0.00,
              "selected_threshold": 8.0, "holdout_truth_count": 8,
              "insufficient_truth": False}]
    out = SE.summarize(folds)
    assert out["summary"]["passed"] is False
    assert out["summary"]["checks"]["precision_degradation<=0.20"] is False


def test_summarize_reports_all_failed():
    out = SE.summarize([{"fold": "f1", "status": "failed"}])
    assert out["summary"]["passed"] is False
    assert "没有可用于判定的折" in out["summary"]["reason"]


# ---------------- 参数覆盖 ----------------

def test_param_override_applied(tmp_path):
    df = make_weather_frame(n=300, stations=1)
    p = tmp_path / "w.csv"
    df.to_csv(p, index=False)
    req = AnalysisRequest(request_id="ov_ok", input_path=str(p), scene_id="weather",
                          question="分析温度", param_overrides={"anomaly_threshold": 6.0},
                          output_dir=str(tmp_path / "out"))
    res = run_analysis(req, verbose=False)
    assert res.status == "succeeded"
    assert res.data["plan"]["parameters"]["anomaly_threshold"] == 6.0
    assert any(e.node == "param_override" for e in res.events)


def test_unknown_param_override_fails_explicitly(tmp_path):
    df = make_weather_frame(n=200, stations=1)
    p = tmp_path / "w.csv"
    df.to_csv(p, index=False)
    req = AnalysisRequest(request_id="ov_bad", input_path=str(p), scene_id="weather",
                          question="分析温度", param_overrides={"anomaly_treshold": 6.0},
                          output_dir=str(tmp_path / "out2"))
    res = run_analysis(req, verbose=False)
    assert res.status == "failed"
    assert res.error.details["reason"] == "invalid_param_override"
    assert "anomaly_treshold" in res.error.details["unknown"]


# ---------------- 评估专用集 ----------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _eval_fixture():
    data = pd.read_csv(os.path.join(ROOT, "fixtures", SE.EVAL_DATASET))
    labels = pd.read_csv(os.path.join(ROOT, "fixtures", SE.EVAL_LABELS))
    return data, labels


def test_eval_fixture_differs_from_demo():
    """评估集必须与演示集分离，否则又回到「同集调参再报告」。"""
    assert SE.EVAL_DATASET != "weather_v1.csv"
    assert os.path.exists(os.path.join(ROOT, "fixtures", SE.EVAL_DATASET))


def test_eval_fixture_has_enough_truth():
    _, labels = _eval_fixture()
    anom = labels[labels["kind"] == "anomaly"]
    assert len(anom) >= 50
    per = anom.groupby(["column", "station"]).size()
    assert len(per) == 6
    assert int(per.min()) >= 8


def test_eval_fixture_has_both_anomaly_types():
    """评估集必须同时覆盖孤立点与持续段，否则「题型扩展」没有验证对象。"""
    _, labels = _eval_fixture()
    anom = labels[labels["kind"] == "anomaly"]
    assert set(anom["anomaly_type"].unique()) == {"point", "collective"}
    assert "event_id" in anom.columns
    n_point_events = anom[anom["anomaly_type"] == "point"]["event_id"].nunique()
    n_coll_events = anom[anom["anomaly_type"] == "collective"]["event_id"].nunique()
    assert n_point_events == 60
    assert n_coll_events == 24
    # 持续段每段至少 5 个点，否则退化成点异常
    per_event = anom[anom["anomaly_type"] == "collective"].groupby("event_id").size()
    assert int(per_event.min()) >= 5


def test_eval_fixture_covers_every_fold():
    """每折 holdout 的真值数必须达到可判定下限，否则划分无法验证。"""
    data, labels = _eval_fixture()
    anom_ts = pd.to_datetime(labels[labels["kind"] == "anomaly"]["timestamp"])
    for fold in SE.FOLDS:
        _, info = SE.split_by_time(data, "timestamp", fold["dev_ratio"],
                                   fold["dev_ratio"] + fold["holdout_len"])
        lo = pd.Timestamp(info["start"])
        mask = anom_ts >= lo
        if info["end"]:
            mask &= anom_ts < pd.Timestamp(info["end"])
        cnt = int(mask.sum())
        assert cnt >= 10, f"{fold['name']} holdout 真值仅 {cnt} 个，低于可判定要求"

# ---------------- 评估链路护栏 ----------------

def test_holdout_plan_missing_anomaly_is_harness_error():
    """检测器没执行不能按 0 分算：必须显式标成评估链路错误。"""
    assert SE.holdout_plan_problem(["trend", "seasonal"]) is not None
    assert SE.holdout_plan_problem([]) is not None
    assert SE.holdout_plan_problem(["trend", "anomaly"]) is None


def test_rule_planner_env_forces_and_restores(monkeypatch):
    monkeypatch.setenv("MTSA_PLANNER", "auto")
    with SE.rule_planner_env():
        assert os.environ["MTSA_PLANNER"] == "rule"
    assert os.environ["MTSA_PLANNER"] == "auto"


def test_rule_planner_env_removes_when_unset(monkeypatch):
    monkeypatch.delenv("MTSA_PLANNER", raising=False)
    with SE.rule_planner_env():
        assert os.environ["MTSA_PLANNER"] == "rule"
    assert "MTSA_PLANNER" not in os.environ
