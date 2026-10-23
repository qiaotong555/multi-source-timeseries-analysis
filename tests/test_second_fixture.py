# -*- coding: utf-8 -*-
"""C 第二份数据（30 天 / 15 分钟 / 单站点 / DST / 10% 缺失）的固化测试。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from evals.repro_check import compare
from mtsa import engine
from mtsa.models import AnalysisRequest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "fixtures"
CSV = FIX / "weather_short_v1.csv"
LABELS = FIX / "labels_weather_short_v1.csv"
TRUTH = FIX / "truth_weather_short_v1.json"


def test_short_fixture_shape_and_dst_gap():
    df = pd.read_csv(CSV, encoding="utf-8-sig", parse_dates=["timestamp"])
    assert len(df) == 2876
    assert df["station"].nunique() == 1
    assert 0.095 <= df["temperature"].isna().mean() <= 0.105
    ts = df["timestamp"].sort_values().reset_index(drop=True)
    diff = ts.diff().dt.total_seconds().dropna()
    assert diff.max() == 75 * 60          # 01:45 -> 03:00，DST 春季前跳
    assert (diff == 15 * 60).mean() > 0.98


def test_short_labels_and_truth():
    labels = pd.read_csv(LABELS, encoding="utf-8-sig")
    counts = labels["kind"].value_counts().to_dict()
    assert counts.get("anomaly") == 6
    assert counts.get("level_shift_start") == 1
    assert counts.get("missing_gap_start") == 1
    assert counts.get("dst_shift") == 1
    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    t = truth["series"]["temperature@STA_S"]
    assert t["trend_direction"] == "up"
    assert t["seasonal_available"] is False


def _run(out_dir: Path, tag: str) -> dict:
    res = engine.run_analysis(AnalysisRequest(
        request_id=tag,
        input_path=str(CSV),
        scene_id="weather",
        question="分析温度的趋势、日周期与异常",
        output_dir=str(out_dir),
        mode="pipeline",
    ), verbose=False)
    assert res.status == "succeeded"
    return json.loads((out_dir / "result.json").read_text(encoding="utf-8"))


def test_short_pipeline_reproducible_and_warns(tmp_path, monkeypatch):
    monkeypatch.setenv("MTSA_PLANNER", "rule")
    a = _run(tmp_path / "r1", "short_r1")
    b = _run(tmp_path / "r2", "short_r2")
    cmp = compare(a, b)
    assert cmp["identical"] is True, cmp["diffs"]
    assert len(a["conclusions"]) >= 10
    joined = "\n".join(a["warnings"])
    assert "样本量仅" in joined and "最大缺失率" in joined
    assert a["data"]["plan"]["plan_source"] == "rule"