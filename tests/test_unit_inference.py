# -*- coding: utf-8 -*-
"""C1：单位推断的验收测试（先写后测）。

接缝：
- `quality.detect_level_shifts` / `quality.check_quality`：分类行为；
- `WeatherAdapter.plan`：是否自动换算；
- `engine.run_analysis`：真实数据端到端（R08 零换算、STA_A 仍转换）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mtsa import profiler, quality
from mtsa.engine import run_analysis
from mtsa.io_loaders import build_spec
from mtsa.models import AnalysisRequest, RawDataset
from mtsa.scenarios.weather import WeatherAdapter

ROOT = Path(__file__).resolve().parents[1]
R08 = ROOT / "fixtures" / "real_weather" / "r08_berlin_tegel_hourly_2024" / "data.csv"
WEATHER_V1 = ROOT / "fixtures" / "weather_v1.csv"


def _hourly_year(n: int = 4000, seed: int = 7) -> pd.Series:
    """小时级、带日周期与年周期的正常气温（°C），用于模拟 R08。"""
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    v = (12.0 + 8.0 * np.sin(2 * np.pi * i / 24.0)
         + 6.0 * np.sin(2 * np.pi * i / (24.0 * 365.0)) + rng.normal(0, 1.0, n))
    return pd.Series(v)


def _with_fahrenheit_block(s: pd.Series, lo: int, hi: int) -> pd.Series:
    out = s.copy()
    out.iloc[lo:hi] = out.iloc[lo:hi] * 1.8 + 32.0
    return out


def _frame(s: pd.Series, column: str = "temperature") -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=len(s), freq="h"),
        "station": "STA_A",
        column: s.to_numpy(),
    })


def _quality_and_plan(frame: pd.DataFrame):
    raw = RawDataset(frame=frame, spec=build_spec("mem.csv"), file_hash="0" * 40,
                     row_count=len(frame),
                     column_dtypes={c: str(t) for c, t in frame.dtypes.items()})
    profile, parsed = profiler.profile_dataset(raw)
    qrep = quality.check_quality(raw, profile, parsed)
    plan = WeatherAdapter().plan("请检查这份数据的单位变化", profile, qrep, frame)
    return qrep, plan


# ---------- 分类行为 ----------

def test_seasonal_warming_is_not_labeled_as_unit_change():
    hits = quality.detect_level_shifts(_hourly_year(), temperature=True)
    assert not [h for h in hits if h["kind"] == "unit"], hits


def test_true_fahrenheit_block_is_labeled_at_both_boundaries():
    s = _with_fahrenheit_block(_hourly_year(), 1200, 2200)
    hits = [h for h in quality.detect_level_shifts(s, temperature=True)
            if h["kind"] == "unit"]
    assert len(hits) == 2, hits


def test_r08_real_data_has_no_unit_change():
    df = pd.read_csv(R08, encoding="utf-8-sig")
    hits = quality.detect_level_shifts(df["temperature"], temperature=True)
    assert not [h for h in hits if h["kind"] == "unit"], hits


# ---------- 计划层：单侧不换算，成对才换算 ----------

def test_single_transition_is_not_converted():
    s = _with_fahrenheit_block(_hourly_year(), 1200, 4000)   # 只进入、不退出
    qrep, plan = _quality_and_plan(_frame(s))
    assert any(i.code == "POSSIBLE_UNIT_CHANGE" for i in qrep.issues)
    assert plan.parameters["unit_conversions"] == []
    assert plan.parameters.get("unit_change_open_intervals"), plan.parameters


def test_paired_transitions_are_converted_once():
    s = _with_fahrenheit_block(_hourly_year(), 1200, 2200)
    qrep, plan = _quality_and_plan(_frame(s))
    convs = plan.parameters["unit_conversions"]
    assert len(convs) == 1, convs
    assert convs[0]["start_time"] < convs[0]["end_time"]
    assert convs[0]["kind"] == "fahrenheit_to_celsius"


# ---------- 端到端 ----------

def _run(path: Path, out: Path) -> dict:
    res = run_analysis(AnalysisRequest(
        request_id="c1_test", input_path=str(path), scene_id="weather",
        question="分析温度的趋势和异常", mode="pipeline", seed=20260927,
        output_dir=str(out)), verbose=False)
    assert res.status == "succeeded", res.error
    return json.loads((out / "result.json").read_text(encoding="utf-8"))


def test_r08_end_to_end_has_no_conversion(tmp_path):
    doc = _run(R08, tmp_path)
    assert doc["data"]["clean"]["unit_conversions"] == []
    raw_mean = float(pd.read_csv(R08, encoding="utf-8-sig")["temperature"].mean())
    metrics = {m["metric_id"]: m["value"] for m in doc["data"]["metrics"]}
    assert abs(metrics["ts.desc.mean:temperature"] - raw_mean) < 1e-9


@pytest.mark.slow
def test_demo_sta_a_still_converts(tmp_path):
    doc = _run(WEATHER_V1, tmp_path)
    convs = doc["data"]["clean"]["unit_conversions"]
    assert len(convs) == 1, convs
    assert convs[0]["rows"] == 713
    assert abs(convs[0]["median_before"] - 76.9392) < 1e-3
    assert abs(convs[0]["median_after"] - 24.9662) < 1e-3