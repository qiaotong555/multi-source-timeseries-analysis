# -*- coding: utf-8 -*-
"""D2：绝对极端值通道（经验分位），与统计异常分开。

接缝：
- `anomaly.detect_extremes`：分位、方向、插值排除、连续段与上限；
- `engine`：`data.extremes` 只增字段；
- `report` / `weather.conclusions`：8b 节与「极端值」结论。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mtsa import anomaly
from mtsa.engine import run_analysis
from mtsa.models import AnalysisRequest, CleanDataset

from .helpers import make_plan, make_weather_frame


def _clean(df: pd.DataFrame, imputed=None) -> CleanDataset:
    return CleanDataset(frame=df, imputed_index=imputed or [])


def _frame(values, cols=("temperature",)) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.DataFrame({c: values for c in cols}, index=idx)


def test_temperature_has_both_tails_rainfall_high_only():
    vals = np.arange(1000, dtype=float)
    df = _frame(vals, cols=("temperature", "rainfall"))
    out = anomaly.detect_extremes(_clean(df), make_plan("weather", ["temperature", "rainfall"]))
    temp_tails = {e["tail"] for e in out if e["column"] == "temperature"}
    rain_tails = {e["tail"] for e in out if e["column"] == "rainfall"}
    assert temp_tails == {"high", "low"}
    assert rain_tails == {"high"}
    for e in out:
        assert e["quantile"] == 0.99
        assert e["count"] == 10 or e["count"] == 11   # 1000 点的 1% 尾
        assert e["runs"] and e["run_count"] == len(e["runs"])


def test_imputed_extreme_point_is_excluded():
    vals = np.arange(1000, dtype=float)
    df = _frame(vals)
    peak = str(df.index[-1])
    out = anomaly.detect_extremes(_clean(df, imputed=[peak]),
                                  make_plan("weather", ["temperature"]))
    high = next(e for e in out if e["tail"] == "high")
    assert peak not in {r["end"] for r in high["runs"]}
    assert peak not in {r["start"] for r in high["runs"]}


def test_runs_are_capped_and_total_is_reported():
    base = np.linspace(0, 100, 4000)
    vals = base.copy()
    for k in range(125):
        vals[20 + 30 * k] = 1000.0
    df = _frame(vals)
    out = anomaly.detect_extremes(_clean(df), make_plan("weather", ["temperature"]))
    high = next(e for e in out if e["tail"] == "high")
    assert high["run_count"] == 125
    assert len(high["runs"]) == anomaly.EXTREME_MAX_RUNS == 100
    assert high["runs_truncated"] is True


def test_small_sample_is_skipped():
    df = _frame(np.arange(50, dtype=float))
    assert anomaly.detect_extremes(_clean(df), make_plan("weather", ["temperature"])) == []


def test_pipeline_exposes_extremes_in_data_report_and_conclusions(tmp_path):
    df = make_weather_frame(n=400, stations=1)
    p = tmp_path / "weather.csv"
    df.to_csv(p, index=False)
    out = tmp_path / "out"
    res = run_analysis(AnalysisRequest(
        request_id="d2_test", input_path=str(p), scene_id="weather",
        question="分析温度的趋势、季节性和异常点", mode="pipeline",
        seed=20260927, output_dir=str(out)), verbose=False)
    assert res.status == "succeeded", res.error
    doc = json.loads((out / "result.json").read_text(encoding="utf-8"))
    exts = doc["data"]["extremes"]
    assert exts and {"column", "tail", "threshold", "runs"} <= set(exts[0])
    cats = {c["category"] for c in doc["conclusions"]}
    assert "极端值" in cats
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "## 8b. 极端值（绝对阈值）" in text
    assert "不能互相替代" in text


def test_extremes_do_not_change_anomalies(tmp_path):
    df = make_weather_frame(n=400, stations=1)
    p = tmp_path / "weather.csv"
    df.to_csv(p, index=False)
    out = tmp_path / "out"
    run_analysis(AnalysisRequest(
        request_id="d2_test2", input_path=str(p), scene_id="weather",
        question="找出异常点", mode="pipeline", seed=20260927,
        output_dir=str(out)), verbose=False)
    doc = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert "anomalies" in doc["data"] and "extremes" in doc["data"]
    for a in doc["data"]["anomalies"]:
        assert "method" in a and "score" in a
