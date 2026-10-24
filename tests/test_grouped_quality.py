# -*- coding: utf-8 -*-
"""C2：多实体时间轴的重复/乱序必须按实体判定。

接缝：
- `profiler.profile_dataset`：`duplicated_count` / `out_of_order_count` 的口径；
- `quality.check_quality`：issue 是否带 group、是否误报；
- `engine.run_analysis`：R05 端到端不再报 17,568 条重复。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mtsa import profiler, quality
from mtsa.engine import run_analysis
from mtsa.io_loaders import build_spec
from mtsa.models import AnalysisRequest, RawDataset

from .helpers import make_weather_frame

ROOT = Path(__file__).resolve().parents[1]
R05 = ROOT / "fixtures" / "real_weather" / "r05_three_cities_hourly_2024" / "data.csv"


def _profile(df: pd.DataFrame):
    raw = RawDataset(frame=df, spec=build_spec("mem.csv"), file_hash="0" * 40,
                     row_count=len(df),
                     column_dtypes={c: str(t) for c, t in df.dtypes.items()})
    profile, parsed = profiler.profile_dataset(raw)
    return raw, profile, parsed


def _issues(rep, code):
    return [i for i in rep.issues if i.code == code]


def test_two_stations_sharing_a_timeline_are_not_duplicates():
    df = make_weather_frame(n=120, stations=2)
    raw, profile, parsed = _profile(df)
    rep = quality.check_quality(raw, profile, parsed)
    assert profile.time_axis.duplicated_count == 0
    assert rep.duplicate_timestamps == 0
    assert _issues(rep, "DUPLICATE_TIMESTAMPS") == []
    assert rep.duplicate_by_group == {}


def test_duplicate_within_one_station_is_reported_with_group():
    df = make_weather_frame(n=120, stations=2)
    a = df[df["station"] == "STA_0"].iloc[:3]
    df = pd.concat([df, a], ignore_index=True)
    raw, profile, parsed = _profile(df)
    rep = quality.check_quality(raw, profile, parsed)
    issues = _issues(rep, "DUPLICATE_TIMESTAMPS")
    assert len(issues) == 1
    assert issues[0].group == "STA_0"
    assert issues[0].count == 3
    assert rep.duplicate_timestamps == 3
    assert rep.duplicate_by_group == {"STA_0": 3}


def test_interleaved_stations_are_not_out_of_order():
    df = make_weather_frame(n=100, stations=2).sort_values(
        ["timestamp", "station"]).reset_index(drop=True)
    raw, profile, parsed = _profile(df)
    rep = quality.check_quality(raw, profile, parsed)
    assert profile.time_axis.out_of_order_count == 0
    assert _issues(rep, "OUT_OF_ORDER") == []


def test_out_of_order_within_a_station_is_reported_with_group():
    df = make_weather_frame(n=100, stations=2)
    idx = df.index[df["station"] == "STA_0"][:2].tolist()
    df.loc[idx, "timestamp"] = df.loc[idx[::-1], "timestamp"].to_numpy()
    raw, profile, parsed = _profile(df)
    rep = quality.check_quality(raw, profile, parsed)
    issues = _issues(rep, "OUT_OF_ORDER")
    assert issues and issues[0].group == "STA_0"
    assert rep.out_of_order >= 1


def test_single_series_duplicate_keeps_global_issue():
    df = make_weather_frame(n=60, stations=1)
    df = pd.concat([df, df.iloc[:2]], ignore_index=True)
    raw, profile, parsed = _profile(df)
    rep = quality.check_quality(raw, profile, parsed)
    issues = _issues(rep, "DUPLICATE_TIMESTAMPS")
    assert len(issues) == 1 and issues[0].group is None
    assert rep.duplicate_timestamps == 2


def test_r05_end_to_end_reports_no_duplicates(tmp_path):
    res = run_analysis(AnalysisRequest(
        request_id="c2_test", input_path=str(R05), scene_id="weather",
        question="检查重复时间戳", mode="pipeline", seed=20260927,
        output_dir=str(tmp_path)), verbose=False)
    assert res.status == "succeeded", res.error
    doc = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    q = doc["data"]["quality"]
    assert q["duplicate_timestamps"] == 0
    assert "DUPLICATE_TIMESTAMPS" not in {i["code"] for i in q["issues"]}
    assert doc["data"]["profile"]["time_axis"]["duplicated_count"] == 0
    assert doc["data"]["clean"]["rows"] == 8784
    assert len(doc["data"]["plan"]["target_columns"]) == 15