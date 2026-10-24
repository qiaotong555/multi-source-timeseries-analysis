# -*- coding: utf-8 -*-
"""D1：L2 官方事件清单与事件级异常评分。

接缝：
- `fixtures/real_weather/l2_verified_events.csv` 与 `l2_sources/*` 的可追溯性；
- `evals.real_data_eval.score_anomaly_events` 的 Recall / Precision / F1 与 N/A 语义。
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import pytest

from evals.real_data_eval import (load_l2_events, l2_source_status,
                                  score_anomaly_events, _anomaly_episodes)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "fixtures" / "real_weather" / "l2_sources" / "noaa_stormevents_selected.csv"


def _events():
    return load_l2_events()


def test_l2_covers_multiple_datasets_columns_and_types():
    ev = _events()
    assert len(ev) >= 12
    assert len({e["dataset"] for e in ev}) >= 2
    assert len({e["column"] for e in ev}) >= 3
    types = {t for e in ev for t in (e.get("event_types") or "").split(";") if t}
    assert len(types) >= 3


def test_every_event_traces_to_archived_source_row():
    n_ids, n_found = l2_source_status()
    assert n_ids >= 12
    assert n_ids == n_found
    have = {r["EVENT_ID"] for r in csv.DictReader(SRC.open(encoding="utf-8-sig"))}
    assert have


def test_source_manifest_has_urls_and_hashes():
    p = ROOT / "fixtures" / "real_weather" / "l2_sources" / "sources.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert doc["event_count"] == len(_events())
    for year in range(2018, 2025):
        f = doc["files"][str(year)]
        assert f["url"].startswith("https://www.ncei.noaa.gov/")
        assert len(f["sha1"]) == 40


def test_event_windows_fall_inside_dataset_range():
    import pandas as pd
    reg = json.loads((ROOT / "fixtures" / "real_weather" / "registry.json")
                     .read_text(encoding="utf-8"))["datasets"]
    span = {m["id"]: (pd.Timestamp(m["start"]), pd.Timestamp(m["end"])) for m in reg}
    for e in _events():
        s, t = pd.Timestamp(e["start"]), pd.Timestamp(e["end"])
        lo, hi = span[e["dataset"]]
        assert lo <= s <= t <= hi


def _ev(start, end, col="temperature"):
    return {"dataset": "R03", "column": col, "start": start, "end": end,
            "event_types": "Heat", "station": "PHX"}


def test_score_counts_hits_and_episodes():
    events = [_ev("2023-07-01", "2023-07-05"), _ev("2023-08-01", "2023-08-02")]
    anomalies = [
        {"timestamp": "2023-07-02 00:00:00", "column": "temperature"},
        {"timestamp": "2023-07-03 00:00:00", "column": "temperature"},
        {"timestamp": "2023-07-20 00:00:00", "column": "temperature"},
    ]
    r = score_anomaly_events(events, anomalies)
    assert r["status"] == "scored"
    assert r["matched"] == 1 and r["recall"] == 0.5
    assert r["episodes"] == 2 and r["hit_episodes"] == 1
    assert r["precision"] == 0.5 and r["f1"] == 0.5


def test_episode_clusters_consecutive_days_only():
    import pandas as pd
    pts = [(pd.Timestamp("2023-07-01"), "t"), (pd.Timestamp("2023-07-02"), "t"),
           (pd.Timestamp("2023-07-05"), "t")]
    assert len(_anomaly_episodes(pts)) == 2


def test_tool_not_run_is_na_not_zero():
    r = score_anomaly_events([_ev("2023-07-01", "2023-07-05")], [], executed=False)
    assert r["status"] == "tool_not_run"
    assert r["recall"] is None and r["precision"] is None and r["f1"] is None


def test_no_l2_is_na():
    r = score_anomaly_events([], [{"timestamp": "2023-07-02", "column": "temperature"}])
    assert r["status"] == "no_l2" and r["recall"] is None
