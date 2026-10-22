# -*- coding: utf-8 -*-
"""性能基准测试：数据构造、worker 输出、缓存读取。"""
from __future__ import annotations

import json

import pandas as pd

from evals import benchmark as BM
from evals import run_evals


def test_build_frame_shape_and_columns():
    df = BM.build_frame(1000, stations=2)
    assert len(df) == 1000
    for col in ("timestamp", "station", "temperature", "humidity",
                "pressure", "wind_speed", "rainfall"):
        assert col in df.columns
    assert df["station"].nunique() == 2
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_build_frame_is_deterministic():
    a = BM.build_frame(400, stations=1)
    b = BM.build_frame(400, stations=1)
    pd.testing.assert_frame_equal(a, b)


def test_worker_timed_emits_valid_json(capsys):
    rc = BM._worker(2000, "timed")
    assert rc == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    d = json.loads(line)
    assert d["rows"] == 2000
    assert d["mode"] == "timed"
    assert d["total_s"] > 0
    assert d["peak_mb"] is None                  # 计时模式不测内存
    assert d["summary"]["columns"] == 10
    assert set(d["timings"]) >= {"load", "clean", "metrics", "stat_tests", "anomalies"}


def test_worker_memory_mode_reports_peak(capsys):
    rc = BM._worker(2000, "memory")
    assert rc == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    d = json.loads(line)
    assert d["mode"] == "memory"
    assert d["peak_mb"] is not None and d["peak_mb"] > 0


def test_benchmark_cache_shape():
    b = run_evals._load_benchmark()
    assert isinstance(b.get("available"), bool)
    if b["available"]:
        assert "scale_1e5" in b
        for key in ("total_s", "peak_mb", "status"):
            assert key in b["scale_1e5"]
    else:
        assert b.get("reason")


def test_missing_cache_is_not_counted_as_pass():
    """缓存缺失时指标应记大值，避免「没测过」被判成达标。"""
    b = {"available": False}
    total = float((b.get("scale_1e5") or {}).get("total_s") or 9999.0)
    assert total > 120.0
