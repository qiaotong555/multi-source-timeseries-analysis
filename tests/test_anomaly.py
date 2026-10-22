# -*- coding: utf-8 -*-
"""异常检测与变化点测试：对照注入真值计算 Precision / Recall。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mtsa import anomaly, profiler, preprocess
from mtsa.io_loaders import build_spec
from mtsa.models import RawDataset
from .helpers import make_weather_frame

MIN_PRECISION = 0.70
MIN_RECALL = 0.70


def _pipeline(df: pd.DataFrame, plan):
    raw = RawDataset(frame=df, spec=build_spec("mem.csv"), file_hash="0" * 40,
                     row_count=len(df), column_dtypes={c: str(t) for c, t in df.dtypes.items()})
    profile, parsed = profiler.profile_dataset(raw)
    clean = preprocess.clean_dataset(raw, profile, parsed, plan)
    return clean


def test_anomaly_precision_recall(make_plan):
    n = 3000
    df = make_weather_frame(n=n, seed=21, stations=1)
    rng = np.random.default_rng(99)
    sd = float(df["temperature"].std())
    truth = set()
    for _ in range(12):
        i = int(rng.integers(200, n - 200))
        df.loc[i, "temperature"] += (1 if rng.random() > 0.5 else -1) * (7.0 + 2.0 * rng.random()) * sd
        truth.add(str(df.loc[i, "timestamp"]))

    plan = make_plan("weather", ["temperature"], seasonal_period=24,
                     rolling_window=24, anomaly_threshold=4.0)
    clean = _pipeline(df, plan)
    detected = anomaly.detect_anomalies(clean, plan)
    got = {a.timestamp for a in detected if a.column == "temperature"}
    tp = len(truth & got)
    precision = tp / len(got) if got else 0.0
    recall = tp / len(truth)
    assert recall >= MIN_RECALL, f"Recall={recall:.3f} 低于 {MIN_RECALL}"
    assert precision >= MIN_PRECISION, f"Precision={precision:.3f} 低于 {MIN_PRECISION}"


def test_no_false_alarm_on_clean_series(make_plan):
    df = make_weather_frame(n=3000, seed=33, stations=1)
    plan = make_plan("weather", ["temperature"], seasonal_period=24, anomaly_threshold=4.5)
    clean = _pipeline(df, plan)
    detected = anomaly.detect_anomalies(clean, plan)
    rate = len(detected) / len(clean.frame)
    assert rate < 0.005, f"干净序列误报率过高：{rate:.4%}"


def test_change_point_locates_level_shift(make_plan):
    n = 2400
    df = make_weather_frame(n=n, seed=44, stations=1)
    shift_at = 1200
    df.loc[shift_at:, "temperature"] += 6.0      # 持续水平抬升 6°C
    plan = make_plan("weather", ["temperature"], seasonal_period=24, change_point_depth=2)
    clean = _pipeline(df, plan)
    cps = [c for c in anomaly.detect_change_points(clean, plan)
           if c.column == "temperature" and c.kind == "mean"]
    assert cps, "未检出变化点"
    best = min(cps, key=lambda c: abs(c.index - shift_at))
    assert abs(best.index - shift_at) <= 24, f"变化点定位偏差过大：{best.index} vs {shift_at}"


def test_anomaly_output_schema(make_plan):
    df = make_weather_frame(n=1500, seed=55, stations=1)
    sd = float(df["temperature"].std())
    df.loc[500, "temperature"] += 9 * sd
    plan = make_plan("weather", ["temperature"], seasonal_period=24)
    clean = _pipeline(df, plan)
    pts = anomaly.detect_anomalies(clean, plan)
    assert pts
    p = pts[0]
    assert isinstance(p.value, float) and isinstance(p.score, float)
    assert p.method and p.column == "temperature"
