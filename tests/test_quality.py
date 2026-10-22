# -*- coding: utf-8 -*-
"""质量检查测试：缺失、重复、乱序、缺口、异常候选、疑似单位变化。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mtsa import profiler, quality
from mtsa.io_loaders import build_spec
from mtsa.models import RawDataset
from .helpers import make_weather_frame


def _profile(df: pd.DataFrame):
    raw = RawDataset(frame=df, spec=build_spec("mem.csv"), file_hash="0" * 40,
                     row_count=len(df), column_dtypes={c: str(t) for c, t in df.dtypes.items()})
    return raw, *profiler.profile_dataset(raw)


def _codes(report):
    return {i.code for i in report.issues}


def test_detects_missing_values():
    df = make_weather_frame(n=600)
    df.loc[100:160, "temperature"] = np.nan
    raw, profile, parsed = _profile(df)
    rep = quality.check_quality(raw, profile, parsed)
    assert "MISSING_VALUES" in _codes(rep)
    assert rep.missing_ratio["temperature"] > 0.05


def test_detects_duplicates_and_out_of_order():
    df = make_weather_frame(n=600)
    df = pd.concat([df, df.iloc[200:210]], ignore_index=True)
    block = df.iloc[300:310]
    rest = pd.concat([df.iloc[:300], df.iloc[310:]], ignore_index=True)
    df = pd.concat([rest.iloc[:350], block, rest.iloc[350:]], ignore_index=True)
    raw, profile, parsed = _profile(df)
    rep = quality.check_quality(raw, profile, parsed)
    c = _codes(rep)
    assert "DUPLICATE_TIMESTAMPS" in c
    assert "OUT_OF_ORDER" in c


def test_detects_sampling_gap():
    df = make_weather_frame(n=600)
    df = df.drop(index=range(200, 260)).reset_index(drop=True)
    raw, profile, parsed = _profile(df)
    rep = quality.check_quality(raw, profile, parsed)
    assert "SAMPLING_GAP" in _codes(rep)
    assert rep.gap_count >= 1


def test_detects_outlier_candidates():
    df = make_weather_frame(n=1200)
    sd = float(df["temperature"].std())
    df.loc[300, "temperature"] += 9 * sd
    df.loc[700, "temperature"] -= 9 * sd
    raw, profile, parsed = _profile(df)
    rep = quality.check_quality(raw, profile, parsed)
    assert "OUTLIER_CANDIDATE" in _codes(rep)
    assert rep.anomaly_candidate_count >= 2


def test_detects_possible_unit_change():
    df = make_weather_frame(n=2000)
    df.loc[1000:1400, "temperature"] = df.loc[1000:1400, "temperature"] * 1.8 + 32.0
    raw, profile, parsed = _profile(df)
    rep = quality.check_quality(raw, profile, parsed)
    unit = [i for i in rep.issues if i.code == "POSSIBLE_UNIT_CHANGE"]
    assert unit, "未检出疑似单位变化"
    assert 2.0 < unit[0].count or "比值" in (unit[0].detail or "")


def test_robust_zscore_not_contaminated():
    x = pd.Series(np.concatenate([np.random.default_rng(1).normal(0, 1, 500), [80.0]]))
    z = quality.robust_zscore(x)
    assert abs(z.iloc[-1]) > 10
    assert z.abs().median() < 1.5


def test_time_axis_missing_reports_error():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    raw, profile, parsed = _profile(df)
    rep = quality.check_quality(raw, profile, parsed)
    assert "TIME_AXIS_NOT_FOUND" in _codes(rep)
