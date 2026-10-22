# -*- coding: utf-8 -*-
"""字段与采样摘要测试：时间轴识别、变量角色、缺口检测。"""
from __future__ import annotations

import pandas as pd

from mtsa import profiler
from mtsa.io_loaders import build_spec
from mtsa.models import RawDataset
from .helpers import make_weather_frame


def _raw(df: pd.DataFrame, **kw) -> RawDataset:
    return RawDataset(frame=df, spec=build_spec("mem.csv", **kw),
                      file_hash="0" * 40, row_count=len(df),
                      column_dtypes={c: str(t) for c, t in df.dtypes.items()})


def test_detect_time_axis_by_name():
    df = make_weather_frame(n=200)
    col, parsed, ratio, is_epoch = profiler.detect_time_axis(df)
    assert col == "timestamp"
    assert ratio == 1.0
    assert not is_epoch


def test_detect_epoch_column():
    df = pd.DataFrame({"ts": [1700000000 + i * 3600 for i in range(50)], "v": range(50)})
    col, parsed, ratio, is_epoch = profiler.detect_time_axis(df)
    assert col == "ts" and is_epoch


def test_no_time_axis():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    col, _, _, _ = profiler.detect_time_axis(df)
    assert col is None


def test_profile_weather():
    df = make_weather_frame(n=1200)
    profile, parsed = profiler.profile_dataset(_raw(df))
    assert profile.time_axis.column == "timestamp"
    assert profile.time_axis.median_interval_seconds == 3600.0
    assert profile.sampling.interval_stable is True
    assert profile.sampling.gap_count == 0
    assert "temperature" in profile.numeric_columns
    assert "station" in profile.group_columns


def test_profile_detects_gap():
    df = make_weather_frame(n=600)
    df = df.drop(index=range(200, 260)).reset_index(drop=True)  # 60 小时缺口
    profile, _ = profiler.profile_dataset(_raw(df))
    assert profile.sampling.gap_count >= 1


def test_profile_detects_out_of_order_and_duplicates():
    df = make_weather_frame(n=400)
    block = df.iloc[100:110]
    rest = pd.concat([df.iloc[:100], df.iloc[110:]], ignore_index=True)
    df2 = pd.concat([rest.iloc[:150], block, rest.iloc[150:]], ignore_index=True)
    df2 = pd.concat([df2, df2.iloc[300:305]], ignore_index=True)
    profile, _ = profiler.profile_dataset(_raw(df2))
    assert profile.time_axis.out_of_order_count >= 1
    assert profile.time_axis.duplicated_count >= 1
