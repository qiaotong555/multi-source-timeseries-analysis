# -*- coding: utf-8 -*-
"""预处理测试：排序去重、插值、单位换算、重采样、只读语义。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mtsa import profiler, preprocess
from mtsa.io_loaders import build_spec
from mtsa.models import RawDataset
from .helpers import make_weather_frame


def _prep(df: pd.DataFrame, plan):
    raw = RawDataset(frame=df, spec=build_spec("mem.csv"), file_hash="0" * 40,
                     row_count=len(df), column_dtypes={c: str(t) for c, t in df.dtypes.items()})
    profile, parsed = profiler.profile_dataset(raw)
    return raw, profile, parsed, preprocess.clean_dataset(raw, profile, parsed, plan)


def test_sorts_and_deduplicates(make_plan):
    df = make_weather_frame(n=600)
    df = pd.concat([df, df.iloc[100:105]], ignore_index=True)
    df = df.sample(frac=1.0, random_state=3).reset_index(drop=True)
    plan = make_plan("weather", ["temperature"])
    _, _, _, clean = _prep(df, plan)
    assert clean.frame.index.is_monotonic_increasing
    assert not clean.frame.index.duplicated().any()
    assert any(o["op"] == "aggregate_duplicate_timestamps" for o in clean.operations)


def test_interpolates_missing(make_plan):
    # 单站点：多站点共享时间戳时，缺失会被同时刻另一站点的取值填补，无法验证插值
    df = make_weather_frame(n=600, stations=1)
    df.loc[100:130, "temperature"] = np.nan
    plan = make_plan("weather", ["temperature"])
    _, _, _, clean = _prep(df, plan)
    assert clean.imputed_cells > 0
    assert clean.frame["temperature"].isna().sum() == 0


def test_unit_conversion_fahrenheit_to_celsius(make_plan):
    df = make_weather_frame(n=2000, stations=1)
    start_label = df.index[1200]
    df.loc[1200:1600, "temperature"] = df.loc[1200:1600, "temperature"] * 1.8 + 32.0
    plan = make_plan("weather", ["temperature"], unit_conversions=[{
        "column": "temperature", "kind": "fahrenheit_to_celsius",
        "start_index": start_label, "end_index": df.index[1600],
        "from_unit": "°F", "to_unit": "°C"}])
    _, _, _, clean = _prep(df, plan)
    assert clean.unit_conversions, "未执行单位换算"
    conv = clean.unit_conversions[0]
    assert conv["rows"] == 401
    # 换算后应回到摄氏度区间，且不再出现华氏量级
    assert 5.0 < conv["median_after"] < 30.0
    assert conv["median_before"] > 50.0


def test_unit_conversion_respects_group(make_plan):
    """多实体数据必须只换算指定分组，不能影响其他实体。"""
    df = make_weather_frame(n=1200, stations=2)
    rows_a = df.index[df["station"] == "STA_0"]
    df.loc[rows_a[600:900], "temperature"] = df.loc[rows_a[600:900], "temperature"] * 1.8 + 32.0
    plan = make_plan("weather", ["temperature"], unit_conversions=[{
        "column": "temperature", "kind": "fahrenheit_to_celsius",
        "group_column": "station", "group_value": "STA_0",
        "start_index": rows_a[600], "end_index": rows_a[900],
        "from_unit": "°F", "to_unit": "°C"}])
    plan.group_column = "station"
    _, _, _, clean = _prep(df, plan)
    conv = clean.unit_conversions[0]
    assert conv["rows"] == 301          # 起止标签均为闭区间
    assert conv["group_value"] == "STA_0"
    col_b = [c for c in clean.frame.columns if c.startswith("temperature@") and c.endswith("STA_1")]
    assert col_b and clean.frame[col_b[0]].median() < 30.0   # STA_1 未被误换算


def test_resample_hourly_to_daily(make_plan):
    df = make_weather_frame(n=24 * 30)
    plan = make_plan("weather", ["temperature"])
    plan.resample_rule = "1D"
    _, _, _, clean = _prep(df, plan)
    assert len(clean.frame) == 30
    assert any(o["op"] == "resample" for o in clean.operations)


def test_does_not_modify_input_frame(make_plan):
    df = make_weather_frame(n=600)
    before = df.copy()
    plan = make_plan("weather", ["temperature"])
    _prep(df, plan)
    pd.testing.assert_frame_equal(df, before)
