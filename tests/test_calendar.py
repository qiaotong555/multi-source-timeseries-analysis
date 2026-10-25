# -*- coding: utf-8 -*-
"""日历热力图（D3）：字段映射错一格，图上就会把某天的值挪到别的星期。

这些断言不依赖渲染：只看 `go.Heatmap` 的 z 矩阵与标签，所以不需要浏览器。
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from mtsa import viz
from mtsa.scenarios.weather import WeatherAdapter

from .helpers import make_plan


def _ctx(frame: pd.DataFrame):
    return types.SimpleNamespace(clean=types.SimpleNamespace(frame=frame))


def _daily_frame(start="2024-01-01", end="2024-12-31", drop=()):
    idx = pd.date_range(start, end, freq="D")
    idx = idx[~idx.isin([pd.Timestamp(d) for d in drop])]
    return pd.DataFrame({"temperature": np.arange(len(idx), dtype=float)}, index=idx)


def _z(fig):
    trace = fig.data[0]
    return np.array(trace.z, dtype=float), list(trace.x), list(trace.y)


def test_calendar_grid_is_seven_rows_by_iso_weeks():
    frame = _daily_frame()
    z, x, y = _z(viz._fig_calendar(_ctx(frame), {"columns": ["temperature"]}))
    assert z.shape == (7, len(x))
    assert y == ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    assert x[0] == "2024-W01"
    assert all(len(lbl.split("-W")) == 2 and lbl[:4].isdigit() for lbl in x)
    # 每天一格：非空格子数等于天数（2024 是闰年 366 天）
    assert int((~np.isnan(z)).sum()) == len(frame)


def test_known_date_lands_in_its_own_weekday_cell():
    frame = _daily_frame()
    spec = {"columns": ["temperature"]}
    fig = viz._fig_calendar(_ctx(frame), spec)
    z, x, _ = _z(fig)
    row, col = 2, x.index("2024-W01")          # 2024-01-03 是周三
    assert not np.isnan(z[row, col])
    assert "2024-01-03" in fig.data[0].text[row][col]
    # 同一周的周一（01-01）在第 0 行，值不同——两格没有串位
    assert z[0, col] != z[row, col]


def test_missing_days_stay_blank_instead_of_zero():
    frame = _daily_frame(drop=("2024-03-05", "2024-03-06", "2024-03-07"))
    z, x, _ = _z(viz._fig_calendar(_ctx(frame), {"columns": ["temperature"]}))
    assert int((~np.isnan(z)).sum()) == len(frame)        # 少 3 天就少 3 格
    col = x.index("2024-W10")
    blank = [r for r in range(7) if np.isnan(z[r, col])]
    assert len(blank) == 3                                 # 没有被填成 0


def test_agg_switch_changes_the_daily_value():
    idx = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 12:00"])
    frame = pd.DataFrame({"rainfall": [1.0, 3.0]}, index=idx)
    z_mean, _, _ = _z(viz._fig_calendar(_ctx(frame), {"columns": ["rainfall"], "agg": "mean"}))
    z_sum, _, _ = _z(viz._fig_calendar(_ctx(frame), {"columns": ["rainfall"], "agg": "sum"}))
    assert z_mean[0, 0] == pytest.approx(2.0)
    assert z_sum[0, 0] == pytest.approx(4.0)


def test_calendar_uses_only_the_first_column():
    """多列时只画第一列：一张日历画两个变量会互相覆盖。"""
    frame = _daily_frame()
    frame["humidity"] = 50.0
    fig = viz._fig_calendar(_ctx(frame), {"columns": ["temperature", "humidity"]})
    assert "temperature" in fig.data[0].text[0][0]
    assert "humidity" not in fig.data[0].text[0][0]


def test_calendar_rejects_bad_spec():
    frame = _daily_frame()
    with pytest.raises(ValueError, match="mean/sum/max/min"):
        viz._fig_calendar(_ctx(frame), {"columns": ["temperature"], "agg": "median"})
    with pytest.raises(ValueError, match="时间索引"):
        viz._fig_calendar(_ctx(frame.reset_index(drop=True)), {"columns": ["temperature"]})
    with pytest.raises(ValueError, match="没有可用列"):
        viz._fig_calendar(_ctx(frame), {"columns": ["nope"]})


def test_weather_scene_declares_a_calendar_figure():
    specs = WeatherAdapter().figure_specs(make_plan("weather", ["temperature", "humidity"]))
    kinds = [s["kind"] for s in specs]
    assert "calendar" in kinds
    cal = next(s for s in specs if s["kind"] == "calendar")
    assert cal["columns"] == ["temperature"]        # 只用第一列
    assert cal["agg"] == "mean"
    assert viz._BUILDERS.get("calendar") is viz._fig_calendar