# -*- coding: utf-8 -*-
"""测试辅助：构造固定、脱敏的小样本数据，避免依赖公网与真实业务接口。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mtsa.models import AnalysisPlan


def make_weather_frame(n: int = 2000, seed: int = 7, stations: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    hour = idx.hour.to_numpy()
    frames = []
    for s in range(stations):
        temp = 15 + 6 * np.sin(2 * np.pi * (hour - 14) / 24) + rng.normal(0, 0.6, n)
        frames.append(pd.DataFrame({
            "timestamp": idx,
            "station": f"STA_{s}",
            "temperature": temp,
            "humidity": 60 - 0.5 * (temp - 15) + rng.normal(0, 1.5, n),
            "pressure": 1013 + rng.normal(0, 1.0, n),
            "wind_speed": np.abs(3 + rng.normal(0, 1.0, n)),
            "rainfall": np.where(rng.random(n) < 0.9, 0.0, rng.exponential(2.0, n)),
        }))
    return pd.concat(frames, ignore_index=True)


def make_stock_frame(n: int = 500, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    rets = 0.0004 + 0.011 * rng.normal(0, 1, n)
    close = 12.0 * np.exp(np.cumsum(rets))
    bench = 3200.0 * np.exp(np.cumsum(0.0003 + 0.009 * rng.normal(0, 1, n)))
    return pd.DataFrame({
        "date": idx.strftime("%Y-%m-%d"),
        "symbol": "DEMO.SZ",
        "open": close * (1 + rng.normal(0, 0.004, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "close": close,
        "volume": 1.2e7 * np.exp(rng.normal(0, 0.2, n)),
        "benchmark": bench,
    })


def make_plan(scene_id: str, columns: list[str], **params) -> AnalysisPlan:
    base = {"seasonal_period": None, "rolling_window": 24, "max_lag": 12,
            "anomaly_threshold": 4.0, "close_column": None, "volume_column": None,
            "benchmark_column": None}
    base.update(params)
    return AnalysisPlan(scene_id=scene_id, scene_confidence=1.0, question="测试",
                        target_columns=columns, group_column=None,
                        resample_rule=None, analyses=["trend", "corr", "anomaly", "changepoint"],
                        parameters=base)
