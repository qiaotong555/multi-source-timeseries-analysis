# -*- coding: utf-8 -*-
"""生成 C 任务用的第二份天气 fixture：30 天 / 15 分钟 / 单站点 / DST / 10% 缺失。

与 `gen_fixtures.py::gen_weather` 的区别是刻意改变采样结构，用来验证结论是否依赖
单一 fixture。数据是确定性的：固定 seed，重复运行输出一致。

产物：
    weather_short_v1.csv
    labels_weather_short_v1.csv
    truth_weather_short_v1.json

运行：
    python fixtures/gen_weather_short.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261023
START = "2024-03-01 00:00:00"
DAYS = 30
FREQ = "15min"
STATION = "STA_S"
# 美东 2024-03-10 02:00 春季前跳：本地时钟从 01:45 直接到 03:00。
DST_CUT = pd.Timestamp("2024-03-10 02:00:00")
DST_RESUME = pd.Timestamp("2024-03-10 03:00:00")


def _ar1(rng, n: int, phi: float, sigma: float) -> np.ndarray:
    x, out = 0.0, []
    for _ in range(n):
        x = phi * x + rng.normal(0.0, sigma)
        out.append(x)
    return np.asarray(out)


def gen() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED)
    idx = pd.date_range(START, periods=DAYS * 24 * 4, freq=FREQ)
    hour = idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0
    day = (idx - idx[0]).total_seconds().to_numpy() / 86400.0

    n = len(idx)
    temp_noise = _ar1(rng, n, 0.90, 0.35)
    temp = 12.0 + 3.0 * np.sin(2 * np.pi * (hour - 8) / 24.0) + 0.05 * day + temp_noise
    hum_noise = _ar1(rng, n, 0.92, 1.8)
    humidity = np.clip(65.0 - 0.6 * (temp - 12.0) + hum_noise, 10.0, 100.0)
    pres_noise = _ar1(rng, n, 0.85, 0.9)
    pressure = 1013.0 + 2.0 * np.sin(2 * np.pi * day / 7.0) + pres_noise
    wind = np.abs(3.0 + _ar1(rng, n, 0.88, 0.8) + 1.0 * np.sin(2 * np.pi * hour / 24.0))
    rain_u = rng.random(n)
    rainfall = np.where(rain_u < 0.95, 0.0,
                        rng.exponential(2.0, n) * (rain_u - 0.94) / 0.06)

    df = pd.DataFrame({
        "timestamp": idx,
        "station": STATION,
        "temperature": temp,
        "humidity": humidity,
        "pressure": pressure,
        "wind_speed": wind,
        "rainfall": rainfall,
    })
    # DST 春季前跳：删掉本地 02:00–02:45 的 4 个点，保留跳变后的 03:00。
    df = df[~((df["timestamp"] >= DST_CUT) & (df["timestamp"] < DST_RESUME))].reset_index(drop=True)
    df["timezone"] = "America/New_York"

    labels: list[dict[str, str]] = []
    labels.append({"column": "timestamp", "timestamp": str(DST_RESUME),
                   "kind": "dst_shift", "station": STATION})

    # 10% 缺失：6 小时连续块 + 其余随机点。
    n = len(df)
    block_start = int(n * 0.48)
    block = list(range(block_start, block_start + 24))
    remain = [i for i in range(n) if i not in set(block)]
    n_missing = int(round(n * 0.10))
    random_missing = rng.choice(remain, size=max(0, n_missing - len(block)), replace=False).tolist()
    missing = sorted(set(block + random_missing))
    df.loc[missing, "temperature"] = np.nan
    labels.append({"column": "temperature", "timestamp": str(df.loc[block_start, "timestamp"]),
                   "kind": "missing_gap_start", "station": STATION})

    # 小幅异常：5.5 倍残差标准差，避开缺失点。
    def inject(column: str, count: int, sd: float, sign: int = 1) -> None:
        candidates = [i for i in range(n) if i not in set(missing)]
        rows = rng.choice(candidates, size=count, replace=False)
        for r in rows:
            df.loc[r, column] = float(df.loc[r, column]) + sign * 5.5 * sd
            labels.append({"column": column, "timestamp": str(df.loc[r, "timestamp"]),
                           "kind": "anomaly", "station": STATION})

    inject("temperature", 3, float(np.std(temp_noise)), 1)
    inject("pressure", 2, float(np.std(pres_noise)), 1)
    inject("humidity", 1, float(np.std(hum_noise)), -1)

    # 一个 +10 hPa 的水平跳变（变化点真值；是否检出由检测器决定）。
    cp = int(n * 0.60)
    df.loc[cp:, "pressure"] = df.loc[cp:, "pressure"] + 10.0
    labels.append({"column": "pressure", "timestamp": str(df.loc[cp, "timestamp"]),
                   "kind": "level_shift_start", "station": STATION})

    truth = {
        "dataset": "weather_short_v1",
        "generator": "fixtures/gen_weather_short.py",
        "seed": SEED,
        "start": START,
        "rows": int(len(df)),
        "station": STATION,
        "tolerances": {
            "trend_direction": "sign",
            "seasonal_amplitude_abs": None,
            "seasonal_month_tolerance": None,
            "changepoint_samples": 2,
        },
        "series": {
            f"temperature@{STATION}": {
                "trend_per_day": 0.05,
                "trend_per_year": round(0.05 * 365.25, 4),
                "trend_direction": "up",
                "diurnal_amplitude": 6.0,
                "diurnal_period_hours": 24.0,
                "annual_amplitude_peak_to_trough": None,
                "annual_amplitude": None,
                "peak_month": None,
                "trough_month": None,
                "seasonal_available": False,
            }
        },
        "not_covered": ["annual_seasonality", "humidity", "pressure", "wind_speed", "rainfall"],
        "note": "30 天不足以估计年周期；只把趋势方向与日周期作为真值，年季节性标 unlabeled。",
    }
    return df, pd.DataFrame(labels), truth


def main() -> None:
    df, labels, truth = gen()
    df.to_csv(os.path.join(HERE, "weather_short_v1.csv"), index=False, encoding="utf-8-sig")
    labels.to_csv(os.path.join(HERE, "labels_weather_short_v1.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(HERE, "truth_weather_short_v1.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("已生成 weather_short_v1.csv / labels_weather_short_v1.csv / truth_weather_short_v1.json")
    print("rows=%d labels=%d" % (len(df), len(labels)))


if __name__ == "__main__":
    main()