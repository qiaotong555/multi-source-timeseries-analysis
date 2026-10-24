# -*- coding: utf-8 -*-
"""生成方案 2 的独立评估集 weather_eval_v2。

与 weather_v1 / weather_eval_v1 的区别：
- 独立 seed（20261024）与 365 天长度；
- 5 列 × 2 站点的异常/变化点排布不同；
- 标注覆盖异常点/持续段、均值跳变、缺失缺口、趋势与年周期。

运行：
    python fixtures/gen_weather_eval_v2.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from fixtures.gen_fixtures import _ar1

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261024
START = "2024-01-01 00:00:00"
HOURS = 8760
STATIONS = ("STA_A", "STA_B")
COLUMNS = ("temperature", "humidity", "pressure", "wind_speed", "rainfall")

# 残差尺度（去趋势/季节后的近似标准差），用于把注入幅度换算成稳健 Z
RESID_SIGMA = {
    "temperature": 0.55,
    "humidity": 1.6,
    "pressure": 1.2,
    "wind_speed": 0.9,
    "rainfall": 2.2,
}

# (column, station, day, delta, direction)
CHANGEPOINTS = [
    ("pressure", "STA_A", 60, 60.0, "up"),
    ("pressure", "STA_B", 200, 60.0, "up"),
]

def _allowed(rng, n: int, exclude_days: list[int], gap_day: int, k: int,
             span: int = 1) -> list[int]:
    """在 [0,n) 内取 k 个位置，避开变化点 ±3 天与缺失缺口。"""
    mask = np.ones(n, dtype=bool)
    for d in exclude_days:
        lo = max(0, d * 24 - 72)
        hi = min(n, d * 24 + 72 + span)
        mask[lo:hi] = False
    lo = max(0, gap_day * 24 - 24)
    hi = min(n, gap_day * 24 + 24 + span)
    mask[lo:hi] = False
    allowed = np.flatnonzero(mask)
    if len(allowed) < k:
        raise ValueError(f"可用位置不足：{len(allowed)} < {k}")
    picked = sorted(int(x) for x in rng.choice(allowed, size=k, replace=False))
    # 同一列的点之间至少间隔 2 天，避免被合并成一个事件
    for a, b in zip(picked, picked[1:]):
        if b - a < 48:
            raise ValueError("异常位置过近，需调整抽样")
    return picked


def gen() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED)
    idx = pd.date_range(START, periods=HOURS, freq="h")
    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    t_year = np.arange(HOURS) / (365.0 * 24.0)
    gap_day = 200

    frames = []
    for s_i, station in enumerate(STATIONS):
        phase = 0.0
        annual = 10.0 * np.sin(2 * np.pi * (doy - 100) / 365.0 + phase)
        diurnal = 6.0 * np.sin(2 * np.pi * (hour - 14) / 24.0 + phase)
        trend = 0.8 * t_year
        temp = 15.0 + annual + diurnal + trend + _ar1(rng, HOURS, 0.95, 0.45) + (1.0 if s_i == 0 else -1.0)

        humidity = 65.0 - 0.7 * (temp - 15.0) + _ar1(rng, HOURS, 0.92, 2.0)
        humidity = np.clip(humidity, 5.0, 100.0)
        pressure = 1013.0 + 6.0 * np.sin(2 * np.pi * doy / 365.0 + phase) + _ar1(rng, HOURS, 0.90, 1.0)
        wind = np.abs(3.0 + _ar1(rng, HOURS, 0.90, 0.9) + 1.0 * np.sin(2 * np.pi * hour / 24.0))
        rain_u = rng.random(HOURS)
        rainfall = np.where(rain_u < 0.93, 0.0,
                            rng.exponential(2.2, HOURS) * (rain_u - 0.92) / 0.08)

        frames.append(pd.DataFrame({
            "timestamp": idx, "station": station,
            "temperature": temp, "humidity": humidity, "pressure": pressure,
            "wind_speed": wind, "rainfall": rainfall,
        }))

    data = pd.concat(frames, ignore_index=True)

    # ---- 变化点：按站点在指定日期之后整体平移 ----
    labels = []
    cp_days = [d for _, _, d, _, _ in CHANGEPOINTS]
    for col, station, day, delta, direction in CHANGEPOINTS:
        m = data["station"] == station
        ts = data.loc[m, "timestamp"]
        mask = ts >= (pd.Timestamp(START) + pd.Timedelta(days=day))
        rows = data.index[m][mask.to_numpy()]
        data.loc[rows, col] = data.loc[rows, col] + delta
        if col == "humidity":
            data[col] = data[col].clip(5.0, 100.0)
        labels.append({
            "column": col, "timestamp": str(pd.Timestamp(START) + pd.Timedelta(days=day)),
            "kind": "level_shift_start", "station": station,
            "direction": direction, "delta": delta,
        })

    # ---- 异常：每站每列 2 个孤立点 + 1 个 8 小时持续段 ----
    for station in STATIONS:
        m = data["station"] == station
        rows = data.index[m].to_numpy()
        n_st = len(rows)
        for col in COLUMNS:
            sd = RESID_SIGMA[col]
            pts = _allowed(rng, n_st, cp_days, gap_day, k=2)
            for j, p in enumerate(pts):
                r = int(rows[p])
                z = float(rng.uniform(18.0, 24.0))
                sg = 1.0 if rng.random() < 0.5 else -1.0
                data.loc[r, col] = data.loc[r, col] + sg * z * sd
                labels.append({
                    "column": col, "timestamp": str(data.loc[r, "timestamp"]),
                    "kind": "anomaly", "station": station,
                    "anomaly_type": "point", "event_id": f"{col}_{station}_p{j}",
                })
            starts = _allowed(rng, n_st, cp_days, gap_day, k=1, span=8)
            p = starts[0]
            z = float(rng.uniform(15.0, 18.0))
            sg = 1.0 if rng.random() < 0.5 else -1.0
            eid = f"{col}_{station}_c0"
            for r in rows[p:p + 8]:
                jitter = 0.9 + 0.2 * float(rng.random())
                data.loc[int(r), col] = data.loc[int(r), col] + sg * z * sd * jitter
                labels.append({
                    "column": col, "timestamp": str(data.loc[int(r), "timestamp"]),
                    "kind": "anomaly", "station": station,
                    "anomaly_type": "collective", "event_id": eid,
                })

    # ---- 缺失：8% 随机 + 一个 12 小时连续缺口 ----
    anomaly_rows = {i for i, x in enumerate(labels) if x["kind"] == "anomaly"}
    # 标签里存的是 timestamp，转换成行号集合以便跳过
    anomaly_ts = {(x["station"], x["timestamp"], x["column"]) for x in labels if x["kind"] == "anomaly"}
    gap_mask = (data["station"] == "STA_A") & (
        (data["timestamp"] >= pd.Timestamp(START) + pd.Timedelta(days=gap_day)) &
        (data["timestamp"] < pd.Timestamp(START) + pd.Timedelta(days=gap_day, hours=12))
    )
    gap_rows = data.index[gap_mask]
    data.loc[gap_rows, "temperature"] = np.nan
    labels.append({
        "column": "temperature", "timestamp": str(pd.Timestamp(START) + pd.Timedelta(days=gap_day)),
        "kind": "missing_gap_start", "station": "STA_A",
    })
    # 缺失必须避开异常所在的时间戳（两个站点都不遮蔽）：
    # clean 的 imputed_index 是行级的，同一时刻任一列插值都会让整行被异常检测跳过。
    protected_ts = {pd.Timestamp(x["timestamp"]) for x in labels if x["kind"] == "anomaly"}
    protected_ts |= {pd.Timestamp(START) + pd.Timedelta(days=gap_day)}
    for col in COLUMNS:
        candidates = [i for i in data.index if data.at[i, "timestamp"] not in protected_ts]
        k = int(round(0.08 * len(candidates)))
        drop = rng.choice(candidates, size=k, replace=False)
        data.loc[drop, col] = np.nan

    # 排序保持时间顺序（多站点共享时间戳）
    data = data.sort_values(["timestamp", "station"]).reset_index(drop=True)

    truth = {
        "dataset": "weather_eval_v2",
        "generator": "fixtures/gen_weather_eval_v2.py",
        "seed": SEED,
        "start": START,
        "hours": HOURS,
        "stations": list(STATIONS),
        "tolerances": {
            "trend_direction": "sign",
            "seasonal_amplitude_abs": 3.0,
            "seasonal_month_tolerance": 1,
            "changepoint_samples": 2,
        },
        "series": {
            "temperature@STA_A": {
                "trend_per_year": 0.8, "trend_direction": "up",
                "annual_amplitude": 10.0, "annual_amplitude_peak_to_trough": 20.0,
                "peak_month": 7, "trough_month": 1,
                "diurnal_amplitude": 6.0, "diurnal_period_hours": 24.0,
            },
            "temperature@STA_B": {
                "trend_per_year": 0.8, "trend_direction": "up",
                "annual_amplitude": 10.0, "annual_amplitude_peak_to_trough": 20.0,
                "peak_month": 7, "trough_month": 1,
                "diurnal_amplitude": 6.0, "diurnal_period_hours": 24.0,
            },
        },
        "not_covered": ["humidity", "pressure", "wind_speed", "rainfall",
                        "period", "correlation", "lag"],
        "note": ("温度真值：年振幅 10°C（峰谷差 20）、峰月 7、谷月 1、日周期 6°C、"
                 "线性趋势 +0.8°C/年。变化点：pressure@STA_A day 60 +60 hPa；"
                 "pressure@STA_B day 200 +60 hPa。缺失缺口：temperature@STA_A "
                 "day 200 起连续 12 小时。"),
    }
    return data, pd.DataFrame(labels), truth


def main() -> None:
    data, labels, truth = gen()
    data.to_csv(os.path.join(HERE, "weather_eval_v2.csv"), index=False, encoding="utf-8-sig")
    labels.to_csv(os.path.join(HERE, "labels_weather_eval_v2.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(HERE, "truth_weather_eval_v2.json"), "w", encoding="utf-8") as f:
        json.dump(truth, f, ensure_ascii=False, indent=2)
    print("已生成 weather_eval_v2.csv / labels_weather_eval_v2.csv / truth_weather_eval_v2.json")
    print("rows=%d labels=%d anomaly_events=%d" % (
        len(data), len(labels),
        labels[labels["kind"] == "anomaly"]["event_id"].nunique()))


if __name__ == "__main__":
    main()
