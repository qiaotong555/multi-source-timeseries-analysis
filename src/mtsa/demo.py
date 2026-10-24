# -*- coding: utf-8 -*-
"""`mtsa demo`：生成一份合成天气数据，跑完整链路并给出报告。

为什么要有它：刚装好包的人手里没有符合格式的数据，`mtsa run --input ...` 会直接
file not found。demo 用固定 seed 合成一份数据，让「装完 → 看到报告」之间不需要
准备数据、不需要模型、不需要联网。

数据是**合成的**，只用于演示链路与报告格式，不能当作真实气候结论。
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from .engine import run_analysis
from .models import DEFAULT_SEED, AnalysisRequest
from .timeout_guard import DEFAULT_TIMEOUT_S, run_with_timeout

DEMO_START = "2024-06-01 00:00:00"
DEMO_DAYS = 92
DEMO_ROWS_PER_DAY = 24
DEMO_CSV_NAME = "input.csv"
DEMO_QUESTION = "分析温度与气压的趋势、异常与变化点，并说明数据质量"

# 注入的「已知事件」：演示报告能不能把它们找出来
MISSING_DAY = 30
MISSING_HOURS = 6
HEAT_DAY = 45
HEAT_HOURS = 5
HEAT_DELTA_C = 10.0
STEP_DAY = 60
STEP_DELTA_HPA = -8.0
DUPLICATE_ROWS = 3


def build_demo_frame(seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """合成 92 天逐小时单站天气数据（2,208 行），并注入四类已知情况。"""
    rng = np.random.default_rng(seed)
    n = DEMO_DAYS * DEMO_ROWS_PER_DAY
    ts = pd.date_range(DEMO_START, periods=n, freq="h")
    hour = ts.hour.to_numpy()
    day = np.arange(n) / DEMO_ROWS_PER_DAY

    temp = (18.0
            + 6.0 * np.sin(2 * np.pi * (hour - 14) / 24)
            + 3.0 * np.sin(2 * np.pi * day / 90)
            + rng.normal(0.0, 0.6, n))
    humidity = np.clip(65.0 - 0.8 * (temp - 18.0) + rng.normal(0.0, 3.0, n), 20.0, 100.0)
    pressure = 1013.0 + 2.0 * np.sin(2 * np.pi * day / 7) + rng.normal(0.0, 0.4, n)
    wind = np.clip(3.0 + rng.normal(0.0, 1.0, n), 0.0, None)
    rain = np.where(rng.random(n) < 0.012, rng.uniform(2.0, 8.0, n), 0.0)

    # ① 一段持续高温（异常点 + 持续段）
    heat = int(HEAT_DAY * DEMO_ROWS_PER_DAY) + 13
    temp[heat:heat + HEAT_HOURS] += HEAT_DELTA_C

    # ② 气压台阶（变化点）
    step = int(STEP_DAY * DEMO_ROWS_PER_DAY)
    pressure[step:] += STEP_DELTA_HPA

    # ③ 一段连续缺失（质量检查要能定位到区间）
    miss = int(MISSING_DAY * DEMO_ROWS_PER_DAY) + 5
    for col in (temp, humidity, pressure, wind):
        col[miss:miss + MISSING_HOURS] = np.nan

    frame = pd.DataFrame({
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "station": "DEMO",
        "temperature": np.round(temp, 2),
        "humidity": np.round(humidity, 2),
        "pressure": np.round(pressure, 2),
        "wind_speed": np.round(wind, 2),
        "rainfall": np.round(rain, 2),
    })

    # ④ 三条重复时间戳（质量检查要能报出来）
    if DUPLICATE_ROWS:
        dup = frame.iloc[:DUPLICATE_ROWS].copy()
        frame = pd.concat([frame, dup], ignore_index=True)
    return frame


def write_demo_csv(path: str, frame: pd.DataFrame) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    return path


def run_demo(out_dir: str = "mtsa_demo", seed: int = DEFAULT_SEED,
             question: str = DEMO_QUESTION,
             timeout: float = DEFAULT_TIMEOUT_S):
    """生成数据 → 跑 pipeline → 返回 AnalysisResult（产物写在 out_dir）。"""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = write_demo_csv(os.path.join(out_dir, DEMO_CSV_NAME), build_demo_frame(seed))

    req = AnalysisRequest(
        request_id="demo",
        input_path=csv_path,
        question=question,
        scene_id="weather",
        seed=seed,
        output_dir=out_dir,
        mode="pipeline",
    )
    if timeout and timeout > 0:
        return run_with_timeout(req, timeout_s=timeout, verbose=False)
    return run_analysis(req)
