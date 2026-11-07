# -*- coding: utf-8 -*-
"""量 stat_tests 的并行潜力：按列拆开，看单列耗时与总耗时的关系。

结论要能回答两件事：
1. 串行总耗时里有多少是真计算、多少是固定开销（可以靠并行摊掉）；
2. 若按列多进程，理论加速比与实测加速比差多少（Windows spawn 的代价）。
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mtsa.models import AnalysisPlan, CleanDataset  # noqa: E402
from mtsa.stats_tests import run_stat_tests  # noqa: E402

N = int(os.environ.get("N", "200000"))
NCOLS = int(os.environ.get("NCOLS", "10"))
SEED = 20260927


def make_frame(n: int = N, ncols: int = NCOLS) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    data = {"timestamp": idx}
    for i in range(ncols):
        base = 20.0 + 5.0 * np.sin(2 * np.pi * idx.hour.to_numpy() / 24.0) + i
        data[f"c{i}"] = base + rng.standard_normal(n) * 0.8
    return pd.DataFrame(data)


def make_plan(cols: list[str]) -> AnalysisPlan:
    return AnalysisPlan(
        scene_id="weather", scene_confidence=1.0, question="profile",
        target_columns=cols,
        analyses=["test", "stationarity", "autocorr", "normality", "trend"],
        parameters={"seasonal_period": 24, "anomaly_threshold": 7.0},
        not_supported=[],
    )


def one_column(args: tuple[pd.Series, AnalysisPlan]) -> float:
    """子进程：只跑一列的检验，返回耗时。"""
    s, plan = args
    frame = pd.DataFrame({"x": s})
    p = plan.model_copy(update={"target_columns": ["x"]})
    t0 = time.perf_counter()
    run_stat_tests(CleanDataset(frame=frame), p)
    return time.perf_counter() - t0


def main() -> int:
    df = make_frame()
    cols = [c for c in df.columns if c != "timestamp"]
    plan = make_plan(cols)
    clean = CleanDataset(frame=df)

    print(f"数据 {len(df)} 行 × {len(cols)} 列，Python {sys.version.split()[0]}")
    print(f"CPU 逻辑核 {os.cpu_count()}\n")

    total = float("nan")
    if os.environ.get("SKIP_TOTAL") == "1":
        print("跳过整体串行计时（SKIP_TOTAL=1）\n")
    else:
        t0 = time.perf_counter()
        rows = run_stat_tests(clean, plan)
        total = time.perf_counter() - t0
        print(f"串行总耗时 {total:.2f} s，产出 {len(rows)} 行\n")
    # 逐列计时（串行）
    per: list[float] = []
    for c in cols:
        t1 = time.perf_counter()
        p = plan.model_copy(update={"target_columns": [c]})
        run_stat_tests(CleanDataset(frame=df[["timestamp", c]].set_index("timestamp")), p)
        per.append(time.perf_counter() - t1)
    print("逐列耗时：" + "、".join(f"{x:.2f}" for x in per))
    print(f"逐列合计 {sum(per):.2f} s（串行总耗时 {total:.2f} s，差额是准备与汇总）\n")

    # 多进程
    workers = max(1, min(os.cpu_count() or 1, len(cols)))
    tasks = [(df[c], plan) for c in cols]
    t2 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one_column, tasks))
    par = time.perf_counter() - t2
    print(f"多进程 {workers} 进程 {par:.2f} s，相对逐列合计加速 {sum(per) / par:.2f}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
