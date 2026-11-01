# -*- coding: utf-8 -*-
"""一次性对照实验：回测 origin 的密度值多少。

现在 `DEFAULT_ORIGINS = 40`，测试段只取最后 40% 的 origin，即 16 个起点 × horizon 24 =
**384 个点**。384 点上的覆盖率标准误约 1.5–2 个百分点——这意味着「某列 0.794 出界」
和「某列 0.806 在带内」的差别大半是抽样噪声。

本实验把 origins 从 40 拉到 200，看三件事：
1. 覆盖率点估计与参考值（200）差多少；
2. 事件点数（零膨胀列）能涨到多少；
3. 耗时倍率。

不改产线代码，只喂不同的 origins_n。

运行：
    PYTHONPATH=src python scripts/exp_origins_density.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(ROOT, "src"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mtsa import forecast, io_loaders, preprocess, prob_forecast, profiler, quality  # noqa: E402
from mtsa.scenarios.weather import WeatherAdapter  # noqa: E402

DENSITIES = (40, 60, 100, 150, 200)
REF = 200
SPECS = {
    "r10_toronto_hourly_2024": ["temperature", "humidity"],
    "r13_addis_ababa_hourly_2024": ["temperature", "rainfall"],
    "r12_quito_hourly_2024": ["rainfall"],
    "r17_nyc_asos_hourly_2018_2024": ["temperature", "rainfall"],
}


def _load_clean(ds: str):
    path = os.path.join(ROOT, "fixtures", "real_weather", ds, "data.csv")
    raw = io_loaders.load_dataset(path)
    profile, parsed = profiler.profile_dataset(raw)
    qrep = quality.check_quality(raw, profile, parsed)
    plan = WeatherAdapter().plan("预测评估", profile, qrep, raw.frame)
    plan.scene_confidence = 1.0
    clean = preprocess.clean_dataset(raw, profile, parsed, plan)
    return clean, plan


def _se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n else float("nan")


def main() -> int:
    rows = []
    for ds, cols in SPECS.items():
        clean, plan = _load_clean(ds)
        period = int(plan.parameters.get("seasonal_period") or 0) or forecast.infer_period(clean.frame.index)
        horizon = max(1, min(int(plan.parameters.get("forecast_horizon") or period or 24),
                             forecast.MAX_HORIZON))
        print("=" * 92)
        print("%s（%d 行，period=%d，horizon=%d）" % (ds, len(clean.frame), period, horizon))
        for col in cols:
            s = pd.to_numeric(clean.frame[col], errors="coerce").interpolate(
                limit_direction="both").ffill().bfill()
            s.name = col
            rec = {"dataset": ds, "column": col, "points": int(len(s)), "runs": {}}
            for k in DENSITIES:
                t0 = time.perf_counter()
                fit = forecast._fit_column(s, period, horizon, k)
                dt = time.perf_counter() - t0
                rec["runs"][str(k)] = {
                    "coverage": fit.get("interval_coverage"),
                    "coverage_points": fit.get("interval_coverage_points"),
                    "origins_used": fit.get("origins_used"),
                    "mase": fit.get("selected_mase"),
                    "seconds": round(dt, 2),
                }
            # 零膨胀列再看事件评估
            if fit.get("zero_inflated") or True:
                ev = {}
                for k in DENSITIES:
                    r = prob_forecast.evaluate_events(
                        s.to_numpy(float), s.index, period, horizon, origins_n=k)
                    ev[str(k)] = {"status": r.get("status"),
                                  "event_points": r.get("event_points"),
                                  "bss": r.get("bss")}
                rec["events"] = ev
            rows.append(rec)

            ref = rec["runs"][str(REF)]["coverage"]
            print("\n  %s" % col)
            print("    %6s %10s %8s %8s %10s %8s %8s" % (
                "origins", "测试点数", "覆盖率", "SE", "与200之差", "事件点", "耗时s"))
            for k in DENSITIES:
                r = rec["runs"][str(k)]
                cov, n = r["coverage"], r["coverage_points"] or 0
                diff = None if (cov is None or ref is None) else cov - ref
                ep = (rec["events"].get(str(k)) or {}).get("event_points")
                print("    %6d %10d %8.4f %8.4f %10s %8s %8.2f" % (
                    k, n, cov if cov is not None else -1, _se(cov or 0, n),
                    "—" if diff is None else "%+.4f" % diff, ep, r["seconds"]))

    dst = os.path.join(ROOT, "artifacts", "exp_origins_density.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("\n结果：%s" % dst)

    # 汇总：40 / 100 相对 200 的覆盖率偏差
    print("\n相对 origins=200 的覆盖率偏差（%d 列）" % len(rows))
    for k in (40, 60, 100):
        d = [abs(r["runs"][str(k)]["coverage"] - r["runs"][str(REF)]["coverage"])
             for r in rows
             if r["runs"][str(k)]["coverage"] is not None
             and r["runs"][str(REF)]["coverage"] is not None]
        t = sum(r["runs"][str(k)]["seconds"] for r in rows)
        tref = sum(r["runs"][str(REF)]["seconds"] for r in rows)
        if d:
            print("  origins=%3d：平均 |偏差| %.4f，最大 %.4f，耗时 %.0f%% of 200" % (
                k, float(np.mean(d)), float(np.max(d)), 100.0 * t / tref if tref else 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
