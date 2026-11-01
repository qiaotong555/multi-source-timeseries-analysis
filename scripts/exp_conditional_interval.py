# -*- coding: utf-8 -*-
"""一次性对照实验：降水列在「真下雨时」的区间覆盖不足，能不能用条件标定修好。

现状：18 个零膨胀列里 13 列的非零观测覆盖率低于 0.80，最低 0.37
（`interval_coverage_nonzero`）。原因是区间按**全部误差**标定，而 79%–99% 的时刻
误差恰为 0，把尺度压得很小；真下雨时误差大，区间不够宽。

方案对比（同一批误差矩阵，只改标定用的样本）：
  full   现状：全部误差标定，只在非零观测上统计覆盖率
  nz1    只用「非零观测」的误差标定（不分箱）
  nz2    同上，按 horizon 分 2 箱
  nz4    同上，按 horizon 分 4 箱（与产线一致）
样本不足时回退 full。

运行：
    PYTHONPATH=src python scripts/exp_conditional_interval.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(ROOT, "src"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mtsa import forecast, io_loaders, preprocess, profiler, quality  # noqa: E402
from mtsa.scenarios.weather import WeatherAdapter  # noqa: E402

SPECS = [
    ("r05_three_cities_hourly_2024", "rainfall@TYO"),
    ("r05_three_cities_hourly_2024", "rainfall@SYD"),
    ("r12_quito_hourly_2024", "rainfall"),
    ("r13_addis_ababa_hourly_2024", "rainfall"),
    ("r17_nyc_asos_hourly_2018_2024", "rainfall"),
    ("r01_berlin_hourly_2024", "rainfall"),
    ("r04_reykjavik_hourly_2024", "rainfall"),
]
LEVEL = forecast.INTERVAL_LEVEL
MIN_NZ = 20


def _load_clean(ds: str):
    path = os.path.join(ROOT, "fixtures", "real_weather", ds, "data.csv")
    raw = io_loaders.load_dataset(path)
    profile, parsed = profiler.profile_dataset(raw)
    qrep = quality.check_quality(raw, profile, parsed)
    plan = WeatherAdapter().plan("预测评估", profile, qrep, raw.frame)
    plan.scene_confidence = 1.0
    return preprocess.clean_dataset(raw, profile, parsed, plan), plan


def _errors(y, period, horizon, method, origins_n=150):
    origins = forecast._origins(len(y), horizon, origins_n)
    min_valid = max(3, int(np.ceil(horizon * forecast.MIN_ORIGIN_VALID)))
    rows, used = [], []
    for t in origins:
        pred = forecast._forecast_from(y[:t], period, horizon, method)
        actual = y[t:t + horizon]
        ok = np.isfinite(actual) & np.isfinite(pred)
        if int(ok.sum()) < min_valid:
            continue
        rows.append(np.where(ok, actual - pred, np.nan))
        used.append(int(t))
    return np.array(rows, float), used


def _cal(errs, mask=None, bins=1):
    """标定 (lo, hi)；给 mask 时只用 mask 为真的误差。样本不足返回 None。"""
    arr = np.asarray(errs, float)
    h = arr.shape[1]
    if mask is None:
        m = np.isfinite(arr)
    else:
        m = np.asarray(mask, bool) & np.isfinite(arr)
    edges = np.linspace(0, h, bins + 1).astype(int)
    lo, hi = np.zeros(h), np.zeros(h)
    tail = (1 - LEVEL) / 2
    for a, b in zip(edges[:-1], edges[1:]):
        sub = arr[:, a:b][m[:, a:b]]
        sub = sub[np.isfinite(sub)]
        if sub.size < MIN_NZ:
            # 该箱非零样本太少：退回用整个窗口的（含零）误差
            allsub = arr[:, a:b][np.isfinite(arr[:, a:b])]
            if allsub.size < MIN_NZ:
                return None
            sub, scale_src = allsub, allsub
        else:
            scale_src = sub
        scale = float(np.mean(np.abs(scale_src)))
        if not np.isfinite(scale) or scale <= 0:
            return None
        u = np.sort(sub / scale)
        mm = u.size
        k_hi = min(int(math.ceil((mm + 1) * (1 - tail))) - 1, mm - 1)
        k_lo = max(int(math.floor((mm + 1) * tail)) - 1, 0)
        lo[a:b], hi[a:b] = u[k_lo] * scale, u[k_hi] * scale
    return lo, hi


def _coverage(errs, mask, window, bins, conditional):
    k = len(errs)
    start = max(1, int(np.floor(k * (1 - forecast.TEST_FRACTION))))
    hits = tot = 0
    for i in range(start, k):
        m_hist = mask[max(0, i - window):i] if conditional else None
        cal = _cal(errs[max(0, i - window):i], m_hist, bins)
        if cal is None:
            continue
        lo, hi = cal
        row, mk = errs[i], mask[i]
        sel = mk & np.isfinite(row)
        if not sel.any():
            continue
        hits += int(np.sum((row[sel] >= lo[sel]) & (row[sel] <= hi[sel])))
        tot += int(sel.sum())
    return (hits / tot if tot else None), tot


def main() -> int:
    rows = []
    for ds, col in SPECS:
        clean, plan = _load_clean(ds)
        period = int(plan.parameters.get("seasonal_period") or 0) or forecast.infer_period(clean.frame.index)
        horizon = max(1, min(int(plan.parameters.get("forecast_horizon") or period or 24),
                             forecast.MAX_HORIZON))
        s = pd.to_numeric(clean.frame[col], errors="coerce").interpolate(
            limit_direction="both").ffill().bfill()
        y = s.to_numpy(float)
        s.name = col
        fit = forecast._fit_column(s, period, horizon, forecast.DEFAULT_ORIGINS)
        errs, used = _errors(y, period, horizon, fit["selected"])
        mask = np.array([[abs(y[t + j]) > 0 if np.isfinite(y[t + j]) else False
                          for j in range(horizon)] for t in used], dtype=bool)
        rec = {"dataset": ds, "column": col, "selected": fit["selected"],
               "free": fit.get("interval_coverage"),
               "free_nz": fit.get("interval_coverage_nonzero"),
               "plans": {}}
        for name, bins, cond in (("full", 1, False), ("nz1", 1, True),
                                 ("nz2", 2, True), ("nz4", 4, True)):
            cov, n = _coverage(errs, mask, forecast.CALIB_WINDOW, bins, cond)
            rec["plans"][name] = {"coverage": None if cov is None else round(cov, 4), "n": n}
        rows.append(rec)
        print("=" * 88)
        print("%-32s %-12s sel=%-14s 整体=%.4f 非零=%.4f" % (
            ds[:32], col, fit["selected"], fit.get("interval_coverage") or -1,
            fit.get("interval_coverage_nonzero") or -1))
        for name in ("full", "nz1", "nz2", "nz4"):
            p = rec["plans"][name]
            print("   %-5s 非零覆盖率=%s (n=%s)" % (name, p["coverage"], p["n"]))

    dst = os.path.join(ROOT, "artifacts", "exp_conditional_interval.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("\n结果：%s" % dst)
    for name in ("full", "nz1", "nz2", "nz4"):
        inband = sum(1 for r in rows
                     if r["plans"][name]["coverage"] is not None
                     and 0.80 <= r["plans"][name]["coverage"] <= 0.98)
        print("  %-5s 落在 [0.80,0.98] 的列数：%d/%d" % (name, inband, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
