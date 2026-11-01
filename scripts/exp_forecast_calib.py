# -*- coding: utf-8 -*-
"""一次性对照实验：几种区间标定方案在出界列上的表现。

不改产线代码，只把同一批误差矩阵喂给不同标定器，比较：
- 整体覆盖率（名义 0.90）
- 区间均宽（不能靠放宽区间换覆盖率）
- 逐 horizon 段的实际覆盖率（看名义 90% 是否在各段都成立）
- 降水列额外看「非零观测覆盖率」——零膨胀序列上整体覆盖率衡量的是什么

方案：
  pooled      现状：u=e/scale_h 跨 horizon 池化取 5%/95%
  binned4     按 horizon 分 4 箱，每箱内池化（允许误差分布随 horizon 变形）
  binned4c    binned4 + conformal 有限样本校正（分位取第 ceil((m+1)*0.95) 个）
  pooled8     现状但只用最近 8 个 origin

运行：
    PYTHONPATH=src python scripts/exp_forecast_calib.py
"""
from __future__ import annotations

import json
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

SPECS = {
    # r05 是三城市宽表，列名带 @站点后缀（plan.target_columns 在 scene_confidence=1 时也是带后缀的）
    "r05_three_cities_hourly_2024": ["rainfall@TYO", "temperature@TYO"],
    "r10_toronto_hourly_2024": ["temperature", "humidity", "pressure"],
    "r13_addis_ababa_hourly_2024": ["rainfall"],
    "r15_cairo_hourly_2024": ["rainfall"],
    "r17_nyc_asos_hourly_2018_2024": ["temperature", "humidity", "pressure"],
}
LEVEL = forecast.INTERVAL_LEVEL


def _load_clean(ds: str):
    path = os.path.join(ROOT, "fixtures", "real_weather", ds, "data.csv")
    raw = io_loaders.load_dataset(path)
    profile, parsed = profiler.profile_dataset(raw)
    qrep = quality.check_quality(raw, profile, parsed)
    plan = WeatherAdapter().plan("预测评估", profile, qrep, raw.frame)
    plan.scene_confidence = 1.0
    clean = preprocess.clean_dataset(raw, profile, parsed, plan)
    return clean, plan


def _errors(y, period, horizon, method):
    """返回 (误差矩阵, 实际计入的 origin 位置列表)。"""
    origins = forecast._origins(len(y), horizon, forecast.DEFAULT_ORIGINS)
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
    return (np.array(rows, dtype=float) if rows else np.empty((0, horizon))), used


def _cal_pooled(errs, bins=1, conformal=False):
    """返回 (lo, hi)，长度 = horizon。bins=1 时等价于现状。"""
    arr = np.asarray(errs, float)
    h = arr.shape[1]
    scale = np.nanmean(np.abs(arr), axis=0)
    base = float(np.nanmean(np.abs(arr))) if arr.size else np.nan
    if not np.isfinite(base) or base <= 0:
        base = 1e-12
    scale = np.where(np.isfinite(scale) & (scale > 0.1 * base), scale, 0.1 * base)
    u = arr / scale[None, :]
    lo = np.zeros(h)
    hi = np.zeros(h)
    edges = np.linspace(0, h, bins + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        pool = u[:, a:b].ravel()
        pool = pool[np.isfinite(pool)]
        if pool.size < 8:
            lo[a:b], hi[a:b] = -1.645 * scale[a:b], 1.645 * scale[a:b]
            continue
        s = np.sort(pool)
        if conformal:
            m = s.size
            k_hi = int(np.ceil((m + 1) * (1 + LEVEL) / 2)) - 1
            k_lo = int(np.floor((m + 1) * (1 - LEVEL) / 2))
            k_hi = min(k_hi, m - 1)
            k_lo = max(k_lo, 0)
            qlo, qhi = s[k_lo], s[k_hi]
        else:
            qlo, qhi = np.quantile(s, [(1 - LEVEL) / 2, (1 + LEVEL) / 2])
        lo[a:b], hi[a:b] = qlo * scale[a:b], qhi * scale[a:b]
    return lo, hi


def _coverage(errs, window, bins=1, conformal=False):
    """返回 (命中矩阵, 每行对应的 errs 行号)。滚动标定：每个测试 origin 只用它之前的误差。"""
    k = len(errs)
    if k < 4:
        return None, None
    start = max(1, int(np.floor(k * (1.0 - forecast.TEST_FRACTION))))
    hits = []
    idxs = []
    for i in range(start, k):
        lo, hi = _cal_pooled(errs[max(0, i - window):i], bins, conformal)
        row = errs[i]
        ok = np.isfinite(row)
        if not ok.any():
            continue
        hits.append(((row[ok] >= lo[ok]) & (row[ok] <= hi[ok])).astype(float))
        idxs.append(i)
    if not hits:
        return None, None
    return np.vstack(hits), idxs


def _seg(mat):
    """按 horizon 三等分的实际覆盖率，用来验证「名义 90% 在各段都成立」。"""
    h = mat.shape[1]
    third = max(1, h // 3)
    return [round(float(np.nanmean(mat[:, :third])), 4),
            round(float(np.nanmean(mat[:, third:2 * third])), 4),
            round(float(np.nanmean(mat[:, 2 * third:])), 4)]


def main() -> int:
    rows = []
    for ds, cols in SPECS.items():
        clean, plan = _load_clean(ds)
        period = int(plan.parameters.get("seasonal_period") or 0) or forecast.infer_period(clean.frame.index)
        horizon = max(1, min(int(plan.parameters.get("forecast_horizon") or period or 24), forecast.MAX_HORIZON))
        for col in cols:
            s = pd.to_numeric(clean.frame[col], errors="coerce").interpolate(
                limit_direction="both").ffill().bfill()
            y = s.to_numpy(float)
            s.name = col
            fit = forecast._fit_column(s, period, horizon, forecast.DEFAULT_ORIGINS)
            errs, oidx = _errors(y, period, horizon, fit["selected"])
            if len(errs) < 6:
                continue
            rec = {"dataset": ds, "column": col, "selected": fit["selected"],
                   "current": fit["interval_coverage"],
                   "zero_share": round(float(np.mean(y == 0)), 4), "plans": {}}
            for name, bins, conformal, window in [
                ("pooled", 1, False, forecast.CALIB_WINDOW),
                ("pooled8", 1, False, 8),
                ("binned4", 4, False, forecast.CALIB_WINDOW),
                ("binned4c", 4, True, forecast.CALIB_WINDOW),
            ]:
                mat, idxs = _coverage(errs, window, bins, conformal)
                if mat is None:
                    continue
                lo, hi = _cal_pooled(errs[-window:], bins, conformal)
                rec["plans"][name] = {
                    "coverage": round(float(np.nanmean(mat)), 4),
                    "seg": _seg(mat),
                    "mean_width": round(float(np.mean(hi - lo)), 4),
                }
            # 零膨胀：整体覆盖率 vs 只在「观测非零」的点上统计（同一套滚动标定）
            mat, idxs = _coverage(errs, forecast.CALIB_WINDOW)
            if mat is not None:
                mask_nz = np.array([
                    [abs(y[oidx[i] + j]) > 0 for j in range(horizon)] for i in idxs
                ], dtype=bool)
                if mask_nz.any():
                    rec["coverage_nonzero_actual"] = round(float(mat[mask_nz].mean()), 4)
                    rec["nonzero_points"] = int(mask_nz.sum())
                    rec["nonzero_hit"] = int(round(mat[mask_nz].sum()))
            rows.append(rec)

    for r in rows:
        print("=" * 96)
        print("%-32s %-12s sel=%-14s 现状 cov=%.4f  零值=%.1f%%" % (
            r["dataset"][:32], r["column"], r["selected"], r["current"], r["zero_share"] * 100))
        print("  %-10s %8s %22s %10s" % ("方案", "覆盖率", "逐段(h/3)", "区间均宽"))
        for name, p in r["plans"].items():
            print("  %-10s %8.4f %22s %10.4f" % (name, p["coverage"], str(p["seg"]), p["mean_width"]))
        if "coverage_nonzero_actual" in r:
            print("  非零观测覆盖率=%.4f（%d 点，现状方案）" % (
                r["coverage_nonzero_actual"], r["nonzero_points"]))

    dst = os.path.join(ROOT, "artifacts", "exp_forecast_calib.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("\n结果：%s" % dst)

    # 汇总：每种方案有几列落在带内
    print("\n方案 × 落在 [0.80,0.98] 的列数（共 %d 列）" % len(rows))
    for name in ("pooled", "pooled8", "binned4", "binned4c"):
        hit = sum(1 for r in rows if name in r["plans"] and 0.80 <= r["plans"][name]["coverage"] <= 0.98)
        print("  %-10s %d/%d" % (name, hit, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
