# -*- coding: utf-8 -*-
"""一次性诊断：为什么 6 列的 90% 区间覆盖率跑出 [0.80, 0.98]。

不改任何东西，只把 `_fit_column` 的内部量摊开：误差矩阵、逐 horizon 覆盖率、
区间半宽与误差尺度的比、误差分位与偏态、标定窗口内的尺度漂移。

运行：
    PYTHONPATH=src python scripts/diag_forecast_band.py
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

# 出界列（来自 evals/forecast_results.json）
TARGETS = [
    ("r05_three_cities_hourly_2024", "rainfall@TYO"),
    ("r10_toronto_hourly_2024", "temperature"),
    ("r13_addis_ababa_hourly_2024", "rainfall"),
    ("r15_cairo_hourly_2024", "rainfall"),
    ("r17_nyc_asos_hourly_2018_2024", "temperature"),
    ("r17_nyc_asos_hourly_2018_2024", "humidity"),
    # 对照：同一份数据里覆盖率正常的列
    ("r17_nyc_asos_hourly_2018_2024", "pressure"),
    ("r10_toronto_hourly_2024", "humidity"),
]


def _load_clean(ds: str):
    path = os.path.join(ROOT, "fixtures", "real_weather", ds, "data.csv")
    raw = io_loaders.load_dataset(path)
    profile, parsed = profiler.profile_dataset(raw)
    qrep = quality.check_quality(raw, profile, parsed)
    plan = WeatherAdapter().plan("预测评估", profile, qrep, raw.frame)
    plan.scene_confidence = 1.0
    clean = preprocess.clean_dataset(raw, profile, parsed, plan)
    return clean, plan


def _errors(y: np.ndarray, period: int, horizon: int, method: str):
    origins = forecast._origins(len(y), horizon, forecast.DEFAULT_ORIGINS)
    min_valid = max(3, int(np.ceil(horizon * forecast.MIN_ORIGIN_VALID)))
    rows = []
    for t in origins:
        pred = forecast._forecast_from(y[:t], period, horizon, method)
        actual = y[t:t + horizon]
        ok = np.isfinite(actual) & np.isfinite(pred)
        if int(ok.sum()) < min_valid:
            continue
        rows.append(np.where(ok, actual - pred, np.nan))
    return np.array(rows, dtype=float) if rows else np.empty((0, horizon))


def _rolling_hits(errs: np.ndarray):
    """复刻 _coverage，但同时返回逐 horizon 的命中矩阵，便于看看是哪一段出问题。"""
    k = len(errs)
    if k < 4:
        return None, None
    start = max(1, int(np.floor(k * (1.0 - forecast.TEST_FRACTION))))
    hits = np.full(errs.shape, np.nan)
    for i in range(start, k):
        cal = forecast._calibrate(errs[max(0, i - forecast.CALIB_WINDOW):i])
        if cal["lo"] is None:
            continue
        row = errs[i]
        ok = np.isfinite(row)
        h = (row[ok] >= cal["lo"][ok]) & (row[ok] <= cal["hi"][ok])
        hits[i, ok] = h.astype(float)
    return hits[start:], start


def _scale_detail(errs: np.ndarray):
    """_calibrate 里的 scale 是否被下限截断（0.1 * base）。"""
    arr = np.asarray(errs, float)
    absarr = np.abs(arr)
    scale = np.nanmean(absarr, axis=0)
    base = float(np.nanmean(absarr))
    floor = 0.1 * base if np.isfinite(base) and base > 0 else 1e-12
    clipped = int(np.sum(~(scale > floor)))
    return scale, base, floor, clipped


def diag(ds: str, col: str) -> dict:
    clean, plan = _load_clean(ds)
    period = int(plan.parameters.get("seasonal_period") or 0) or forecast.infer_period(clean.frame.index)
    horizon = int(plan.parameters.get("forecast_horizon") or period or 24)
    horizon = max(1, min(horizon, forecast.MAX_HORIZON))
    frame = clean.frame
    s = pd.to_numeric(frame[col], errors="coerce").interpolate(limit_direction="both").ffill().bfill()
    y = s.to_numpy(dtype=float)
    s.name = col

    fit = forecast._fit_column(s, period, horizon, forecast.DEFAULT_ORIGINS)
    method = fit["selected"]
    errs = _errors(y, period, horizon, method)
    hits, start = _rolling_hits(errs)

    scale, base, floor, clipped = _scale_detail(errs)
    tail = errs[-forecast.CALIB_WINDOW:]
    cal = forecast._calibrate(tail)
    width = float(np.nanmean(cal["hi"] - cal["lo"])) if cal["lo"] is not None else None
    mae = float(np.nanmean(np.abs(errs[start:]))) if hits is not None else None

    if hits is not None:
        cov_h = np.nanmean(hits, axis=0)
        third = max(1, horizon // 3)
        seg = [float(np.nanmean(cov_h[:third])), float(np.nanmean(cov_h[third:2 * third])),
               float(np.nanmean(cov_h[2 * third:]))]
        cov_by_origin = np.nanmean(hits, axis=1)
        origin_span = float(np.nanmax(cov_by_origin) - np.nanmin(cov_by_origin))
    else:
        seg, origin_span = [None] * 3, None

    e = errs[np.isfinite(errs)]
    q = np.quantile(e, [0.01, 0.05, 0.5, 0.95, 0.99]) if e.size else [np.nan] * 5
    skew = float(pd.Series(e).skew()) if e.size > 3 else None

    # 标定窗口内每个 origin 的 MAE，看尺度是否在漂移
    per_origin_mae = [float(np.nanmean(np.abs(r))) for r in errs[-forecast.CALIB_WINDOW:]]

    return {
        "dataset": ds, "column": col, "points": int(len(y)),
        "zero_share": round(float(np.mean(y == 0)), 4),
        "period": period, "horizon": horizon, "method": method,
        "origins_used": int(len(errs)),
        "coverage": fit["interval_coverage"],
        "coverage_by_horizon_third": [None if v is None else round(v, 4) for v in seg],
        "coverage_by_origin_span": None if origin_span is None else round(origin_span, 4),
        "mean_width": None if width is None else round(width, 4),
        "mean_abs_err_test": None if mae is None else round(mae, 4),
        "width_over_err": None if (width is None or not mae) else round(width / mae, 3),
        "err_q01_q05_q50_q95_q99": [round(float(x), 4) for x in q],
        "err_skew": None if skew is None else round(skew, 3),
        "base_abs_err": round(base, 6),
        "scale_floor": round(floor, 6),
        "scale_clipped_horizons": clipped,
        "calib_method": cal["method"], "calib_n": cal["n"],
        "per_origin_mae_last12": [round(v, 4) for v in per_origin_mae],
    }


def main() -> int:
    out = []
    for ds, col in TARGETS:
        d = diag(ds, col)
        out.append(d)
        print("=" * 100)
        print("%s / %s  (n=%d, zero=%.2f%%, %s, h=%d)" % (
            d["dataset"], d["column"], d["points"], d["zero_share"] * 100, d["method"], d["horizon"]))
        print("  coverage=%.4f  逐段(h/3)=%s  逐 origin 极差=%s" % (
            d["coverage"], d["coverage_by_horizon_third"], d["coverage_by_origin_span"]))
        print("  区间均宽=%.4f  测试段 MAE=%.4f  宽度/误差=%s" % (
            d["mean_width"], d["mean_abs_err_test"], d["width_over_err"]))
        print("  误差分位 q01/05/50/95/99=%s  skew=%s" % (d["err_q01_q05_q50_q95_q99"], d["err_skew"]))
        print("  scale 下限截断的 horizon 数=%d（base=%.4g, floor=%.4g）calib=%s n=%d" % (
            d["scale_clipped_horizons"], d["base_abs_err"], d["scale_floor"],
            d["calib_method"], d["calib_n"]))
        print("  最近 12 个 origin 的 MAE=%s" % d["per_origin_mae_last12"])
    dst = os.path.join(ROOT, "artifacts", "diag_forecast_band.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n结果：%s" % dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
