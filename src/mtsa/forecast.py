# -*- coding: utf-8 -*-
"""预测：点预测 + 90% 区间 + walk-forward 回测（仅天气，默认关闭）。

设计约束（见 `spec/当前开发任务.md` 的 F 任务书）：

1. 只用 origin 及以前的数据。`_forecast_from` 是纯函数，签名里只有 train，
   因此「改动观测之后的未来段」在结构上不可能影响任何一次回测预测（F2）。
2. 基线是 seasonal naive。候选方法必须把 MASE 相对基线压低 5% 以上才允许替换，
   否则强制回退基线，并把原因写进 `selected_reason`（F4）。
3. 区间用「滚动窗口 + 尺度归一」的 conformal 式校准：取最近 `CALIB_WINDOW` 个
   origin 的误差，按每个 horizon 的平均绝对误差归一成一个无量纲残差池，
   再取 5%/95% 经验分位。这样既跟随误差尺度的漂移（季节变化），
   也不会因为窗口内样本太少而退化成「最小值/最大值」的过宽区间。
4. 覆盖率只在最后 40% 的 origin 上统计，且每个 origin 的区间只用它之前的误差标定，
   不存在「用同一批数据既拟合区间又评估区间」（F3）。
5. 不用 `CleanDataset.imputed_index` 做逐列剔除：它是行级标记（某行任一列被插补），
   在 v2 上覆盖 56% 的时间戳，按列剔除会误删一半以上的观测。预测与指标、
   检验、异常检测一样，统一以清洗后的序列为口径。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import AnalysisPlan, CleanDataset

METHODS = ("seasonal_naive", "moving_average", "drift")
# origin 数量决定回测的稳定性。v2 一年小时数据下 40 个 origin 相隔约 18 天，
# 标定窗口 12 个仍覆盖约半年，够用且不至于把一年前的误差尺度带进来。
DEFAULT_ORIGINS = 40
CALIB_WINDOW = 12
TEST_FRACTION = 0.4
MAX_HORIZON = 168
INTERVAL_LEVEL = 0.90
# 候选方法必须把 MASE 压到基线的 95% 以下才算「优于基线」
MIN_IMPROVEMENT = 0.95
# 一个 origin 的未来段至少一半是可比较点，才计入回测
MIN_ORIGIN_VALID = 0.5
MIN_POINTS_PER_PERIOD = 3
MIN_POINTS_ABS = 30
MIN_CALIB_POINTS = 20


def infer_period(index: pd.Index) -> int:
    """从时间间隔推断季节周期：小时→24，日→7，周→4，月→12，其他按一天步数。"""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return 1
    deltas = np.diff(index.values).astype("timedelta64[s]").astype(float)
    pos = deltas[deltas > 0]
    if len(pos) == 0:
        return 1
    step = float(np.median(pos))
    if step <= 0:
        return 1
    if 20 * 3600 <= step <= 28 * 3600:      # 日数据 → 周周期
        return 7
    if 5 * 86400 <= step <= 9 * 86400:      # 周数据 → 月周期
        return 4
    if 25 * 86400 <= step <= 32 * 86400:    # 月数据 → 年周期
        return 12
    per_day = 86400.0 / step
    p = int(round(per_day))
    return p if 2 <= p <= MAX_HORIZON else 1


def _forecast_from(train: np.ndarray, period: int, horizon: int, method: str) -> np.ndarray:
    """纯函数：只用 train 生成 horizon 步预测。

    签名里没有「未来」这个参数，调用方只能传 train；F2 因此是结构性质，
    不依赖调用约定。
    """
    y = np.asarray(train, dtype=float)
    n = len(y)
    if n == 0:
        return np.full(horizon, np.nan)
    if method == "seasonal_naive":
        p = max(1, min(period, n))
        reps = int(np.ceil(horizon / p))
        return np.tile(y[-p:], reps)[:horizon]
    if method == "moving_average":
        p = max(1, min(period, n))
        return np.full(horizon, float(np.mean(y[-p:])))
    if method == "drift":
        p = max(1, min(period, n))
        if n <= p:
            return np.full(horizon, float(y[-1]))
        slope = (float(y[-1]) - float(y[-p - 1])) / p
        return float(y[-1]) + slope * np.arange(1, horizon + 1)
    raise ValueError("unknown forecast method: %s" % method)


def _mase(y_true: np.ndarray, y_pred: np.ndarray, train: np.ndarray,
          period: int, mask: np.ndarray | None = None) -> float | None:
    """MASE = 平均绝对误差 / train 内 p 阶差分的平均绝对值（同尺度，可跨列比较）。"""
    p = max(1, min(period, len(train) - 1))
    if len(train) <= p:
        return None
    denom = float(np.mean(np.abs(train[p:] - train[:-p])))
    if not np.isfinite(denom) or denom <= 0:
        return None
    # 分母只剩浮点噪声时（常数序列、完美周期序列），MASE 是一个被放大的比值：
    # 2e-14 的差分能除出 2.0 的 MASE。这种情况必须返回 None，而不是伪造小数。
    scale = float(np.mean(np.abs(train)))
    if denom <= 1e-9 * max(scale, 1e-12):
        return None
    if mask is not None and not mask.any():
        return None
    err = np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))
    if mask is not None:
        err = err[mask]
    return float(np.mean(err) / denom)


def _origins(n: int, horizon: int, k: int) -> list[int]:
    """在最后 30% 内等距取 origin；保证每个 origin 后面还有完整 horizon 个点。"""
    end = n - horizon
    start = max(int(n * 0.7), 2 * horizon)
    if end <= start:
        return []
    k = max(3, min(k, end - start + 1))
    pts = np.linspace(start, end, k).astype(int)
    return sorted({int(x) for x in pts})


def _jnum(x):
    """JSON 安全：非有限浮点一律 null，避免序列化出 NaN。"""
    v = float(x)
    return round(v, 6) if np.isfinite(v) else None


def _calibrate(errs: np.ndarray) -> dict:
    """把误差矩阵标定成 90% 区间偏移量。

    先按 horizon 归一成尺度无关的残差 `u = e / mean|e_h|`，再取 u 的 5%/95% 分位。
    分位池跨 horizon 汇总（窗口 12 × horizon 24 = 288 个值），比逐 horizon 取分位稳。
    """
    arr = np.asarray(errs, dtype=float)
    if arr.size == 0:
        return {"lo": None, "hi": None, "n": 0, "method": "none"}
    absarr = np.abs(arr)
    scale = np.nanmean(absarr, axis=0)
    base = float(np.nanmean(absarr))
    if not np.isfinite(base) or base <= 0:
        base = 1e-12
    scale = np.where(np.isfinite(scale) & (scale > 0.1 * base), scale, 0.1 * base)
    u = (arr / scale).ravel()
    u = u[np.isfinite(u)]
    if u.size < MIN_CALIB_POINTS:
        # 样本太少时退回正态近似，而不是用 3 个点估分位
        return {"lo": -1.645 * scale, "hi": 1.645 * scale, "n": int(u.size),
                "method": "normal_approx"}
    qlo, qhi = np.quantile(u, (1.0 - INTERVAL_LEVEL) / 2.0), np.quantile(u, (1.0 + INTERVAL_LEVEL) / 2.0)
    return {"lo": qlo * scale, "hi": qhi * scale, "n": int(u.size), "method": "empirical_pooled"}


def _coverage(errs: np.ndarray, window: int) -> tuple[float | None, int]:
    """滚动覆盖率：最后 40% 的 origin，每个只用它之前的误差标定区间。"""
    k = len(errs)
    if k < 4:
        return None, 0
    start = max(1, int(np.floor(k * (1.0 - TEST_FRACTION))))
    hits = total = 0
    for i in range(start, k):
        cal = _calibrate(errs[max(0, i - window):i])
        if cal["lo"] is None:
            continue
        row = errs[i]
        ok = np.isfinite(row)
        if not ok.any():
            continue
        hits += int(np.sum((row[ok] >= cal["lo"][ok]) & (row[ok] <= cal["hi"][ok])))
        total += int(ok.sum())
    return (float(hits) / total if total else None), total


def _fit_column(s: pd.Series, period: int, horizon: int, origins_n: int) -> dict:
    name = str(s.name)
    y = s.to_numpy(dtype=float)
    n = len(y)
    need = max(MIN_POINTS_PER_PERIOD * period, MIN_POINTS_ABS)
    if n < need:
        return {"column": name, "status": "insufficient_data", "points": int(n),
                "period": period, "required_points": int(need),
                "note": "样本不足 %d 点，未做预测" % int(need)}

    origins = _origins(n, horizon, origins_n)
    if not origins:
        return {"column": name, "status": "insufficient_data", "points": int(n),
                "period": period, "required_points": int(need),
                "note": "序列太短，最后 30%% 内放不下 3 个 %d 步回测窗口" % horizon}

    min_valid = max(3, int(np.ceil(horizon * MIN_ORIGIN_VALID)))
    per_method: dict[str, dict] = {}
    errors_by_h: dict[str, np.ndarray] = {}
    for m in METHODS:
        abs_err: list[float] = []
        sq_err: list[float] = []
        mases: list[float] = []
        rows: list[np.ndarray] = []
        for t in origins:
            pred = _forecast_from(y[:t], period, horizon, m)
            actual = y[t:t + horizon]
            ok = np.isfinite(actual) & np.isfinite(pred)
            if int(ok.sum()) < min_valid:
                continue
            e = np.where(ok, actual - pred, np.nan)
            abs_err.extend(np.abs(e[ok]).tolist())
            sq_err.extend((e[ok] ** 2).tolist())
            mase = _mase(actual, pred, y[:t], period, mask=ok)
            if mase is not None:
                mases.append(mase)
            rows.append(e)
        errors_by_h[m] = np.array(rows, dtype=float) if rows else np.empty((0, horizon))
        per_method[m] = {
            "method": m,
            "origins": len(rows),
            "n": len(abs_err),
            "mae": round(float(np.mean(abs_err)), 4) if abs_err else None,
            "rmse": round(float(np.sqrt(np.mean(sq_err))), 4) if sq_err else None,
            "mase": round(float(np.mean(mases)), 4) if mases else None,
        }

    base = per_method["seasonal_naive"]
    base_mase = base["mase"]
    cand = [m for m in METHODS if per_method[m]["mase"] is not None]
    if not cand:
        selected, reason = "seasonal_naive", "候选方法 MASE 不可用，回退 seasonal_naive（基线）"
    else:
        best = min(cand, key=lambda m: per_method[m]["mase"])
        if best == "seasonal_naive":
            selected, reason = "seasonal_naive", "基线（候选未优于基线）"
        elif base_mase is not None and per_method[best]["mase"] < MIN_IMPROVEMENT * base_mase:
            selected, reason = best, "优于基线（MASE 相对基线下降 ≥5%）"
        else:
            selected = "seasonal_naive"
            reason = "候选未优于基线（回退 seasonal_naive）"
    beats = reason.startswith("优于基线")

    errs = errors_by_h.get(selected)
    if errs is None or len(errs) == 0:
        errs = np.empty((0, horizon))
    coverage, coverage_n = _coverage(errs, CALIB_WINDOW)
    tail = errs[-CALIB_WINDOW:] if len(errs) else errs
    final_cal = _calibrate(tail)
    lo = final_cal["lo"] if final_cal["lo"] is not None else np.zeros(horizon)
    hi = final_cal["hi"] if final_cal["hi"] is not None else np.zeros(horizon)

    point = _forecast_from(y, period, horizon, selected)
    step = None
    if isinstance(s.index, pd.DatetimeIndex) and len(s.index) > 1:
        d = pd.Series(s.index).diff().dropna()
        if len(d):
            step = d.median()
    idx = [s.index[-1] + step * (i + 1) for i in range(horizon)] if step is not None else []

    return {
        "column": name, "status": "ok", "points": int(n),
        "period": period, "horizon": horizon, "origins": len(origins),
        "origins_used": int(len(errs)),
        "methods": [per_method[m] for m in METHODS],
        "selected": selected, "selected_reason": reason,
        "selected_mase": per_method[selected]["mase"],
        "beats_baseline": bool(beats),
        "baseline": "seasonal_naive", "baseline_mase": base_mase,
        "interval_level": INTERVAL_LEVEL,
        "interval_method": final_cal["method"],
        "interval_calib_points": final_cal["n"],
        "interval_coverage": round(coverage, 4) if coverage is not None else None,
        "interval_coverage_points": coverage_n,
        "interval_note": ("最近 %d 个 origin 的误差按 horizon 归一后取 5%%/95%% 分位；"
                          "覆盖率只用最后 %d%% 的 origin 且逐个只向前取误差"
                          % (CALIB_WINDOW, int(TEST_FRACTION * 100))),
        "forecast": [
            {"timestamp": str(idx[i]) if idx else None,
             "value": _jnum(point[i]),
             "lo": _jnum(point[i] + lo[i]),
             "hi": _jnum(point[i] + hi[i])}
            for i in range(horizon)
        ],
    }


def build_forecast(clean: CleanDataset, plan: AnalysisPlan,
                   horizon: int | None = None) -> dict:
    """对 `plan.target_columns` 逐列回测并预测；返回可直接写入 `data.forecast` 的字典。"""
    frame = clean.frame
    period = int(plan.parameters.get("seasonal_period") or 0) or infer_period(frame.index)
    h = int(horizon or plan.parameters.get("forecast_horizon") or period or 24)
    h = max(1, min(h, MAX_HORIZON))
    cols = []
    for col in plan.target_columns:
        if col not in frame.columns:
            continue
        s = pd.to_numeric(frame[col], errors="coerce")
        if s.notna().sum() < 3:
            continue
        # 仅剩的边缘缺失用最近值补齐：预测需要连续序列，否则位置错位会破坏 p 阶季节对齐
        s = s.interpolate(limit_direction="both").ffill().bfill()
        if s.isna().all():
            continue
        s = s.astype(float)
        s.name = col
        cols.append(_fit_column(s, period, h, DEFAULT_ORIGINS))
    return {
        "enabled": True, "period": period, "horizon": h,
        "interval_level": INTERVAL_LEVEL,
        "methods": list(METHODS), "baseline": "seasonal_naive",
        "origins": DEFAULT_ORIGINS, "calib_window": CALIB_WINDOW,
        "columns": cols,
        "disclaimer": "预测是回测口径下的条件推断，不是事实；horizon 越长误差越大。",
    }
