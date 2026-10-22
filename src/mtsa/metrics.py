# -*- coding: utf-8 -*-
"""时域量化指标：描述统计、趋势、波动、相关性、滞后相关、自相关与极值。

所有指标带 method 与 formula，可通过 code_ref 定位复算实现。
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .models import AnalysisPlan, CleanDataset, MetricRow, MetricTable

YEAR_SECONDS = 365.25 * 86400.0


def periods_per_year(index: pd.Index) -> float:
    """按中位采样间隔估算每年的采样点数，用于年化。"""
    if isinstance(index, pd.DatetimeIndex) and len(index) > 2:
        deltas = np.diff(index.values).astype("timedelta64[s]").astype(float)
        med = float(np.median(deltas[deltas > 0])) if np.any(deltas > 0) else 0.0
        if med > 0:
            return YEAR_SECONDS / med
    return 252.0


def _fmt_period(seconds: Optional[float]) -> str:
    if seconds is None or seconds <= 0:
        return "未知"
    if seconds < 90:
        return f"{seconds:.0f} 秒"
    if seconds < 5400:
        return f"{seconds / 60:.1f} 分钟"
    if seconds < 172800:
        return f"{seconds / 3600:.2f} 小时"
    return f"{seconds / 86400:.2f} 天"


def _rolling_window(index_len: int, wanted: int) -> int:
    return max(3, min(wanted, max(3, index_len // 10)))


def lag_correlation(a: pd.Series, b: pd.Series, max_lag: int) -> tuple[int, float]:
    """返回使 |corr| 最大的 lag 与对应相关系数；lag>0 表示 a 领先 b。"""
    best_lag, best_corr = 0, 0.0
    x = a.to_numpy(dtype=float)
    y = b.to_numpy(dtype=float)
    n = len(x)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            xs, ys = x[: n - lag] if lag else x, y[lag:]
        else:
            xs, ys = x[-lag:], y[: n + lag]
        mask = ~(np.isnan(xs) | np.isnan(ys))
        if mask.sum() < 10:
            continue
        if np.std(xs[mask]) == 0 or np.std(ys[mask]) == 0:
            continue
        c = float(np.corrcoef(xs[mask], ys[mask])[0, 1])
        if abs(c) > abs(best_corr):
            best_lag, best_corr = lag, c
    return best_lag, best_corr


def compute_metrics(clean: CleanDataset, plan: AnalysisPlan) -> MetricTable:
    frame = clean.frame
    rows: list[MetricRow] = []
    cols = [c for c in plan.target_columns if c in frame.columns]
    ppy = periods_per_year(frame.index)
    max_lag = int(plan.parameters.get("max_lag", 24))
    max_lag = max(1, min(max_lag, max(1, len(frame) // 20)))
    group_col = plan.group_column if plan.group_column in frame.columns else None

    def add(metric_id, name, column, value, unit, method, formula, window=None, group=None):
        rows.append(MetricRow(
            metric_id=metric_id, name=name, column=column, group=group,
            value=None if value is None or (isinstance(value, float) and not np.isfinite(value)) else float(value),
            unit=unit, window=window, method=method, formula=formula,
        ))

    groups: list[tuple[Optional[str], pd.DataFrame]] = [("__all__", frame)]
    if group_col:
        groups = [(str(g), sub) for g, sub in frame.groupby(group_col)]

    for gname, sub in groups:
        g = None if gname == "__all__" else gname
        for c in cols:
            s = pd.to_numeric(sub[c], errors="coerce").dropna()
            if s.empty:
                continue
            add(f"ts.desc.mean:{c}", "均值", c, s.mean(), None, "arithmetic_mean", "sum(x)/n", group=g)
            add(f"ts.desc.std:{c}", "标准差", c, s.std(ddof=1), None, "sample_std", "sqrt(sum((x-mu)^2)/(n-1))", group=g)
            add(f"ts.desc.min:{c}", "最小值", c, s.min(), None, "min", "min(x)", group=g)
            add(f"ts.desc.max:{c}", "最大值", c, s.max(), None, "max", "max(x)", group=g)
            add(f"ts.desc.median:{c}", "中位数", c, s.median(), None, "median", "median(x)", group=g)

            # 趋势：OLS 斜率（按天）与 Mann-Kendall 检验
            if isinstance(sub.index, pd.DatetimeIndex) and len(s) > 3:
                x = (sub.index - sub.index[0]).total_seconds().to_numpy() / 86400.0
                y = pd.to_numeric(sub[c], errors="coerce").to_numpy(dtype=float)
                ok = ~np.isnan(y)
                if ok.sum() > 3 and np.std(x[ok]) > 0:
                    slope = float(np.polyfit(x[ok], y[ok], 1)[0])
                    add(f"ts.trend.ols_slope_per_day:{c}", "OLS 趋势斜率", c, slope,
                        "单位/天", "ols_linear", "polyfit(t_day, x, 1)[0]", group=g)
                kt = stats.kendalltau(np.arange(len(s)), s.to_numpy())
                add(f"ts.trend.mann_kendall_tau:{c}", "Mann-Kendall tau", c, float(kt.statistic),
                    None, "kendalltau", "kendalltau(rank(t), x)", group=g)
                add(f"ts.trend.mann_kendall_p:{c}", "Mann-Kendall p 值", c, float(kt.pvalue),
                    None, "kendalltau", "kendalltau(rank(t), x)", group=g)

            # 波动
            sd = float(s.std(ddof=1))
            add(f"ts.vol.std:{c}", "标准差", c, sd, None, "sample_std", "sqrt(var)", group=g)
            add(f"ts.vol.annualized:{c}", "年化波动", c, sd * math.sqrt(ppy), None,
                "std_sqrt_ppy", f"std*sqrt({ppy:.1f})", group=g)
            w = _rolling_window(len(s), int(plan.parameters.get("rolling_window", 24)))
            roll = pd.to_numeric(sub[c], errors="coerce").rolling(w, min_periods=max(3, w // 2)).std()
            if roll.notna().any():
                add(f"ts.vol.rolling_mean:{c}", "滚动波动均值", c, float(roll.mean()), None,
                    "rolling_std_mean", f"mean(rolling({w}).std())", window=str(w), group=g)
                add(f"ts.vol.rolling_max:{c}", "滚动波动峰值", c, float(roll.max()), None,
                    "rolling_std_max", f"max(rolling({w}).std())", window=str(w), group=g)

            # 极值
            q95 = float(s.quantile(0.95))
            q05 = float(s.quantile(0.05))
            hi_ex = int((s > q95).sum())
            lo_ex = int((s < q05).sum())
            add(f"ts.extreme.p95:{c}", "95 分位阈值", c, q95, None, "quantile", "quantile(x, 0.95)", group=g)
            add(f"ts.extreme.p05:{c}", "5 分位阈值", c, q05, None, "quantile", "quantile(x, 0.05)", group=g)
            add(f"ts.extreme.high_count:{c}", "超 95 分位次数", c, hi_ex, "次", "count", "count(x > p95)", group=g)
            add(f"ts.extreme.low_count:{c}", "低于 5 分位次数", c, lo_ex, "次", "count", "count(x < p05)", group=g)

            # 一阶自相关（自行实现，不依赖 pandas 版本相关的 autocorr）
            vals = s.to_numpy(dtype=float)
            if len(vals) > 10:
                x0, x1 = vals[:-1], vals[1:]
                if np.std(x0) > 0 and np.std(x1) > 0:
                    ac1 = float(np.corrcoef(x0, x1)[0, 1])
                    add(f"ts.acf.lag1:{c}", "一阶自相关", c, ac1, None, "pearson_lag1",
                        "corr(x[:-1], x[1:])", group=g)

    # 相关性矩阵（全样本）
    sub_all = frame
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            sa = pd.to_numeric(sub_all[a], errors="coerce")
            sb = pd.to_numeric(sub_all[b], errors="coerce")
            mask = sa.notna() & sb.notna()
            if mask.sum() < 10:
                continue
            pr = float(np.corrcoef(sa[mask], sb[mask])[0, 1])
            sp = float(stats.spearmanr(sa[mask], sb[mask]).statistic)
            add(f"ts.corr.pearson:{a}~{b}", f"Pearson 相关 {a} × {b}", f"{a}~{b}", pr, None,
                "pearson", "cov/(sd_a*sd_b)")
            add(f"ts.corr.spearman:{a}~{b}", f"Spearman 相关 {a} × {b}", f"{a}~{b}", sp, None,
                "spearman", "pearson(rank(a), rank(b))")
            if "lag" in plan.analyses or "corr" in plan.analyses:
                l, c = lag_correlation(sa, sb, max_lag)
                add(f"ts.corr.best_lag:{a}~{b}", f"最优滞后 {a} → {b}", f"{a}~{b}", l, "采样点",
                    "lag_scan", f"argmax_lag |corr(a[t], b[t+lag])|, lag∈[-{max_lag},{max_lag}]")
                add(f"ts.corr.best_lag_corr:{a}~{b}", f"最优滞后相关 {a} → {b}", f"{a}~{b}", c, None,
                    "lag_scan", "max_lag |corr(a[t], b[t+lag])|")

    return MetricTable(rows=rows, generated_by="mtsa.metrics.compute_metrics",
                       code_ref="src/mtsa/metrics.py::compute_metrics")
