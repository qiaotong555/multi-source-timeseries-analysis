# -*- coding: utf-8 -*-
"""天气场景适配器（主场景）。

覆盖规范要求的：重采样、缺失插值、日/月/季节统计、趋势与周期分解、极端天气识别、
变量相关与滞后关系、站点对比。
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..models import (
    AnalysisPlan,
    CleanDataset,
    Conclusion,
    DatasetProfile,
    Evidence,
    MetricRow,
    MetricTable,
    QualityReport,
)
from .base import ScenarioAdapter, ScenarioContext, register

CANON = {
    "temperature": ("temperature", "temp", "air_temperature", "ta", "温度", "气温"),
    "humidity": ("humidity", "rh", "relative_humidity", "湿度"),
    "pressure": ("pressure", "pres", "air_pressure", "气压"),
    "wind": ("wind", "wind_speed", "ws", "风速"),
    "rainfall": ("rainfall", "rain", "precip", "precipitation", "降雨", "降水量"),
}
STATION_HINT = ("station", "site", "city", "站点", "城市", "station_id")


def _canon_map(columns) -> dict[str, str]:
    """把实际列名映射到规范字段名。"""
    mapping: dict[str, str] = {}
    lowered = {str(c): c for c in columns}
    for canon, aliases in CANON.items():
        for alias in aliases:
            for low, orig in lowered.items():
                if low.lower() == alias or low.lower().startswith(alias + "_") or alias in low.lower():
                    mapping.setdefault(canon, orig)
    return mapping


class WeatherAdapter(ScenarioAdapter):
    scene_id = "weather"
    name = "天气时序分析"

    # ---------------- 场景识别 ----------------
    def detect(self, frame: pd.DataFrame, profile: DatasetProfile) -> float:
        mapping = _canon_map(frame.columns)
        hit = len(mapping)
        if hit == 0:
            return 0.0
        score = min(1.0, 0.24 * hit)
        # 出现股票字段则降低天气置信度
        stockish = sum(
            1 for c in frame.columns
            if re.search(r"(open|high|low|close|volume|symbol|开盘|收盘|成交量)", str(c), re.I)
        )
        if stockish >= 2:
            score *= 0.3
        return round(score, 4)

    # ---------------- 分析计划 ----------------
    def plan(self, question: str, profile: DatasetProfile, quality: QualityReport,
             frame: pd.DataFrame) -> AnalysisPlan:
        mapping = _canon_map(frame.columns)
        targets = [c for c in mapping.values() if c in frame.columns]
        if not targets:
            targets = profile.numeric_columns[:5]

        group = None
        for c in frame.columns:
            if any(h in str(c).lower() for h in STATION_HINT):
                group = c
                break
        if group is None and profile.group_columns:
            group = profile.group_columns[0]

        med_sec = profile.time_axis.median_interval_seconds if profile.time_axis else None
        period = None
        if med_sec and 0 < med_sec < 86400:
            period = int(round(86400.0 / med_sec))
        if med_sec and med_sec >= 86400 * 20:
            period = 12

        # 单位变化修复：温度列前后中位数比值落在 [2.4, 4.2] 判定为华氏度段。
        # 一次真实的量纲错误会产生两个转移点（进入异常量纲 / 回到正常量纲），
        # 必须配对成区间换算；否则第二个转移点会把已修正的数据再换算一遍。
        group_col = profile.group_columns[0] if profile.group_columns else None
        pending: dict[tuple, list] = {}
        for iss in quality.issues:
            if iss.code != "POSSIBLE_UNIT_CHANGE" or not iss.start:
                continue
            canon = next((k for k, v in mapping.items() if v == iss.column), None)
            if canon != "temperature":
                continue
            m = re.search(r"比值\s*([0-9.]+)", iss.detail or "")
            if not m:
                continue
            ratio = float(m.group(1))
            # 这里只用比值判断「进入」还是「退出」异常量纲（>1 进入，<1 退出）。
            # 比值能否作为量纲证据已由质量层判定：温度列必须同时满足
            # 「阶跃幅度」与「换算后边界连续」。不再用固定区间 [2.4, 4.2]
            # 过滤——该区间依赖基线温度（比值 = 1.8 + 32/T），冷站点会误伤。
            pending.setdefault((iss.column, iss.group), []).append((iss.start, ratio))

        conversions = []
        open_intervals = []
        for (col, gval), items in pending.items():
            items.sort(key=lambda x: x[0])
            i = 0
            while i + 1 < len(items):
                (t0, r0), (t1, r1) = items[i], items[i + 1]
                if (r0 > 1) != (r1 > 1):          # 方向相反的一对构成一个异常量纲区间
                    conversions.append({
                        "column": col, "kind": "fahrenheit_to_celsius",
                        "start_time": t0, "end_time": t1,
                        "group_column": group_col, "group_value": gval,
                        "from_unit": "°F", "to_unit": "°C",
                    })
                    i += 2
                else:
                    i += 1
            if i == len(items) - 1:
                # 只有一个转移点：无法区分「量纲变化」与「真实水平跳变」，
                # 不自动换算，只交给人工核对（避免一次误判改写整份数据）。
                open_intervals.append({
                    "column": col, "start_time": items[i][0],
                    "group_value": gval, "ratio": items[i][1],
                    "reason": "single_transition",
                })

        analyses = [
            "trend", "corr", "lag", "extreme", "spectrum", "changepoint", "anomaly",
            "test", "stationarity", "autocorr", "normality", "seasonal",
        ]
        params = {
            "seasonal_period": period,
            "rolling_window": 24 if (med_sec and med_sec < 86400) else 7,
            "max_lag": 48 if (med_sec and med_sec < 86400) else 12,
            # 阈值在开发集（带真值标签）上按精确率/召回率权衡选定：
            # 单条序列约 1.75 万个采样点且残差存在重尾，|z|<=5 的判定会产生大量误报。
            # 已知限制：未用独立保留集复核该阈值。
            "anomaly_threshold": 8.0,
            "unit_conversions": conversions,
            "unit_change_open_intervals": open_intervals,
        }
        return AnalysisPlan(
            scene_id="weather",
            scene_confidence=0.0,  # 由编排层写入真实置信度
            question=question,
            target_columns=targets,
            group_column=group,
            resample_rule=None,
            analyses=analyses,
            parameters=params,
            not_supported=[],
        )

    # ---------------- 专属指标 ----------------
    def extra_metrics(self, clean: CleanDataset, plan: AnalysisPlan) -> MetricTable:
        frame = clean.frame
        rows: list[MetricRow] = []

        def add(mid, name, col, value, unit, method, formula, window=None, group=None):
            rows.append(MetricRow(
                metric_id=mid, name=name, column=col, group=group,
                value=None if value is None or not np.isfinite(value) else float(value),
                unit=unit, window=window, method=method, formula=formula,
            ))

        is_dt = isinstance(frame.index, pd.DatetimeIndex)
        for col in plan.target_columns:
            if col not in frame.columns:
                continue
            s = pd.to_numeric(frame[col], errors="coerce").dropna()
            if s.empty:
                continue
            if is_dt:
                sub = pd.to_numeric(frame[col], errors="coerce")
                monthly = sub.groupby(sub.index.month).mean()
                if len(monthly) >= 3:
                    add(f"wx.seasonal.monthly_amplitude:{col}", "月均振幅", col,
                        float(monthly.max() - monthly.min()), None, "groupby_month_mean_range",
                        "max(mean_by_month) - min(mean_by_month)")
                    add(f"wx.seasonal.hottest_month:{col}", "最热月份", col,
                        float(monthly.idxmax()), "月", "groupby_month_mean_argmax",
                        "argmax(mean_by_month)")
                    add(f"wx.seasonal.coldest_month:{col}", "最冷月份", col,
                        float(monthly.idxmin()), "月", "groupby_month_mean_argmin",
                        "argmin(mean_by_month)")
                # 用年月均值拟合趋势，削弱季节周期对趋势检验的干扰
                ym = sub.groupby([sub.index.year, sub.index.month]).mean()
                if len(ym) >= 4:
                    xm = np.arange(len(ym), dtype=float)
                    ymv = ym.to_numpy(dtype=float)
                    ok = ~np.isnan(ymv)
                    if ok.sum() >= 4 and np.std(xm[ok]) > 0:
                        add(f"wx.trend.monthly_ols_slope:{col}", "月度均值 OLS 斜率", col,
                            float(np.polyfit(xm[ok], ymv[ok], 1)[0]), "单位/月",
                            "ols_on_monthly_mean", "polyfit(month_index, mean_by_month, 1)[0]")
                if plan.parameters.get("seasonal_period"):
                    hourly = sub.groupby(sub.index.hour).mean()
                    add(f"wx.seasonal.diurnal_range:{col}", "日内温差（小时均值极差）", col,
                        float(hourly.max() - hourly.min()), None, "groupby_hour_mean_range",
                        "max(mean_by_hour) - min(mean_by_hour)")
                # 极端事件
                p95, p99 = float(s.quantile(0.95)), float(s.quantile(0.99))
                above = (sub > p95)
                if above.any():
                    runs = _max_run_length(above.to_numpy())
                    add(f"wx.extreme.max_consecutive_high:{col}", "超 95 分位最长连续时长", col,
                        float(runs), "采样点", "max_run_length", "最长连续 True 段长度")
                if "rainfall" in col.lower() or "降雨" in col:
                    dry = (sub.fillna(0) <= 0.0)
                    add(f"wx.extreme.max_dry_spell:{col}", "最长连续无降水时长", col,
                        float(_max_run_length(dry.to_numpy())), "采样点", "max_run_length",
                        "最长连续降水量<=0 的段长度")
                    add(f"wx.extreme.heavy_rain_count:{col}", "强降水事件数（超 99 分位）", col,
                        float(int((sub > p99).sum())), "次", "count", "count(x > p99)")
        # 站点对比：清洗阶段已把多实体透视成 `列名@分组` 的宽表
        base_cols = sorted({c.split("@")[0] for c in plan.target_columns if "@" in c})
        for base in base_cols:
            members = [c for c in plan.target_columns if c.split("@")[0] == base]
            if len(members) < 2 or not any(m in frame.columns for m in members):
                continue
            gmean = pd.DataFrame({m: pd.to_numeric(frame[m], errors="coerce") for m in members}).mean()
            add(f"wx.station.max_mean:{base}", "分组均值最高", base, float(gmean.max()),
                None, "group_mean_max", "max(mean_by_group)")
            add(f"wx.station.spread:{base}", "分组均值极差", base,
                float(gmean.max() - gmean.min()), None, "group_mean_range",
                "max(mean_by_group) - min(mean_by_group)")
            add(f"wx.station.argmax:{base}", "均值最高的分组", base,
                float(list(gmean.index).index(gmean.idxmax())), "序号", "group_mean_argmax",
                "argmax(mean_by_group)")

        return MetricTable(rows=rows, generated_by="mtsa.scenarios.weather.extra_metrics",
                           code_ref="src/mtsa/scenarios/weather.py::WeatherAdapter.extra_metrics")

    # ---------------- 图表 ----------------
    def figure_specs(self, plan: AnalysisPlan) -> list[dict]:
        specs = [
            {"id": "wx_series", "kind": "series", "title": "气象要素时间序列",
             "columns": plan.target_columns, "downsample": 4000},
            {"id": "wx_corr", "kind": "heatmap", "title": "变量相关矩阵", "columns": plan.target_columns},
            {"id": "wx_spectrum", "kind": "spectrum", "title": "功率谱密度（Welch）", "columns": plan.target_columns},
            {"id": "wx_month", "kind": "bar", "title": "月度均值", "columns": plan.target_columns[:4],
             "groupby": "month"},
            {"id": "wx_box_month", "kind": "box", "title": "按月分布箱线图",
             "columns": plan.target_columns[:2], "groupby": "month"},
            {"id": "wx_anomaly", "kind": "anomaly", "title": "异常点与变化点", "columns": plan.target_columns[:3]},
        ]
        if any("@" in c for c in plan.target_columns):
            # 清洗阶段已透视成 `列名@站点` 宽表，直接比较各列均值即为站点对比
            specs.append({"id": "wx_station", "kind": "bar", "title": "各站点要素均值对比",
                          "columns": plan.target_columns[:8]})
        return specs

    # ---------------- 结论 ----------------
    def conclusions(self, ctx: ScenarioContext) -> tuple[list[Conclusion], list[Evidence]]:
        conclusions: list[Conclusion] = []
        evidences: list[Evidence] = []
        nid = [0]

        def add(category, text, metric_ids, artifact_ids, code_ref, caveat=""):
            nid[0] += 1
            eid = f"ev_{nid[0]:03d}"
            cid = f"con_{nid[0]:03d}"
            evidences.append(Evidence(evidence_id=eid, claim=text, metric_ids=metric_ids,
                                      artifact_ids=artifact_ids, code_ref=code_ref, caveat=caveat))
            conclusions.append(Conclusion(conclusion_id=cid, category=category, text=text,
                                          evidence_ids=[eid], limitations=caveat))

        frame = ctx.clean.frame
        conv = ctx.clean.unit_conversions
        if conv:
            for c in conv:
                seg = f"{c.get('start_time') or '起始'} ~ {c.get('end_time') or '序列末尾'}"
                who = f"（分组 {c['group_value']}）" if c.get("group_value") else ""
                add("数据口径",
                    f"字段 {c['column']}{who} 在 {seg} 区间被识别为华氏度段，共 {c['rows']} 个采样点，"
                    f"已统一换算为摄氏度（区间中位数 {c['median_before']:.2f} → {c['median_after']:.2f}）。",
                    [], ["art_clean_csv"], "src/mtsa/preprocess.py::_apply_unit_conversions",
                    "换算依据为前后中位数比值落在华氏—摄氏区间，属推断，需与数据来源方确认")

        for col in ctx.plan.target_columns:
            mk = ctx.metric(f"ts.trend.mann_kendall_p:{col}")
            tau = ctx.metric(f"ts.trend.mann_kendall_tau:{col}")
            slope = ctx.metric(f"wx.trend.monthly_ols_slope:{col}")
            if mk and tau and mk.value is not None:
                sig = mk.value < 0.05
                extra = f"；月度均值 OLS 斜率 {slope.value:+.4f} 单位/月" if slope and slope.value is not None else ""
                add("趋势",
                    f"{col}：Mann-Kendall tau={tau.value:.4f}，p={mk.value:.4g}，"
                    f"{'存在' if sig else '未发现'}显著单调趋势"
                    f"{'，方向' + ('上升' if tau.value > 0 else '下降') if sig else ''}{extra}。",
                    [r.metric_id for r in (tau, mk, slope) if r], ["art_fig_series"],
                    "src/mtsa/metrics.py::compute_metrics",
                    "Mann-Kendall 要求观测独立；未做季节调整时强季节周期会污染趋势检验，"
                    "应以月度均值斜率与 STL 趋势项为准")

            spec = next((s for s in ctx.spectrum if s.column == col), None)
            if spec and spec.dominant_freq:
                add("周期",
                    f"{col}：功率谱主频 {spec.dominant_freq:.6g} Hz，对应周期 {spec.dominant_period}；"
                    f"低/中/高频能量占比 {spec.band_power.get('低频', 0):.2%} / "
                    f"{spec.band_power.get('中频', 0):.2%} / {spec.band_power.get('高频', 0):.2%}。",
                    [], ["art_fig_spectrum"], "src/mtsa/freq.py::analyze_spectrum",
                    "Welch 法主频分辨率受 nperseg 限制；短样本下周期估计不稳定")

            amp = ctx.metric(f"wx.seasonal.monthly_amplitude:{col}")
            hot = ctx.metric(f"wx.seasonal.hottest_month:{col}")
            cold = ctx.metric(f"wx.seasonal.coldest_month:{col}")
            if amp and hot and cold:
                add("季节性",
                    f"{col}：月均振幅 {amp.value:.3f}，峰值月 {int(hot.value)} 月，谷值月 {int(cold.value)} 月。",
                    [amp.metric_id, hot.metric_id, cold.metric_id], ["art_fig_month"],
                    "src/mtsa/scenarios/weather.py::extra_metrics",
                    "月度统计按日历月聚合，跨年样本合并了不同年份")

        # 相关性结论
        corr_rows = [r for r in ctx.metrics.rows if r.metric_id.startswith("ts.corr.pearson:")]
        strong = sorted(corr_rows, key=lambda r: abs(r.value or 0), reverse=True)[:3]
        for r in strong:
            if r.value is None or abs(r.value) < 0.3:
                continue
            add("相关性",
                f"{r.column}：Pearson 相关系数 {r.value:.3f}（{'正' if r.value > 0 else '负'}相关，"
                f"{'强' if abs(r.value) >= 0.6 else '中等'}）。",
                [r.metric_id], ["art_fig_corr"], "src/mtsa/metrics.py::compute_metrics",
                "相关不等于因果；两列可能共同受季节驱动")

        lag_rows = [r for r in ctx.metrics.rows if r.metric_id.startswith("ts.corr.best_lag:")]
        for r in lag_rows[:2]:
            if r.value in (None, 0):
                continue
            c = ctx.metric(r.metric_id.replace("best_lag:", "best_lag_corr:"))
            add("滞后关系",
                f"{r.column}：最优滞后 {int(r.value)} 个采样点"
                f"（相关系数 {c.value:.3f}），{'前者领先后者' if r.value > 0 else '后者领先前者'}。",
                [r.metric_id] + ([c.metric_id] if c else []), ["art_fig_series"],
                "src/mtsa/metrics.py::lag_correlation",
                "滞后扫描在有限 lag 范围内取最大绝对相关，多重比较下需谨慎")

        # 异常与变化点
        if ctx.anomalies:
            by_col: dict[str, list] = {}
            for a in ctx.anomalies:
                by_col.setdefault(a.column, []).append(a)
            for col, items in list(by_col.items())[:3]:
                worst = max(items, key=lambda x: abs(x.score))
                add("异常检测",
                    f"{col}：检出 {len(items)} 个异常点，最显著者位于 {worst.timestamp}，"
                    f"取值 {worst.value:.3f}，稳健 Z={worst.score:.2f}（方法：{worst.method}）。",
                    [], ["art_fig_anomaly"], "src/mtsa/anomaly.py::detect_anomalies",
                    "阈值与窗口为可配置参数；极端值与真实事件需业务确认")
        if ctx.change_points:
            cp = ctx.change_points[0]
            add("变化点",
                f"{cp.column} 在 {cp.timestamp}（第 {cp.index} 个采样点）存在均值突变，"
                f"前后中位数 {cp.before:.3f} → {cp.after:.3f}，统计量 {cp.statistic:.2f}。",
                [], ["art_fig_anomaly"], "src/mtsa/anomaly.py::detect_change_points",
                "二分分割受最大深度限制，密集变化点可能被合并")

        if not conclusions:
            add("总体", "样本量或字段不足，未生成可支持的结构化结论。", [], [],
                "src/mtsa/scenarios/weather.py::conclusions", "需补充数据后重跑")
        return conclusions, evidences


def _max_run_length(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


register(WeatherAdapter())
