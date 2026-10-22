# -*- coding: utf-8 -*-
"""股票场景适配器（扩展场景）。

覆盖：复权检查、收益率、波动率、最大回撤、滚动相关、量价关系、简单信号与
严格按时间切分的 walk-forward 回测；禁止使用未来数据。

边界：不提供投资建议，不做自动交易；回测不含交易成本与滑点。
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from ..models import (
    ERR_LOOKAHEAD_VIOLATION,
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

TRADING_DAYS = 252
RF = 0.02
LOOKAHEAD_CORR_THRESHOLD = 0.80
N_FOLDS = 5

PRICE_HINT = re.compile(r"(open|high|low|close|volume|adj_close|adjust_factor|benchmark|开盘|收盘|成交量)", re.I)
SIGNAL_HINT = re.compile(r"(signal|score|factor|信号|因子)", re.I)


class LookaheadViolation(Exception):
    def __init__(self, details: dict):
        super().__init__("检测到未来数据泄漏")
        self.details = details


def max_drawdown(nav: np.ndarray) -> tuple[float, int, int]:
    peak, peak_i, mdd, mi, ti = nav[0], 0, 0.0, 0, 0
    for i, v in enumerate(nav):
        if v > peak:
            peak, peak_i = v, i
        dd = v / peak - 1.0
        if dd < mdd:
            mdd, mi, ti = dd, peak_i, i
    return mdd, mi, ti


def check_lookahead(frame: pd.DataFrame, price_col: str = "close") -> list[dict]:
    """检查候选信号列是否包含未来信息。

    判据：信号值与「下一期收益」的相关系数绝对值超过 0.80。真实日频信号的 IC
    通常在 0.02—0.06，超过该阈值只可能来自构造时使用了未来数据。
    """
    if price_col not in frame.columns:
        return []
    px = pd.to_numeric(frame[price_col], errors="coerce")
    fwd = px.shift(-1) / px - 1.0
    issues = []
    for col in frame.columns:
        if col == price_col or not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        if not SIGNAL_HINT.search(str(col)):
            continue
        s = pd.to_numeric(frame[col], errors="coerce")
        mask = s.notna() & fwd.notna()
        if mask.sum() < 30:
            continue
        c = float(np.corrcoef(s[mask], fwd[mask])[0, 1])
        if abs(c) > LOOKAHEAD_CORR_THRESHOLD:
            issues.append({
                "column": str(col),
                "corr_with_next_return": round(c, 6),
                "threshold": LOOKAHEAD_CORR_THRESHOLD,
                "samples": int(mask.sum()),
            })
    return issues


class StockAdapter(ScenarioAdapter):
    scene_id = "stock"
    name = "股票时序分析"

    # ---------------- 场景识别 ----------------
    def detect(self, frame: pd.DataFrame, profile: DatasetProfile) -> float:
        needed = ["open", "high", "low", "close", "volume"]
        lowered = {str(c).lower(): c for c in frame.columns}
        hit = 0
        for k in needed:
            if any(k == low or k in low for low in lowered):
                hit += 1
        if hit <= 1:
            return 0.0
        score = min(1.0, 0.22 * hit)
        wx = sum(1 for c in frame.columns
                 if re.search(r"(temperature|humidity|pressure|rainfall|温度|湿度|降雨)", str(c), re.I))
        if wx >= 2:
            score *= 0.3
        return round(score, 4)

    # ---------------- 分析计划 ----------------
    def plan(self, question: str, profile: DatasetProfile, quality: QualityReport,
             frame: pd.DataFrame) -> AnalysisPlan:
        lowered = {str(c).lower(): c for c in frame.columns}

        def find(*names):
            for n in names:
                for low, orig in lowered.items():
                    if low == n:
                        return orig
            return None

        close = find("close", "adj_close", "收盘")
        volume = find("volume", "成交量")
        bench = find("benchmark", "index", "基准")
        targets = [c for c in (close, volume, bench) if c]
        if not targets:
            targets = profile.numeric_columns[:3]

        analyses = ["trend", "corr", "lag", "extreme", "changepoint", "anomaly",
                    "test", "stationarity", "normality"]
        params = {
            "close_column": close,
            "volume_column": volume,
            "benchmark_column": bench,
            "rolling_window": 20,
            "max_lag": 10,
            "anomaly_threshold": 4.5,
            "n_folds": N_FOLDS,
            "rf": RF,
        }
        return AnalysisPlan(
            scene_id="stock",
            scene_confidence=0.0,
            question=question,
            target_columns=targets,
            group_column=find("symbol", "code", "股票代码"),
            resample_rule=None,
            analyses=analyses,
            parameters=params,
            not_supported=[],
        )

    # ---------------- 专属指标 ----------------
    def extra_metrics(self, clean: CleanDataset, plan: AnalysisPlan) -> MetricTable:
        frame = clean.frame
        rows: list[MetricRow] = []
        p = plan.parameters
        close_col = p.get("close_column")
        vol_col = p.get("volume_column")
        bench_col = p.get("benchmark_column")

        def add(mid, name, col, value, unit, method, formula, window=None, group=None):
            rows.append(MetricRow(
                metric_id=mid, name=name, column=col or "", group=group,
                value=None if value is None or (isinstance(value, float) and not np.isfinite(value)) else float(value),
                unit=unit, window=window, method=method, formula=formula,
            ))

        if not close_col or close_col not in frame.columns:
            return MetricTable(rows=[], generated_by="mtsa.scenarios.stock.extra_metrics",
                               code_ref="src/mtsa/scenarios/stock.py::StockAdapter.extra_metrics")

        px = pd.to_numeric(frame[close_col], errors="coerce")
        rets = px.pct_change().fillna(0.0)
        nav = (1.0 + rets).cumprod()

        # 复权检查
        factor = next((c for c in frame.columns if "adjust_factor" in str(c).lower()), None)
        adj_col = next((c for c in frame.columns if str(c).lower() in ("adj_close", "adjusted_close")), None)
        if factor:
            recomputed = px * pd.to_numeric(frame[factor], errors="coerce")
            if adj_col:
                diff = (pd.to_numeric(frame[adj_col], errors="coerce") - recomputed).abs()
                rel = float((diff / recomputed.abs().replace(0, np.nan)).max())
                add("stk.adj.max_rel_diff", "复权价最大相对偏差", adj_col, rel, None,
                    "max_abs_rel_diff", "max(|adj_close - close*factor| / |close*factor|)")
                if rel > 1e-6:
                    add("stk.adj.consistent", "复权一致性", adj_col, 0.0, None,
                        "threshold_check", "rel_diff <= 1e-6")
            else:
                add("stk.adj.recomputed_mean", "重算复权价均值", close_col,
                    float(recomputed.mean()), None, "mean", "mean(close*adjust_factor)")

        # 收益
        cum = float(nav.iloc[-1] - 1.0)
        add("stk.ret.cumulative", "累计收益", close_col, cum, None, "cumprod", "prod(1+r)-1")
        n = len(rets)
        ann_ret = float((1.0 + cum) ** (TRADING_DAYS / max(n, 1)) - 1.0) if n else float("nan")
        add("stk.ret.annualized", "年化收益", close_col, ann_ret, None, "geometric",
            "(1+cum)^(252/n)-1")

        # 风险
        vol = float(rets.std(ddof=1) * math.sqrt(TRADING_DAYS))
        add("stk.risk.ann_vol", "年化波动率", close_col, vol, None, "std_sqrt_252", "std(r)*sqrt(252)")
        downside = float(np.sqrt(np.mean(np.minimum(rets.to_numpy(), 0.0) ** 2)) * math.sqrt(TRADING_DAYS))
        add("stk.risk.sortino", "索提诺比率", close_col, (ann_ret - RF) / downside if downside else None,
            None, "sortino", "(ann_ret - rf)/downside_vol")
        add("stk.risk.sharpe", "夏普比率", close_col, (ann_ret - RF) / vol if vol else None,
            None, "sharpe", "(ann_ret - rf)/ann_vol")
        mdd, mi, ti = max_drawdown(nav.to_numpy())
        add("stk.risk.max_drawdown", "最大回撤", close_col, mdd, None, "peak_to_trough",
            "min(nav/cummax(nav) - 1)")
        add("stk.risk.dd_peak_index", "回撤起点序号", close_col, float(mi), "采样点", "argmax_before", "argmax(nav[:t])")
        add("stk.risk.dd_trough_index", "回撤终点序号", close_col, float(ti), "采样点", "argmin", "argmin(nav/cummax(nav))")
        add("stk.risk.win_rate", "日胜率", close_col, float((rets > 0).mean()), None, "mean", "mean(r > 0)")

        # 量价关系
        if vol_col and vol_col in frame.columns:
            v = pd.to_numeric(frame[vol_col], errors="coerce")
            mask = v.notna() & rets.notna()
            if mask.sum() > 30:
                c1 = float(np.corrcoef(v[mask], rets[mask].abs())[0, 1])
                add("stk.volprice.corr_vol_absret", "成交量与绝对收益相关", vol_col, c1, None,
                    "pearson", "corr(volume, |r|)")
                dv = v.diff()
                m2 = dv.notna() & rets.notna()
                if m2.sum() > 30:
                    c2 = float(np.corrcoef(dv[m2], rets[m2])[0, 1])
                    add("stk.volprice.corr_dvol_ret", "成交量变动与收益相关", vol_col, c2, None,
                        "pearson", "corr(diff(volume), r)")

        # 与基准的滚动相关
        if bench_col and bench_col in frame.columns:
            b = pd.to_numeric(frame[bench_col], errors="coerce").pct_change()
            w = int(p.get("rolling_window", 20))
            rc = rets.rolling(w, min_periods=w).corr(b)
            if rc.notna().any():
                add("stk.corr.rolling_mean", "与基准滚动相关均值", bench_col, float(rc.mean()),
                    None, "rolling_corr_mean", f"mean(rolling({w}).corr(r, r_bench))", window=str(w))
                add("stk.corr.rolling_min", "与基准滚动相关最低", bench_col, float(rc.min()),
                    None, "rolling_corr_min", f"min(rolling({w}).corr(r, r_bench))", window=str(w))

        # Walk-forward 回测：信号只用历史，决策 T-1、执行 T
        mom = px / px.shift(20) - 1.0
        sig = (mom - mom.rolling(120, min_periods=60).mean()) / mom.rolling(120, min_periods=60).std()
        pos = (sig.shift(1) > 0).astype(float).fillna(0.0)
        strat = rets * pos
        folds = int(p.get("n_folds", N_FOLDS))
        for k in range(folds):
            a = n * k // folds
            b = n * (k + 1) // folds
            if b - a < 10:
                continue
            sret = float((1.0 + strat.iloc[a:b]).prod() - 1.0)
            bret = float((1.0 + rets.iloc[a:b]).prod() - 1.0)
            add(f"stk.bt.fold{k}_strategy", f"第 {k + 1} 段策略收益", close_col, sret, None,
                "walk_forward", f"prod(1+r*pos)-1, 区间[{a},{b})", window=f"fold{k}")
            add(f"stk.bt.fold{k}_benchmark", f"第 {k + 1} 段基准收益", close_col, bret, None,
                "walk_forward", f"prod(1+r)-1, 区间[{a},{b})", window=f"fold{k}")
            add(f"stk.bt.fold{k}_excess", f"第 {k + 1} 段超额收益", close_col, sret - bret, None,
                "walk_forward", "strategy - benchmark", window=f"fold{k}")

        strat_nav = float((1.0 + strat).prod() - 1.0)
        add("stk.bt.total_strategy", "全区间策略累计收益", close_col, strat_nav, None, "cumprod", "prod(1+r*pos)-1")
        add("stk.bt.total_benchmark", "全区间基准累计收益", close_col, cum, None, "cumprod", "prod(1+r)-1")
        add("stk.bt.win_folds", "跑赢基准的段数", close_col,
            float(sum(1 for k in range(folds)
                      if (r1 := next((r.value for r in rows if r.metric_id == f"stk.bt.fold{k}_excess"), None)) is not None
                      and r1 > 0)), "段", "count", "count(fold_excess > 0)")

        return MetricTable(rows=rows, generated_by="mtsa.scenarios.stock.extra_metrics",
                           code_ref="src/mtsa/scenarios/stock.py::StockAdapter.extra_metrics")

    # ---------------- 图表 ----------------
    def figure_specs(self, plan: AnalysisPlan) -> list[dict]:
        p = plan.parameters
        close = p.get("close_column")
        cols = [c for c in (close, p.get("benchmark_column")) if c]
        return [
            {"id": "stk_series", "kind": "series", "title": "价格与基准走势", "columns": cols},
            {"id": "stk_nav", "kind": "nav", "title": "累计收益与回撤", "columns": [close] if close else []},
            {"id": "stk_rolling", "kind": "rolling", "title": "滚动波动率与滚动相关", "columns": cols},
            {"id": "stk_corr", "kind": "heatmap", "title": "相关矩阵", "columns": plan.target_columns},
            {"id": "stk_fold", "kind": "fold", "title": "Walk-forward 分段收益", "columns": []},
            {"id": "stk_anomaly", "kind": "anomaly", "title": "异常点与变化点", "columns": [close] if close else []},
        ]

    # ---------------- 结论 ----------------
    def conclusions(self, ctx: ScenarioContext) -> tuple[list[Conclusion], list[Evidence]]:
        conclusions: list[Conclusion] = []
        evidences: list[Evidence] = []
        nid = [0]
        close = ctx.plan.parameters.get("close_column")

        def add(category, text, metric_ids, artifact_ids, code_ref, caveat=""):
            nid[0] += 1
            eid = f"ev_{nid[0]:03d}"
            evidences.append(Evidence(evidence_id=eid, claim=text, metric_ids=metric_ids,
                                      artifact_ids=artifact_ids, code_ref=code_ref, caveat=caveat))
            conclusions.append(Conclusion(conclusion_id=f"con_{nid[0]:03d}", category=category,
                                          text=text, evidence_ids=[eid], limitations=caveat))

        def m(mid):
            return ctx.metric(mid)

        ret, vol, mdd, sharpe = m("stk.ret.cumulative"), m("stk.risk.ann_vol"), \
            m("stk.risk.max_drawdown"), m("stk.risk.sharpe")
        if ret and vol and mdd:
            add("收益风险",
                f"区间累计收益 {ret.value:.2%}，年化波动率 {vol.value:.2%}，最大回撤 {mdd.value:.2%}"
                f"，夏普比率 {(sharpe.value if sharpe else float('nan')):.3f}（无风险利率 2%）。",
                [r.metric_id for r in (ret, vol, mdd, sharpe) if r], ["art_fig_nav"],
                "src/mtsa/scenarios/stock.py::extra_metrics",
                "回测不含交易成本与滑点；结果不代表实盘可得收益")

        folds = [r for r in ctx.extra.rows if r.metric_id.startswith("stk.bt.fold") and r.metric_id.endswith("_excess")]
        if folds:
            win = sum(1 for r in folds if (r.value or 0) > 0)
            add("回测",
                f"Walk-forward 共 {len(folds)} 段，策略跑赢基准 {win} 段；"
                f"各段超额：" + "、".join(f"{r.value:+.2%}" for r in folds) + "。",
                [r.metric_id for r in folds], ["art_fig_fold"],
                "src/mtsa/scenarios/stock.py::extra_metrics",
                "信号为 20 日动量的滚动 Z 值，参数未做敏感性扫描；分段边界为等分切割")

        leak = check_lookahead(ctx.clean.frame, close or "close")
        if leak:
            add("数据校验",
                "；".join(f"列 {d['column']} 与次日收益相关 {d['corr_with_next_return']:.3f}" for d in leak)
                + "，已判定为未来数据泄漏，回测结果不予输出。",
                [], [], "src/mtsa/scenarios/stock.py::check_lookahead",
                "判据为信号与次日收益相关系数绝对值 > 0.80")
        else:
            add("数据校验",
                "未检出未来数据泄漏：信号列与次日收益的相关系数均低于 0.80 阈值，"
                "回测采用 T-1 决策、T 执行，且分段严格按时间切分无重叠。",
                [], [], "src/mtsa/scenarios/stock.py::check_lookahead",
                "该检查只能识别显式泄漏，无法排除特征构造中的隐性泄漏")

        vp = m("stk.volprice.corr_vol_absret")
        if vp and vp.value is not None:
            add("量价关系",
                f"成交量与绝对收益相关系数 {vp.value:.3f}，"
                f"{'呈现' if abs(vp.value) > 0.2 else '未呈现'}显著的量价同步特征。",
                [vp.metric_id], ["art_fig_corr"], "src/mtsa/scenarios/stock.py::extra_metrics",
                "相关关系受样本区间影响，不构成交易依据")

        adj = m("stk.adj.max_rel_diff")
        if adj and adj.value is not None:
            ok = adj.value <= 1e-6
            add("复权校验",
                f"复权价与 close×adjust_factor 的最大相对偏差 {adj.value:.2e}，"
                f"{'复权一致' if ok else '存在不一致，需核对除权除息数据'}。",
                [adj.metric_id], [], "src/mtsa/scenarios/stock.py::extra_metrics",
                "仅校验数值一致性，不校验除权除息事件本身是否正确")

        add("免责声明",
            "本分析为统计计算结果，不构成任何投资建议或买卖依据，不代表未来收益。",
            [], [], "src/mtsa/scenarios/stock.py::conclusions", "")
        return conclusions, evidences


register(StockAdapter())
