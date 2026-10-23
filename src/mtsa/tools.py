# -*- coding: utf-8 -*-
"""分析层工具：把六个分析节点包装为可独立调用的 Tool。

设计约束
--------
1. **只包装分析层**。数据管道（导入 → 摘要 → 质量 → 计划 → 清洗）与收尾
   （图表 → 报告 → 证据校验）不在此列：前者有严格前置依赖，后者需聚合全部
   分析结果，都不适合由模型自主编排。
2. **工具执行时使用全量分析开关**。「是否执行」由决策层决定（pipeline 由计划决定，
   agent 由模型决定），但一旦决定执行某个工具，就必须跑完这个工具的**完整分析**。
   底层函数内部还会按 `plan.analyses` 过滤一次（例如 `run_stat_tests` 只在
   `analyses` 含 `stationarity`/`test` 时才跑 ADF），若沿用它，模型选了
   「统计检验」却只得到计划层精简后剩下的那一小类，会形成「执行了但没做全」的
   静默失效。因此工具内统一用 `_plan_for_tool()` 换成全量开关的副本。
   参数（阈值、窗口等）仍取原计划，不受影响。
3. **`ToolResult` 分两路**：
   - `summary`：结构化短摘要，将进入模型上下文。**不得包含原始数据数组**。
   - `data`：完整结果，仅代码使用（图表、报告、评测）。
   这一条是「模型不接触原始数值」的落点，也是控制 token 成本的关键。
4. **幂等**：同一 `ToolContext` 重复调用同一工具，结果必须相同，且不得修改
   `ctx` 中的任何对象。由 `test_tools.py` 断言。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from . import anomaly as anomaly_mod
from . import freq as freq_mod
from . import metrics as metrics_mod
from . import stats_tests as stats_mod
from .models import (AnalysisPlan, AnomalyPoint, ChangePoint, CleanDataset,
                     MetricTable, SpectrumResult, TestResultRow)
from .scenarios.base import ScenarioAdapter


@dataclass
class ToolContext:
    """工具执行所需的只读上下文。"""

    clean: CleanDataset
    plan: AnalysisPlan
    adapter: Optional[ScenarioAdapter] = None


@dataclass
class ToolResult:
    """工具执行结果。

    outcome 语义：
    - ``ok``      执行成功（可能产出为空，见 summary 中的 result_count）
    - ``skipped`` 前置条件不满足，未执行（如非股票场景的泄漏检查）
    - ``failed``  执行中抛异常，error 给出原因
    """

    tool: str
    outcome: str
    summary: dict[str, Any] = field(default_factory=dict)
    data: Any = None
    elapsed_ms: int = 0
    error: Optional[str] = None
    reason: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.outcome == "ok"

    def to_summary_dict(self) -> dict[str, Any]:
        """给模型/轨迹看的字典，不含完整 data。"""
        out = {"tool": self.tool, "outcome": self.outcome, "summary": self.summary,
               "elapsed_ms": self.elapsed_ms}
        if self.reason:
            out["reason"] = self.reason
        if self.error:
            out["error"] = self.error
        return out


def _round(x: Any, nd: int = 6) -> Any:
    """数值统一保留位数，避免摘要出现 0.30000000000000004 这类噪声。"""
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, float):
        return round(x, nd)
    return x


# 与下游消费点一一对应（由 test_tools 的一致性测试锁死）
ALL_ANALYSES: tuple[str, ...] = (
    "trend", "volatility", "corr", "lag", "extreme", "spectrum",
    "changepoint", "anomaly", "test", "stationarity", "autocorr",
    "normality", "seasonal", "randomness",
)


def _plan_for_tool(plan: AnalysisPlan) -> AnalysisPlan:
    """把分析开关全部打开，供工具内部使用。

    底层函数会按 `plan.analyses` 二次过滤（如 `run_stat_tests` 只在含
    `stationarity`/`test` 时才跑 ADF）。模型选择了某工具，意味着要它跑完整分析；
    若沿用计划层可能很精简的 `analyses`，就会出现「工具执行了但只做了一小部分」
    的静默失效——比报错更难发现。

    只改 `analyses`，`parameters`（阈值、窗口、列映射）原样保留：那些是标定结果，
    不能因开关而变。
    """
    if set(ALL_ANALYSES).issubset(set(plan.analyses)):
        return plan
    expanded = plan.model_copy(deep=True)
    expanded.analyses = list(ALL_ANALYSES)
    return expanded


class Tool:
    """工具基类。子类只需声明元信息并实现 run。"""

    name: str = ""
    description: str = ""
    when_to_use: str = ""
    requires_scene: Optional[str] = None

    def run(self, ctx: ToolContext) -> ToolResult:      # pragma: no cover - 抽象
        raise NotImplementedError

    def describe(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description,
                "when_to_use": self.when_to_use}


class ComputeMetricsTool(Tool):
    name = "compute_metrics"
    description = "计算时域量化指标：均值、标准差、分位数、趋势斜率，以及场景专属指标（如股票收益与回撤）。"
    when_to_use = "问题涉及数值大小、波动程度、收益风险，或需要总体画像时。"

    def run(self, ctx: ToolContext) -> ToolResult:
        plan = _plan_for_tool(ctx.plan)
        mtab = metrics_mod.compute_metrics(ctx.clean, plan)
        extra = ctx.adapter.extra_metrics(ctx.clean, plan) if ctx.adapter else MetricTable(
            rows=[], generated_by="none", code_ref="")
        highlights: dict[str, Any] = {}
        for r in list(mtab.rows) + list(extra.rows):
            if r.value is None:
                continue
            key = f"{r.name}@{r.column}"
            if key in highlights:
                continue
            highlights[key] = _round(r.value)
            if len(highlights) >= 8:
                break
        return ToolResult(
            tool=self.name, outcome="ok",
            summary={"generic_metrics": len(mtab.rows), "scenario_metrics": len(extra.rows),
                     "highlights": highlights},
            data={"generic": mtab, "scenario": extra},
        )


class AnalyzeSpectrumTool(Tool):
    name = "analyze_spectrum"
    description = "Welch 功率谱分析，给出各列的主频与周期长度，用于识别周期性成分。"
    when_to_use = "问题涉及周期、季节性、振荡节律，或需要判断是否存在固定周期时。"

    def run(self, ctx: ToolContext) -> ToolResult:
        spectra: list[SpectrumResult] = freq_mod.analyze_spectrum(ctx.clean, _plan_for_tool(ctx.plan))
        peaks = [{"column": s.column,
                  "dominant_period": s.dominant_period,
                  "dominant_freq": _round(s.dominant_freq),
                  "sampling_rate_hz": _round(s.sampling_rate_hz)}
                 for s in spectra]
        return ToolResult(
            tool=self.name, outcome="ok",
            summary={"result_count": len(spectra), "peaks": peaks},
            data=spectra,
        )


class RunStatTestsTool(Tool):
    name = "run_stat_tests"
    description = "统计检验：ADF/KPSS 平稳性、Ljung-Box 自相关、Jarque-Bera 正态性、Mann-Kendall 趋势、游程随机性。"
    when_to_use = "需要推断性结论（是否显著、是否平稳、是否有趋势）而非仅描述时。"

    def run(self, ctx: ToolContext) -> ToolResult:
        rows: list[TestResultRow] = stats_mod.run_stat_tests(ctx.clean, _plan_for_tool(ctx.plan))
        adf = {}
        for r in rows:
            if r.test_id.startswith("adf:") and r.p_value is not None:
                adf[r.column] = {"p_value": _round(r.p_value),
                                 "reject_unit_root": bool(r.reject_null)}
        trend = {}
        for r in rows:
            if r.test_id.startswith("mann_kendall:") and r.p_value is not None:
                trend[r.column] = {"p_value": _round(r.p_value),
                                   "significant": bool(r.reject_null)}
        skipped = [r.column for r in rows if r.test_id.startswith("skip:")]
        return ToolResult(
            tool=self.name, outcome="ok",
            summary={"test_count": len(rows), "stationarity": adf, "trend": trend,
                     "insufficient_sample_columns": skipped},
            data=rows,
        )


class DetectAnomaliesTool(Tool):
    name = "detect_anomalies"
    description = "基于稳健 Z 分数的异常点检测，定位离群值与故障点，给出每列异常数量。"
    when_to_use = "关注数据质量、设备故障、极端值，或问题中出现「异常」「离群」「故障」时。"

    def run(self, ctx: ToolContext) -> ToolResult:
        points: list[AnomalyPoint] = anomaly_mod.detect_anomalies(ctx.clean, _plan_for_tool(ctx.plan))
        by_column: dict[str, int] = {}
        for p in points:
            by_column[p.column] = by_column.get(p.column, 0) + 1
        return ToolResult(
            tool=self.name, outcome="ok",
            summary={"count": len(points), "by_column": by_column,
                     "threshold": ctx.plan.parameters.get("anomaly_threshold")},
            data=points,
        )


class DetectChangePointsTool(Tool):
    name = "detect_change_points"
    description = "均值与方差突变检测，定位结构性变化的时刻。"
    when_to_use = "问题涉及「何时发生变化」「拐点」「水平跳变」「行为改变」时。"

    def run(self, ctx: ToolContext) -> ToolResult:
        points: list[ChangePoint] = anomaly_mod.detect_change_points(ctx.clean, _plan_for_tool(ctx.plan))
        by_column: dict[str, int] = {}
        for p in points:
            by_column[p.column] = by_column.get(p.column, 0) + 1
        first = [{"column": p.column, "timestamp": p.timestamp, "kind": p.kind,
                  "before": _round(p.before, 4), "after": _round(p.after, 4)}
                 for p in points[:5]]
        return ToolResult(
            tool=self.name, outcome="ok",
            summary={"count": len(points), "by_column": by_column, "first_points": first},
            data=points,
        )


class CheckLookaheadTool(Tool):
    name = "check_lookahead"
    description = "未来数据泄漏检查：信号列与下一期收益的相关性是否异常偏高。仅股票场景有效。"
    when_to_use = "股票场景、涉及回测或信号有效性时。其他场景不适用。"
    requires_scene = "stock"

    def run(self, ctx: ToolContext) -> ToolResult:
        if ctx.plan.scene_id != self.requires_scene:
            return ToolResult(tool=self.name, outcome="skipped",
                              reason=f"仅 {self.requires_scene} 场景适用，当前场景为 {ctx.plan.scene_id}")
        close_col = ctx.plan.parameters.get("close_column") or "close"
        issues = _stock().check_lookahead(ctx.clean.frame, close_col)
        return ToolResult(
            tool=self.name, outcome="ok",
            summary={"issue_count": len(issues),
                     "issues": [{k: _round(v) for k, v in i.items()} for i in issues[:5]]},
            data=issues,
        )


def _stock():
    from .scenarios import stock as stock_mod
    return stock_mod


_REGISTRY: dict[str, Tool] = {}


def _register(tool: Tool) -> Tool:
    _REGISTRY[tool.name] = tool
    return tool


for _t in (ComputeMetricsTool(), AnalyzeSpectrumTool(), RunStatTestsTool(),
           DetectAnomaliesTool(), DetectChangePointsTool(), CheckLookaheadTool()):
    _register(_t)


def tool_names() -> list[str]:
    """按固定顺序返回全部工具名，保证提示词与轨迹可复读。"""
    return list(_REGISTRY.keys())


def get_tool(name: str) -> Optional[Tool]:
    return _REGISTRY.get(name)


def describe_tools() -> list[dict[str, str]]:
    """给模型看的工具清单。"""
    return [t.describe() for t in _REGISTRY.values()]


def iter_summary_values(summary: Any, prefix: str = ""):
    """遍历摘要中的全部叶子值，产出 (路径, 值)。

    用途：检查摘要是否混入了原始数组、是否超长。观察摘要器（T2.3）在压缩
    上下文时同样依赖这一遍历，因此放在此处而非测试内。
    """
    if isinstance(summary, dict):
        for k, v in summary.items():
            yield from iter_summary_values(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(summary, (list, tuple)):
        yield prefix, summary
        for i, v in enumerate(summary):
            yield from iter_summary_values(v, f"{prefix}[{i}]")
    else:
        yield prefix, summary


def run_tool(name: str, ctx: ToolContext) -> ToolResult:
    """执行工具并统一计时与异常包装。

    不在此处吞掉异常：只把异常转成 ``outcome=failed`` 的结果，
    由调用方（pipeline 或 agent 循环）决定是中止还是换工具。
    """
    tool = _REGISTRY.get(name)
    if tool is None:
        return ToolResult(tool=name, outcome="failed",
                          error=f"未注册的工具：{name}",
                          reason="unknown_tool")
    t0 = time.perf_counter()
    try:
        res = tool.run(ctx)
    except Exception as exc:                                  # noqa: BLE001 - 边界处兜底
        return ToolResult(tool=name, outcome="failed",
                          elapsed_ms=int((time.perf_counter() - t0) * 1000),
                          error=f"{type(exc).__name__}: {exc}")
    res.elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return res
