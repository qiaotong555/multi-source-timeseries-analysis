# -*- coding: utf-8 -*-
"""场景适配器基类与注册表。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from ..models import (
    AnalysisPlan,
    AnomalyPoint,
    ChangePoint,
    CleanDataset,
    Conclusion,
    DatasetProfile,
    Evidence,
    MetricTable,
    QualityReport,
    SpectrumResult,
    TestResultRow,
)


# 工具函数名 → 触发它所需的分析项（`plan.analyses` 里的名字）。
# pipeline 模式没有 `executed_tools`，只能靠这张表判断某个工具到底跑没跑。
#
# **只列真正被计划门控的工具**：引擎里是 `if "anomaly" in plan.analyses: ...` 这种写法。
# `compute_metrics` / `run_stat_tests` 是无条件执行的，不能列进来——否则计划里没写
# trend 时会得出「指标没算」的错误结论，把假话从「未检出」换成「未执行」。
TOOL_ANALYSES: dict[str, frozenset[str]] = {
    "detect_anomalies": frozenset({"anomaly"}),
    "detect_change_points": frozenset({"changepoint"}),
    "analyze_spectrum": frozenset({"spectrum"}),
}


class ScenarioContext:
    """适配器生成结论时可访问的只读上下文。"""

    def __init__(
        self,
        clean: CleanDataset,
        plan: AnalysisPlan,
        metrics: MetricTable,
        spectrum: list[SpectrumResult],
        tests: list[TestResultRow],
        anomalies: list[AnomalyPoint],
        change_points: list[ChangePoint],
        extra: MetricTable,
        quality: QualityReport,
        executed_tools: Optional[set[str]] = None,
        extremes: Optional[list[dict]] = None,
        forecast: Optional[dict] = None,
    ):
        self.clean = clean
        self.plan = plan
        self.metrics = metrics
        self.spectrum = spectrum
        self.tests = tests
        self.anomalies = anomalies
        self.change_points = change_points
        self.extra = extra
        self.quality = quality
        # None 表示固定流水线（所有分析项都执行过）；agent 模式传入实际成功的工具，
        # 用于区分「执行了但没检出」与「根本没执行」。
        self.executed_tools = set(executed_tools) if executed_tools is not None else None
        # 绝对极端值（经验分位），与统计异常分开；流水线与 Agent 都会计算。
        self.extremes = list(extremes or [])
        # 预测（F）：默认 None（关闭）；开启时为包含回测与区间的字典。
        self.forecast = forecast

    def tool_ran(self, name: str) -> bool:
        """该工具本次是否**真的执行过**。

        agent 模式下 `executed_tools` 是权威答案。pipeline 模式下它是 `None`，
        过去直接返回 True——等于假设「计划里的 12 项全都跑过」。可是模型出力计划时
        常常只选 3–5 项（默认 `--planner auto` 有模型就交给模型），于是
        `anomaly` 没被选中、`anomalies` 为空，结论却仍写
        「未发现异常点（已对 N 条目标序列执行检测）」。

        这正是项目硬约束禁止的：没跑 ≠ 没检出。所以 `None` 时必须回落到
        `plan.analyses`，用 `TOOL_ANALYSES` 判断这个工具由哪些分析项触发。
        未知工具名保持 True，避免以后新增工具时静默变成「永不执行」。
        """
        if self.executed_tools is not None:
            return name in self.executed_tools
        required = TOOL_ANALYSES.get(name)
        if required is None:
            return True
        return bool(required & set(getattr(self.plan, "analyses", None) or ()))

    def metric(self, metric_id: str):
        rows = [r for r in self.metrics.rows if r.metric_id == metric_id]
        if rows:
            return rows[0]
        rows = [r for r in self.extra.rows if r.metric_id == metric_id]
        return rows[0] if rows else None


class ScenarioAdapter(ABC):
    scene_id: str = ""
    name: str = ""

    @abstractmethod
    def detect(self, frame: pd.DataFrame, profile: DatasetProfile) -> float:
        """返回该场景的匹配置信度，取值 [0, 1]。"""

    @abstractmethod
    def plan(
        self,
        question: str,
        profile: DatasetProfile,
        quality: QualityReport,
        frame: pd.DataFrame,
    ) -> AnalysisPlan:
        """生成分析计划：目标列、重采样规则、分析项与参数。"""

    @abstractmethod
    def extra_metrics(self, clean: CleanDataset, plan: AnalysisPlan) -> MetricTable:
        """场景专属指标（引擎通用指标之外的部分）。"""

    @abstractmethod
    def figure_specs(self, plan: AnalysisPlan) -> list[dict]:
        """图表规格列表，由 viz 按 kind 渲染。"""

    @abstractmethod
    def conclusions(self, ctx: ScenarioContext) -> tuple[list[Conclusion], list[Evidence]]:
        """生成结论与证据绑定；每条结论必须有 evidence。"""


_REGISTRY: dict[str, ScenarioAdapter] = {}


def register(adapter: ScenarioAdapter) -> None:
    _REGISTRY[adapter.scene_id] = adapter


def get_adapter(scene_id: str) -> ScenarioAdapter:
    if scene_id not in _REGISTRY:
        raise KeyError(f"未注册的场景适配器：{scene_id}，已注册：{sorted(_REGISTRY)}")
    return _REGISTRY[scene_id]


def detect_scene(
    frame: pd.DataFrame, profile: DatasetProfile
) -> tuple[Optional[ScenarioAdapter], float, dict[str, float]]:
    """自动识别场景：返回（最佳适配器，置信度，全部候选置信度）。"""
    scores = {sid: ad.detect(frame, profile) for sid, ad in _REGISTRY.items()}
    if not scores:
        return None, 0.0, {}
    best_id = max(scores, key=lambda k: scores[k])
    return _REGISTRY[best_id], scores[best_id], scores
