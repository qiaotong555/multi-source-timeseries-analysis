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
        return True if self.executed_tools is None else name in self.executed_tools

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
