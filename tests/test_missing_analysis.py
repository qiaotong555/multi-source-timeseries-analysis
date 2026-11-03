# -*- coding: utf-8 -*-
"""「未执行」与「未检出」必须分开说。

项目硬约束：模型选的分析项可能少于规则计划，报告里必须说明「未跑」而非「未检出」。

原来的实现在两处都违反了它：
1. `ScenarioContext.tool_ran` 在 pipeline 模式（`executed_tools is None`）直接返回 True，
   等于假设「计划里的项全都跑过」；
2. 报告第 8 节只看结果列表是否为空，空就写「未检出」。

于是模型只选 3 项、恰好没选 `anomaly` 时，报告与结论都会写「未发现异常点
（已对 N 条目标序列执行检测）」——而实际一次检测都没跑。
"""

from __future__ import annotations

import os

import pytest

from mtsa import engine
from mtsa.models import AnalysisRequest
from mtsa.scenarios.base import TOOL_ANALYSES, ScenarioContext

FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")
WEATHER_CSV = os.path.join(FIX, "weather_v1.csv")


class _RestrictedPlanner:
    """只保留指定分析项的规则计划器替身（模拟模型挑了几项就收手）。"""

    last_error = None

    def __init__(self, keep: set[str]) -> None:
        self.keep = set(keep)

    def plan(self, question, profile, qrep, adapter, frame):
        base = adapter.plan(question, profile, qrep, frame)
        base.analyses = [a for a in base.analyses if a in self.keep]
        base.target_columns = [c for c in base.target_columns if c in frame.columns]
        base.plan_source = "rule"
        return base, "rule"


# ---------------- tool_ran 的语义 ----------------

def test_tool_ran_falls_back_to_plan_when_no_executed_tools():
    """pipeline 模式：没在计划里的分析项，对应工具就是没跑过。"""

    class _Plan:
        analyses = ["trend", "seasonal"]

    class _Clean:
        frame = None

    ctx = ScenarioContext.__new__(ScenarioContext)
    ctx.executed_tools = None
    ctx.plan = _Plan()

    assert ctx.tool_ran("detect_anomalies") is False
    assert ctx.tool_ran("detect_change_points") is False
    # compute_metrics 不受计划门控（引擎里无条件执行），保持 True
    assert ctx.tool_ran("compute_metrics") is True
    assert "compute_metrics" not in TOOL_ANALYSES


def test_tool_ran_trusts_executed_tools_in_agent_mode():
    """agent 模式：`executed_tools` 是权威答案，不再看计划。"""

    class _Plan:
        analyses = ["anomaly", "changepoint"]

    ctx = ScenarioContext.__new__(ScenarioContext)
    ctx.executed_tools = {"detect_anomalies"}
    ctx.plan = _Plan()

    assert ctx.tool_ran("detect_anomalies") is True
    assert ctx.tool_ran("detect_change_points") is False   # 计划里有，但实际没跑


def test_tool_ran_keeps_unknown_tools_permissive():
    """未知工具名保持 True：以后新增工具时不能静默变成「永不执行」。"""
    ctx = ScenarioContext.__new__(ScenarioContext)
    ctx.executed_tools = None
    ctx.plan = type("_P", (), {"analyses": []})()
    assert ctx.tool_ran("some_future_tool") is True
    assert "some_future_tool" not in TOOL_ANALYSES


# ---------------- 端到端：报告与 warnings ----------------

@pytest.fixture(scope="module")
def restricted_run(tmp_path_factory):
    """只跑 trend 的流水线结果（真实 run_analysis，不是桩）。"""
    out = str(tmp_path_factory.mktemp("scope"))
    req = AnalysisRequest(request_id="req_scope", input_path=WEATHER_CSV, scene_id="weather",
                          question="分析温度的趋势与季节性", seed=20260927,
                          output_dir=out, planner="rule")
    original = engine.build_planner
    engine.build_planner = lambda *a, **k: _RestrictedPlanner({"trend"})
    try:
        res = engine.run_analysis(req, verbose=False)
    finally:
        engine.build_planner = original
    assert res.status == "succeeded", res.error
    report = next(a.path for a in res.artifacts if a.kind == "report_md")
    with open(report, encoding="utf-8") as f:
        return res, f.read()


def test_report_says_not_run_for_omitted_analysis(restricted_run):
    res, report = restricted_run
    assert "anomaly" not in res.data["plan"]["analyses"]
    assert "异常点：**未执行**" in report
    assert "变化点：**未执行**" in report


def test_report_never_claims_zero_hits_when_not_run(restricted_run):
    """旧实现会写出「未发现异常点（…已对 N 条目标序列执行检测）」。"""
    _, report = restricted_run
    assert "未发现异常点" not in report
    assert "未发现均值突变" not in report


def test_pipeline_warns_about_omitted_analysis(restricted_run):
    res, _ = restricted_run
    warn = " ".join(res.warnings or [])
    assert "未包含以下分析项" in warn
    assert "不能解读为「未发现问题」" in warn


def test_extremes_not_computed_when_not_planned(restricted_run):
    """计划里既没有 `extreme` 也没有 `anomaly` 时，极值不该被算出来。

    这条曾经是死键：引擎无条件调 `detect_extremes`，于是模型只选 trend/seasonal 时，
    报告 8b 照样打印整张极值表，而 `extreme` 在引擎里没有任何消费者。
    """
    res, report = restricted_run
    assert res.data["extremes"] == []
    assert "极端值：**未执行**" in report


def _run_with(tmp_path, keep, tag):
    req = AnalysisRequest(request_id="req_" + tag, input_path=WEATHER_CSV,
                          scene_id="weather", question="分析温度的趋势与季节性",
                          seed=20260927, output_dir=str(tmp_path / tag), planner="rule")
    original = engine.build_planner
    engine.build_planner = lambda *a, **k: _RestrictedPlanner(keep)
    try:
        return engine.run_analysis(req, verbose=False)
    finally:
        engine.build_planner = original


def test_extreme_switch_is_live(tmp_path):
    """`extreme` 必须是一个真的开关：选了就有极值，不选就没有。

    对比只取决于这一项，所以两次运行除 keep 集合外完全相同。
    """
    with_ext = _run_with(tmp_path, {"trend", "extreme"}, "with_ext")
    without = _run_with(tmp_path, {"trend"}, "without_ext")
    assert with_ext.status == "succeeded" and without.status == "succeeded"

    assert with_ext.data["extremes"], "选了 extreme 却没算出极值"
    assert without.data["extremes"] == [], "没选 extreme 却算出了极值"
    # anomaly 不在计划里，但 extreme 在——极值随它一起产出
    assert "anomaly" not in with_ext.data["plan"]["analyses"]
