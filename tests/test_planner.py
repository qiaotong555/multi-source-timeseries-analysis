# -*- coding: utf-8 -*-
"""计划层测试：规则兜底、三级校验、缓存、无配置降级。

这些测试全部不联网。无 API key 是常态而非异常，测试必须能在离线环境完整运行。
"""
from __future__ import annotations

import pytest

from mtsa import io_loaders, profiler, quality
from mtsa.llm_client import LLMError
from mtsa.models import AnalysisPlan
from mtsa.planner import (FORCED_ANALYSES, PARAM_LOCKED, PLANNER_VERSION, TOOL_SPECS,
                          RulePlanner, LLMPlanner, PlannerValidationError, build_planner)
from mtsa.scenarios import stock as _stock  # noqa: F401  适配器靠 import 副作用注册
from mtsa.scenarios import weather as _weather  # noqa: F401
from mtsa.scenarios.base import get_adapter
from .helpers import make_weather_frame


class FakeClient:
    """替身客户端：可返回指定 JSON、或抛出指定异常，并统计调用次数。"""

    def __init__(self, payload=None, exc=None):
        self.payload = payload
        self.exc = exc
        self.calls = 0

    def chat_json(self, system: str, user: str):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.payload


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("planner")
    path = tmp / "w.csv"
    make_weather_frame(n=800, stations=2).to_csv(path, index=False)
    raw = io_loaders.load_dataset(str(path))
    profile, parsed = profiler.profile_dataset(raw)
    qrep = quality.check_quality(raw, profile, parsed)
    adapter = get_adapter("weather")
    return raw.frame, profile, qrep, adapter


def _plan(planner, ctx, question="分析温度的趋势与异常"):
    frame, profile, qrep, adapter = ctx
    return planner.plan(question, profile, qrep, adapter, frame)


# ---------------- 规则计划 ----------------

def test_rule_planner_marks_source(ctx):
    plan, source = _plan(RulePlanner(), ctx)
    assert source == "rule"
    assert plan.plan_source == "rule"
    assert plan.planner_version == PLANNER_VERSION
    assert plan.analyses and plan.target_columns


def test_build_planner_falls_back_without_config(ctx):
    """没有任何 LLM 配置时必须静默降级到规则计划，而不是报错。"""
    p = build_planner(ctx[3], env={})
    assert isinstance(p, RulePlanner)
    plan, source = _plan(p, ctx)
    assert source == "rule"

    p2 = build_planner(ctx[3], env={"MTSA_LLM_BASE_URL": "https://api.deepseek.com/v1"})
    assert isinstance(p2, RulePlanner)          # 缺 key 同样降级

    p3 = build_planner(ctx[3], env={"MTSA_PLANNER": "rule",
                                    "MTSA_LLM_BASE_URL": "https://x", "MTSA_LLM_API_KEY": "k"})
    assert isinstance(p3, RulePlanner)          # 显式 rule 优先


def test_partial_llm_config_is_reported_not_silent(ctx):
    """配了一半必须留痕：用户以为模型生效、实际走规则，是最难发现的配置错误。"""
    p = build_planner(ctx[3], env={"MTSA_LLM_API_KEY": "sk-test"})
    assert isinstance(p, RulePlanner)
    assert p.last_error and "MTSA_LLM_BASE_URL" in p.last_error
    assert "MTSA_LLM_API_KEY" not in p.last_error      # 只报真正缺的那个

    p2 = build_planner(ctx[3], env={"MTSA_LLM_BASE_URL": "https://api.deepseek.com/v1"})
    assert isinstance(p2, RulePlanner)
    assert p2.last_error and "MTSA_LLM_API_KEY" in p2.last_error


def test_fully_absent_llm_config_stays_silent(ctx):
    """一个都没配是正常的规则模式，不应产生告警——否则每次运行都多一条噪音。"""
    p = build_planner(ctx[3], env={})
    assert isinstance(p, RulePlanner)
    assert p.last_error is None


def test_explicit_llm_mode_without_config_reports_error(ctx):
    """显式要求用模型却没配，必须报出来，不能静默降级。"""
    p = build_planner(ctx[3], env={"MTSA_PLANNER": "llm"})
    assert isinstance(p, RulePlanner)
    assert p.last_error and "MTSA_LLM_BASE_URL" in p.last_error
    assert "MTSA_LLM_API_KEY" in p.last_error


def test_build_planner_uses_llm_when_configured(ctx):
    p = build_planner(ctx[3], env={
        "MTSA_LLM_BASE_URL": "https://api.deepseek.com/v1",
        "MTSA_LLM_API_KEY": "sk-test",
        "MTSA_LLM_MODEL": "deepseek-chat",
    })
    assert isinstance(p, LLMPlanner)


# ---------------- 校验与兜底 ----------------

def test_valid_llm_plan_is_adopted(ctx):
    client = FakeClient(payload={"analyses": ["trend", "anomaly"],
                                 "target_columns": ["temperature"],
                                 "parameters": {"seasonal_period": 24},
                                 "reason": "问题关注趋势与异常"})
    plan, source = _plan(LLMPlanner(client, cache_dir=None), ctx)
    assert source == "llm"
    assert plan.plan_source == "llm"
    assert plan.analyses == ["trend", "anomaly"]
    assert plan.target_columns == ["temperature"]
    assert plan.parameters["seasonal_period"] == 24
    assert client.calls == 1


def test_unknown_tool_falls_back(ctx):
    client = FakeClient(payload={"analyses": ["trend", "magic_forecast"],
                                 "target_columns": ["temperature"], "reason": "x"})
    planner = LLMPlanner(client, cache_dir=None)
    plan, source = _plan(planner, ctx)
    assert source == "rule"
    assert "未知分析项" in (planner.last_error or "")
    assert "magic_forecast" not in plan.analyses


def test_missing_column_falls_back(ctx):
    client = FakeClient(payload={"analyses": ["trend"],
                                 "target_columns": ["no_such_column"], "reason": "x"})
    planner = LLMPlanner(client, cache_dir=None)
    _, source = _plan(planner, ctx)
    assert source == "rule"
    assert "不存在的列" in (planner.last_error or "")


def test_locked_parameter_falls_back(ctx):
    """已标定的阈值不允许模型改动，否则会破坏已有评估结论。"""
    payload = {"analyses": ["anomaly"], "target_columns": ["temperature"],
               "parameters": {"anomaly_threshold": 4}, "reason": "x"}
    planner = LLMPlanner(FakeClient(payload=payload), cache_dir=None)
    _, source = _plan(planner, ctx)
    assert source == "rule"
    assert "禁止模型修改" in (planner.last_error or "")


def test_out_of_whitelist_parameter_falls_back(ctx):
    payload = {"analyses": ["trend"], "target_columns": ["temperature"],
               "parameters": {"made_up_param": 3}, "reason": "x"}
    planner = LLMPlanner(FakeClient(payload=payload), cache_dir=None)
    _, source = _plan(planner, ctx)
    assert source == "rule"
    assert "白名单" in (planner.last_error or "")


def test_negative_parameter_falls_back(ctx):
    payload = {"analyses": ["trend"], "target_columns": ["temperature"],
               "parameters": {"max_lag": -5}, "reason": "x"}
    planner = LLMPlanner(FakeClient(payload=payload), cache_dir=None)
    _, source = _plan(planner, ctx)
    assert source == "rule"


def test_empty_analyses_falls_back(ctx):
    planner = LLMPlanner(FakeClient(payload={"analyses": [], "target_columns": ["temperature"]}),
                         cache_dir=None)
    _, source = _plan(planner, ctx)
    assert source == "rule"


def test_client_error_falls_back(ctx):
    planner = LLMPlanner(FakeClient(exc=LLMError("网络或超时：timed out")), cache_dir=None)
    plan, source = _plan(planner, ctx)
    assert source == "rule"
    assert "LLMError" in (planner.last_error or "")
    assert plan.analyses                      # 回退后仍有可用计划


# ---------------- 缓存 ----------------

def test_cache_hit_skips_model(ctx, tmp_path):
    payload = {"analyses": ["trend"], "target_columns": ["temperature"],
               "parameters": {}, "reason": "关注趋势"}
    client = FakeClient(payload=payload)
    planner = LLMPlanner(client, cache_dir=str(tmp_path / "cache"))

    _, first = _plan(planner, ctx)
    assert first == "llm" and client.calls == 1

    plan2, second = _plan(planner, ctx)
    assert second == "cache"
    assert client.calls == 1                  # 命中缓存后不再调用模型
    assert plan2.plan_source == "cache"
    assert plan2.analyses == ["trend"]


def test_corrupted_cache_is_ignored(ctx, tmp_path):
    cache_dir = tmp_path / "cache"
    client = FakeClient(payload={"analyses": ["trend"], "target_columns": ["temperature"],
                                 "reason": "x"})
    planner = LLMPlanner(client, cache_dir=str(cache_dir))
    _plan(planner, ctx)
    assert client.calls == 1

    for f in cache_dir.glob("*.json"):
        f.write_text("{not json", encoding="utf-8")

    _, source = _plan(planner, ctx)
    assert source == "llm"                    # 缓存损坏按未命中处理
    assert client.calls == 2


def test_locked_params_are_declared(ctx):
    """锁定参数清单必须非空，否则「模型不得改阈值」这条约束没有落点。"""
    assert PARAM_LOCKED
    assert "anomaly_threshold" in PARAM_LOCKED


# ---------------- 与下游的一致性 ----------------

def test_tool_names_match_downstream_consumers():
    """工具名必须与下游真正判断的字符串一致。

    下游消费点：engine 判断 spectrum / anomaly / changepoint；metrics 判断 corr / lag；
    stats_tests 判断 stationarity / autocorr / normality / trend / test / randomness。
    若这里改了名字而下游没改，模型选的工具会被静默忽略——计划看起来成功，实际什么都没跑。
    """
    downstream = {
        "trend", "corr", "lag", "extreme", "spectrum", "changepoint", "anomaly",
        "test", "stationarity", "autocorr", "normality", "seasonal", "randomness",
    }
    assert set(TOOL_SPECS) == downstream


def test_rule_plan_analyses_all_selectable(ctx):
    """规则计划产出的分析项必须都在工具白名单内，否则模型无从表达同样的选择。"""
    plan, _ = _plan(RulePlanner(), ctx)
    unexpected = set(plan.analyses) - set(TOOL_SPECS) - set(FORCED_ANALYSES)
    assert not unexpected, f"规则计划的这些分析项不在白名单内：{sorted(unexpected)}"


def test_forced_analysis_survives_model_selection(ctx):
    """场景必需项不受模型控制：base 里有、模型没选，合并后仍要保留。"""
    frame = ctx[0]
    base = AnalysisPlan(scene_id="stock", scene_confidence=1.0, question="回测",
                        target_columns=["temperature"],
                        analyses=["trend", "scenario_metrics"], parameters={})
    planner = LLMPlanner(FakeClient(payload={}), cache_dir=None)
    merged = planner._validate_and_merge(
        {"analyses": ["trend"], "target_columns": ["temperature"],
         "parameters": {}, "reason": "只关注趋势"},
        base, frame)
    assert "scenario_metrics" in merged.analyses
    assert merged.analyses[0] == "trend"
