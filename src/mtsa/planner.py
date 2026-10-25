# -*- coding: utf-8 -*-
"""计划层：决定「用哪些工具、分析哪些列、什么参数」。

改造的边界
----------
本模块只替换引擎的第 4 个节点（场景识别与计划），节点 5 至 12 完全不变。
模型输出的是**结构化计划**，不是自然语言指令；所有数值结论仍由确定性代码产出。

兜底链
------
    缓存 → LLM → 规则 → 最小可用计划

任一环节失败都不抛到引擎，而是向下退一级。这样做的理由很实际：计划层是把
不确定的外部服务引入原本确定的流水线，若不设兜底，一次网络抖动就会让整次分析失败。

三级校验
--------
一级 格式：JSON 能否解析、字段类型是否正确
二级 schema：必填字段是否齐全、是否为非空列表
三级 语义：工具名是否在白名单、列是否真实存在、参数是否越界

三级校验中语义校验最关键：它拦住的是模型最常见的三类错误——编造不存在的分析项、
指定数据集里没有的列、把窗口设成负数或把标定好的阈值改掉。
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Optional, Protocol

import pandas as pd

from .models import AnalysisPlan

PLANNER_VERSION = "v1"

# 模型可选的分析项（工具）。
# 键必须与下游真正的消费点一致：engine 判断 spectrum/anomaly/changepoint，
# metrics 判断 corr/lag，stats_tests 判断 stationarity/autocorr/normality/trend/test/randomness。
# 用粗粒度的自造名字（如 correlation）会导致「模型选了但下游不执行」，是静默失效。
TOOL_SPECS: dict[str, str] = {
    "trend": "趋势与斜率：序列是否存在单调变化、趋势强度。适用于有长期走向的数据。",
    "corr": "列间相关：多列之间的同步关系。需要至少两列。",
    "lag": "滞后相关：变量间的领先滞后关系，给出最优滞后阶。需要至少两列。",
    "extreme": "极值统计：最大最小、分位数与超阈值事件。关注极端情形时使用。",
    "spectrum": "频域分析：Welch 功率谱、主频与周期。要求采样均匀且样本量足够。",
    "changepoint": "变化点检测：均值与方差突变。关注结构性变化或行为拐点时使用。",
    "anomaly": "异常检测：离群点与故障点定位。关注数据质量或设备异常时使用。",
    "test": "统计检验总开关：一次开启平稳性、自相关、正态性、趋势四类检验。",
    "stationarity": "平稳性检验（ADF 与 KPSS）：序列是否适合直接建模。",
    "autocorr": "自相关检验（Ljung-Box）：残差是否仍有结构。",
    "normality": "正态性检验（Jarque-Bera）：是否需要改用非参数方法。",
    "seasonal": "季节分解：剥离周期成分，输出趋势项与季节项。存在明显周期时使用。",
    "randomness": "随机性检验（游程）：序列是否近似随机。",
}

# 场景必需项：由适配器决定，不受模型控制。
# 例如股票的 scenario_metrics 承载回测与风险指标，若被模型裁掉，专属指标会静默消失。
FORCED_ANALYSES = frozenset({"scenario_metrics"})

# 允许模型调整的参数键（分析配置类）
PARAM_OVERRIDABLE = frozenset({"seasonal_period", "max_lag", "rolling_window", "change_point_depth"})
# 禁止模型调整的参数键（统计标定类，改动会破坏已有评估结论）
PARAM_LOCKED = frozenset({"anomaly_threshold", "change_point_sse_ratio", "close_column"})

MIN_ANALYSES = 1


class PlannerValidationError(ValueError):
    """计划未通过校验。校验失败一律回退，不做「尽力修补」。"""


class Planner(Protocol):
    source: str
    last_error: Optional[str]

    def plan(self, question: str, profile, qrep, adapter, frame: pd.DataFrame) -> tuple[AnalysisPlan, str]:
        ...


# ---------------- 规则计划 ----------------

class RulePlanner:
    """现有逻辑的包装：由场景适配器按规则给出计划。永远可用，是兜底链的底座。"""

    source = "rule"
    last_error: Optional[str] = None

    def plan(self, question: str, profile, qrep, adapter, frame: pd.DataFrame):
        plan = adapter.plan(question, profile, qrep, frame)
        plan.plan_source = "rule"
        plan.planner_version = PLANNER_VERSION
        return plan, "rule"


# ---------------- LLM 计划 ----------------

class LLMPlanner:
    """模型生成计划，规则兜底。缓存命中时不调用模型。"""

    source = "llm"
    last_error: Optional[str] = None

    def __init__(self, client, cache_dir: Optional[str] = None) -> None:
        self.client = client
        self.cache_dir = cache_dir
        self.last_error = None

    # ---- 入口 ----

    def plan(self, question: str, profile, qrep, adapter, frame: pd.DataFrame):
        # 基底永远由适配器给出：保证场景专属参数（如股票的 close_column）不丢，
        # 也让「模型只覆盖可覆盖的部分」这条边界天然成立。
        base = adapter.plan(question, profile, qrep, frame)
        self.last_error = None

        key = self._cache_key(question, profile, frame, base.scene_id)
        cached = self._cache_load(key)
        if cached is not None:
            return cached, "cache"

        try:
            payload = self.client.chat_json(self._system_prompt(), self._user_prompt(question, profile, qrep, frame))
            final = self._validate_and_merge(payload, base, frame)
        except Exception as exc:                      # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"[:300]
            base.plan_source = "rule"
            base.planner_version = PLANNER_VERSION
            return base, "rule"

        final.plan_source = "llm"
        final.planner_version = PLANNER_VERSION
        self._cache_save(key, final)
        return final, "llm"

    # ---- 校验 ----

    def _validate_and_merge(self, payload: dict, base: AnalysisPlan,
                            frame: pd.DataFrame) -> AnalysisPlan:
        # 一级：格式
        if not isinstance(payload, dict):
            raise PlannerValidationError("计划不是 JSON 对象")
        analyses = payload.get("analyses")
        if not isinstance(analyses, list) or not analyses:
            raise PlannerValidationError("analyses 必须是非空列表")

        columns = payload.get("target_columns") or base.target_columns
        if not isinstance(columns, list) or not columns:
            raise PlannerValidationError("target_columns 必须是非空列表")

        params = payload.get("parameters") or {}
        if not isinstance(params, dict):
            raise PlannerValidationError("parameters 必须是对象")

        # 二级：元素类型
        if not all(isinstance(a, str) for a in analyses):
            raise PlannerValidationError("analyses 元素必须为字符串")
        if not all(isinstance(c, str) for c in columns):
            raise PlannerValidationError("target_columns 元素必须为字符串")

        # 三级：语义
        unknown_tools = sorted(set(analyses) - set(TOOL_SPECS))
        if unknown_tools:
            raise PlannerValidationError(f"未知分析项：{unknown_tools}")

        absent_cols = [c for c in columns if c not in frame.columns]
        if absent_cols:
            raise PlannerValidationError(f"数据中不存在的列：{absent_cols}")

        locked = sorted(set(params) & PARAM_LOCKED)
        if locked:
            raise PlannerValidationError(f"禁止模型修改的参数：{locked}")

        illegal = sorted(set(params) - PARAM_OVERRIDABLE)
        if illegal:
            raise PlannerValidationError(f"不在白名单内的参数：{illegal}")

        for key, val in params.items():
            if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
                raise PlannerValidationError(f"参数 {key} 必须为正整数，实际为 {val!r}")

        # 通过校验：在基底上覆盖，未覆盖的字段保持适配器默认
        merged = base.model_copy(deep=True)
        chosen = list(dict.fromkeys(analyses))                 # 去重并保持顺序
        # 场景必需项不受模型控制，否则股票的 scenario_metrics 会被裁掉，
        # 回测与风险指标静默消失，而评估里的「方法选择准确率」会立刻失败。
        for forced in FORCED_ANALYSES:
            if forced in base.analyses and forced not in chosen:
                chosen.append(forced)
        merged.analyses = chosen
        merged.target_columns = list(dict.fromkeys(columns))
        merged.parameters.update(params)
        return merged

    # ---- 提示词 ----

    @staticmethod
    def _system_prompt() -> str:
        tools = "\n".join(f"- {k}：{v}" for k, v in TOOL_SPECS.items())
        locked = "、".join(sorted(PARAM_LOCKED))
        allowed = "、".join(sorted(PARAM_OVERRIDABLE))
        return (
            "你是时序分析的计划器。根据数据摘要与用户问题，从下列工具中选择合适的分析项。\n\n"
            f"可用工具：\n{tools}\n\n"
            "要求：\n"
            "1. 只输出 JSON 对象，不要任何解释性文字或代码块标记。\n"
            "2. analyses 必须是上述工具名的子集，至少一项，一般不超过五项；"
            "没有依据的工具不要选。'test' 会一次性开启四类检验，"
            "若只需其中某一类，直接选 stationarity、autocorr、normality 或 trend。\n"
            "3. target_columns 只能取数据摘要中列出的数值列。\n"
            f"4. parameters 只允许调整这些键：{allowed}，且必须为正整数。\n"
            f"5. 严禁修改这些已标定的参数：{locked}。\n"
            "6. reason 字段用一句话说明选择理由，不得为空。\n\n"
            '输出格式：{"analyses": [...], "target_columns": [...], '
            '"parameters": {...}, "reason": "..."}'
        )

    @staticmethod
    def _user_prompt(question: str, profile, qrep, frame: pd.DataFrame) -> str:
        numeric = list(getattr(profile, "numeric_columns", []) or [])
        summary: dict[str, Any] = {
            "question": question,
            "rows": int(len(frame)),
            "numeric_columns": numeric[:20],
            "sampling_interval": getattr(getattr(profile, "sampling", None), "inferred_freq", None),
            "gaps": int(getattr(getattr(profile, "sampling", None), "gap_count", 0) or 0),
            "quality_issues": len(getattr(qrep, "issues", []) or []),
        }
        return json.dumps(summary, ensure_ascii=False)

    # ---- 缓存 ----

    def _cache_key(self, question: str, profile, frame: pd.DataFrame, scene_id: str) -> str:
        parts = [
            PLANNER_VERSION, scene_id, question.strip(),
            ",".join(map(str, getattr(profile, "numeric_columns", []) or [])),
            str(len(frame)),
        ]
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:20]

    def _cache_path(self, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"{key}.json")

    def _cache_load(self, key: str) -> Optional[AnalysisPlan]:
        path = self._cache_path(key)
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            plan = AnalysisPlan(**data)
            plan.plan_source = "cache"
            return plan
        except Exception:                             # noqa: BLE001
            return None                               # 缓存损坏当作未命中

    def _cache_save(self, key: str, plan: AnalysisPlan) -> None:
        path = self._cache_path(key)
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(plan.model_dump(), f, ensure_ascii=False, indent=2)
        except OSError:
            pass                                      # 缓存写不进去不影响本次结果


# ---------------- 工厂 ----------------

def _llm_config_problem(env: dict[str, str], explicit: bool) -> Optional[str]:
    """诊断 LLM 配置为何不可用；完全未配置且未显式要求时返回 None（属正常规则模式）。

    「一个都没配」与「配了一半」必须区别对待：前者是用户本就要用规则模式，
    不该产生告警；后者是用户明确想用模型却写错了变量名，静默回退会让人
    误以为模型生效了，属于必须暴露的配置错误。
    """
    base = (env.get("MTSA_LLM_BASE_URL") or "").strip()
    key = (env.get("MTSA_LLM_API_KEY") or "").strip()
    if base and key:
        return None
    if not base and not key and not explicit:
        return None
    missing = [n for n, v in (("MTSA_LLM_BASE_URL", base), ("MTSA_LLM_API_KEY", key)) if not v]
    return ("已要求使用模型计划，但缺少环境变量：" + "、".join(missing)
            if explicit else "LLM 配置不完整，已回退规则计划，缺少环境变量：" + "、".join(missing))


def build_planner(adapter, env: Optional[dict[str, str]] = None,
                  mode: Optional[str] = None) -> Planner:
    """按配置选择计划器。无配置即规则计划，保证零成本可用。

    `mode` 是**请求级**覆盖（`AnalysisRequest.planner`）：给了就以它为准，
    没给才读环境变量 `MTSA_PLANNER`。这样调用方不必为了换个计划器去改服务端配置，
    而既有调用（只设环境变量、不传该字段）行为完全不变。
    """
    import os

    from .llm_client import client_from_env

    env = env if env is not None else os.environ
    if mode is None:
        mode = env.get("MTSA_PLANNER", "auto").strip().lower() or "auto"
    else:
        mode = str(mode).strip().lower()
        if mode not in ("auto", "rule", "llm"):
            mode = "auto"

    if mode == "rule":
        return RulePlanner()
    if mode in ("llm", "auto"):
        client = client_from_env(env)
        if client is None:
            planner = RulePlanner()
            # 配置不全时留下原因，engine 会把它写进 result.warnings，避免静默回退
            planner.last_error = _llm_config_problem(env, explicit=(mode == "llm"))
            return planner
        cache_dir = env.get("MTSA_PLAN_CACHE", "artifacts/plan_cache")
        return LLMPlanner(client, cache_dir.strip() or None)
    return RulePlanner()
