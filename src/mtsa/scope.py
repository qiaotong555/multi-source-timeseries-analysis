# -*- coding: utf-8 -*-
"""问题范围下限（T2.11）：把「这个问题至少要覆盖哪些分析」变成确定性规则。

为什么需要它
------------
T2.8 / T2.9 暴露出一个可复现性问题：同一个问题跑两次，步数会从 3 步变成 5 步。
原因不在温度参数（已是 0），而在于**结束条件是主观的**——提示词说
「当你认为已覆盖问题所需的分析时输出 finish」，"所需"由模型每次自行判断，
判断边界漂移，步数就漂移。

本模块把「所需」外置为一条可复核规则：从问题关键词推出**下限工具集**。
下限只规定「不能少于」，不规定顺序，也不禁止模型在下限之外追加有限步数。
这样既保住「顺序由模型决定」，又让「最少做到什么」可复现、可测试、可审计。

边界
----
1. **纯函数、无模型、无 IO**：同样的 (问题, 白名单) 永远得到同样的下限。
2. **只做加法，不做减法**：不因为没提某词就禁止某个工具。
3. **基础指标恒在下限内**：`compute_metrics` 承载趋势/季节/极值/相关等描述性指标，
   报告与结论都以它为底座，缺了它下游只能拿到空指标表。
4. **关键词表是白盒**：命中的词会写进轨迹，便于人工复核「为什么这一步是必须的」。

已知限制
--------
- 关键词匹配不理解否定与语境（如「不需要检验」仍会命中「检验」）。下限只会让
  模型多做一步，不会让它少做，因此错的代价是成本而非正确性。
- 中文同义词覆盖靠人工枚举，新表述需要补词并补测试。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

# 恒在下限内的基础工具
BASE_TOOL = "compute_metrics"

# 工具 -> 触发词。顺序即输出顺序，保证下限集合稳定可比较。
FLOOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "run_stat_tests": (
        "平稳", "单位根", "adf", "kpss", "自相关", "自协方差", "正态",
        "随机性", "白噪声", "显著性", "假设检验", "统计检验",
    ),
    "analyze_spectrum": (
        "周期", "频谱", "主频", "功率谱", "振荡", "谐波", "频域",
    ),
    "detect_anomalies": (
        "异常", "离群", "故障", "越界点", "离群点",
    ),
    "detect_change_points": (
        "变化点", "突变", "拐点", "结构变化", "转折点", "变点",
    ),
}


@dataclass(frozen=True)
class ScopeFloor:
    """一次运行的范围下限。"""

    tools: tuple[str, ...]
    matched: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    source: str = "rule_keywords"

    def missing(self, results: Mapping[str, object]) -> tuple[str, ...]:
        """下限里尚未成功的工具。

        只有 `ok` / `skipped` 算覆盖：`skipped` 表示前置条件不满足，已处理；
        `failed` 不算覆盖，但也不无限重试——是否放行由运行时结合重试次数决定。
        """
        out = []
        for name in self.tools:
            res = results.get(name)
            outcome = getattr(res, "outcome", None)
            if outcome not in ("ok", "skipped"):
                out.append(name)
        return tuple(out)

    def met(self, results: Mapping[str, object]) -> bool:
        return not self.missing(results)

    def to_dict(self, results: Mapping[str, object] | None = None) -> dict:
        out = {"source": self.source, "tools": list(self.tools),
               "matched": {k: list(v) for k, v in self.matched.items()}}
        if results is not None:
            missing = self.missing(results)
            out["missing"] = list(missing)
            out["met"] = not missing
        return out


def infer_floor(question: str, allowed_tools: Sequence[str]) -> ScopeFloor:
    """按关键词推出下限工具集；结果只包含白名单内的工具。

    `compute_metrics` 若不在白名单（例如未来的极简场景），则不加；
    下限至少保留白名单中的第一个工具，避免出现「空下限」让任何一次结束都合法。
    """
    allowed = list(dict.fromkeys(allowed_tools))
    allowed_set = set(allowed)
    text = (question or "").lower()

    chosen: list[str] = []
    matched: dict[str, tuple[str, ...]] = {}

    for tool, words in FLOOR_KEYWORDS.items():
        if tool not in allowed_set:
            continue
        hits = tuple(w for w in words if w in text)
        if hits:
            chosen.append(tool)
            matched[tool] = hits

    if BASE_TOOL in allowed_set:
        chosen.insert(0, BASE_TOOL)
        matched.setdefault(BASE_TOOL, ("基础指标（恒在下限内）",))

    if not chosen and allowed:
        chosen.append(allowed[0])
        matched[allowed[0]] = ("白名单首项兜底",)

    # 按白名单顺序输出，保证同一问题在两次运行中下限顺序一致
    order = {name: i for i, name in enumerate(allowed)}
    tools = tuple(sorted(dict.fromkeys(chosen), key=lambda t: order[t]))
    return ScopeFloor(tools=tools, matched=matched)


def describe(floor: ScopeFloor) -> str:
    """把下限写成一句给模型看的话。"""
    return "、".join(floor.tools) if floor.tools else "（无）"
