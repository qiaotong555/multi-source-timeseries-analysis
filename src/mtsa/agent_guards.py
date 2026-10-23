# -*- coding: utf-8 -*-
"""决策安全边界（T2.4）：模型能选什么、不能改什么、越界几次后停下。

设计原则
--------
1. **边界独立于运行时**。运行时只负责循环；「动作是否合法」由本模块判断，
   这样规则可以被单独测试，也不会随着提示词改动而漂移。
2. **拒绝必须有理由，且理由要回灌模型**。只返回 False 会让模型反复尝试同一动作。
3. **拒绝要留痕**。每次越界记录 kind、动作、步序与说明，供报告与后续指标统计。
4. **失败可重试一次**。工具失败可能是瞬时的；但成功或 skipped 的工具不得重复，
   否则会白白消耗步数与 token。

边界只覆盖「选择」：不改参数、不改数据、不改计算。锁定的标定参数直接拒绝，
未知参数也拒绝——静默忽略会让模型误以为参数生效。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from .planner import PARAM_LOCKED

# 越界类型。报告与指标按这个分类统计，不靠自由文本解析。
VIOLATION_TOOL_NOT_ALLOWED = "tool_not_allowed"
VIOLATION_DUPLICATE = "duplicate_tool"
VIOLATION_LOCKED_PARAM = "locked_parameter"
VIOLATION_UNKNOWN_ARG = "unknown_argument"
VIOLATION_STEP_BUDGET = "step_budget"


@dataclass(frozen=True)
class GuardPolicy:
    """一次运行的硬边界。"""

    allowed_tools: tuple[str, ...]
    max_steps: int = 8
    max_consecutive_violations: int = 3
    # 失败工具允许重试的次数。成功或 skipped 的工具不允许重复。
    max_retries_per_tool: int = 1
    locked_parameters: frozenset[str] = field(default_factory=lambda: frozenset(PARAM_LOCKED))
    # 当前工具全部从计划取参数，因此默认不接受任何入参。
    accepts_args: bool = False


@dataclass
class GuardDecision:
    allowed: bool
    kind: str = "ok"
    detail: str = ""
    is_retry: bool = False


class ActionGuard:
    """按策略检查动作，并累计越界与重试统计。"""

    def __init__(self, policy: GuardPolicy) -> None:
        self.policy = policy
        self.violations: list[dict[str, Any]] = []
        self._consecutive = 0
        self._retries: dict[str, int] = {}
        self._ok: dict[str, int] = {}

    @property
    def consecutive_violations(self) -> int:
        return self._consecutive

    def retry_count(self, tool: str) -> int:
        """某个工具已用掉的重试次数（运行时判断「已失败且无法再试」时需要）。"""
        return self._retries.get(tool, 0)

    def check(self, action: str, args: Mapping[str, Any],
              results: Mapping[str, Any], step_index: int) -> GuardDecision:
        """判断动作是否可执行。results 为 tool -> ToolResult（含 outcome）。"""
        if step_index >= self.policy.max_steps:
            return GuardDecision(False, VIOLATION_STEP_BUDGET,
                                 f"已达步数上限 {self.policy.max_steps}，只能结束")

        if action not in self.policy.allowed_tools:
            return GuardDecision(
                False, VIOLATION_TOOL_NOT_ALLOWED,
                f"工具 {action} 不在本场景白名单中，可选：{list(self.policy.allowed_tools)}")

        params = dict(args or {})
        locked = sorted(set(params) & set(self.policy.locked_parameters))
        if locked:
            return GuardDecision(
                False, VIOLATION_LOCKED_PARAM,
                f"参数 {locked} 属于已标定参数，禁止模型修改；如需调整请用 --param 显式覆盖")
        if params and not self.policy.accepts_args:
            return GuardDecision(
                False, VIOLATION_UNKNOWN_ARG,
                f"{action} 不接受参数，实际收到 {sorted(params)}")

        prev = results.get(action)
        if prev is not None:
            if getattr(prev, "outcome", None) == "failed":
                used = self._retries.get(action, 0)
                if used < self.policy.max_retries_per_tool:
                    return GuardDecision(True, "retry", f"{action} 上次失败，允许重试一次",
                                         is_retry=True)
                return GuardDecision(
                    False, VIOLATION_DUPLICATE,
                    f"{action} 已重试 {used} 次仍失败，不得继续重复调用")
            return GuardDecision(
                False, VIOLATION_DUPLICATE,
                f"{action} 已执行过（{getattr(prev, 'outcome', 'ok')}），不得重复调用")

        return GuardDecision(True)

    def record_violation(self, decision: GuardDecision, action: str, step_index: int) -> None:
        self._consecutive += 1
        self.violations.append({
            "step": step_index, "action": action,
            "kind": decision.kind, "detail": decision.detail,
        })

    def record_execution(self, action: str, outcome: str, is_retry: bool = False) -> None:
        if is_retry:
            self._retries[action] = self._retries.get(action, 0) + 1
        if outcome in ("ok", "skipped"):
            self._consecutive = 0
            self._ok[action] = self._ok.get(action, 0) + 1

    def stats(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        for v in self.violations:
            by_kind[v["kind"]] = by_kind.get(v["kind"], 0) + 1
        return {
            "violations": len(self.violations),
            "by_kind": by_kind,
            "consecutive_violations": self._consecutive,
            "retries": dict(self._retries),
            "ok_executions": dict(self._ok),
            "policy": {
                "allowed_tools": list(self.policy.allowed_tools),
                "max_steps": self.policy.max_steps,
                "max_consecutive_violations": self.policy.max_consecutive_violations,
                "max_retries_per_tool": self.policy.max_retries_per_tool,
                "locked_parameters": sorted(self.policy.locked_parameters),
                "accepts_args": self.policy.accepts_args,
            },
        }
