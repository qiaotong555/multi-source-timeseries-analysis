# -*- coding: utf-8 -*-
"""T2.4 决策安全边界：白名单、锁定参数、重复、重试与留痕。"""
from __future__ import annotations

from mtsa.agent_guards import (VIOLATION_DUPLICATE, VIOLATION_LOCKED_PARAM,
                               VIOLATION_STEP_BUDGET, VIOLATION_TOOL_NOT_ALLOWED,
                               VIOLATION_UNKNOWN_ARG, ActionGuard, GuardPolicy)
from mtsa.tools import ToolResult


def _guard(**kw) -> ActionGuard:
    base = {"allowed_tools": ("a", "b"), "max_steps": 3, "max_retries_per_tool": 1}
    base.update(kw)
    return ActionGuard(GuardPolicy(**base))


def _res(outcome: str) -> ToolResult:
    return ToolResult(tool="x", outcome=outcome)


def test_unknown_tool_is_rejected_with_choices():
    d = _guard().check("c", {}, {}, 0)
    assert not d.allowed
    assert d.kind == VIOLATION_TOOL_NOT_ALLOWED
    assert "['a', 'b']" in d.detail


def test_locked_parameter_is_rejected_before_unknown_arg():
    d = _guard().check("a", {"anomaly_threshold": 5.0}, {}, 0)
    assert not d.allowed
    assert d.kind == VIOLATION_LOCKED_PARAM
    assert "标定参数" in d.detail


def test_unknown_argument_is_rejected():
    d = _guard().check("a", {"columns": ["x"]}, {}, 0)
    assert not d.allowed
    assert d.kind == VIOLATION_UNKNOWN_ARG
    assert "不接受参数" in d.detail


def test_step_budget_is_rejected():
    d = _guard().check("a", {}, {}, 3)
    assert not d.allowed
    assert d.kind == VIOLATION_STEP_BUDGET


def test_successful_tool_cannot_repeat():
    d = _guard().check("a", {}, {"a": _res("ok")}, 1)
    assert not d.allowed
    assert d.kind == VIOLATION_DUPLICATE


def test_failed_tool_can_retry_once_then_blocked():
    g = _guard()
    results = {"a": _res("failed")}
    d1 = g.check("a", {}, results, 1)
    assert d1.allowed and d1.is_retry
    g.record_execution("a", "failed", is_retry=True)
    d2 = g.check("a", {}, results, 2)
    assert not d2.allowed
    assert "已重试 1 次" in d2.detail


def test_skipped_tool_counts_as_handled_not_retryable():
    d = _guard().check("a", {}, {"a": _res("skipped")}, 1)
    assert not d.allowed
    assert d.kind == VIOLATION_DUPLICATE


def test_consecutive_violations_reset_on_progress():
    g = _guard()
    d = g.check("c", {}, {}, 0)
    g.record_violation(d, "c", 0)
    g.record_violation(d, "c", 0)
    assert g.consecutive_violations == 2
    g.record_execution("a", "ok")
    assert g.consecutive_violations == 0


def test_stats_track_kinds_and_policy():
    g = _guard()
    for action, args in (("c", {}), ("a", {"anomaly_threshold": 1}), ("a", {"x": 1})):
        d = g.check(action, args, {}, 0)
        g.record_violation(d, action, 0)
    s = g.stats()
    assert s["violations"] == 3
    assert s["by_kind"][VIOLATION_TOOL_NOT_ALLOWED] == 1
    assert s["by_kind"][VIOLATION_LOCKED_PARAM] == 1
    assert s["by_kind"][VIOLATION_UNKNOWN_ARG] == 1
    assert s["policy"]["max_steps"] == 3
