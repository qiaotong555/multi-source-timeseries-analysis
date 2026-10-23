# -*- coding: utf-8 -*-
"""D 探针评估的纯函数测试：不调模型、不跑引擎。"""
from __future__ import annotations

import pytest

from evals.probe_eval import format_table, parse_plan, summarize


def _row(base: str, scope: str, repeat: int, steps: int = 5,
         concl: int = 39, tokens: int = 7000, reason: str = "model_finished") -> dict:
    return {"base_case": base, "scope": scope, "repeat": repeat,
            "status": "succeeded", "steps": steps, "conclusions": concl,
            "elapsed_ms": 15000, "tokens": {"total_tokens": tokens},
            "terminate_reason": reason, "scope_met": (True if scope == "guided" else None)}


def test_parse_plan():
    assert parse_plan("free:5,guided:1") == {"free": 5, "guided": 1}
    with pytest.raises(ValueError):
        parse_plan("bad:2")


def test_free_distribution_30_runs():
    probes = ["a", "b", "c", "d", "e", "f"]
    rows = [_row(p, "free", r) for p in probes for r in range(1, 6)]
    rows += [_row(p, "guided", 1, steps=2, concl=28, tokens=3200,
                  reason="budget_exhausted") for p in probes]
    s = summarize(rows)
    assert s["passed"] is True
    free = s["by_scope"]["free"]
    assert free["runs"] == 30 and free["cases"] == 6
    assert free["steps_freq"] == {"5": 30}
    assert s["by_scope"]["guided"]["avg_steps"] == 2.0
    assert all(c["steps_range"] == 0 for c in s["cases"] if c["scope"] == "free")


def test_free_with_mixed_steps_is_reported_not_hidden():
    probes = ["a", "b", "c", "d", "e", "f"]
    rows = []
    for p in probes:
        for r in range(1, 6):
            rows.append(_row(p, "free", r, steps=(3 if (p == "a" and r == 1) else 5)))
    s = summarize(rows)
    a = next(c for c in s["cases"] if c["base_case"] == "a" and c["scope"] == "free")
    assert a["steps_freq"] == {"3": 1, "5": 4}
    assert a["steps_range"] == 2


def test_format_table_has_rows():
    rows = [_row("a", "free", r) for r in range(1, 6)]
    rows += [_row("a", "guided", 1, steps=2, concl=28, tokens=3200)]
    out = format_table({"summary": summarize(rows)})
    assert "| a | free | 5 |" in out