# -*- coding: utf-8 -*-
"""稳定性评估的纯函数测试：不跑引擎、不调模型。"""
from __future__ import annotations

from evals.stability_eval import format_table, summarize


def _row(case: str, scope: str, repeat: int, steps: int, concl: int = 20,
         need: float | None = 1.0, reason: str = "model_finished",
         status: str = "succeeded") -> dict:
    return {"case": case, "scope": scope, "repeat": repeat, "status": status,
            "steps": steps, "terminate_reason": reason, "conclusions": concl,
            "need_coverage": need, "need_missing": [], "scope_met": True,
            "extra_used": 0, "wall_ms": 1000, "tokens": None}


def test_stable_steps_pass():
    rows = [_row("wx_trend", "guided", i, 2) for i in (1, 2, 3)]
    s = summarize(rows, "guided")
    assert s["passed"] is True
    assert s["cases"][0]["steps_range"] == 0


def test_range_of_one_still_passes():
    rows = [_row("wx_trend", "guided", 1, 2), _row("wx_trend", "guided", 2, 3)]
    s = summarize(rows, "guided")
    assert s["cases"][0]["steps_range"] == 1
    assert s["passed"] is True


def test_range_of_two_fails():
    rows = [_row("wx_stationary", "guided", 1, 3), _row("wx_stationary", "guided", 2, 5)]
    s = summarize(rows, "guided")
    assert s["cases"][0]["steps_range"] == 2
    assert s["passed"] is False
    assert s["checks"][0]["steps_range_ok"] is False


def test_scope_unmet_fails_even_if_steps_stable():
    rows = [_row("wx_anomaly", "guided", i, 2, reason="scope_unmet") for i in (1, 2)]
    s = summarize(rows, "guided")
    assert s["passed"] is False
    assert s["checks"][0]["scope_unmet_free"] is False


def test_missing_requirement_fails():
    rows = [_row("wx_normal", "guided", 1, 2, need=0.5)]
    s = summarize(rows, "guided")
    assert s["checks"][0]["need_coverage_ok"] is False
    assert s["passed"] is False


def test_probe_without_requirements_is_allowed():
    rows = [_row("wx_open", "guided", 1, 2, need=None)]
    s = summarize(rows, "guided")
    assert s["passed"] is True


def test_failed_run_fails_case():
    rows = [_row("wx_trend", "free", 1, 3, status="failed")]
    s = summarize(rows, "free")
    assert s["passed"] is False


def test_format_table_lists_every_case():
    rows = [_row("wx_trend", "guided", 1, 2), _row("wx_normal", "guided", 1, 2)]
    text = format_table({"scopes": [summarize(rows, "guided")]})
    assert "wx_trend" in text and "wx_normal" in text
    assert "通过" in text
