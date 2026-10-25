# -*- coding: utf-8 -*-
"""C2 的三项指标脚本：口径与输出结构不能悄悄变。"""
from __future__ import annotations

from evals import table_metrics


def test_metrics_have_the_three_rows_and_are_bounded():
    doc = table_metrics.collect()
    rate = doc["structured_output_valid_rate"]
    assert rate["runs"] > 0
    assert 0.0 <= rate["rate"] <= 1.0
    assert rate["valid"] <= rate["runs"]

    ev = doc["evidence_support_rate"]
    for value in [ev.get("mode_compare_pooled")] + list((ev.get("real_data_by_mode") or {}).values()):
        assert value is None or 0.0 <= value <= 1.0

    assert doc["cost"]["estimated"] is True          # 金额永远是估算
    assert doc["cost"]["total_cny"] > 0
    assert "不是账单" in doc["cost"]["caveat"]


def test_definitions_are_spelled_out():
    """口径写在输出里，避免下一个人重新解释一遍。"""
    doc = table_metrics.collect()
    assert set(doc["definitions"]) == {
        "structured_output_valid_rate", "evidence_support_rate", "cost_cny"}
