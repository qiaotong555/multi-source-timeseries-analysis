# -*- coding: utf-8 -*-
"""报告数值格式化：NaN/inf 必须落成占位符，不能让收尾层崩溃。"""
from __future__ import annotations

from mtsa.report import _fmt


def test_nan_formats_as_placeholder():
    assert _fmt(float("nan")) == "—"


def test_inf_formats_as_placeholder():
    assert _fmt(float("inf")) == "—"
    assert _fmt(float("-inf")) == "—"


def test_normal_float_still_formats():
    assert _fmt(3.0) == "3"
    assert _fmt(0.5) == "0.5"
    assert _fmt(None) == "—"


def test_md_table_escapes_pipe_inside_cells():
    """单元格里的竖线必须转义，否则整行多出一列。

    实测：数据质量表的「说明」写成 `稳健 |Z| > 5.0` 时，7 列的表被冲成 9 列，
    markdown 渲染直接错位。
    """
    from mtsa.report import _md_table

    md = _md_table(["列", "说明"], [["降水量", "稳健 |Z| > 5.0 的候选"]])
    lines = md.split("\n")
    assert len(lines) == 3
    for line in lines:
        # 去掉转义后的 \| 再数，应恰好等于列数 + 1
        assert line.replace("\\|", "").count("|") == 3, line
    assert "\\|Z\\|" in md
