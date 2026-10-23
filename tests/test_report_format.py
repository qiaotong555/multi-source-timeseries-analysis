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
