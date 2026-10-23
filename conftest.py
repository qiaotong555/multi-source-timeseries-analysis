# -*- coding: utf-8 -*-
"""把 src/ 加入导入路径，并提供测试共用的 fixture 工厂。"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
for _p in (SRC, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tests.helpers import make_plan as _make_plan, make_stock_frame, make_weather_frame  # noqa: E402


@pytest.fixture
def make_plan():
    return _make_plan


@pytest.fixture
def weather_df():
    return make_weather_frame()


@pytest.fixture
def stock_df():
    return make_stock_frame()

@pytest.fixture(autouse=True)
def _offline_planner(monkeypatch):
    """测试默认走规则计划。

    开发者本机可能配置了 MTSA_LLM_*；若不隔离，pytest 会意外联网、变慢，
    且结果随模型输出波动。需要测 LLM 的用例都显式传入 env dict，不受此影响。
    """
    monkeypatch.setenv("MTSA_PLANNER", "rule")
