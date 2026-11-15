# -*- coding: utf-8 -*-
"""D1：`reproduce.py` 必须带 `--planner` 与 `--mode`。

这条不是格式洁癖。不带 `--planner` 时，服务端按 `MTSA_PLANNER`（缺省 auto）重算计划：
配了模型就换成模型选的项——实测同一份数据同一句问题，**12 项变 4 项**，结论从 51 条变 12 条。
「复现脚本」复现出另一份报告，比没有脚本更糟：它看起来是可核对的。

手工验证过一次端到端（2026-11-15）：按脚本重跑，结论 51/51 逐字相同、指标 350+79、
分析项 12、产物 21。这里锁住「参数确实写进去了」，端到端留给发版的手工检查。
"""
from __future__ import annotations

import os
import py_compile

import mtsa.engine as engine
from mtsa.models import AnalysisRequest

FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")


def _run(tmp_path, *, planner="rule", scene="weather", question="分析温度的趋势与季节性"):
    req = AnalysisRequest(request_id="d1", input_path=os.path.join(FIX, "weather_v1.csv"),
                          scene_id=scene, question=question, output_dir=str(tmp_path),
                          mode="pipeline", planner=planner)
    res = engine.run_analysis(req, verbose=False)
    assert res.status == "succeeded", res.error
    return res


def _script(res):
    path = next(a.path for a in res.artifacts if a.kind == "script")
    return path, open(path, encoding="utf-8").read()


def test_rule_plan_writes_planner_and_mode(tmp_path):
    res = _run(tmp_path)
    assert res.data["plan"]["plan_source"] == "rule"
    path, src = _script(res)
    assert '"--planner", "rule"' in src
    assert '"--mode", "pipeline"' in src
    py_compile.compile(path, doraise=True)


def test_planner_matches_actual_plan_source(tmp_path):
    """写进去的计划器必须与这次实际用的计划一致，不能写个默认值糊弄。"""
    res = _run(tmp_path, planner="rule")
    _, src = _script(res)
    assert f'"--planner", "{res.data["plan"]["plan_source"]}"' in src
