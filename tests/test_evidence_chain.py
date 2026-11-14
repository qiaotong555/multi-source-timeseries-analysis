# -*- coding: utf-8 -*-
"""A5：结论的每个数字都要能指回一个已落盘节点。

这条测试锁三件事：

1. 两个场景的结论都带 `data_refs`（周期/极值/异常点/变化点这类不在指标表里的量）；
2. 所有引用都能在 `result.data` 里解析到——引用写错就是「看起来有出处，其实指不到」；
3. 报告里印出「证据可定位率」，且含小数的结论覆盖率为 1.0。

不锁具体条数：结论条数会随结论通道增删而变，锁死数字只会让测试变成维护负担。
"""
from __future__ import annotations

import os

import pytest

import mtsa.engine as engine
import mtsa.verify as verify
from mtsa.models import AnalysisRequest

FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")
WEATHER_CSV = os.path.join(FIX, "weather_v1.csv")
STOCK_CSV = os.path.join(FIX, "stock_v1.csv")


def _run(scene, path, question, out):
    req = AnalysisRequest(request_id=f"a5_{scene}", input_path=path, scene_id=scene,
                          question=question, output_dir=str(out), mode="pipeline",
                          planner="rule", strict_verify=True)
    res = engine.run_analysis(req, verbose=False)
    assert res.status == "succeeded", res.error
    return res


@pytest.mark.parametrize("scene,path,question", [
    ("weather", WEATHER_CSV, "分析温度的趋势、季节性与极端事件"),
    ("stock", STOCK_CSV, "计算收益风险指标并做 walk-forward 回测"),
])
def test_all_data_refs_resolve(scene, path, question, tmp_path):
    res = _run(scene, path, question, tmp_path / scene)
    refs = sorted({r for e in res.evidence for r in (e.data_refs or [])})
    assert refs, "该场景应当有非指标类证据引用"
    broken = [r for r in refs if not verify.resolve_ref(res.data, r)[0]]
    assert broken == [], f"引用解析不到：{broken}"


@pytest.mark.parametrize("scene,path,question", [
    ("weather", WEATHER_CSV, "分析温度的趋势、季节性与极端事件"),
    ("stock", STOCK_CSV, "计算收益风险指标并做 walk-forward 回测"),
])
def test_numeric_conclusions_are_fully_locatable(scene, path, question, tmp_path):
    res = _run(scene, path, question, tmp_path / scene)
    cov = verify.evidence_coverage(res.conclusions, res.evidence)
    assert cov["numeric_conclusions"] > 0
    assert cov["numeric_locatable_rate"] == 1.0, cov


def test_report_prints_coverage(tmp_path):
    res = _run("weather", WEATHER_CSV, "分析温度的趋势、季节性与极端事件", tmp_path / "wx")
    report = next(a.path for a in res.artifacts if a.kind == "report_md")
    md = open(report, encoding="utf-8").read()
    assert "证据可定位率" in md
    assert "数据位置" in md
