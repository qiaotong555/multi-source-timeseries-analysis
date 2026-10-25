# -*- coding: utf-8 -*-
"""校验与「依校验结果改参重跑」。

边界（写死在这里，改动必须先改 spec）：
1. 自动修复只允许动图表 spec —— 不改阈值、不删结论、不碰任何指标；
2. 正常输入下 result.json 不出现 verify 块，事件摘要也不多键；
3. 修不动就停，不允许空转。
"""
from __future__ import annotations

import pytest

import mtsa.engine as engine
from mtsa import verify
from mtsa.models import AnalysisRequest, Conclusion, Evidence
from mtsa.scenarios.weather import WeatherAdapter
from tests.helpers import make_weather_frame


def _csv(tmp_path, rows=200):
    path = tmp_path / "wx.csv"
    make_weather_frame(n=rows, stations=1).to_csv(path, index=False)
    return str(path)


def _run(tmp_path, **kw):
    req = AnalysisRequest(request_id="vf", input_path=_csv(tmp_path), scene_id="weather",
                          question="分析温度的趋势与异常", seed=7,
                          output_dir=str(tmp_path / "out"), **kw)
    return engine.run_analysis(req, verbose=False)


def test_clean_run_has_no_verify_block(tmp_path):
    """正常输入：不出现 verify 块，verify 事件摘要也保持原样（逐位不变）。"""
    res = _run(tmp_path)
    assert res.status == "succeeded", res.error
    assert "verify" not in res.data
    node = [e for e in res.events if e.node == "verify"]
    assert node and node[0].summary == {"no_evidence_conclusions": 0}


def test_missing_figure_column_is_repaired_with_one_rerun(tmp_path, monkeypatch):
    original = WeatherAdapter.figure_specs

    def with_bad_spec(self, plan):
        specs = original(self, plan)
        specs.append({"id": "wx_bad", "kind": "series", "title": "坏图",
                      "columns": ["not_a_column"]})
        return specs

    monkeypatch.setattr(WeatherAdapter, "figure_specs", with_bad_spec)
    res = _run(tmp_path)

    assert res.status == "succeeded", res.error
    block = res.data["verify"]
    assert block["passed"] is True          # 修好了，不算失败
    assert block["reruns"] and block["reruns"][0]["dropped_specs"] == ["wx_bad"]
    assert [e.node for e in res.events].count("rerun_figures") == 1
    # 没有把异常列拿去渲染
    assert not any("wx_bad" in a.path for a in res.artifacts)


def test_unrepairable_figure_failure_stops_without_spinning(tmp_path, monkeypatch):
    original = WeatherAdapter.figure_specs

    def with_unknown_kind(self, plan):
        specs = original(self, plan)
        specs.append({"id": "wx_unknown", "kind": "no_such_kind",
                      "title": "不可渲染", "columns": ["temperature"]})
        return specs

    monkeypatch.setattr(WeatherAdapter, "figure_specs", with_unknown_kind)
    res = _run(tmp_path)

    assert res.status == "succeeded", res.error     # 图表问题不改变成败
    block = res.data["verify"]
    assert block["passed"] is False
    assert any(i["kind"] == "figure_not_rendered" for i in block["issues"])
    assert any("没有产出任何文件" in w for w in res.warnings)
    # 修不动：一次重跑都不该发生，避免空转
    assert not any(e.node == "rerun_figures" for e in res.events)


def test_conclusion_without_evidence_is_reported(tmp_path, monkeypatch):
    def bad_conclusions(self, ctx):
        return ([Conclusion(conclusion_id="c_bad", category="测试", text="没有证据的结论",
                            evidence_ids=[])], [])

    monkeypatch.setattr(WeatherAdapter, "conclusions", bad_conclusions)
    res = _run(tmp_path)
    assert res.status == "succeeded", res.error
    assert res.data["verify"]["passed"] is False
    assert any("未绑定证据" in w for w in res.warnings)


def test_strict_verify_surfaces_advisory_without_failing(tmp_path, monkeypatch):
    original = WeatherAdapter.conclusions

    def with_odd_number(self, ctx):
        cons, evs = original(self, ctx)
        if cons and evs:
            evs[0].metric_ids = ["some.metric.id"]   # 有引用才会被比对
            cons[0].text = cons[0].text + "（补充：99.99）"
        return cons, evs

    monkeypatch.setattr(WeatherAdapter, "conclusions", with_odd_number)
    res = _run(tmp_path, strict_verify=True)

    assert res.status == "succeeded", res.error
    block = res.data["verify"]
    assert block["passed"] is True                  # 抽查只提示，不改成败
    assert block["advisory"] >= 1
    kinds = [i["kind"] for i in block["issues"]]
    assert "conclusion_number_unmatched" in kinds


# ---------------- 纯函数 ----------------

def _ev(eid, metrics):
    return Evidence(evidence_id=eid, claim="c", metric_ids=metrics, artifact_ids=[],
                    code_ref="test")


def test_number_check_only_flags_unknown_decimals():
    cons = [Conclusion(conclusion_id="c1", category="x",
                       text="均值 21.44，差 3.7，共 12 个点", evidence_ids=["e1"])]
    evs = [_ev("e1", ["m1"])]
    pool = verify.numeric_pool([{"value": 21.44}])
    issues = verify.check_conclusion_numbers(cons, evs, pool)
    assert issues and issues[0].severity == "advisory"
    flagged = [s["number"] for s in issues[0].payload["samples"]]
    assert flagged == ["3.7"]        # 21.44 命中；12 是整数，不参与


def test_number_check_skips_conclusions_without_metric_ids():
    cons = [Conclusion(conclusion_id="c2", category="x", text="差值 99.9",
                       evidence_ids=["e2"])]
    issues = verify.check_conclusion_numbers(cons, [_ev("e2", [])], {1.0})
    assert issues == []


def test_repair_only_touches_columns():
    specs = [{"id": "a", "kind": "series", "title": "T", "columns": ["x", "gone"],
              "groupby": "month"}]
    fixed, info = verify.repair_figure_specs(specs, ["x"])
    assert fixed == [{"id": "a", "kind": "series", "title": "T", "columns": ["x"],
                      "groupby": "month"}]
    assert info["dropped_columns"] == {"a": ["gone"]}
