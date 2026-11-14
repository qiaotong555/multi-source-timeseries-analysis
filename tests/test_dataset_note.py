# -*- coding: utf-8 -*-
"""A1：`dataset_note`（数据说明）入参。

课程要求「输入数据」九项里有一项是数据说明。它不参与任何计算，只做两件事：
写进 `datasets[0]`、原样出现在报告第 1 节。因此这里锁四条：

1. 给了就落盘、没给就不出现（**不能给所有运行平白多一个键**）；
2. 自由文本（竖线、换行）不能把报告表格冲坏；
3. 复现脚本要带这个参数，且生成的脚本仍是合法 Python；
4. API 的 `AnalysisIn` 能收到这个字段并透传给 `AnalysisRequest`。
"""
from __future__ import annotations

import os
import py_compile

import mtsa.engine as engine
from mtsa.api import AnalysisIn
from mtsa.models import AnalysisRequest

FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")
WEATHER_CSV = os.path.join(FIX, "weather_v1.csv")


def _run(tmp_path, note=None):
    req = AnalysisRequest(request_id="a1", input_path=WEATHER_CSV, scene_id="weather",
                          question="分析温度的趋势", output_dir=str(tmp_path),
                          mode="pipeline", planner="rule", dataset_note=note)
    res = engine.run_analysis(req, verbose=False)
    assert res.status == "succeeded", res.error
    return res


def test_note_recorded_and_absent_by_default(tmp_path):
    res = _run(tmp_path / "plain")
    assert "dataset_note" not in res.datasets[0]


def test_note_round_trips(tmp_path):
    note = '来源：NOAA GHCN-Daily；注意 "湿度" 列有整段缺失'
    res = _run(tmp_path / "with_note", note)
    assert res.datasets[0]["dataset_note"] == note

    report = next(a.path for a in res.artifacts if a.kind == "report_md")
    md = open(report, encoding="utf-8").read()
    assert "数据说明" in md
    assert 'NOAA GHCN-Daily' in md


def test_note_with_pipe_and_newline_keeps_table_shape(tmp_path):
    res = _run(tmp_path / "messy", "第一行 | 带竖线\n第二行")
    report = next(a.path for a in res.artifacts if a.kind == "report_md")
    lines = open(report, encoding="utf-8").read().splitlines()
    row = next(l for l in lines if l.startswith("| 数据说明 |"))
    # 两列 → 结构上 3 个分隔符；正文里的竖线被转义成 \|（算第 4 个字符），换行被压平
    assert row.count("|") == 4 and "\\|" in row and "第二行" in row


def test_repro_script_carries_note(tmp_path):
    note = '含 "引号" 与反斜杠 C:\\data'
    res = _run(tmp_path / "repro", note)
    script = next(a.path for a in res.artifacts if a.kind == "script")
    src = open(script, encoding="utf-8").read()
    assert "--dataset-note" in src
    py_compile.compile(script, doraise=True)      # 转义没做对这里会失败


def test_api_accepts_note(tmp_path):
    body = AnalysisIn(input_path=WEATHER_CSV, question="分析温度的趋势",
                      dataset_note="来自数据提供方的说明")
    kwargs = body.model_dump(exclude={"input_path", "question", "mode"})
    req = AnalysisRequest(request_id="api", input_path=body.input_path,
                          question=body.question, **kwargs)
    assert req.dataset_note == "来自数据提供方的说明"
