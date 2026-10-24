# -*- coding: utf-8 -*-
"""C3：质量问题与零检出都要进入结论，且不得把「未执行」说成「未发现」。

接缝：`engine.run_analysis` 的 `conclusions`（公开产物）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mtsa.engine as engine
from mtsa.agent_runtime import AgentRunner
from mtsa.models import AnalysisRequest


class FakeClient:
    model = "fake-model"

    def __init__(self, decisions):
        self._decisions = list(decisions)

    def chat_json(self, system: str, user: str) -> dict:
        return self._decisions.pop(0)


def _act(name):
    return {"action": name, "args": {}, "reason": "test"}


def _finish():
    return {"action": "finish", "args": {}, "reason": "done"}


def _patch_runner(monkeypatch, decisions):
    def fake(ctx, question, env=None, max_steps=8, allowed_tools=None,
             scope_mode="guided", max_extra_steps=1, max_tokens=0):
        return AgentRunner(ctx, FakeClient(decisions), question, max_steps=max_steps,
                           allowed_tools=allowed_tools, scope_mode=scope_mode,
                           max_extra_steps=max_extra_steps, max_tokens=max_tokens)
    monkeypatch.setattr(engine, "build_agent_runner", fake)


def _frame(n=900, seed=5, dirty=False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    temp = 15.0 + rng.normal(0, 0.5, n)
    humidity = 60.0 + rng.normal(0, 1.0, n)
    df = pd.DataFrame({"timestamp": idx, "station": "STA_A",
                       "temperature": temp, "humidity": humidity})
    if dirty:
        df.loc[df.index[100:130], "temperature"] = np.nan
        df = pd.concat([df, df.iloc[200:205]], ignore_index=True)
    return df


def _run(tmp_path, name, frame, mode="pipeline", **kw) -> dict:
    path = tmp_path / ("%s.csv" % name)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    out = tmp_path / name
    res = engine.run_analysis(AnalysisRequest(
        request_id=name, input_path=str(path), scene_id="weather",
        question="检查数据质量并分析异常", mode=mode, seed=20260927,
        output_dir=str(out), **kw), verbose=False)
    assert res.status == "succeeded", res.error
    return json.loads((out / "result.json").read_text(encoding="utf-8"))


def _cat(doc, category):
    return [c for c in doc["conclusions"] if c["category"] == category]


def test_quality_issues_become_conclusions(tmp_path):
    doc = _run(tmp_path, "dirty", _frame(dirty=True))
    texts = " ".join(c["text"] for c in _cat(doc, "数据口径"))
    assert "MISSING_VALUES" in texts
    assert "DUPLICATE_TIMESTAMPS" in texts
    ids = {e["evidence_id"] for e in doc["evidence"]}
    for c in _cat(doc, "数据口径"):
        assert c["evidence_ids"] and set(c["evidence_ids"]) <= ids


def test_clean_data_gets_explicit_no_quality_issue_conclusion(tmp_path):
    doc = _run(tmp_path, "clean", _frame())
    texts = " ".join(c["text"] for c in _cat(doc, "数据口径"))
    assert "未发现数据质量问题" in texts


def test_zero_detection_is_stated_in_pipeline(tmp_path):
    doc = _run(tmp_path, "zero", _frame())
    a = " ".join(c["text"] for c in _cat(doc, "异常检测"))
    cp = " ".join(c["text"] for c in _cat(doc, "变化点"))
    assert "未发现异常" in a
    assert "未发现均值突变" in cp
    assert doc["data"]["anomalies"] == []
    assert doc["data"]["change_points"] == []


def test_agent_without_detector_does_not_claim_none(monkeypatch, tmp_path):
    _patch_runner(monkeypatch, [_act("compute_metrics"), _finish()])
    doc = _run(tmp_path, "agent_skip", _frame(), mode="agent", agent_scope="free")
    assert "detect_anomalies" not in doc["data"]["agent"]["tools_run"]
    assert _cat(doc, "异常检测") == []
    assert _cat(doc, "变化点") == []
    # 质量问题与工具无关，仍应给出
    assert _cat(doc, "数据口径")