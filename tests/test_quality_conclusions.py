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


def test_agent_report_attributes_extreme_to_engine(monkeypatch, tmp_path):
    """覆盖矩阵要把「引擎做的」单列一行，不能记在 compute_metrics 名下（#29 遗留）。"""
    _patch_runner(monkeypatch, [_act("compute_metrics"), _finish()])
    doc = _run(tmp_path, "agent_cov", _frame(), mode="agent", agent_scope="free")
    cov = doc["data"]["agent"]["coverage"]
    assert cov["engine_covers"] == ["extreme"]
    assert "extreme" in cov["covered_analyses"]

    report = (tmp_path / "agent_cov" / "report.md").read_text(encoding="utf-8")
    assert "（引擎）" in report
    assert "极值由引擎统一计算" in report


# ---------------- P1：否定句必须带前提 ----------------

def test_zero_detection_reports_actual_scope_and_window(tmp_path):
    """「已对 N 条序列执行检测」的 N 是**实检列数**，窗口是 `anomaly_window`。

    回归：旧实现写 `len(plan.target_columns)`（计划列数）与 `rolling_window`
    （指标模块的窗口，天气场景默认 24），而检测器实际按 `max(12, n // 50)` 取窗。
    900 点 → 18 点，报 24 是错的。
    """
    doc = _run(tmp_path, "scope", _frame())
    a = " ".join(c["text"] for c in _cat(doc, "异常检测"))
    assert "已对 2/2 条目标序列执行检测" in a
    assert "滚动窗口 18 点" in a
    assert "滚动窗口 24 点" not in a

    report = (tmp_path / "scope" / "report.md").read_text(encoding="utf-8")
    assert "异常点：未检出（已对 2/2 条目标序列执行" in report
    assert "滚动窗口 18 点" in report


def test_quality_negative_states_scope_and_blind_spots(tmp_path):
    """「未检出数据质量问题」要写明查了什么、以及查不到什么。"""
    doc = _run(tmp_path, "clean_scope", _frame())
    texts = " ".join(c["text"] for c in _cat(doc, "数据口径"))
    assert "已检查 2 个数值列" in texts

    report = (tmp_path / "clean_scope" / "report.md").read_text(encoding="utf-8")
    assert "未检出数据质量问题（已检查 2 个数值列" in report
    assert "有效点少于 60 的序列不做水平跳变" in report


def test_examined_series_skips_missing_and_short_columns():
    """实检列数必须排除「列不存在」与「点太少」两种情况。"""
    from mtsa.anomaly import EXTREME_MIN_POINTS, MIN_ANOMALY_POINTS, examined_series
    frame = pd.DataFrame({"long": np.arange(200.0),
                          "short": [1.0, 2.0, 3.0, 4.0, 5.0] + [np.nan] * 195,
                          "empty": [np.nan] * 200})
    assert examined_series(frame, ["long", "short", "empty", "absent"],
                           MIN_ANOMALY_POINTS) == 1
    assert examined_series(frame, ["long", "short"], 3) == 2
    assert EXTREME_MIN_POINTS > 0


def test_extreme_examined_counts_only_observed_points():
    """极值的「已检查列数」按观测点算：全被插补的列不算检查过。"""
    from mtsa.anomaly import EXTREME_MIN_POINTS, extreme_examined
    from mtsa.models import AnalysisPlan, CleanDataset

    n = EXTREME_MIN_POINTS + 20
    idx = pd.date_range("2024-01-01", periods=n, freq="h").astype(str)
    frame = pd.DataFrame({"obs": np.arange(float(n)),
                          "filled": np.arange(float(n))}, index=idx)
    clean = CleanDataset(frame=frame, imputed_index=list(idx),
                         imputed_columns={"filled": list(idx)})
    plan = AnalysisPlan(scene_id="weather", scene_confidence=1.0, question="t",
                        target_columns=["obs", "filled", "absent"], analyses=["anomaly"])
    assert extreme_examined(clean, plan) == (1, 3)

def test_quality_conclusion_does_not_repeat_group(tmp_path):
    """分组只出现一次：`group` 已是结构化字段，`detail` 里不再内嵌。

    旧文案是「列 timestamp，分组 STA_A 分组 STA_A 内重复 5 行」——同一句里说了两遍，
    报告第 2 节的表还另有「分组」列，于是三处重复。
    必须用两站数据：单站会被当成常量列，走的是无分组分支。
    """
    rng = np.random.default_rng(11)
    idx = pd.date_range("2024-01-01", periods=400, freq="h")
    parts = []
    for st in ("STA_A", "STA_B"):
        parts.append(pd.DataFrame({"timestamp": idx, "station": st,
                                   "temperature": 15.0 + rng.normal(0, 0.5, 400),
                                   "humidity": 60.0 + rng.normal(0, 1.0, 400)}))
    df = pd.concat(parts, ignore_index=True)
    df = pd.concat([df, df[df["station"] == "STA_A"].iloc[10:13]], ignore_index=True)

    doc = _run(tmp_path, "dup_group", df)
    dup = [c["text"] for c in _cat(doc, "数据口径") if "DUPLICATE_TIMESTAMPS" in c["text"]]
    assert dup, [c["text"] for c in _cat(doc, "数据口径")]
    assert dup[0].count("分组 STA_A") == 1, dup[0]
    assert "同一时刻重复" in dup[0], dup[0]


def test_changepoint_report_states_time_resolution(tmp_path):
    """变化点时间戳必须带定位分辨率，否则用户会当成突变发生的准确时刻。

    实测：注入 2/15 00:00 开始的台阶，检出时间戳是 2/14 15:00（偏 9 小时），
    而报告当时只给一个精确到小时的时刻，没有说窗口有多宽。
    """
    rng = np.random.default_rng(3)
    n = 900
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    temp = 15.0 + rng.normal(0, 0.4, n)
    temp[450:] += 5.0
    df = pd.DataFrame({"timestamp": idx, "station": "STA_A", "temperature": temp})

    doc = _run(tmp_path, "cp_window", df)
    cps = doc["data"]["change_points"]
    assert cps, "注入 5 度台阶后没有变化点"
    assert all(c.get("window") for c in cps), cps

    md = (tmp_path / "cp_window" / "report.md").read_text(encoding="utf-8")
    assert "时间戳不是精确时刻" in md
    assert "定位不确定度" in md
