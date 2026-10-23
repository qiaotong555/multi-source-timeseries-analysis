# -*- coding: utf-8 -*-
"""T2.5 轨迹哈希与离线重放。

重点不是「能跑」，而是「重放不调模型且结果一致」：
- 哈希忽略耗时/时间戳，但覆盖动作、理由与摘要；
- 不一致时明确失败（REPLAY_MISMATCH），不静默放过。
"""
from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

import mtsa.engine as engine
from mtsa.agent_runtime import AgentRunner
from mtsa.models import AnalysisRequest
from mtsa.replay import ReplayError, diff_paths, load_replay, trajectory_hash

FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")
WEATHER_CSV = os.path.join(FIX, "weather_v1.csv")

STEPS = [
    {"action": "compute_metrics", "args": {}, "reason": "先看总体", "outcome": "ok",
     "summary": {"count": 38}},
    {"action": "detect_anomalies", "args": {}, "reason": "找异常", "outcome": "ok",
     "summary": {"count": 3}},
]


class FakeClient:
    model = "fake-model"

    def __init__(self, decisions):
        self._decisions = list(decisions)

    def chat_json(self, system, user):
        return self._decisions.pop(0)


def _fake_build(ctx, question, env=None, max_steps=8, allowed_tools=None, scope_mode="guided", max_extra_steps=1, max_tokens=0):
    decisions = [
        {"action": "compute_metrics", "args": {}, "reason": "先看总体"},
        {"action": "detect_anomalies", "args": {}, "reason": "找异常"},
        {"action": "finish", "args": {}, "reason": "完成"},
    ]
    return AgentRunner(ctx, FakeClient(decisions), question,
                       max_steps=max_steps, allowed_tools=allowed_tools,
                           scope_mode=scope_mode, max_extra_steps=max_extra_steps,
                           max_tokens=max_tokens)


@pytest.fixture(scope="module")
def agent_run(tmp_path_factory):
    if not os.path.exists(WEATHER_CSV):
        from fixtures.gen_fixtures import main as gen_main
        gen_main()
    out = str(tmp_path_factory.mktemp("replay_src"))
    req = AnalysisRequest(request_id="req_replay_src", input_path=WEATHER_CSV,
                          scene_id="weather", question="分析温度异常",
                          output_dir=out, mode="agent")
    with patch.object(engine, "build_agent_runner", _fake_build):
        res = engine.run_analysis(req, verbose=False)
    assert res.status == "succeeded", res.error
    return res, out


# ---------------- 纯函数 ----------------

def test_hash_ignores_timing_but_covers_decisions():
    base = {"model": "m", "allowed_tools": ["a"], "steps": STEPS}
    noisy = json.loads(json.dumps(base))
    noisy["steps"][0]["elapsed_ms"] = 999
    noisy["steps"][0]["started_at"] = "2026-01-01T00:00:00"
    assert trajectory_hash(base) == trajectory_hash(noisy)

    changed = json.loads(json.dumps(base))
    changed["steps"][0]["reason"] = "另一个理由"
    assert trajectory_hash(base) != trajectory_hash(changed)


def test_diff_paths_reports_first_mismatches():
    a = {"metrics": [1, 2], "plan": {"analyses": ["trend"]}}
    b = {"metrics": [1], "plan": {"analyses": ["trend", "anomaly"]}}
    diffs = diff_paths(a, b)
    assert any("metrics: 长度" in d for d in diffs)
    assert any("plan/analyses: 长度" in d for d in diffs)


def test_load_replay_rejects_non_agent_result(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"data": {"plan": {}}, "datasets": []}), encoding="utf-8")
    with pytest.raises(ReplayError, match="data.agent"):
        load_replay(str(p))


def test_load_replay_rejects_missing_file(tmp_path):
    with pytest.raises(ReplayError, match="不存在"):
        load_replay(str(tmp_path / "nope.json"))


# ---------------- 引擎集成 ----------------

def test_engine_replay_roundtrip_matches(agent_run, tmp_path):
    _, src_dir = agent_run
    src = os.path.join(src_dir, "result.json")
    out = str(tmp_path / "replay")
    req = AnalysisRequest(request_id="req_replay", input_path=WEATHER_CSV,
                          question="分析温度异常", output_dir=out,
                          mode="agent", replay_from=src)
    res = engine.run_analysis(req, verbose=False)
    assert res.status == "succeeded", res.error
    replay = res.data["agent"]["replay"]
    assert res.data["agent"]["scope"]["tools"] == ["compute_metrics", "detect_anomalies"]
    assert replay["trajectory_match"] is True
    assert replay["data_match"] is True
    assert replay["match"] is True
    assert replay["mismatches"] == []


def test_engine_replay_detects_tampered_source(agent_run, tmp_path):
    _, src_dir = agent_run
    src = json.loads(open(os.path.join(src_dir, "result.json"), encoding="utf-8").read())
    src["data"]["metrics"] = []                     # 人为删掉指标
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(src, ensure_ascii=False), encoding="utf-8")

    out = str(tmp_path / "replay_bad")
    req = AnalysisRequest(request_id="req_bad", input_path=WEATHER_CSV,
                          question="分析温度异常", output_dir=out,
                          mode="agent", replay_from=str(tampered))
    res = engine.run_analysis(req, verbose=False)
    assert res.status == "failed"
    assert res.error.code == "REPLAY_MISMATCH"
    assert res.error.details["mismatches"]


def test_cli_allows_replay_without_input_or_question():
    from mtsa.cli import _build_parser
    args = _build_parser().parse_args(["run", "--replay", "x.json"])
    assert args.replay == "x.json"
    assert args.input is None and args.question is None


def test_diff_treats_nan_and_null_as_same_missing_value():
    """落盘把 NaN 写成 null；重放内存里仍是 nan，二者语义相同，不能判不一致。"""
    assert diff_paths({"a": float("nan")}, {"a": None}) == []
    assert diff_paths({"a": None}, {"a": float("nan")}) == []
    assert diff_paths({"a": 1.0}, {"a": None})       # 真差异仍要报
