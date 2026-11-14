# -*- coding: utf-8 -*-
"""A2：七个状态全部可达——取消与「等输入」。

这一组测的是状态机，不是引擎。每条用例对准一条边界：

1. 排队中取消 → 立即 `cancelled`（没有进程要杀）；
2. 运行中取消 → 短暂 `cancel_requested`，随后 `cancelled`，且**没有结果文件**；
3. 取消是幂等的；已结束的任务不能取消（409）；
4. 缺字段的失败进 `waiting_for_input` 并给出 `input_required`；补参后能跑成功；
5. 不是 `waiting_for_input` 的任务不能 resume。
"""
from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

from mtsa.api import create_app

TOKEN = "test-token"
H = {"X-API-Token": TOKEN}
SHORT = "fixtures/weather_short_v1.csv"
NEUTRAL = "fixtures/neutral_no_scene_v1.csv"
QUESTION = "分析温度的趋势、季节性和异常点"


def _get(client, run_id):
    return client.get("/api/v1/analyses/%s" % run_id, headers=H).json()


def _wait(client, run_id, limit_s=180, until=None):
    terminal = until or ("succeeded", "failed", "cancelled", "waiting_for_input")
    deadline = time.time() + limit_s
    while time.time() < deadline:
        doc = _get(client, run_id)
        if doc["status"] in terminal:
            return doc
        time.sleep(0.1)
    raise AssertionError("等待超时：%s 停在 %s" % (run_id, _get(client, run_id)["status"]))


def _wait_status(client, run_id, want, limit_s=60):
    deadline = time.time() + limit_s
    while time.time() < deadline:
        if _get(client, run_id)["status"] == want:
            return
        time.sleep(0.05)
    raise AssertionError("等不到 %s：当前 %s" % (want, _get(client, run_id)["status"]))


@pytest.fixture(scope="module")
def client():
    app = create_app(TOKEN, workers=1, run_timeout=180, db_path="off")
    with TestClient(app) as c:
        yield c


def _submit(client, **over):
    body = {"input_path": SHORT, "question": QUESTION, "mode": "pipeline"}
    body.update(over)
    r = client.post("/api/v1/analyses", headers=H, json=body)
    assert r.status_code == 202, r.text
    return r.json()


def test_cancel_while_queued_is_immediate(client):
    """workers=1：先占住唯一槽位，第二条必然排队，取消应该是即时的。"""
    first = _submit(client)
    _wait_status(client, first["run_id"], "running")
    second = _submit(client)
    assert _get(client, second["run_id"])["status"] == "queued"

    r = client.post("/api/v1/analyses/%s/cancel" % second["run_id"], headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "cancelled"
    assert r.json()["duration_s"] == 0.0

    # 幂等：再取消一次仍是 200 + cancelled
    again = client.post("/api/v1/analyses/%s/cancel" % second["run_id"], headers=H)
    assert again.status_code == 200 and again.json()["status"] == "cancelled"
    # 没有结果文件可拿
    res = client.get("/api/v1/analyses/%s/result" % second["run_id"], headers=H)
    assert res.status_code == 409 and res.json()["detail"]["code"] == "RUN_CANCELLED"

    _wait(client, first["run_id"])


def test_cancel_while_running_reaches_cancelled(client):
    """运行中取消：可能先看到 cancel_requested，终态必须是 cancelled 且无结果。"""
    rec = _submit(client)
    run_id = rec["run_id"]
    _wait_status(client, run_id, "running")

    r = client.post("/api/v1/analyses/%s/cancel" % run_id, headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["status"] in ("cancel_requested", "cancelled")

    doc = _wait(client, run_id, until=("cancelled", "succeeded", "failed"))
    assert doc["status"] == "cancelled", doc
    assert doc["error"]["code"] == "CANCELLED"
    assert doc["result_available"] is False
    assert not os.path.exists(os.path.join(doc["out_dir"], "result.json"))


def test_cancel_rejected_after_finish(client):
    rec = _submit(client)
    doc = _wait(client, rec["run_id"])
    assert doc["status"] == "succeeded"
    r = client.post("/api/v1/analyses/%s/cancel" % rec["run_id"], headers=H)
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "CANCEL_NOT_ALLOWED"


def test_missing_input_goes_to_waiting_then_resumes(client):
    """场景识别不出 → waiting_for_input（不是 failed），补上 scene_id 后能跑成功。"""
    rec = _submit(client, input_path=NEUTRAL)
    doc = _wait(client, rec["run_id"])
    assert doc["status"] == "waiting_for_input", doc
    assert doc["input_required"] == ["scene_id"]
    assert doc["resume_url"].endswith("/resume")
    assert doc["error"]["code"] == "SCENE_NOT_DETECTED"
    # 等输入的任务不该占并发槽：此时再提交一条应当能立刻开跑
    other = _submit(client)
    _wait_status(client, other["run_id"], "running")
    _wait(client, other["run_id"])

    # 空 body 补不上字段 → 400，且仍停在 waiting_for_input
    bad = client.post("/api/v1/analyses/%s/resume" % rec["run_id"], headers=H, json={})
    assert bad.status_code == 400
    assert bad.json()["detail"]["code"] == "INPUT_STILL_MISSING"

    ok = client.post("/api/v1/analyses/%s/resume" % rec["run_id"], headers=H,
                     json={"scene_id": "weather"})
    assert ok.status_code == 202, ok.text
    assert ok.json()["status"] == "queued"
    final = _wait(client, rec["run_id"])
    assert final["status"] == "succeeded", final
    assert final["input_required"] == []


def test_resume_rejected_for_non_waiting_run(client):
    rec = _submit(client)
    doc = _wait(client, rec["run_id"])
    assert doc["status"] == "succeeded"
    r = client.post("/api/v1/analyses/%s/resume" % rec["run_id"], headers=H,
                    json={"scene_id": "weather"})
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "RESUME_NOT_ALLOWED"
