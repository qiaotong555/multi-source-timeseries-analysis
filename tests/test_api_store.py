# -*- coding: utf-8 -*-
"""E3 任务仓储：状态跨进程重启存活，重启时把未完成任务标为中断。

接缝：
- `create_app(db_path=...)` 的持久化开关；
- 同一个库文件两次 `create_app`，第二次能查到第一次的 run；
- 重启前 `running` 的记录必须变成 `failed` + `INTERRUPTED`，不能假装还在跑；
- `db_path="off"` 时退回「重启即丢」，行为与 E3 之前一致。
"""
from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from mtsa.api import create_app
from mtsa.storage import RunStore

TOKEN = "test-token-e3"
H = {"X-API-Token": TOKEN}
QUESTION = "分析温度的趋势"
SHORT = "fixtures/weather_short_v1.csv"


def _wait(client: TestClient, run_id: str, timeout: float = 180.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        doc = client.get("/api/v1/analyses/%s" % run_id, headers=H).json()
        if doc["status"] not in ("queued", "running"):
            return doc
        time.sleep(0.3)
    raise AssertionError("等待运行结束超时：%s" % run_id)


def test_state_survives_restart(tmp_path):
    """第一次进程跑完的 run，第二个进程（同一库）仍能查到状态与产物清单。"""
    db = str(tmp_path / "state.sqlite3")

    app1 = create_app(TOKEN, workers=1, run_timeout=180, db_path=db)
    with TestClient(app1) as c1:
        r = c1.post("/api/v1/analyses", headers=H, json={
            "input_path": SHORT, "question": QUESTION, "mode": "pipeline"})
        assert r.status_code == 202, r.text
        run_id = r.json()["run_id"]
        first = _wait(c1, run_id)
        assert first["status"] == "succeeded", first
        names = [a["name"] for a in first["artifacts"]]
        assert names                      # 至少有一批图表产物
        assert first["result_available"] is True

    # 第二个进程：全新的 app、全新的内存 dict，只有库是同一个
    app2 = create_app(TOKEN, workers=1, run_timeout=180, db_path=db)
    with TestClient(app2) as c2:
        again = c2.get("/api/v1/analyses/%s" % run_id, headers=H)
        assert again.status_code == 200, again.text
        doc = again.json()
        assert doc["run_id"] == run_id
        assert doc["status"] == "succeeded"
        assert doc["result_available"] is True
        assert [a["name"] for a in doc["artifacts"]] == names
        # 产物路由重启后仍可下载
        res = c2.get("/api/v1/analyses/%s/result" % run_id, headers=H)
        assert res.status_code == 200
        assert json.loads(res.text)["request_id"] == run_id


def test_running_record_becomes_interrupted_on_restart(tmp_path):
    """进程被杀留下的 running 记录，下次启动必须标成 failed + INTERRUPTED。"""
    db = str(tmp_path / "state.sqlite3")
    store = RunStore(db)
    store.init()
    assert store.enabled
    store.save({
        "run_id": "api_deadbeef0001", "question": "分析温度的异常",
        "input_path": "fixtures/weather_short_v1.csv", "mode": "pipeline",
        "out_dir": str(tmp_path / "artifacts" / "api_deadbeef0001"),
        "kwargs": {}, "status": "running", "submitted_at": "2026-11-02T10:00:00",
        "started_at": "2026-11-02T10:00:01", "finished_at": None,
        "duration_s": None, "engine_run_id": None, "error": None, "artifacts": [],
    })

    app = create_app(TOKEN, workers=1, run_timeout=180, db_path=db)
    with TestClient(app) as c:
        doc = c.get("/api/v1/analyses/api_deadbeef0001", headers=H).json()
        assert doc["status"] == "failed"
        assert doc["error"]["code"] == "INTERRUPTED"
        assert doc["error"]["details"]["previous_status"] == "running"
        assert doc["finished_at"] is not None


def test_db_off_falls_back_to_memory(tmp_path):
    """db_path="off" 时不落库：第二个进程查不到，与 E3 之前的行为一致。"""
    app1 = create_app(TOKEN, workers=1, run_timeout=180, db_path="off")
    with TestClient(app1) as c1:
        r = c1.post("/api/v1/analyses", headers=H, json={
            "input_path": SHORT, "question": QUESTION, "mode": "pipeline"})
        run_id = r.json()["run_id"]
        _wait(c1, run_id)

    app2 = create_app(TOKEN, workers=1, run_timeout=180, db_path="off")
    with TestClient(app2) as c2:
        assert c2.get("/api/v1/analyses/%s" % run_id, headers=H).status_code == 404


def test_unknown_run_is_404(tmp_path):
    app = create_app(TOKEN, workers=1, run_timeout=180,
                     db_path=str(tmp_path / "state.sqlite3"))
    with TestClient(app) as c:
        r = c.get("/api/v1/analyses/api_does_not_exist", headers=H)
        assert r.status_code == 404
        assert r.json()["detail"]["code"] == "RUN_NOT_FOUND"


def test_store_survives_broken_db_path(tmp_path):
    """库路径不可写时服务仍能起——只是退回「重启即丢」，不能让整个 API 挂掉。"""
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")      # 父路径是文件，makedirs 必然失败
    app = create_app(TOKEN, workers=1, run_timeout=180,
                     db_path=str(blocker / "state.sqlite3"))
    with TestClient(app) as c:
        assert c.get("/healthz").json()["ok"] is True
