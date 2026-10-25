# -*- coding: utf-8 -*-
"""HTTP 薄壳（E1）：契约、鉴权、排队上限与"失败也要有错误码"。

薄壳的价值全在边界上，所以每条断言都对准一条边界，而不是重复测引擎。
单个分析走子进程，约 8 s；共享一次成功运行，避免每个用例都付这个成本。
"""
from __future__ import annotations

import os
import time

import pytest
from fastapi.testclient import TestClient

from mtsa.api import OUT_ROOT, create_app

TOKEN = "test-token"
H = {"X-API-Token": TOKEN}
SHORT = "fixtures/weather_short_v1.csv"
QUESTION = "分析温度的趋势、季节性和异常点，并检测变化点"


def _wait(client: TestClient, run_id: str, limit_s: float = 180) -> dict:
    deadline = time.time() + limit_s
    while time.time() < deadline:
        doc = client.get("/api/v1/analyses/%s" % run_id, headers=H).json()
        if doc["status"] not in ("queued", "running"):
            return doc
        time.sleep(0.3)
    raise AssertionError("等待运行结束超时：%s" % run_id)


@pytest.fixture(scope="module")
def client():
    app = create_app(TOKEN, workers=2, run_timeout=180)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def done_run(client):
    r = client.post("/api/v1/analyses", headers=H, json={
        "input_path": SHORT, "question": QUESTION, "mode": "pipeline"})
    assert r.status_code == 202, r.text
    run_id = r.json()["run_id"]
    return run_id, _wait(client, run_id)


def test_healthz_is_open_but_carries_nothing(client):
    doc = client.get("/healthz").json()
    assert doc["ok"] is True
    assert set(doc) == {"ok", "version", "active"}      # 不泄露运行内容


def test_missing_or_wrong_token_is_401(client):
    body = {"input_path": SHORT, "question": QUESTION}
    assert client.post("/api/v1/analyses", json=body).status_code == 401
    assert client.post("/api/v1/analyses", json=body,
                       headers={"X-API-Token": "wrong"}).status_code == 401
    assert client.get("/api/v1/analyses/whatever").status_code == 401


def test_missing_token_refuses_to_start():
    """未配置 token 时不能"先跑起来再说"——startup 必须失败。"""
    app = create_app("")
    with pytest.raises(RuntimeError, match="MTSA_API_TOKEN"):
        with TestClient(app):
            pass


def test_submit_poll_result_contract(client, done_run):
    run_id, doc = done_run
    assert doc["run_id"] == run_id
    assert doc["status"] == "succeeded"
    assert doc["error"] is None
    assert doc["duration_s"] and doc["duration_s"] > 0
    assert doc["result_available"] is True
    # result.json 不是 Artifact（由引擎直接写盘），它走 /result 单独取
    assert {"report.md", "clean_data.csv"} <= {a["name"] for a in doc["artifacts"]}

    r = client.get("/api/v1/analyses/%s/result" % run_id, headers=H)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    # 原样是 result.json 的字节：状态、结论、产物三个字段必须与 CLI 完全一致
    assert body["status"] == "succeeded"
    assert body["conclusions"] and body["artifacts"]
    # 两个 id：API 句柄在 request_id；引擎自己的 run_id 是另一个
    assert body["request_id"] == run_id
    assert body["run_id"] == doc["engine_run_id"] != run_id


def test_artifact_download_is_whitelisted(client, done_run):
    run_id, _ = done_run
    r = client.get("/api/v1/analyses/%s/artifacts/report.md" % run_id, headers=H)
    assert r.status_code == 200 and "# " in r.text
    assert client.get("/api/v1/analyses/%s/artifacts/nope.md" % run_id,
                      headers=H).status_code == 404
    # 路径穿越：basename 化之后不在白名单里，直接 404
    assert client.get("/api/v1/analyses/%s/artifacts/..%%2F..%%2Fresult.json" % run_id,
                      headers=H).status_code in (404, 405)


def test_unknown_run_id_is_404(client):
    r = client.get("/api/v1/analyses/api_does_not_exist", headers=H)
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "RUN_NOT_FOUND"


def test_result_before_finish_is_409(client):
    r = client.post("/api/v1/analyses", headers=H, json={
        "input_path": SHORT, "question": QUESTION, "mode": "pipeline"})
    run_id = r.json()["run_id"]
    early = client.get("/api/v1/analyses/%s/result" % run_id, headers=H)
    assert early.status_code == 409
    assert early.json()["detail"]["code"] == "RESULT_NOT_READY"
    _wait(client, run_id)


def test_missing_input_is_400_with_code(client):
    r = client.post("/api/v1/analyses", headers=H, json={
        "input_path": "fixtures/does_not_exist.csv", "question": QUESTION})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "INPUT_NOT_FOUND"


def test_failed_run_still_carries_error_code(client, tmp_path):
    bad = tmp_path / "no_time.csv"
    bad.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
    r = client.post("/api/v1/analyses", headers=H,
                    json={"input_path": str(bad), "question": "分析一下"})
    assert r.status_code == 202
    doc = _wait(client, r.json()["run_id"])
    assert doc["status"] == "failed"
    assert doc["error"]["code"] == "TIME_AXIS_NOT_FOUND"
    assert doc["error"]["last_node"] == "profile_dataset"   # 停在哪一步，从 events 读
    assert doc["error"]["details"]["hint"]                   # 失败要带可操作的提示
    # 失败运行同样落 result.json，可下载（留档，不掩盖）
    assert doc["result_available"] is True


def test_preview_writes_nothing(client, tmp_path):
    r = client.post("/api/v1/preview", headers=H,
                    json={"input_path": SHORT, "question": QUESTION})
    assert r.status_code == 200
    body = r.json()
    assert body["plan"]["plan_source"] and body["tools"]
    # 预演不建产物目录：目录名就是 run_id 前缀的 api_*，这里不该出现新的
    assert not os.path.exists(os.path.join(OUT_ROOT, "api_preview"))


def test_upload_sanitizes_filename(client, tmp_path):
    """`../../evil.csv` 只能落到 uploads/<run_id>/evil.csv，不能跳目录。"""
    r = client.post("/api/v1/analyses/upload", headers=H,
                    files={"file": ("../../evil.csv", b"a,b\n1,2\n", "text/csv")},
                    data={"question": "分析一下"})
    assert r.status_code == 202, r.text
    run_id = r.json()["run_id"]
    assert os.path.exists(os.path.join(OUT_ROOT, "uploads", run_id, "evil.csv"))
    assert not os.path.exists(os.path.join(OUT_ROOT, "evil.csv"))
    _wait(client, run_id)


def test_upload_rejects_bad_filename(client):
    r = client.post("/api/v1/analyses/upload", headers=H,
                    files={"file": ("..", b"x", "text/csv")}, data={"question": "q"})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "BAD_FILENAME"


def test_queue_limit_is_enforced():
    """排队上限不是"建议"：满了就 429，不无限收。"""
    app = create_app(TOKEN, workers=1, queue_limit=1, run_timeout=180)
    with TestClient(app) as c:
        first = c.post("/api/v1/analyses", headers=H, json={
            "input_path": SHORT, "question": QUESTION, "mode": "pipeline"})
        assert first.status_code == 202
        second = c.post("/api/v1/analyses", headers=H, json={
            "input_path": SHORT, "question": QUESTION, "mode": "pipeline"})
        assert second.status_code == 429
        assert second.json()["detail"]["code"] == "QUEUE_FULL"
        assert second.headers.get("retry-after") == "30"
        _wait(c, first.json()["run_id"])

def test_planner_can_be_chosen_per_request(client):
    """服务端配了模型也要能按请求拿规则计划——API 输出要能与已验收口径对齐。"""
    r = client.post("/api/v1/preview", headers=H, json={
        "input_path": SHORT, "question": QUESTION, "planner": "rule"})
    assert r.status_code == 200, r.text
    plan = r.json()["plan"]
    assert plan["plan_source"] == "rule"
    assert plan["planner_requested"] == "rule"
    assert len(plan["analyses"]) >= 10        # 规则计划 12 项，模型那种收窄不会出现


def test_planner_llm_without_model_falls_back_and_says_so(client, monkeypatch):
    monkeypatch.delenv("MTSA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("MTSA_LLM_API_KEY", raising=False)
    r = client.post("/api/v1/preview", headers=H, json={
        "input_path": SHORT, "question": QUESTION, "planner": "llm"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["plan"]["plan_source"] == "rule"      # 回退
    assert body["plan"]["planner_requested"] == "llm"  # 但请求意图要留痕
    assert any("回退" in w for w in body["warnings"])  # 且必须说出来
