# -*- coding: utf-8 -*-
"""运维面：任务列表、/metrics、保留期清理。

接缝：
- `GET /api/v1/analyses` 的分页、状态过滤、非法状态 400；
- `GET /metrics` 的计数/状态分布/队列水位；
- `MTSA_API_RETENTION_DAYS` 启动时清旧记录，且**不动产物目录**；
- 这三个都走鉴权。
"""
from __future__ import annotations

import time

from fastapi.testclient import TestClient

from mtsa.api import create_app
from mtsa.storage import RunStore

TOKEN = "test-token-ops"
H = {"X-API-Token": TOKEN}


def _seed(db: str, rows: list[dict]) -> None:
    store = RunStore(db)
    store.init()
    for r in rows:
        store.save(r)


def _rec(run_id: str, status: str = "succeeded", submitted: str = "2026-11-02T10:00:00+08:00",
         duration: float | None = 8.0, artifacts: int = 2) -> dict:
    return {
        "run_id": run_id, "question": "分析温度的趋势" * 20, "input_path": "fixtures/x.csv",
        "mode": "pipeline", "out_dir": "artifacts/api/%s" % run_id, "kwargs": {},
        "status": status, "submitted_at": submitted,
        "started_at": submitted, "finished_at": submitted,
        "duration_s": duration, "engine_run_id": None, "error": None,
        "artifacts": [{"name": "fig_%d.png" % i, "kind": "figure", "bytes": 100}
                      for i in range(artifacts)],
    }


def test_list_is_paged_newest_first_and_filterable(tmp_path):
    db = str(tmp_path / "state.sqlite3")
    _seed(db, [
        _rec("api_a", submitted="2026-11-02T10:00:00+08:00"),
        _rec("api_b", status="failed", submitted="2026-11-02T11:00:00+08:00"),
        _rec("api_c", submitted="2026-11-02T12:00:00+08:00"),
    ])
    app = create_app(TOKEN, workers=1, db_path=db)
    with TestClient(app) as c:
        all_rows = c.get("/api/v1/analyses", headers=H).json()
        assert all_rows["total"] == 3
        assert [i["run_id"] for i in all_rows["items"]] == ["api_c", "api_b", "api_a"]
        # 摘要把 question 截断到 80 字、artifacts 只给数量
        assert len(all_rows["items"][0]["question"]) == 80
        assert all_rows["items"][0]["artifacts"] == 2

        page = c.get("/api/v1/analyses?limit=1&offset=1", headers=H).json()
        assert [i["run_id"] for i in page["items"]] == ["api_b"]
        assert page["limit"] == 1 and page["offset"] == 1

        failed = c.get("/api/v1/analyses?status=failed", headers=H).json()
        assert failed["total"] == 1
        assert [i["run_id"] for i in failed["items"]] == ["api_b"]
        assert failed["items"][0]["error_code"] is None       # 这条记录没写 error


def test_list_rejects_unknown_status(tmp_path):
    app = create_app(TOKEN, workers=1, db_path=str(tmp_path / "s.sqlite3"))
    with TestClient(app) as c:
        r = c.get("/api/v1/analyses?status=whatever", headers=H)
        assert r.status_code == 400
        assert r.json()["detail"]["code"] == "BAD_STATUS_FILTER"


def test_metrics_reports_counts_and_queue(tmp_path):
    db = str(tmp_path / "state.sqlite3")
    _seed(db, [
        _rec("api_1", duration=8.0),
        _rec("api_2", status="failed", duration=3.0),
        _rec("api_3", duration=12.0),
    ])
    app = create_app(TOKEN, workers=2, queue_limit=5, db_path=db)
    with TestClient(app) as c:
        m = c.get("/metrics", headers=H).json()
        assert m["runs"]["enabled"] is True
        assert m["runs"]["total"] == 3
        assert m["runs"]["by_status"] == {"failed": 1, "succeeded": 2}
        assert m["runs"]["duration_s"]["count"] == 3
        assert m["runs"]["duration_s"]["max"] == 12.0
        assert m["queue"]["limit"] == 5 and m["queue"]["workers"] == 2
        assert m["queue"]["active"] == 0
        assert m["process"]["uptime_s"] >= 0


def test_metrics_works_with_store_off(tmp_path):
    app = create_app(TOKEN, workers=1, db_path="off")
    with TestClient(app) as c:
        m = c.get("/metrics", headers=H).json()
        assert m["runs"] == {"enabled": False}
        assert "queue" in m and "process" in m


def test_retention_purges_old_records_only(tmp_path):
    db = str(tmp_path / "state.sqlite3")
    _seed(db, [
        _rec("api_old", submitted="2020-01-01T00:00:00+08:00"),
        _rec("api_new", submitted="2026-11-02T10:00:00+08:00"),
    ])
    app = create_app(TOKEN, workers=1, db_path=db, retention_days=1)
    with TestClient(app) as c:
        doc = c.get("/api/v1/analyses", headers=H).json()
        assert [i["run_id"] for i in doc["items"]] == ["api_new"]
        # 记录删了，但产物目录不动（这里本来就不存在，只验证接口不 500）
        assert c.get("/api/v1/analyses/api_old", headers=H).status_code == 404


def test_retention_disabled_by_default(tmp_path):
    db = str(tmp_path / "state.sqlite3")
    _seed(db, [_rec("api_old", submitted="2020-01-01T00:00:00+08:00")])
    app = create_app(TOKEN, workers=1, db_path=db)
    with TestClient(app) as c:
        assert c.get("/api/v1/analyses", headers=H).json()["total"] == 1


def test_ops_endpoints_require_token(tmp_path):
    app = create_app(TOKEN, workers=1, db_path=str(tmp_path / "s.sqlite3"))
    with TestClient(app) as c:
        assert c.get("/api/v1/analyses").status_code == 401
        assert c.get("/metrics").status_code == 401
