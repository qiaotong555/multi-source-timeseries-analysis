# -*- coding: utf-8 -*-
"""HTTP 薄壳（E1）：把 `run_analysis` 包成 5 个路由，不做业务逻辑。

设计边界写死在这里，避免以后被"顺手加个功能"撑成第二个内核：

- **不判场景、不映射参数、不改 `result.json` 契约**：请求 → `AnalysisRequest` → 现有引擎；
- **不做持久化**：运行状态在进程内 dict，重启即丢（产物仍落在 `artifacts/api/<run_id>/`，
  重启后手工取）；重启后 `GET` 会返回 404，这是设计不是 bug；
- **不做取消、不做流式、不做多租户、不做用户体系**；
- **鉴权只有单 token**（`X-API-Token`）；未配置 `MTSA_API_TOKEN` 时**拒绝启动**——
  默认无鉴权的服务等于没门；
- **每个请求都走 `run_with_timeout`**：子进程隔离 + 硬超时，跑飞的分析杀得掉，
  代价是每次多约 3 s 的进程启动（见 `doc/改进记录.md` #8）；
- **请求体不接受 `output_dir` / `replay_from` / `price_table`**：这三个都是服务端路径，
  交给客户端等于开放任意读写。

两个 id，别混
-----------
API 句柄是 `run_id`（形如 `api_1a2b3c4d`，本层生成，用来查状态）；
引擎自己还会生成一个 `run_id`（形如 `run_20261025_1a2b3c4d`），写在 `result.json` 里。
`result.json` 的 `request_id` 就是 API 句柄，`run_id` 是引擎那个——状态响应里两个都给
（`run_id` / `engine_run_id`），免得对不上账。

为什么状态只有 4 个
------------------
`models.Status` 里有 7 个词，其中 `waiting_for_input` / `cancel_requested` / `cancelled`
在本项目里**没有实现语义**（只出现在 Literal 定义里）。薄壳只写
`queued → running → succeeded | failed`，不假装支持另外三个。

启动
----
    MTSA_API_TOKEN=dev mtsa-api --host 127.0.0.1 --port 8000
    # 或
    MTSA_API_TOKEN=dev uvicorn mtsa.api:app --port 8000

环境变量
--------
| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `MTSA_API_TOKEN` | 无 | 必填；缺失时启动失败 |
| `MTSA_API_WORKERS` | 2 | 并发执行的分析数（线程 + 子进程） |
| `MTSA_API_QUEUE` | 32 | 排队上限，超过直接 429，不无限收 |
| `MTSA_API_RUN_TIMEOUT` | 300 | 单次分析的硬超时秒数 |
| `MTSA_API_MAX_UPLOAD_MB` | 50 | 上传文件大小上限 |
| `MTSA_API_OUT` | `./artifacts/api` | 产物根目录（相对当前工作目录） |
"""
from __future__ import annotations

import argparse
import os
import secrets
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

from fastapi import (Depends, FastAPI, File, Form, Header, HTTPException, Request,
                     UploadFile, status as http_status)
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .engine import run_analysis
from .models import DEFAULT_SEED, AnalysisRequest
from .scenarios import stock as _stock        # noqa: F401  注册适配器
from .scenarios import weather as _weather    # noqa: F401
from .timeout_guard import run_with_timeout

# 产物目录按**当前工作目录**解析，与 CLI 的 `artifacts/<run_id>` 同一规则。
# 不能用包的安装位置：pip 装进 site-packages 后，那里是 /usr/local/lib/python3.13，
# 容器里既不是数据目录，也可能不可写（镜像的 WORKDIR 是 /data）。
OUT_ROOT = os.path.abspath(os.environ.get("MTSA_API_OUT",
                                          os.path.join("artifacts", "api")))

Mode = Literal["pipeline", "agent", "auto"]
SceneId = Literal["weather", "stock"]
ScopeMode = Literal["guided", "free"]

ACTIVE = ("queued", "running")


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("mtsa-agent")
    except Exception:                                   # noqa: BLE001  未安装时退回
        return "0.0.0"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------- 请求体

class AnalysisIn(BaseModel):
    """提交一次分析。字段是 `AnalysisRequest` 的子集，理由见模块 docstring。"""

    input_path: str = Field(..., description="服务端可读的文件路径")
    question: str = Field(..., min_length=1)
    scene_id: Optional[SceneId] = None
    time_column: Optional[str] = None
    value_columns: Optional[list[str]] = None
    group_column: Optional[str] = None
    mode: Mode = "pipeline"
    # 请求级计划器；None = 跟随服务端 MTSA_PLANNER。见 models.AnalysisRequest.planner
    planner: Optional[Literal["auto", "rule", "llm"]] = Field(
        default=None, description="rule=固定 12 项规则计划；llm=强制模型；None=跟随服务端")
    seed: int = DEFAULT_SEED
    param_overrides: dict[str, float | int | str | bool] = Field(default_factory=dict)
    forecast: bool = False
    forecast_horizon: Optional[int] = None
    agent_max_steps: int = 8
    agent_scope: ScopeMode = "guided"
    agent_max_tokens: int = 0
    max_cost: Optional[float] = None
    strict_verify: bool = False


@dataclass
class RunRecord:
    run_id: str
    question: str
    input_path: str
    mode: str
    out_dir: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    status: str = "queued"
    submitted_at: str = field(default_factory=_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_s: Optional[float] = None
    engine_run_id: Optional[str] = None
    error: Optional[dict[str, Any]] = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------- 执行

def _execute(app: FastAPI, run_id: str) -> None:
    rec: RunRecord = app.state.records[run_id]
    rec.status = "running"
    rec.started_at = _now()
    t0 = time.time()
    try:
        req = AnalysisRequest(request_id=run_id, input_path=rec.input_path,
                              question=rec.question, output_dir=rec.out_dir,
                              mode=rec.mode, **rec.kwargs)
        res = run_with_timeout(req, timeout_s=app.state.run_timeout, verbose=False)
        rec.status = "succeeded" if res.status == "succeeded" else "failed"
        rec.engine_run_id = getattr(res, "run_id", None)
        if res.error is not None:
            # `ErrorDetail` 只有 code / message / details，没有 node 字段；
            # 「停在哪一步」只能从 events 的最后一条读。两者别混成一个字段，
            # 否则会像 R8 那样查一个永远不存在的键（详见提交说明）。
            events = list(getattr(res, "events", None) or [])
            rec.error = {"code": res.error.code, "message": res.error.message,
                         "details": res.error.details or {},
                         "last_node": events[-1].node if events else None}
        rec.artifacts = [
            {"name": os.path.basename(a.path), "kind": a.kind,
             "bytes": os.path.getsize(a.path) if os.path.exists(a.path) else None}
            for a in res.artifacts
        ]
    except Exception as exc:                            # noqa: BLE001  兜底，不让线程静默死掉
        rec.status = "failed"
        rec.error = {"code": "INTERNAL_ERROR", "node": "api",
                     "message": "%s: %s" % (type(exc).__name__, exc)[:300]}
    finally:
        rec.finished_at = _now()
        rec.duration_s = round(time.time() - t0, 1)


def _submit(app: FastAPI, rec: RunRecord) -> None:
    app.state.records[rec.run_id] = rec
    os.makedirs(rec.out_dir, exist_ok=True)
    app.state.pool.submit(_execute, app, rec.run_id)


def _check_capacity(app: FastAPI) -> None:
    with app.state.lock:
        active = sum(1 for r in app.state.records.values() if r.status in ACTIVE)
    if active >= app.state.queue_limit:
        raise HTTPException(
            status_code=http_status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"code": "QUEUE_FULL",
                    "message": "排队已达上限 %d，请稍后重试" % app.state.queue_limit},
            headers={"Retry-After": "30"})


def _new_run(*, input_path: str, question: str, mode: str, **kwargs) -> RunRecord:
    run_id = "api_" + secrets.token_hex(6)
    return RunRecord(run_id=run_id, question=question, input_path=input_path,
                     mode=mode, out_dir=os.path.join(OUT_ROOT, run_id), kwargs=kwargs)


def _created(rec: RunRecord) -> JSONResponse:
    base = "/api/v1/analyses/%s" % rec.run_id
    return JSONResponse(status_code=http_status.HTTP_202_ACCEPTED, content={
        "run_id": rec.run_id, "status": rec.status,
        "status_url": base, "result_url": base + "/result",
        "out_dir": rec.out_dir,
    })


def _envelope(rec: RunRecord) -> dict[str, Any]:
    base = "/api/v1/analyses/%s" % rec.run_id
    done = rec.status not in ACTIVE
    return {
        "run_id": rec.run_id, "engine_run_id": rec.engine_run_id,
        "status": rec.status,
        "submitted_at": rec.submitted_at, "started_at": rec.started_at,
        "finished_at": rec.finished_at, "duration_s": rec.duration_s,
        "mode": rec.mode, "question": rec.question, "input_path": rec.input_path,
        "out_dir": rec.out_dir,
        "error": rec.error,
        "artifacts": rec.artifacts,
        "result_available": done and os.path.exists(os.path.join(rec.out_dir, "result.json")),
        "result_url": base + "/result",
        "artifact_url": base + "/artifacts/{name}",
    }


# ---------------------------------------------------------------- 应用

def create_app(token: Optional[str] = None,
               *, workers: Optional[int] = None,
               queue_limit: Optional[int] = None,
               run_timeout: Optional[float] = None,
               max_upload_mb: Optional[int] = None) -> FastAPI:
    token = token if token is not None else os.environ.get("MTSA_API_TOKEN", "")
    workers = workers if workers is not None else int(os.environ.get("MTSA_API_WORKERS", "2"))
    queue_limit = (queue_limit if queue_limit is not None
                   else int(os.environ.get("MTSA_API_QUEUE", "32")))
    run_timeout = (run_timeout if run_timeout is not None
                   else float(os.environ.get("MTSA_API_RUN_TIMEOUT", "300")))
    max_upload_mb = (max_upload_mb if max_upload_mb is not None
                     else int(os.environ.get("MTSA_API_MAX_UPLOAD_MB", "50")))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not app.state.token:
            raise RuntimeError(
                "未配置 MTSA_API_TOKEN。这是有意为之：默认无鉴权的服务等于没门。\n"
                "本地试跑：MTSA_API_TOKEN=dev mtsa-api --port 8000")
        os.makedirs(OUT_ROOT, exist_ok=True)
        try:
            yield
        finally:
            app.state.pool.shutdown(wait=False)

    app = FastAPI(title="mtsa API（薄壳）", version=_version(), lifespan=lifespan)
    app.state.token = token
    app.state.pool = ThreadPoolExecutor(max_workers=max(1, workers),
                                        thread_name_prefix="mtsa-api")
    app.state.records: dict[str, RunRecord] = {}
    app.state.lock = threading.Lock()
    app.state.queue_limit = max(1, queue_limit)
    app.state.run_timeout = run_timeout
    app.state.max_upload = max(1, max_upload_mb) * 1024 * 1024

    def require_token(request: Request,
                      x_api_token: Optional[str] = Header(default=None,
                                                          alias="X-API-Token")) -> None:
        expected = request.app.state.token
        if not expected or not x_api_token or not secrets.compare_digest(x_api_token, expected):
            raise HTTPException(
                status_code=http_status.HTTP_401_UNAUTHORIZED,
                detail={"code": "UNAUTHORIZED", "message": "缺少或错误的 X-API-Token"},
                headers={"WWW-Authenticate": "X-API-Token"})

    guard = [Depends(require_token)]

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        """不鉴权：只回报存活与版本，不含任何数据。"""
        return {"ok": True, "version": _version(), "active": sum(
            1 for r in app.state.records.values() if r.status in ACTIVE)}

    @app.post("/api/v1/preview", dependencies=guard)
    def preview(body: AnalysisIn) -> dict[str, Any]:
        """同步预演（B1 的 dry-run）：不写文件、不跑检测器，先确认分析目标。"""
        if not os.path.isfile(body.input_path):
            raise HTTPException(400, {"code": "INPUT_NOT_FOUND",
                                      "message": "文件不存在：%s" % body.input_path})
        req = AnalysisRequest(request_id="api_preview", input_path=body.input_path,
                              question=body.question, scene_id=body.scene_id,
                              time_column=body.time_column,
                              value_columns=body.value_columns,
                              group_column=body.group_column, mode=body.mode,
                              planner=body.planner,
                              seed=body.seed, param_overrides=body.param_overrides,
                              forecast=body.forecast,
                              forecast_horizon=body.forecast_horizon,
                              agent_max_steps=body.agent_max_steps,
                              agent_scope=body.agent_scope,
                              agent_max_tokens=body.agent_max_tokens,
                              max_cost=body.max_cost, strict_verify=body.strict_verify)
        res = run_analysis(req, verbose=False, dry_run=True)
        if res.status != "succeeded":
            code = res.error.code if res.error else "INTERNAL_ERROR"
            raise HTTPException(400, {"code": code,
                                      "message": res.error.message if res.error else "预演失败"})
        return res.data["preview"]

    @app.post("/api/v1/analyses", status_code=http_status.HTTP_202_ACCEPTED,
              dependencies=guard)
    def create(body: AnalysisIn) -> JSONResponse:
        if not os.path.isfile(body.input_path):
            raise HTTPException(400, {"code": "INPUT_NOT_FOUND",
                                      "message": "文件不存在：%s" % body.input_path})
        _check_capacity(app)
        kwargs = body.model_dump(exclude={"input_path", "question", "mode"})
        rec = _new_run(input_path=body.input_path, question=body.question,
                       mode=body.mode, **kwargs)
        _submit(app, rec)
        return _created(rec)

    @app.post("/api/v1/analyses/upload", status_code=http_status.HTTP_202_ACCEPTED,
              dependencies=guard)
    async def create_from_upload(file: UploadFile = File(...),
                                 question: str = Form(...),
                                 mode: Mode = Form("pipeline"),
                                 scene_id: Optional[SceneId] = Form(None)) -> JSONResponse:
        name = os.path.basename((file.filename or "upload.csv").replace("\\", "/"))
        if not name or name in (".", "..") or name.startswith("."):
            raise HTTPException(400, {"code": "BAD_FILENAME",
                                      "message": "文件名不可用：%r" % file.filename})
        _check_capacity(app)
        rec = _new_run(input_path="", question=question, mode=mode,
                       scene_id=scene_id)
        updir = os.path.join(OUT_ROOT, "uploads", rec.run_id)
        os.makedirs(updir, exist_ok=True)
        dest = os.path.join(updir, name)
        written = 0
        try:
            with open(dest, "wb") as fh:
                while True:
                    chunk = await file.read(1 << 20)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > app.state.max_upload:
                        raise HTTPException(
                            413, {"code": "UPLOAD_TOO_LARGE",
                                  "message": "超过上限 %d MB" % (app.state.max_upload // (1 << 20))})
                    fh.write(chunk)
        except HTTPException:
            # 半截文件不留在磁盘上，避免后续误当完整数据
            if os.path.exists(dest):
                os.remove(dest)
            raise
        finally:
            await file.close()
        rec.input_path = dest
        _submit(app, rec)
        return _created(rec)

    def _record(run_id: str) -> RunRecord:
        rec = app.state.records.get(run_id)
        if rec is None:
            raise HTTPException(404, {"code": "RUN_NOT_FOUND",
                                      "message": "没有这个 run_id（重启后状态会丢）：%s" % run_id})
        return rec

    @app.get("/api/v1/analyses/{run_id}", dependencies=guard)
    def get_status(run_id: str) -> dict[str, Any]:
        return _envelope(_record(run_id))

    @app.get("/api/v1/analyses/{run_id}/result", dependencies=guard)
    def get_result(run_id: str) -> FileResponse:
        rec = _record(run_id)
        path = os.path.join(rec.out_dir, "result.json")
        if rec.status in ACTIVE or not os.path.exists(path):
            raise HTTPException(409, {"code": "RESULT_NOT_READY",
                                      "message": "状态为 %s，结果文件尚未生成" % rec.status})
        # 原样返回字节，不重新序列化：契约与 CLI 完全一致
        return FileResponse(path, media_type="application/json")

    @app.get("/api/v1/analyses/{run_id}/artifacts/{name}", dependencies=guard)
    def get_artifact(run_id: str, name: str) -> FileResponse:
        rec = _record(run_id)
        if rec.status in ACTIVE:
            raise HTTPException(409, {"code": "ARTIFACT_NOT_READY",
                                      "message": "状态为 %s" % rec.status})
        safe = os.path.basename(name)
        known = {a["name"] for a in rec.artifacts}
        if safe not in known:
            raise HTTPException(404, {"code": "ARTIFACT_NOT_FOUND",
                                      "available": sorted(known)})
        path = os.path.join(rec.out_dir, safe)
        if not os.path.isfile(path):
            raise HTTPException(404, {"code": "ARTIFACT_MISSING", "path": safe})
        return FileResponse(path)

    return app


app = create_app()


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="mtsa-api", description="mtsa HTTP 薄壳")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="开发用热重载")
    args = ap.parse_args(argv)
    if not os.environ.get("MTSA_API_TOKEN"):
        print("未配置 MTSA_API_TOKEN：默认无鉴权的服务等于没门。\n"
              "本地试跑：$env:MTSA_API_TOKEN='dev'; mtsa-api --port 8000", file=sys.stderr)
        return 2
    import uvicorn
    uvicorn.run("mtsa.api:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())