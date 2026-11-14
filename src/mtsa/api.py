# -*- coding: utf-8 -*-
"""HTTP 薄壳（E1）：把 `run_analysis` 包成 5 个路由，不做业务逻辑。

设计边界写死在这里，避免以后被"顺手加个功能"撑成第二个内核：

- **不判场景、不映射参数、不改 `result.json` 契约**：请求 → `AnalysisRequest` → 现有引擎；
- **状态持久化在 SQLite**（E3，2026-11-02）：`MTSA_API_DB` 默认 `artifacts/api/state.sqlite3`，
  按 `run_id` 落一条记录；重启后 `GET` 仍能查到状态与产物清单。
  重启前处于 `queued` / `running` 的任务会被标成 `failed` + `INTERRUPTED`——
  进程已经死了，不假装它还在跑。`MTSA_API_DB=off` 退回「重启即丢」的内存行为；
- **不做取消、不做流式、不做多租户、不做用户体系**；
- **运维面只有三个**：`GET /analyses`（分页列表）、`GET /metrics`（计数/状态分布/耗时分位）、
  启动时的保留期清理。都走鉴权。不做 Prometheus 导出、不做日志聚合；
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

七个状态怎么走
--------------
`models.Status` 的 7 个词在这里都有对应的路径（2026-11-14 补齐，此前只有 4 个）：

| 状态 | 何时出现 |
| --- | --- |
| `queued` | 已受理、等执行槽 |
| `running` | 子进程在跑 |
| `waiting_for_input` | 失败原因是**场景识别不出**（唯一一种「补一个字段必然能继续」的情形）；补 `scene_id` 后 `POST /resume` 重新排队 |
| `cancel_requested` | 运行中收到取消，子进程正在被终止 |
| `succeeded` / `failed` | 终态 |
| `cancelled` | 取消完成（排队中取消是即时的，运行中取消要等进程退出） |

**取消是尽力而为，不是即时的。** 父进程每 50 ms 轮询一次取消信号，随后 terminate（宽限期内
不退出再 kill）。因此从 `cancel_requested` 到 `cancelled` 之间有一小段窗口，调用方必须轮询，
不能假设 POST /cancel 返回时任务已经停下。

**`waiting_for_input` 不占并发槽。** 它在等调用方补字段，占着槽只会让队列白堵；
但重启时也不该被标成「中断」——它本来就没在跑。

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
| `MTSA_API_DB` | `$MTSA_API_OUT/state.sqlite3` | 任务仓储；`off` 退回「重启即丢」 |
| `MTSA_API_RETENTION_DAYS` | 0 | 0=不清理；>0 时启动清掉更早的**记录**（产物目录不动） |
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
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timedelta
from typing import Any, Literal, Optional

from fastapi import (Depends, FastAPI, File, Form, Header, HTTPException, Request,
                     UploadFile, status as http_status)
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .engine import run_analysis
from .models import DEFAULT_SEED, AnalysisRequest
from .scenarios import stock as _stock        # noqa: F401  注册适配器
from .scenarios import weather as _weather    # noqa: F401
from .storage import RunStore
from .timeout_guard import run_with_timeout

# 产物目录按**当前工作目录**解析，与 CLI 的 `artifacts/<run_id>` 同一规则。
# 不能用包的安装位置：pip 装进 site-packages 后，那里是 /usr/local/lib/python3.13，
# 容器里既不是数据目录，也可能不可写（镜像的 WORKDIR 是 /data）。
OUT_ROOT = os.path.abspath(os.environ.get("MTSA_API_OUT",
                                          os.path.join("artifacts", "api")))

Mode = Literal["pipeline", "agent", "auto"]
SceneId = Literal["weather", "stock"]
ScopeMode = Literal["guided", "free"]

# 「还占着并发槽」的状态。`cancel_requested` 也算：进程还没死，槽位没释放。
# `waiting_for_input` 不算——它在等调用方补字段，不该占槽。
ACTIVE = ("queued", "running", "cancel_requested")

# 这些错误码的本质是「缺一个字段」，补上就能跑。出现它们时任务进 `waiting_for_input`
# 而不是 `failed`：`failed` 表示重试也没用，`waiting_for_input` 表示等你补参数。
# 只有「补一个字段就一定能跑」的错误才进这张表。宁可判 failed 也别把任务
# 停在 waiting_for_input —— 后者的语义是「等你补参数就能继续」，承诺错了比
# 直接失败更糟：调用方会一直等一个不会到的结果。
#
# - `SCENE_NOT_DETECTED`：场景识别不出，给了 scene_id 必然能进计划阶段。
# - `TIME_AXIS_NOT_FOUND` / `EMPTY_NUMERIC_COLUMN` 故意**不在**表里：profiler 已经
#   把所有列试过一遍，指定一个列名能否救活取决于数据本身，无法事先保证。
#   这两种情况按 failed 处理，错误提示里已经写明「用 --time-column 显式指定」。
# - `DATA_LOAD_FAILED` 也不在：提交时就查过文件存在性（INPUT_NOT_FOUND 400），
#   走不到执行阶段。
FIXABLE_INPUTS: dict[str, list[str]] = {
    "SCENE_NOT_DETECTED": ["scene_id"],
}
DEFAULT_DB = os.path.join(OUT_ROOT, "state.sqlite3")


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
    # 原文记录到 datasets[0].dataset_note 与报告第 1 节；不参与任何计算
    dataset_note: Optional[str] = Field(
        default=None, description="数据说明（来源/采集方式/已知缺陷），原样记录")
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


class ResumeIn(BaseModel):
    """补参数后继续：只允许改「可能缺的输入」，不允许改 mode/成本/安全相关开关。

    这是有意的收窄。`waiting_for_input` 的语义是「差一个字段」，不是
    「给你一次重新配置的机会」——放宽到全量字段会让「补输入」变成绕过队列与预算的后门。
    """

    input_path: Optional[str] = Field(default=None, description="换个文件路径重试")
    question: Optional[str] = None
    scene_id: Optional[SceneId] = None
    time_column: Optional[str] = None
    value_columns: Optional[list[str]] = None
    group_column: Optional[str] = None
    unit: Optional[str] = None
    dataset_note: Optional[str] = None
    param_overrides: Optional[dict[str, float | int | str | bool]] = None


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
    # 仅 `waiting_for_input` 时有值：缺哪些字段才能继续
    input_required: list[str] = field(default_factory=list)


_RECORD_FIELDS = {f.name for f in fields(RunRecord)}


def _persist(app: FastAPI, rec: RunRecord) -> None:
    """状态变了就落一次库。仓储不可用时静默跳过——服务不该因为存不了状态而挂。"""
    store = getattr(app.state, "store", None)
    if store is not None:
        store.save(asdict(rec))


def _record_from_dict(d: dict[str, Any]) -> RunRecord:
    return RunRecord(**{k: v for k, v in d.items() if k in _RECORD_FIELDS})


# ---------------------------------------------------------------- 执行

def _execute(app: FastAPI, run_id: str) -> None:
    rec: RunRecord = app.state.records[run_id]
    with app.state.lock:
        # 排队期间被取消：不要开始算。这条检查必须在置 running 之前，
        # 否则取消会输给「刚好轮到它」的竞态。
        if rec.status == "cancelled":
            return
        rec.status = "running"
        rec.started_at = _now()
    _persist(app, rec)
    t0 = time.time()
    event = app.state.cancel_events.setdefault(run_id, threading.Event())
    try:
        req = AnalysisRequest(request_id=run_id, input_path=rec.input_path,
                              question=rec.question, output_dir=rec.out_dir,
                              mode=rec.mode, **rec.kwargs)
        res = run_with_timeout(req, timeout_s=app.state.run_timeout,
                               verbose=False, cancel_event=event)
        if res.status == "cancelled":
            rec.status = "cancelled"
        elif res.status == "succeeded":
            rec.status = "succeeded"
        else:
            code = res.error.code if res.error else None
            required = FIXABLE_INPUTS.get(code or "")
            # 「补一个字段就能跑」与「跑不动」是两回事：前者挂起等输入，后者判失败。
            rec.status = "waiting_for_input" if required else "failed"
            rec.input_required = list(required or [])
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
        _persist(app, rec)


def _submit(app: FastAPI, rec: RunRecord) -> None:
    app.state.records[rec.run_id] = rec
    app.state.cancel_events[rec.run_id] = threading.Event()
    os.makedirs(rec.out_dir, exist_ok=True)
    _persist(app, rec)
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


def _has_field(rec: RunRecord, name: str) -> bool:
    """判断某个「可补字段」是否已经有着落。空列表、空字典都算没有。"""
    if name == "input_path":
        return bool(rec.input_path)
    if name == "question":
        return bool(rec.question)
    return bool(rec.kwargs.get(name))


def _new_run(*, input_path: str, question: str, mode: str, **kwargs) -> RunRecord:
    run_id = "api_" + secrets.token_hex(6)
    return RunRecord(run_id=run_id, question=question, input_path=input_path,
                     mode=mode, out_dir=os.path.join(OUT_ROOT, run_id), kwargs=kwargs)


def _created(rec: RunRecord) -> JSONResponse:
    base = "/api/v1/analyses/%s" % rec.run_id
    return JSONResponse(status_code=http_status.HTTP_202_ACCEPTED, content={
        "run_id": rec.run_id, "status": rec.status,
        "status_url": base, "result_url": base + "/result",
        "cancel_url": base + "/cancel",
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
        # 只有「等输入」时才有内容；「可取消」按状态算，避免调用方猜
        "input_required": rec.input_required,
        "cancel_url": base + "/cancel",
        "resume_url": base + "/resume" if rec.status == "waiting_for_input" else None,
        "cancellable": rec.status in ACTIVE,
    }


# ---------------------------------------------------------------- 应用

def create_app(token: Optional[str] = None,
               *, workers: Optional[int] = None,
               queue_limit: Optional[int] = None,
               run_timeout: Optional[float] = None,
               max_upload_mb: Optional[int] = None,
               db_path: Optional[str] = None,
               retention_days: Optional[float] = None) -> FastAPI:
    token = token if token is not None else os.environ.get("MTSA_API_TOKEN", "")
    workers = workers if workers is not None else int(os.environ.get("MTSA_API_WORKERS", "2"))
    queue_limit = (queue_limit if queue_limit is not None
                   else int(os.environ.get("MTSA_API_QUEUE", "32")))
    run_timeout = (run_timeout if run_timeout is not None
                   else float(os.environ.get("MTSA_API_RUN_TIMEOUT", "300")))
    max_upload_mb = (max_upload_mb if max_upload_mb is not None
                     else int(os.environ.get("MTSA_API_MAX_UPLOAD_MB", "50")))
    db_path = db_path if db_path is not None else os.environ.get("MTSA_API_DB", DEFAULT_DB)
    retention_days = (retention_days if retention_days is not None
                      else float(os.environ.get("MTSA_API_RETENTION_DAYS", "0")))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not app.state.token:
            raise RuntimeError(
                "未配置 MTSA_API_TOKEN。这是有意为之：默认无鉴权的服务等于没门。\n"
                "本地试跑：MTSA_API_TOKEN=dev mtsa-api --port 8000")
        os.makedirs(OUT_ROOT, exist_ok=True)
        app.state.store.init()
        if app.state.store.enabled:
            left = app.state.store.mark_interrupted()
            if left:
                print("[mtsa-api] 上次进程遗留 %d 个未完成任务，已标记为 INTERRUPTED" % left,
                      file=sys.stderr)
            if app.state.retention_days > 0:
                cutoff = (datetime.now().astimezone()
                          - timedelta(days=app.state.retention_days)
                          ).isoformat(timespec="seconds")
                removed = app.state.store.purge_before(cutoff)
                if removed:
                    print("[mtsa-api] 清理 %d 条超过 %.0f 天的运行记录（产物目录未动）"
                          % (len(removed), app.state.retention_days), file=sys.stderr)
        try:
            yield
        finally:
            app.state.pool.shutdown(wait=False)

    app = FastAPI(title="mtsa API（薄壳）", version=_version(), lifespan=lifespan)
    app.state.token = token
    app.state.pool = ThreadPoolExecutor(max_workers=max(1, workers),
                                        thread_name_prefix="mtsa-api")
    app.state.records: dict[str, RunRecord] = {}
    # 每个任务一个取消信号。放内存而不是仓储：取消是「对正在跑的进程说一声」，
    # 进程没了信号就没有意义；重启后未完成的任务本就被标为 INTERRUPTED。
    app.state.cancel_events: dict[str, threading.Event] = {}
    app.state.store = RunStore(db_path)
    app.state.retention_days = max(0.0, retention_days)
    app.state.started_at = time.time()
    app.state.max_workers = max(1, workers)
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
            saved = app.state.store.get(run_id)
            if saved is not None:
                rec = _record_from_dict(saved)
                app.state.records[run_id] = rec
        if rec is None:
            raise HTTPException(404, {"code": "RUN_NOT_FOUND",
                                      "message": "没有这个 run_id：%s" % run_id})
        return rec

    @app.get("/api/v1/analyses", dependencies=guard)
    def list_analyses(status: Optional[str] = None, limit: int = 50,
                      offset: int = 0) -> dict[str, Any]:
        """列出运行记录（分页 + 可选状态过滤）。

        返回的是「摘要」不是完整信封：`question` 截断到 80 字、`artifacts` 只给数量。
        要看完整字段用 `GET /api/v1/analyses/{run_id}`。
        """
        allowed = set(ACTIVE) | {"waiting_for_input", "succeeded", "failed", "cancelled"}
        if status is not None and status not in allowed:
            raise HTTPException(400, {"code": "BAD_STATUS_FILTER",
                                      "message": "status 只能是 %s" % sorted(allowed)})
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        recs = app.state.store.list_runs(status=status, limit=limit, offset=offset)
        items = [{
            "run_id": r.get("run_id"), "status": r.get("status"), "mode": r.get("mode"),
            "question": (r.get("question") or "")[:80],
            "submitted_at": r.get("submitted_at"), "finished_at": r.get("finished_at"),
            "duration_s": r.get("duration_s"),
            "artifacts": len(r.get("artifacts") or []),
            "error_code": (r.get("error") or {}).get("code"),
            "result_available": os.path.exists(os.path.join(r.get("out_dir") or "", "result.json")),
        } for r in recs]
        return {
            "items": items, "limit": limit, "offset": offset,
            "total": app.state.store.count(status),
            "filters": {"status": status},
            "store_enabled": app.state.store.enabled,
        }

    @app.get("/metrics", dependencies=guard)
    def metrics() -> dict[str, Any]:
        """运行计数、状态分布、耗时分位与队列水位。

        不接 Prometheus 客户端：这是单进程薄壳，`/metrics` 给运维看，
        不是给时序数据库抓的。要接抓取器的话，这里就是那个转换点。
        """
        st = app.state.store.stats()
        return {
            "runs": st,
            "queue": {
                "active": sum(1 for r in app.state.records.values() if r.status in ACTIVE),
                "limit": app.state.queue_limit,
                "workers": app.state.max_workers,
            },
            "process": {
                "version": _version(),
                "uptime_s": round(time.time() - app.state.started_at, 1),
                "retention_days": app.state.retention_days,
            },
        }

    @app.get("/api/v1/analyses/{run_id}", dependencies=guard)
    def get_status(run_id: str) -> dict[str, Any]:
        return _envelope(_record(run_id))

    @app.post("/api/v1/analyses/{run_id}/cancel", dependencies=guard)
    def cancel_run(run_id: str):
        """取消。排队中 → 直接 `cancelled`；运行中 → `cancel_requested` 并终止子进程。

        语义边界：只能尽快，不能立刻。运行中的任务要等子进程被终止并把状态写回来，
        中间会短暂停在 `cancel_requested`——调用方应当轮询到终态，而不是假设同步完成。
        """
        rec = _record(run_id)
        with app.state.lock:
            if rec.status == "cancelled":
                return _envelope(rec)                    # 幂等
            if rec.status in ("succeeded", "failed"):
                raise HTTPException(409, {
                    "code": "CANCEL_NOT_ALLOWED",
                    "message": "任务已结束（%s），无法取消" % rec.status})
            if rec.status in ("queued", "waiting_for_input"):
                # 还没开始跑，也没有进程要杀：直接落终态
                rec.status = "cancelled"
                rec.input_required = []
                rec.finished_at = _now()
                rec.duration_s = 0.0
                rec.error = {"code": "CANCELLED", "message": "调用方在开始前取消"}
            else:                                        # running / cancel_requested
                rec.status = "cancel_requested"
                app.state.cancel_events.setdefault(
                    run_id, threading.Event()).set()
        _persist(app, rec)
        return _envelope(rec)

    @app.post("/api/v1/analyses/{run_id}/resume", dependencies=guard)
    def resume_run(run_id: str, body: Optional[ResumeIn] = None):
        """补上缺失字段后重新排队。只对 `waiting_for_input` 有效。"""
        rec = _record(run_id)
        if rec.status != "waiting_for_input":
            raise HTTPException(409, {
                "code": "RESUME_NOT_ALLOWED",
                "message": "只有 waiting_for_input 的任务能补参（当前 %s）" % rec.status})
        data = (body.model_dump(exclude_none=True) if body is not None else {})
        if data.get("input_path"):
            rec.input_path = data.pop("input_path")
        if data.get("question"):
            rec.question = data.pop("question")
        data.pop("input_path", None)
        data.pop("question", None)
        rec.kwargs.update(data)

        # 补完还得查一遍：缺的没补齐就重跑，只会再失败一次，白占一次执行
        still = [f for f in rec.input_required if not _has_field(rec, f)]
        if still:
            raise HTTPException(400, {
                "code": "INPUT_STILL_MISSING",
                "message": "仍需补充：%s" % "、".join(still),
                "input_required": still})

        _check_capacity(app)
        rec.status = "queued"
        rec.input_required = []
        rec.error = None
        rec.started_at = rec.finished_at = rec.duration_s = None
        rec.artifacts = []
        _persist(app, rec)
        _submit(app, rec)
        return _created(rec)

    @app.get("/api/v1/analyses/{run_id}/result", dependencies=guard)
    def get_result(run_id: str) -> FileResponse:
        rec = _record(run_id)
        path = os.path.join(rec.out_dir, "result.json")
        if rec.status == "cancelled":
            raise HTTPException(409, {"code": "RUN_CANCELLED",
                                      "message": "任务已取消，没有结果文件"})
        if rec.status == "waiting_for_input":
            raise HTTPException(409, {"code": "INPUT_REQUIRED",
                                      "message": "任务在等补充输入",
                                      "input_required": rec.input_required})
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