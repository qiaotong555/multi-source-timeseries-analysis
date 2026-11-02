# -*- coding: utf-8 -*-
"""任务仓储（E3）：把 API 的运行记录落到 SQLite。

## 为什么需要它

薄壳最初把 `RunRecord` 放在 `app.state.records` 这个进程内 dict 里，重启即丢：
客户端拿着 `run_id` 回来查状态会得到 404，尽管 `artifacts/api/<run_id>/` 里的
`result.json` 和产物都还在。**产物在磁盘、状态在内存**，这两件事不同步，是这一层
最明显的不一致。

## 为什么是 SQLite，不是 JSON 文件

三个访问模式决定了选型：

1. 按 `run_id` 查单条（每次 `GET /analyses/{id}`）；
2. 启动时扫出所有 `queued` / `running` 的记录（判为中断）；
3. 多个工作线程在分析结束时各写各的。

用 JSON 文件就得「读全量 → 改一条 → 写全量」，两个线程同时收尾会丢写。SQLite 自带
事务和按行更新，且是标准库，不引入依赖。单文件、可直接删、可 `sqlite3` 命令行查。

## 存什么

存整个记录的 JSON（`payload` 列），不拆成列。理由：记录字段少、结构稳定，拆列会让
「加一个字段」变成迁移；而按 `run_id` 取的场景不需要对字段做 SQL 过滤。
`status` / `submitted_at` / `updated_at` 三个是冗余出来做扫描和排序的。

## 中断怎么处理

进程被杀时，库里会留下 `running` 的记录。启动时统一改成 `failed` +
`error.code=INTERRUPTED`。**不假装它还在跑**——客户端看到「已中断」比看到「运行中」
然后永远等下去要有用。产物目录不动，`result.json` 若已落盘仍可下载。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    status       TEXT NOT NULL,
    submitted_at TEXT,
    updated_at   TEXT NOT NULL,
    payload      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_submitted ON runs(submitted_at);
"""

ACTIVE_STATUSES = ("queued", "running")

# 关闭持久化的取值。测试与「无状态部署」两种场景都会用到。
OFF = "off"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class RunStore:
    """运行记录的持久化。`enabled=False` 时所有方法都是空操作。"""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.enabled = bool(path) and str(path).lower() != OFF
        self._lock = threading.Lock()
        self.init_error: Optional[str] = None

    # ------------------------------------------------------------ 连接
    def _connect(self) -> Optional[sqlite3.Connection]:
        if not self.enabled:
            return None
        assert self.path is not None
        try:
            conn = sqlite3.connect(self.path, timeout=15)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            return conn
        except (sqlite3.Error, OSError) as exc:         # pragma: no cover  环境相关
            self.init_error = "%s: %s" % (type(exc).__name__, exc)
            return None

    def init(self) -> None:
        """建表。失败不抛——仓储挂了服务仍应能跑，只是退回「重启即丢」。"""
        if not self.enabled:
            return
        assert self.path is not None
        try:
            parent = os.path.dirname(os.path.abspath(self.path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            conn = self._connect()
            if conn is None:
                return
            try:
                with self._lock:
                    conn.executescript(SCHEMA)
                    conn.commit()
            finally:
                conn.close()
        except (sqlite3.Error, OSError) as exc:        # pragma: no cover  环境相关
            self.enabled = False
            self.init_error = "%s: %s" % (type(exc).__name__, exc)

    # ------------------------------------------------------------ 读写
    def save(self, rec: dict[str, Any]) -> None:
        if not self.enabled:
            return
        conn = self._connect()
        if conn is None:
            return
        try:
            payload = json.dumps(rec, ensure_ascii=False, default=str)
            with self._lock:
                conn.execute(
                    "INSERT INTO runs(run_id, status, submitted_at, updated_at, payload) "
                    "VALUES(?,?,?,?,?) "
                    "ON CONFLICT(run_id) DO UPDATE SET "
                    "status=excluded.status, submitted_at=excluded.submitted_at, "
                    "updated_at=excluded.updated_at, payload=excluded.payload",
                    (rec.get("run_id"), rec.get("status") or "",
                     rec.get("submitted_at"), _now(), payload))
                conn.commit()
        except (sqlite3.Error, OSError):                # pragma: no cover  环境相关
            pass
        finally:
            conn.close()

    def get(self, run_id: str) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None
        conn = self._connect()
        if conn is None:
            return None
        try:
            row = conn.execute("SELECT payload FROM runs WHERE run_id = ?",
                               (run_id,)).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:                     # pragma: no cover  库被手改
            return None

    def mark_interrupted(self, message: str = "服务重启，任务中断，请重新提交") -> int:
        """把上次进程留下的 `queued` / `running` 改成 `failed`，返回条数。"""
        if not self.enabled:
            return 0
        conn = self._connect()
        if conn is None:
            return 0
        changed = 0
        try:
            rows = conn.execute(
                "SELECT run_id, payload FROM runs WHERE status IN (?,?)",
                ACTIVE_STATUSES).fetchall()
            now = _now()
            for run_id, payload in rows:
                try:
                    rec = json.loads(payload)
                except json.JSONDecodeError:             # pragma: no cover
                    continue
                previous = rec.get("status")
                rec["status"] = "failed"
                rec["finished_at"] = now
                rec["error"] = {"code": "INTERRUPTED", "message": message,
                                "details": {"previous_status": previous}}
                with self._lock:
                    conn.execute(
                        "UPDATE runs SET status=?, updated_at=?, payload=? WHERE run_id=?",
                        ("failed", now, json.dumps(rec, ensure_ascii=False, default=str), run_id))
                changed += 1
            with self._lock:
                conn.commit()
        except (sqlite3.Error, OSError):                # pragma: no cover  环境相关
            pass
        finally:
            conn.close()
        return changed

    def count(self, status: Optional[str] = None) -> int:
        if not self.enabled:
            return 0
        conn = self._connect()
        if conn is None:
            return 0
        try:
            if status:
                return int(conn.execute("SELECT COUNT(*) FROM runs WHERE status = ?",
                                        (status,)).fetchone()[0])
            return int(conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])
        finally:
            conn.close()

    # ------------------------------------------------------------ 列表 / 清理 / 统计
    def list_runs(self, status: Optional[str] = None, limit: int = 50,
                  offset: int = 0,
                  newest_first: bool = True) -> list[dict[str, Any]]:
        """按提交时间列出记录。`status` 为 None 时不过滤。

        只读 payload，不在 SQL 层裁剪字段——记录本身小，裁剪逻辑放在 API 层，
        免得「接口要什么字段」和「库怎么存」耦合在一起。
        """
        if not self.enabled:
            return []
        conn = self._connect()
        if conn is None:
            return []
        order = "DESC" if newest_first else "ASC"
        try:
            if status:
                rows = conn.execute(
                    "SELECT payload FROM runs WHERE status = ? "
                    "ORDER BY submitted_at %s LIMIT ? OFFSET ?" % order,
                    (status, max(1, limit), max(0, offset))).fetchall()
            else:
                rows = conn.execute(
                    "SELECT payload FROM runs ORDER BY submitted_at %s LIMIT ? OFFSET ?"
                    % order,
                    (max(1, limit), max(0, offset))).fetchall()
        finally:
            conn.close()
        out = []
        for (payload,) in rows:
            try:
                out.append(json.loads(payload))
            except json.JSONDecodeError:                 # pragma: no cover  库被手改
                continue
        return out

    def purge_before(self, cutoff: str) -> list[str]:
        """删掉 `submitted_at` 早于 cutoff 的记录，返回被删的 run_id。

        只删记录，**不动产物目录**——删文件不可逆，交给调用方显式决定。
        """
        if not self.enabled:
            return []
        conn = self._connect()
        if conn is None:
            return []
        try:
            rows = conn.execute(
                "SELECT run_id FROM runs WHERE submitted_at IS NOT NULL "
                "AND submitted_at < ?", (cutoff,)).fetchall()
            ids = [r[0] for r in rows]
            if ids:
                with self._lock:
                    conn.executemany("DELETE FROM runs WHERE run_id = ?",
                                     [(i,) for i in ids])
                    conn.commit()
            return ids
        except (sqlite3.Error, OSError):                # pragma: no cover  环境相关
            return []
        finally:
            conn.close()

    def stats(self) -> dict[str, Any]:
        """任务计数、状态分布、耗时分位。耗时只统计有 `duration_s` 的记录。"""
        if not self.enabled:
            return {"enabled": False}
        conn = self._connect()
        if conn is None:
            return {"enabled": False}
        try:
            by_status = {row[0]: int(row[1]) for row in conn.execute(
                "SELECT status, COUNT(*) FROM runs GROUP BY status")}
            total = sum(by_status.values())
            row = conn.execute("SELECT MIN(submitted_at), MAX(submitted_at) FROM runs").fetchone()
            durations = [json.loads(p).get("duration_s")
                         for (p,) in conn.execute("SELECT payload FROM runs").fetchall()]
        finally:
            conn.close()
        vals = sorted(float(d) for d in durations if isinstance(d, (int, float)))
        def _q(p: float) -> Optional[float]:
            if not vals:
                return None
            k = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * p))))
            return round(vals[k], 2)
        return {
            "enabled": True,
            "path": self.path,
            "total": total,
            "by_status": dict(sorted(by_status.items())),
            "oldest_submitted_at": row[0] if row else None,
            "newest_submitted_at": row[1] if row else None,
            "duration_s": {
                "count": len(vals),
                "p50": _q(0.5), "p90": _q(0.9), "max": (round(vals[-1], 2) if vals else None),
            },
            "init_error": self.init_error,
        }
