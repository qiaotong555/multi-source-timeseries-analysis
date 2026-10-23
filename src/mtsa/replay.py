# -*- coding: utf-8 -*-
"""轨迹持久化与离线重放（T2.5）。

Agent 的路径由模型决定，因此「同一输入两次运行结果相同」不再由代码保证。
可复现性的替代保证是：**把轨迹存下来，重放时不再调用模型，只按原顺序重跑工具**。

本模块只处理数据，不负责编排：
- `load_replay()` 读取并校验一次 agent 运行的 `result.json`；
- `trajectory_hash()` 对「决策路径」做规范化哈希（不含耗时、时间戳、run_id）；
- `replay_steps()` 按记录顺序重跑工具，构造新的轨迹；
- `diff_paths()` 给出两次结构化结果的差异路径，便于定位。

为什么哈希不包含 `elapsed_ms`：机器负载会改变耗时，但它不改变分析结论。
为什么哈希包含 `reason`：理由属于模型输出的一部分，重放时应与原轨迹一致。

NaN 与 null 视为同一语义
-------------------------
`result.json` 落盘时 NaN 会被序列化成 null，读回来是 None；而重放时内存里的
统计量仍是 float('nan')。两者都表示「该检验不适用」，若判为不一致，
会把正常的重放误报成失败。比对层面做归一化，分析层面不填任何假值。
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from . import tools
from .agent_runtime import AgentStep, AgentTrajectory

REPLAY_VERSION = "replay-v1"


class ReplayError(ValueError):
    """重放源不可用：不是 agent 结果、缺字段、或版本不支持。"""


def load_replay(path: str) -> dict[str, Any]:
    """读取 result.json，返回重放所需的快照。"""
    p = Path(path)
    if not p.exists():
        raise ReplayError(f"重放源不存在：{path}")
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReplayError(f"重放源不是合法 JSON：{exc}") from exc

    agent = (d.get("data") or {}).get("agent")
    if not isinstance(agent, dict):
        raise ReplayError("重放源没有 data.agent：只有 agent 模式的结果可以重放")
    plan = (d.get("data") or {}).get("plan")
    if not isinstance(plan, dict):
        raise ReplayError("重放源缺少 data.plan")
    steps = agent.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ReplayError("重放源没有可重放的步骤")
    for i, s in enumerate(steps):
        if not isinstance(s, dict) or not s.get("action"):
            raise ReplayError(f"第 {i + 1} 步缺少 action")
    return {
        "path": str(p),
        "question": agent.get("question") or d.get("datasets", [{}])[0].get("question", ""),
        "input_path": (d.get("datasets") or [{}])[0].get("input_path", ""),
        "model": agent.get("model", ""),
        "allowed_tools": agent.get("allowed_tools") or [],
        "agent": agent,
        "plan": plan,
        "data": d.get("data") or {},
        "trajectory_sha256": agent.get("trajectory_sha256"),
        "replay_version": agent.get("replay_version", REPLAY_VERSION),
    }


def trajectory_hash(agent: dict[str, Any]) -> str:
    """对决策路径做规范化 SHA-256。"""
    steps = []
    for s in agent.get("steps") or []:
        steps.append({
            "action": s.get("action"),
            "args": s.get("args") or {},
            "reason": s.get("reason") or "",
            "outcome": s.get("outcome"),
            "summary": s.get("summary") or {},
            "error": s.get("error"),
        })
    payload = {
        "model": agent.get("model") or "",
        "allowed_tools": list(agent.get("allowed_tools") or []),
        "steps": steps,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def replay_steps(source: dict[str, Any], ctx: tools.ToolContext) -> AgentTrajectory:
    """按记录顺序重跑工具；不调用模型。"""
    traj = AgentTrajectory(
        question=source.get("question", ""),
        model=source.get("model", ""),
        allowed_tools=list(source.get("allowed_tools") or []),
    )
    for i, s in enumerate(source["agent"]["steps"]):
        action = s["action"]
        started = datetime.now().isoformat(timespec="seconds")
        res = tools.run_tool(action, ctx)
        traj.results[action] = res
        traj.steps.append(AgentStep(
            index=i, action=action, args=s.get("args") or {},
            reason=s.get("reason") or "", outcome=res.outcome,
            summary=res.summary, elapsed_ms=res.elapsed_ms,
            error=res.error, started_at=started,
        ))
    traj.terminate_reason = "replayed"
    return traj


def _missing(value: Any) -> bool:
    """NaN 与 None 都表示「无值」：落盘后 NaN 会变成 null，二者不能算差异。"""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _diff(a: Any, b: Any, path: str, out: list[str], limit: int) -> None:
    if len(out) >= limit:
        return
    if _missing(a) and _missing(b):
        return
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        out.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
        return
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}/{k}: 重放多出字段")
            elif k not in b:
                out.append(f"{path}/{k}: 重放缺少字段")
            else:
                _diff(a[k], b[k], f"{path}/{k}", out, limit)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: 长度 {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")


def diff_paths(recorded: dict[str, Any], replayed: dict[str, Any],
               limit: int = 20) -> list[str]:
    """比较两份 data 载荷，返回差异路径。"""
    out: list[str] = []
    _diff(recorded, replayed, "", out, limit)
    return out
