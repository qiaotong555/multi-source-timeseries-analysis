# -*- coding: utf-8 -*-
"""Agent 运行时：观察—决策—行动循环。

与计划层的区别
--------------
M1 的计划层是**一次性规划**：看完数据摘要就定下 12 个分析项，之后代码按固定
顺序执行，模型看不到任何中间结果。本模块改为**逐步决策**：每执行完一个工具，
就把结果摘要回灌给模型，由它决定下一步做什么，或宣布结束。

这才是「流程长度由模型决定」的落点。计划层仍在准备阶段产出基底计划
（场景、目标列、参数），但分析层走哪几个工具、什么顺序，由这里的循环决定。

边界
----
1. **只编排白名单内的分析工具**。数据管道（导入至清洗）与收尾（图表至报告）不进
   循环：前者有严格前置依赖，后者需聚合全部结果。本轮天气场景白名单为五个工具，
   股票专属的 `check_lookahead` 不在其中。
2. **数值不经模型**。模型只看到 `ToolResult.summary`，看不到 `data`。
3. **不重试工具**。单个工具失败由模型决定换一个还是结束；重试属于 T2.7 的范围。

终止条件
--------
| 原因 | 触发 |
| --- | --- |
| `model_finished` | 模型输出 `finish` |
| `max_steps` | 步数达上限 |
| `idle_loop` | 连续被拒次数达上限（反复选已执行或无效工具） |
| `decision_error` | 模型输出连续无法解析 |
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from . import tools
from .llm_client import LLMError

TERMINATE_MODEL_FINISHED = "model_finished"
TERMINATE_MAX_STEPS = "max_steps"
TERMINATE_IDLE = "idle_loop"
TERMINATE_DECISION_ERROR = "decision_error"

DEFAULT_MAX_STEPS = 8
MAX_CONSECUTIVE_REJECTS = 3
MAX_DECISION_ERRORS = 2

AGENT_VERSION = "agent-v1"

# 本轮 Agent 只实现天气正式场景。股票场景不接入，股票专属的 check_lookahead
# 因此不在此集合内——它仍可由 pipeline 使用，但不作为天气 Agent 的可选动作。
WEATHER_AGENT_TOOLS: tuple[str, ...] = (
    "compute_metrics",
    "analyze_spectrum",
    "run_stat_tests",
    "detect_anomalies",
    "detect_change_points",
)


@dataclass
class AgentStep:
    """一步轨迹。字段即 T2.5 重放所需的最小信息集。"""

    index: int
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    outcome: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int = 0
    error: Optional[str] = None
    # 该步的开始时间。engine 用它写 events，T2.5 重放时也用它对照。
    started_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = {"index": self.index, "action": self.action, "reason": self.reason,
               "outcome": self.outcome, "summary": self.summary,
               "elapsed_ms": self.elapsed_ms, "started_at": self.started_at}
        if self.args:
            out["args"] = self.args
        if self.error:
            out["error"] = self.error
        return out


@dataclass
class AgentTrajectory:
    question: str
    steps: list[AgentStep] = field(default_factory=list)
    terminate_reason: str = ""
    results: dict[str, Any] = field(default_factory=dict)   # tool -> ToolResult（含 data）
    rejects: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    agent_version: str = AGENT_VERSION
    allowed_tools: list[str] = field(default_factory=list)

    # ---- 供 T2.8 指标与报告使用的派生量 ----
    @property
    def tools_run(self) -> list[str]:
        return [s.action for s in self.steps if s.outcome == "ok"]

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def data_of(self, tool: str, default: Any = None) -> Any:
        res = self.results.get(tool)
        return default if res is None else res.data

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_version": self.agent_version,
            "model": self.model,
            "question": self.question,
            "terminate_reason": self.terminate_reason,
            "step_count": self.step_count,
            "allowed_tools": self.allowed_tools,
            "tools_run": self.tools_run,
            "steps": [s.to_dict() for s in self.steps],
            "rejects": self.rejects,
        }


class DecisionError(RuntimeError):
    """模型输出不合法（非 JSON、缺字段、动作无效）。"""


class AgentRunner:
    """一次分析任务的自主编排器。

    用法::

        runner = AgentRunner(ctx, client, question="分析温度趋势")
        traj = runner.run()
    """

    def __init__(self, ctx: tools.ToolContext, client, question: str,
                 max_steps: int = DEFAULT_MAX_STEPS,
                 allowed_tools: Optional[list[str] | tuple[str, ...]] = None) -> None:
        if client is None:
            raise ValueError("AgentRunner 需要 LLM 客户端；无客户端时应走固定流水线")
        # 默认放开全部注册工具，便于单测直接复用；生产入口（engine）必须显式传入
        # 场景白名单，否则一旦新增工具就会被模型看到。
        names = list(allowed_tools) if allowed_tools is not None else tools.tool_names()
        unknown = [n for n in names if n not in tools.tool_names()]
        if unknown:
            raise ValueError(f"白名单包含未注册的工具：{unknown}")
        if not names:
            raise ValueError("白名单不能为空")
        self.allowed_tools = tuple(dict.fromkeys(names))
        self.ctx = ctx
        self.client = client
        self.question = question
        self.max_steps = max(1, max_steps)
        self.traj = AgentTrajectory(question=question,
                                    model=getattr(client, "model", "") or "",
                                    allowed_tools=list(self.allowed_tools))
        self._rejects = 0
        self._decision_errors = 0

    # ---------------- 对外入口 ----------------

    def run(self) -> AgentTrajectory:
        while True:
            if len(self.traj.steps) >= self.max_steps:
                self.traj.terminate_reason = TERMINATE_MAX_STEPS
                break
            if self._rejects >= MAX_CONSECUTIVE_REJECTS:
                self.traj.terminate_reason = TERMINATE_IDLE
                break
            if self._decision_errors >= MAX_DECISION_ERRORS:
                self.traj.terminate_reason = TERMINATE_DECISION_ERROR
                break

            try:
                decision = self._decide()
            except (DecisionError, LLMError) as exc:
                self._decision_errors += 1
                self.traj.rejects.append({"kind": "decision_error",
                                          "detail": str(exc)[:200]})
                continue

            action = decision["action"]
            if action == "finish":
                self.traj.terminate_reason = TERMINATE_MODEL_FINISHED
                break

            reject = self._validate(action, decision.get("args") or {})
            if reject:
                self._rejects += 1
                self.traj.rejects.append({"kind": "invalid_action",
                                          "action": action, "detail": reject})
                continue

            self._execute(action, decision.get("args") or {}, decision.get("reason", ""))

        return self.traj

    # ---------------- 内部实现 ----------------

    def _decide(self) -> dict[str, Any]:
        """调模型取一步决策。"""
        payload = self.client.chat_json(self._system_prompt(), self._user_prompt())
        if not isinstance(payload, dict):
            raise DecisionError("决策必须是 JSON 对象")
        action = payload.get("action")
        if not isinstance(action, str) or not action.strip():
            raise DecisionError("缺少 action 字段")
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            # 理由为空说明模型没在做判断，按不合法处理
            raise DecisionError("reason 不得为空")
        args = payload.get("args")
        if args is not None and not isinstance(args, dict):
            raise DecisionError("args 必须是对象")
        return {"action": action.strip(), "args": args or {}, "reason": reason.strip()}

    def _validate(self, action: str, args: dict[str, Any]) -> Optional[str]:
        """检查动作是否可执行，返回拒绝原因或 None。"""
        if action not in self.allowed_tools:
            return f"工具 {action} 不在本场景白名单中，可选：{list(self.allowed_tools)}"
        if action in self.traj.results:
            return f"{action} 已执行过，不得重复调用"
        if args:
            # 当前六个工具都从 plan 取参数，不接受额外入参。
            # 静默忽略模型给的参数会让它误以为参数生效，宁可拒绝。
            return f"{action} 不接受参数，实际收到 {sorted(args)}"
        return None

    def _execute(self, action: str, args: dict[str, Any], reason: str) -> None:
        started_at = datetime.now().isoformat(timespec="seconds")
        res = tools.run_tool(action, self.ctx)
        self.traj.results[action] = res
        self.traj.steps.append(AgentStep(
            index=len(self.traj.steps),
            action=action, args=args, reason=reason,
            outcome=res.outcome, summary=res.summary,
            elapsed_ms=res.elapsed_ms, error=res.error,
            started_at=started_at,
        ))
        if res.outcome == "ok":
            self._rejects = 0        # 有实质进展则清零，避免累计抖动误判为空转

    # ---------------- 提示词 ----------------

    def _system_prompt(self) -> str:
        catalog = "\n".join(
            f"- {t['name']}：{t['description']} 适用：{t['when_to_use']}"
            for t in tools.describe_tools() if t["name"] in self.allowed_tools)
        return (
            "你是时序分析的执行决策器。每一步从工具清单中选择一个尚未执行的分析工具，"
            "或宣布结束。你会看到此前各步的执行结果摘要。\n\n"
            f"可用工具：\n{catalog}\n\n"
            "规则：\n"
            "1. 只输出 JSON 对象，不要解释性文字或代码块标记。\n"
            "2. 输出格式：{\"action\": \"工具名或 finish\", "
            "\"args\": {}, \"reason\": \"一句话理由\"}。\n"
            "3. action 必须是清单中的工具名，或字符串 finish；不得重复已执行过的工具。\n"
            "4. args 一律留空对象；这些工具不接受额外参数。\n"
            "5. reason 必须说明为什么选它，或为什么结束；不得为空。\n"
            "6. 当已覆盖问题所需的分析时，输出 finish——不必跑完全部工具。\n"
            "7. 若已有结果之间存在矛盾或某列未被覆盖而问题涉及它，应补跑相应工具。"
        )

    def _user_prompt(self) -> str:
        brief = self._data_brief()
        done = self.traj.steps
        lines = [
            f"分析问题：{self.question}",
            f"数据概况：{json.dumps(brief, ensure_ascii=False)}",
            f"已用步数：{len(done)}/{self.max_steps}",
        ]
        if done:
            lines.append("已完成（含结果摘要）：")
            for s in done:
                summary = json.dumps(s.summary, ensure_ascii=False)
                lines.append(f"- {s.action}（{s.outcome}）：{summary}")
        else:
            lines.append("已完成：无")
        pending = [t for t in self.allowed_tools if t not in self.traj.results]
        lines.append(f"尚未执行：{pending}")
        if self.traj.rejects:
            last = self.traj.rejects[-1]
            lines.append(f"上一次动作被拒：{last.get('detail')}。请改选其他工具。")
        lines.append("请给出下一步动作。")
        return "\n".join(lines)

    def _data_brief(self) -> dict[str, Any]:
        """数据概况。只给结构信息，不含任何数值序列。"""
        frame = self.ctx.clean.frame
        plan = self.ctx.plan
        return {
            "scene": plan.scene_id,
            "rows": int(len(frame)),
            "columns": [str(c) for c in plan.target_columns],
            "sampling": str(frame.index.freqstr) if getattr(frame.index, "freqstr", None) else "unknown",
        }


def build_agent_runner(ctx: tools.ToolContext, question: str, env: Optional[dict[str, str]] = None,
                       max_steps: int = DEFAULT_MAX_STEPS,
                       allowed_tools: Optional[list[str] | tuple[str, ...]] = None) -> Optional[AgentRunner]:
    """按环境变量构造运行时；无模型配置返回 None（调用方应回退固定流水线）。"""
    import os

    from .llm_client import client_from_env

    env = env if env is not None else os.environ
    client = client_from_env(env)
    if client is None:
        return None
    return AgentRunner(ctx, client, question, max_steps=max_steps,
                       allowed_tools=allowed_tools)
