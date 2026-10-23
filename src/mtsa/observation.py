# -*- coding: utf-8 -*-
"""观察摘要器（T2.3）：把当前状态压缩成模型可读、可控成本的短上下文。

为什么单独成模块
----------------
T2.2 的运行时直接在 `_user_prompt()` 里拼字符串：每一步把全部历史摘要重复一遍，
既没有长度上限，也没有「已覆盖 / 未覆盖」的结构化视图。7 步以后上下文会持续膨胀，
模型还可能因为看不到计划范围而重复选工具或过早结束。

本模块只做三件事：
1. 生成数据概况、场景、计划参数、质量问题计数；
2. 把轨迹压缩成「最近步骤详细 + 较早步骤一行」；
3. 明确列出已执行、未执行、计划分析项覆盖情况与最近一次拒绝原因。

硬约束
------
- **不包含任何原始数据数组**：只允许统计量、计数、短字符串；
- 单步摘要约 400 token 以内，7 步总量目标低于 6000 token；
- 超预算时优先压缩较早步骤，其次压缩最近步骤，最后裁剪数据概况的列清单。

token 估算用启发式（中文按字、其他按 3.2 字符/token）。它不需要与某个
tokenizer 完全一致，只需稳定、偏保守，能防止上下文无限增长。
"""
from __future__ import annotations

import json
from typing import Any, Optional, Sequence

from .models import AnalysisPlan
from .tools import ToolContext

MAX_STEP_TOKENS = 400
MAX_TOTAL_TOKENS = 6000
RECENT_STEPS = 2

# 工具覆盖的分析开关。用于回答「模型说做了统计检验，计划里的平稳性到底覆盖了吗」。
TOOL_COVERS: dict[str, tuple[str, ...]] = {
    "compute_metrics": ("trend", "volatility", "corr", "lag", "extreme",
                        "seasonal", "scenario_metrics"),
    "analyze_spectrum": ("spectrum",),
    "run_stat_tests": ("test", "stationarity", "autocorr", "normality", "randomness"),
    "detect_anomalies": ("anomaly",),
    "detect_change_points": ("changepoint",),
}


def estimate_tokens(text: str) -> int:
    """保守估算 token 数：CJK 按 1 字 1 token，其余按 3.2 字符 1 token。"""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return int(cjk + other / 3.2) + 1


def _compact(value: Any, max_items: int = 4, max_str: int = 80) -> Any:
    """把任意摘要压成短结构：列表截断、长字符串截断、字典保留前 8 个键。"""
    if isinstance(value, dict):
        out = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= 8:
                out["…"] = f"共 {len(value)} 个键"
                break
            out[k] = _compact(v, max_items, max_str)
        return out
    if isinstance(value, (list, tuple)):
        items = [_compact(v, max_items, max_str) for v in list(value)[:max_items]]
        if len(value) > max_items:
            items.append(f"…共 {len(value)} 项")
        return items
    if isinstance(value, str) and len(value) > max_str:
        return value[:max_str] + "…"
    return value


def _json(value: Any) -> str:
    return json.dumps(_compact(value), ensure_ascii=False, separators=(",", ":"))


def coverage_summary(plan: AnalysisPlan, traj, allowed_tools: Sequence[str]) -> dict[str, Any]:
    """已执行工具与计划分析项的覆盖情况。

    失败的工具有结果对象但不算覆盖；skipped 视为已处理（例如非股票场景的泄漏检查）。
    """
    allowed = list(dict.fromkeys(allowed_tools))
    executed = [t for t in allowed
                if t in traj.results and traj.results[t].outcome in ("ok", "skipped")]
    missing_tools = [t for t in allowed if t not in executed]

    planned = list(dict.fromkeys(plan.analyses))
    covered = {a for t in executed for a in TOOL_COVERS.get(t, ()) if a in planned}
    missing_analyses = [a for a in planned if a not in covered]

    return {
        "allowed_tools": allowed,
        "executed_tools": executed,
        "missing_tools": missing_tools,
        "tool_coverage": round(len(executed) / len(allowed), 4) if allowed else 0.0,
        "planned_analyses": planned,
        "covered_analyses": [a for a in planned if a in covered],
        "missing_analyses": missing_analyses,
        "analysis_coverage": round(len(covered) / len(planned), 4) if planned else 1.0,
    }


def tool_states(traj, allowed_tools: Sequence[str]) -> dict[str, str]:
    """每个白名单工具的最终状态：ok / skipped / failed / not_run。

    失败与跳过必须分开：失败是「试过但没成功」，跳过是「前置条件不满足」，
    未执行是「模型根本没选」。报告不能把三者合并成「未执行」。
    """
    return {
        name: ("not_run" if name not in traj.results else traj.results[name].outcome)
        for name in allowed_tools
    }


def _quality_brief(ctx: ToolContext) -> dict[str, Any]:
    q = ctx.quality
    if q is None:
        return {"available": False}
    issues = list(getattr(q, "issues", []) or [])
    by_sev: dict[str, int] = {}
    for iss in issues:
        sev = getattr(iss, "severity", "unknown")
        by_sev[sev] = by_sev.get(sev, 0) + 1
    codes = sorted({getattr(i, "code", "") for i in issues})
    return {"available": True, "issues": len(issues), "by_severity": by_sev,
            "codes": codes[:8], "codes_total": len(codes)}


def _data_brief(ctx: ToolContext, max_columns: int = 8) -> dict[str, Any]:
    frame = ctx.clean.frame
    plan = ctx.plan
    cols = [str(c) for c in plan.target_columns]
    sampling = "unknown"
    freqstr = getattr(frame.index, "freqstr", None)
    if freqstr:
        sampling = str(freqstr)
    params = {k: v for k, v in plan.parameters.items()
              if k in ("seasonal_period", "rolling_window", "max_lag",
                       "anomaly_threshold", "change_point_sse_ratio", "close_column")}
    return {
        "scene": plan.scene_id,
        "confidence": plan.scene_confidence,
        "rows": int(len(frame)),
        "columns": cols[:max_columns] + ([f"…共 {len(cols)} 列"] if len(cols) > max_columns else []),
        "sampling": sampling,
        "quality": _quality_brief(ctx),
        "plan": {"analyses": list(plan.analyses), "parameters": params},
    }


def _step_digest(step) -> str:
    """较早步骤的一行摘要：只保留最关键的计数。"""
    s = step.summary or {}
    keys = ("count", "result_count", "test_count", "issue_count", "generic_metrics",
            "scenario_metrics", "column")
    parts = [f"{k}={s[k]}" for k in keys if k in s]
    if not parts:
        parts = [f"summary_keys={len(s)}"]
    return "；".join(parts)


def _render(ctx: ToolContext, question: str, traj, allowed_tools: Sequence[str],
            max_steps: int, detail_count: int, max_columns: int) -> str:
    cov = coverage_summary(ctx.plan, traj, allowed_tools)
    lines = [
        f"分析问题：{question}",
        f"数据概况：{_json(_data_brief(ctx, max_columns=max_columns))}",
        f"已用步数：{traj.step_count}/{max_steps}",
        ("覆盖：已执行 {}/{} 个工具（{}）；未覆盖：{}"
         .format(len(cov["executed_tools"]), len(cov["allowed_tools"]),
                 "、".join(cov["executed_tools"]) or "无",
                 "、".join(cov["missing_tools"]) or "无")),
        ("计划分析项覆盖：{}/{}；未覆盖：{}"
         .format(len(cov["covered_analyses"]), len(cov["planned_analyses"]),
                 "、".join(cov["missing_analyses"]) or "无")),
    ]

    steps = list(traj.steps)
    recent = steps[-detail_count:] if detail_count else []
    older = steps[:-detail_count] if detail_count else steps

    if older:
        lines.append("较早步骤（已压缩）：")
        for s in older:
            lines.append(f"- {s.action}（{s.outcome}）：{_step_digest(s)}")

    if recent:
        lines.append("最近步骤（含结果摘要）：")
        for s in recent:
            body = _json(s.summary)
            line = f"- [{s.index + 1}] {s.action}（{s.outcome}, {s.elapsed_ms}ms）：{body}"
            # 单步超预算时退化为 digest，但仍保留动作与结果状态
            if estimate_tokens(line) > MAX_STEP_TOKENS:
                line = f"- [{s.index + 1}] {s.action}（{s.outcome}, {s.elapsed_ms}ms）：{_step_digest(s)}"
            lines.append(line)

    if not steps:
        lines.append("已完成：无")

    pending = [t for t in allowed_tools if t not in traj.results]
    failed = [t for t in allowed_tools
              if t in traj.results and traj.results[t].outcome == "failed"]
    lines.append(f"尚未执行：{pending or '无'}")
    if failed:
        lines.append(f"执行失败：{failed}（可换工具或结束，失败不等于没有问题）")
    if traj.rejects:
        last = traj.rejects[-1]
        lines.append(f"上一次动作被拒：{last.get('detail')}。请改选其他工具。")
    lines.append("请给出下一步动作。")
    return "\n".join(lines)


def build_observation(ctx: ToolContext, question: str, traj,
                      allowed_tools: Sequence[str], max_steps: int,
                      token_budget: int = MAX_TOTAL_TOKENS) -> str:
    """生成一步观察上下文，保证总长度受控。

    依次尝试：最近 2 步详细 → 最近 1 步详细 → 全部压缩 → 再裁剪列清单。
    任何一步先满足预算就返回，避免为「再短一点」牺牲可读性。
    """
    text = _render(ctx, question, traj, allowed_tools, max_steps,
                   detail_count=RECENT_STEPS, max_columns=8)
    if estimate_tokens(text) <= token_budget:
        return text

    text = _render(ctx, question, traj, allowed_tools, max_steps,
                   detail_count=1, max_columns=8)
    if estimate_tokens(text) <= token_budget:
        return text

    text = _render(ctx, question, traj, allowed_tools, max_steps,
                   detail_count=0, max_columns=8)
    if estimate_tokens(text) <= token_budget:
        return text

    # 最后手段：只保留前 4 列。不再继续删，否则模型无法做出有依据的决策。
    return _render(ctx, question, traj, allowed_tools, max_steps,
                   detail_count=0, max_columns=4)
