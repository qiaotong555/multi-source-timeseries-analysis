# -*- coding: utf-8 -*-
"""Agent 模式评估（T2.8）。

为什么要单独一个评估器
----------------------
`evals/llm_eval.py` 评的是「计划层选了哪些分析项」，Agent 模式评的是
「分析层实际跑了哪些工具、跑了几步、为什么停、能不能重放」。
两者口径不同，混在一张表里会把「计划范围」和「执行范围」当成同一件事。

指标口径（先写标准，后跑数据）
------------------------------
| 指标 | 定义 | 用途 |
| --- | --- | --- |
| 平均步数 | `step_count` 算术平均 | 成本与绕路程度 |
| 终止原因分布 | `terminate_reason` 计数 | 区分正常结束 / 步数耗尽 / 空转 |
| 工具覆盖率 | 已执行(ok/skipped)工具 / 白名单工具 | 白名单利用率，不代表覆盖需求 |
| 计划分析项覆盖率 | 已覆盖 / **本次计划**的分析项 | 与计划层对齐程度 |
| 规则计划覆盖率 | 已覆盖 / 天气规则计划 **12 项** | 广度对照，不单独判好坏 |
| 需求覆盖率 | 已覆盖 / 该问题明确需要的分析项 | **防过早结束的主指标** |
| 轨迹可重放率 | 重放后哈希与结构化数据都一致的比例 | 可复现性 |
| 越界拒绝 | `guard_stats.violations` 与 `rejects` 条数 | 安全边界是否生效 |
| 工具失败率 | 失败步骤 / 实际尝试步骤 | 稳定性 |
| 耗时 | `data.agent.elapsed_ms` | 性能 |
| token | 服务端 usage 累计 | 成本；缺失记 None，不估算 |

两个覆盖率为什么都要报
----------------------
规则计划覆盖率的分母固定为 12，窄问题（如「是否平稳」）只需要 1 项，
用它当验收阈值必然冤枉模型；而只看「本次计划」的覆盖率又会被计划层
压缩范围掩盖——计划只列 2 项时，跑 2 个工具就是 100%。
因此验收看需求覆盖率，规则计划覆盖率只做对照与解释。

评估隔离
--------
Agent 只负责分析层，因此默认固定 `MTSA_PLANNER=rule`：两条模式共享同一份
12 项规则计划，差异才只来自分析层。用 `--planner llm` 可放开，但结论必须注明。

真实模型 / Mock 必须分清
------------------------
`--mock` 只验证指标管线与重放链路，结果里 `source=mock`，
不得当作真实模型能力结论。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime
from typing import Any, Optional
from unittest.mock import patch

from mtsa import engine
from mtsa.agent_runtime import WEATHER_AGENT_TOOLS, AgentRunner
from mtsa.llm_client import client_from_env
from mtsa.models import AnalysisRequest
from mtsa.observation import TOOL_COVERS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "fixtures")
DEFAULT_OUT = os.path.join(ROOT, "evals", "agent_eval_results.json")
DEFAULT_OUT_ROOT = os.path.join(ROOT, "artifacts", "agent_eval")

WEATHER_CSV = "weather_v1.csv"

# 天气规则计划的 12 项。它是固定参照，不随模型计划变化。
RULE_PLAN_ITEMS: tuple[str, ...] = (
    "trend", "corr", "lag", "extreme", "spectrum", "changepoint",
    "anomaly", "test", "stationarity", "autocorr", "normality", "seasonal",
)

# 13 个天气用例：11 个有明确需求项 + 2 个探针（T2.12 扩容）。
# expect 是「这个问题明显需要什么」，不是穷举；探针不参与需求覆盖率统计。
CASES: list[dict[str, Any]] = [
    {"id": "wx_trend", "question": "分析温度的趋势、季节性与极端事件",
     "expect": ["trend", "seasonal", "extreme"]},
    {"id": "wx_anomaly", "question": "检查数据中的异常点和突变更点",
     "expect": ["anomaly", "changepoint"]},
    {"id": "wx_corr", "question": "分析温度与湿度之间的相关性以及领先滞后关系",
     "expect": ["corr", "lag"]},
    {"id": "wx_stationary", "question": "评估这条时间序列是否平稳，并说明其自相关结构",
     "expect": ["stationarity", "autocorr"]},
    {"id": "wx_spectrum", "question": "温度序列的周期性与主要频率成分是什么",
     "expect": ["spectrum"]},
    {"id": "wx_change", "question": "找出温度序列中的均值突变位置",
     "expect": ["changepoint"]},
    {"id": "wx_extreme", "question": "统计极端高温事件的频率与强度",
     "expect": ["extreme"]},
    {"id": "wx_normal", "question": "检验温度数据是否服从正态分布",
     "expect": ["normality"]},
    {"id": "wx_open", "question": "帮我看看这两年的温度数据有什么值得注意的地方",
     "expect": [], "probe": True},
    # ---- T2.12 扩容：长问题、缺失数据、否定表述、第二个开放探针 ----
    # 复合问题：一句话里含 5 个需求项，用来测「长问题会不会漏项」。
    {"id": "wx_composite",
     "question": "先看温度的趋势和季节结构，再判断序列是否平稳，最后给出异常点和变化点的位置",
     "expect": ["trend", "seasonal", "stationarity", "anomaly", "changepoint"]},
    # 数据里确实有缺失段与注入异常：问题指向数据质量，期望至少覆盖异常检测。
    {"id": "wx_missing",
     "question": "数据里有缺失和异常值，请说明缺失怎么处理，并给出异常点结论",
     "expect": ["anomaly"]},
    # 否定/条件表述：不能因为「没有」二字就跳过趋势分析。
    {"id": "wx_trend_neg",
     "question": "温度有没有明显上升趋势？如果没有，请说明判断依据",
     "expect": ["trend"]},
    # 第二个开放探针：没有明确分析项，只用于观察范围选择，不参与需求覆盖率。
    {"id": "wx_risk_open",
     "question": "这两年的天气对户外作业有什么风险",
     "expect": [], "probe": True},
]

MIN_SUCCESS_RATE = 0.90
MIN_MEAN_NEED_COVERAGE = 0.80
MIN_ALL_NEEDS_RATE = 0.70
MIN_REPLAYABLE_RATE = 0.95
MAX_TOOL_FAILURE_RATE = 0.10


# ---------------- 指标计算（纯函数，便于离线复核） ----------------

def covered_analyses(executed_tools: list[str]) -> set[str]:
    """已执行工具所覆盖的分析项（与报告里的口径一致）。"""
    out: set[str] = set()
    for tool in executed_tools or []:
        out.update(TOOL_COVERS.get(tool, ()))
    return out


def metrics_from_result(doc: dict[str, Any], expect: Optional[list[str]] = None,
                        replay: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """从一份 result.json 计算单次运行指标。

    只读结果，不重算分析；因此可以离线复核历史运行，也可以直接喂给测试。
    """
    dataset = (doc.get("datasets") or [{}])[0]
    agent = (doc.get("data") or {}).get("agent") or {}
    cov = agent.get("coverage") or {}
    states = agent.get("tool_states") or {}
    allowed = list(agent.get("allowed_tools") or cov.get("allowed_tools") or [])
    executed = list(cov.get("executed_tools") or [])
    if not executed:
        # 历史结果或异常结果可能没有 coverage：按最终状态回推
        executed = [t for t, st in states.items() if st in ("ok", "skipped")]

    covered = covered_analyses(executed)
    rule_hit = covered & set(RULE_PLAN_ITEMS)
    need = [e for e in (expect or []) if e]
    need_hit = [e for e in need if e in covered]

    steps = agent.get("steps") or []
    attempted = [s for s in steps if s.get("outcome") in ("ok", "failed")]
    failed_steps = [s for s in attempted if s.get("outcome") == "failed"]

    guard = agent.get("guard_stats") or {}
    replay_block = replay if replay is not None else (agent.get("replay") or {})

    return {
        "run_id": dataset.get("run_id") or doc.get("run_id"),
        "status": doc.get("status"),
        "conclusions": len(doc.get("conclusions") or []),
        "evidence": len(doc.get("evidence") or []),
        "mode": dataset.get("mode") or agent.get("mode"),
        "steps": int(agent.get("step_count") or 0),
        "max_steps": agent.get("max_steps"),
        "terminate_reason": agent.get("terminate_reason"),
        "allowed_tools": allowed,
        "executed_tools": executed,
        "tool_coverage": (round(len(executed) / len(allowed), 4) if allowed else 0.0),
        "planned_analyses": list(cov.get("planned_analyses") or []),
        "analysis_coverage": cov.get("analysis_coverage"),
        "rule_plan_coverage": round(len(rule_hit) / len(RULE_PLAN_ITEMS), 4),
        "rule_plan_missing": [a for a in RULE_PLAN_ITEMS if a not in rule_hit],
        "need": need,
        "need_coverage": (round(len(need_hit) / len(need), 4) if need else None),
        "need_missing": [e for e in need if e not in covered],
        "attempted_steps": len(attempted),
        "failed_steps": len(failed_steps),
        "tool_failure_rate": (round(len(failed_steps) / len(attempted), 4)
                              if attempted else 0.0),
        "guard_violations": int(guard.get("violations") or 0),
        "guard_by_kind": guard.get("by_kind") or {},
        "rejects": len(agent.get("rejects") or []),
        "elapsed_ms": agent.get("elapsed_ms"),
        "tokens": agent.get("tokens") or None,
        "scope_met": (agent.get("scope") or {}).get("met"),
        "extra_used": (agent.get("scope") or {}).get("extra_used"),
        "trajectory_sha256": agent.get("trajectory_sha256"),
        "replay_ok": (None if not replay_block else bool(replay_block.get("match"))),
        "replay_source": replay_block.get("source"),
    }


def summarize(runs: list[dict[str, Any]], source: str = "real",
              model: str = "", planner: str = "rule") -> dict[str, Any]:
    """汇总；每个指标都标注分母，避免「0 个样本也算 100%」。"""
    total = len(runs)
    if total == 0:
        return {"passed": False, "reason": "无有效用例", "n_runs": 0}

    ok = [r for r in runs if r.get("status") == "succeeded"]

    def _mean(key: str, pool: list[dict[str, Any]]) -> Optional[float]:
        vals = [r[key] for r in pool if r.get(key) is not None]
        return round(statistics.mean(vals), 4) if vals else None

    def _p90(vals: list[float]) -> Optional[float]:
        if not vals:
            return None
        ordered = sorted(vals)
        idx = min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))
        return round(ordered[idx], 2)

    term: dict[str, int] = {}
    for r in runs:
        key = r.get("terminate_reason") or "unknown"
        term[key] = term.get(key, 0) + 1

    need_pool = [r for r in runs if r.get("need_coverage") is not None]
    replay_pool = [r for r in runs if r.get("replay_ok") is not None]
    elapsed = [float(r["elapsed_ms"]) for r in runs if r.get("elapsed_ms") is not None]

    token_runs = [r for r in runs if r.get("tokens")]
    total_tokens = sum(int((r["tokens"] or {}).get("total_tokens") or 0) for r in token_runs)
    prompt_tokens = sum(int((r["tokens"] or {}).get("prompt_tokens") or 0) for r in token_runs)
    completion_tokens = sum(int((r["tokens"] or {}).get("completion_tokens") or 0)
                            for r in token_runs)

    success_rate = round(len(ok) / total, 4)
    mean_need = _mean("need_coverage", need_pool)
    all_needs = (round(sum(1 for r in need_pool if r["need_coverage"] >= 1.0) / len(need_pool), 4)
                 if need_pool else None)
    replayable = (round(sum(1 for r in replay_pool if r["replay_ok"]) / len(replay_pool), 4)
                  if replay_pool else None)
    failure = _mean("tool_failure_rate", runs)

    checks = {
        f"成功率>={MIN_SUCCESS_RATE}": success_rate >= MIN_SUCCESS_RATE,
        f"需求覆盖率>={MIN_MEAN_NEED_COVERAGE}": (mean_need is not None
                                                  and mean_need >= MIN_MEAN_NEED_COVERAGE),
        f"用例全命中率>={MIN_ALL_NEEDS_RATE}": (all_needs is not None
                                                and all_needs >= MIN_ALL_NEEDS_RATE),
        f"工具失败率<={MAX_TOOL_FAILURE_RATE}": (failure is not None
                                                 and failure <= MAX_TOOL_FAILURE_RATE),
    }
    skipped: list[str] = []
    if replayable is None:
        skipped.append(f"可重放率>={MIN_REPLAYABLE_RATE}（本次未测重放）")
    else:
        checks[f"可重放率>={MIN_REPLAYABLE_RATE}"] = replayable >= MIN_REPLAYABLE_RATE

    return {
        "source": source,
        "model": model,
        "planner": planner,
        "n_runs": total,
        "n_succeeded": len(ok),
        "success_rate": success_rate,
        "avg_steps": _mean("steps", runs),
        "median_steps": (round(statistics.median([r["steps"] for r in runs]), 2)
                         if runs else None),
        "max_steps_observed": max((r.get("steps") or 0) for r in runs),
        "terminate_distribution": term,
        "mean_tool_coverage": _mean("tool_coverage", runs),
        "mean_analysis_coverage": _mean("analysis_coverage", runs),
        "mean_rule_plan_coverage": _mean("rule_plan_coverage", runs),
        "min_rule_plan_coverage": min((r.get("rule_plan_coverage") or 0.0) for r in runs),
        "mean_need_coverage": mean_need,
        "all_needs_covered_rate": all_needs,
        "need_pool_size": len(need_pool),
        "replayable_rate": replayable,
        "replay_pool_size": len(replay_pool),
        "guard_violations_total": sum(int(r.get("guard_violations") or 0) for r in runs),
        "guard_violations_per_run": round(
            sum(int(r.get("guard_violations") or 0) for r in runs) / total, 4),
        "rejects_total": sum(int(r.get("rejects") or 0) for r in runs),
        "mean_tool_failure_rate": failure,
        "mean_elapsed_ms": (round(statistics.mean(elapsed), 1) if elapsed else None),
        "p90_elapsed_ms": _p90(elapsed),
        "tokens_available_runs": len(token_runs),
        "tokens_total": (total_tokens if token_runs else None),
        "tokens_avg_per_run": (round(total_tokens / len(token_runs), 1)
                               if token_runs else None),
        "tokens_prompt_total": (prompt_tokens if token_runs else None),
        "tokens_completion_total": (completion_tokens if token_runs else None),
        "checks": checks,
        "checks_skipped": skipped,
        "passed": all(checks.values()),
        "scope_note": ("需求覆盖率只统计有明确期望项的用例；规则计划覆盖率固定以 12 项为分母，"
                       "仅作广度对照，不单独作为通过标准"),
    }


# ---------------- 运行 ----------------

def _ensure_fixtures() -> None:
    if not os.path.exists(os.path.join(FIX, WEATHER_CSV)):
        from fixtures.gen_fixtures import main as gen_main
        gen_main()


def _scripted_tools(expect: list[str], question: str = "") -> list[str]:
    """Mock 用例：按需求项挑最少工具，用来验证指标管线。

    T2.11 起 guided 模式有范围下限：脚本必须把下限工具也跑掉，否则会被
    合法地拒绝结束（scope_unmet），mock 结果就不再代表「指标管线跑通」。
    """
    wanted: set[str] = set()
    for tool, covers in TOOL_COVERS.items():
        if set(covers) & set(expect):
            wanted.add(tool)
    if not wanted:                      # 探针：给两个最通用的工具
        wanted = {"compute_metrics", "detect_anomalies"}
    if question:
        from mtsa.scope import infer_floor
        wanted |= set(infer_floor(question, WEATHER_AGENT_TOOLS).tools)
    order = [t for t in WEATHER_AGENT_TOOLS if t in wanted]
    return order or ["compute_metrics"]


class ScriptedClient:
    """离线脚本客户端：不会联网，只按预设顺序发决策。"""

    def __init__(self, tools_to_run: list[str]) -> None:
        self.model = "mock-scripted"
        self._queue = list(tools_to_run)

    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        if self._queue:
            action = self._queue.pop(0)
            return {"action": action, "args": {}, "reason": f"mock：执行 {action}"}
        return {"action": "finish", "args": {}, "reason": "mock：脚本结束"}


def _patched_runner(client):
    def _build(ctx, question, env=None, max_steps=8, allowed_tools=None, scope_mode="guided", max_extra_steps=1, max_tokens=0):
        return AgentRunner(ctx, client, question, max_steps=max_steps,
                           allowed_tools=allowed_tools,
                           scope_mode=scope_mode, max_extra_steps=max_extra_steps,
                           max_tokens=max_tokens)
    return patch.object(engine, "build_agent_runner", _build)


def run_case(case: dict[str, Any], out_root: str, client=None,
             do_replay: bool = True, verbose: bool = True,
             input_path: Optional[str] = None, scope: str = "guided",
             max_tokens: int = 0) -> dict[str, Any]:
    """跑一个用例；返回单次指标。异常不外抛，转为 status=failed 的记录。"""
    _ensure_fixtures()                  # 单独调用本函数（如测试）时也要保证数据存在
    input_file = input_path or os.path.join(FIX, WEATHER_CSV)
    out_dir = os.path.join(out_root, case["id"])
    req = AnalysisRequest(
        request_id=f"agent_eval_{case['id']}",
        input_path=input_file,
        scene_id="weather",
        question=case["question"],
        output_dir=out_dir,
        mode="agent",
        agent_scope=scope,
        agent_max_tokens=max_tokens,
    )
    # client 可以是实例，也可以是「按用例生成实例」的工厂（mock 必须每次新建，
    # 否则脚本队列会被第一个用例耗尽）。
    case_client = client(case) if callable(client) else client
    try:
        if case_client is None:
            res = engine.run_analysis(req, verbose=False)
        else:
            with _patched_runner(case_client):
                res = engine.run_analysis(req, verbose=False)
    except Exception as exc:            # noqa: BLE001  评估器不能因单个用例崩溃
        row = {"case": case["id"], "status": "crash", "error": f"{type(exc).__name__}: {exc}",
               "steps": 0, "tool_coverage": 0.0, "rule_plan_coverage": 0.0,
               "need_coverage": None, "tool_failure_rate": 0.0, "guard_violations": 0,
               "rejects": 0, "elapsed_ms": None, "terminate_reason": "crash",
               "replay_ok": None, "tokens": None}
        if verbose:
            print("  %-14s 崩溃 %s" % (case["id"], row["error"]))
        return row

    doc_path = os.path.join(out_dir, "result.json")
    try:
        with open(doc_path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        row = {"case": case["id"], "status": "no_result", "error": str(exc),
               "steps": 0, "tool_coverage": 0.0, "rule_plan_coverage": 0.0,
               "need_coverage": None, "tool_failure_rate": 0.0, "guard_violations": 0,
               "rejects": 0, "elapsed_ms": None, "terminate_reason": "no_result",
               "replay_ok": None, "tokens": None}
        if verbose:
            print("  %-14s 无结果文件 %s" % (case["id"], exc))
        return row

    replay_block = None
    if do_replay and doc.get("status") == "succeeded":
        rep = engine.run_analysis(AnalysisRequest(
            request_id=f"agent_eval_replay_{case['id']}",
            input_path=input_file,
            scene_id="weather",
            question=case["question"],
            output_dir=os.path.join(out_dir, "replay"),
            mode="agent",                  # 重放只接受 agent 结果（引擎侧硬校验）
            agent_scope=scope,
            agent_max_tokens=max_tokens,
            replay_from=doc_path,
        ), verbose=False)
        replay_block = ((rep.data or {}).get("agent") or {}).get("replay") or {"match": False}

    row = metrics_from_result(doc, case.get("expect") or [], replay_block)
    row["case"] = case["id"]
    row["question"] = case["question"]
    if verbose:
        cov = "—" if row["need_coverage"] is None else "%.2f" % row["need_coverage"]
        flag = "OK " if doc.get("status") == "succeeded" else "FAIL"
        rep = "—" if row["replay_ok"] is None else ("一致" if row["replay_ok"] else "不一致")
        print("  %-14s %s 步数=%d 需求覆盖=%-5s 规则计划=%.2f 重放=%s 用时=%sms" % (
            case["id"], flag, row["steps"], cov, row["rule_plan_coverage"], rep,
            row["elapsed_ms"]))
    return row


def run_eval(cases: Optional[list[dict[str, Any]]] = None, out_root: str = DEFAULT_OUT_ROOT,
             client=None, do_replay: bool = True, source: str = "real",
             model: str = "", planner: str = "rule", verbose: bool = True,
             input_path: Optional[str] = None, scope: str = "guided",
             max_tokens: int = 0) -> dict[str, Any]:
    _ensure_fixtures()
    cases = cases if cases is not None else CASES
    runs = [run_case(c, out_root, client=client, do_replay=do_replay,
                     verbose=verbose, input_path=input_path, scope=scope,
                     max_tokens=max_tokens)
            for c in cases]
    summary = summarize(runs, source=source, model=model, planner=planner)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "model": model,
        "planner": planner,
        "rule_plan_items": list(RULE_PLAN_ITEMS),
        "cases": runs,
        "input_path": (os.path.relpath(input_path, ROOT) if input_path
                        else os.path.join("fixtures", WEATHER_CSV)),
        "scope": scope,
        "max_tokens": max_tokens,
        "summary": summary,
        "thresholds": {
            "min_success_rate": MIN_SUCCESS_RATE,
            "min_mean_need_coverage": MIN_MEAN_NEED_COVERAGE,
            "min_all_needs_rate": MIN_ALL_NEEDS_RATE,
            "min_replayable_rate": MIN_REPLAYABLE_RATE,
            "max_tool_failure_rate": MAX_TOOL_FAILURE_RATE,
        },
    }


def format_table(result: dict[str, Any]) -> str:
    lines = ["| 用例 | 状态 | 步数 | 终止原因 | 需求覆盖 | 规则计划覆盖 | 工具失败 | 越界 | 重放 | 耗时(ms) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["cases"]:
        need = "—（探针）" if r.get("need_coverage") is None else "%.2f" % r["need_coverage"]
        rep = "—" if r.get("replay_ok") is None else ("一致" if r["replay_ok"] else "不一致")
        lines.append("| %s | %s | %s | %s | %s | %.2f | %s | %s | %s | %s |" % (
            r.get("case"), r.get("status"), r.get("steps"), r.get("terminate_reason"),
            need, r.get("rule_plan_coverage") or 0.0,
            "—" if r.get("tool_failure_rate") is None else "%.2f" % r["tool_failure_rate"],
            r.get("guard_violations"), rep,
            "—" if r.get("elapsed_ms") is None else r["elapsed_ms"]))
    s = result["summary"]
    lines.append("")
    lines.append("汇总：用例 %d，成功 %d，平均步数 %s，需求覆盖率 %s，规则计划覆盖率 %s，"
                 "可重放率 %s，越界 %d 次，token 总量 %s" % (
                     s["n_runs"], s["n_succeeded"], s["avg_steps"],
                     "—" if s["mean_need_coverage"] is None else "%.2f" % s["mean_need_coverage"],
                     "—" if s["mean_rule_plan_coverage"] is None
                     else "%.2f" % s["mean_rule_plan_coverage"],
                     "—" if s["replayable_rate"] is None else "%.2f" % s["replayable_rate"],
                     s["guard_violations_total"],
                     "—" if s["tokens_total"] is None else s["tokens_total"]))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Agent 模式评估（T2.8）")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--cases", default=None, help="只跑指定用例，逗号分隔")
    ap.add_argument("--input", default=None,
                    help="输入 CSV；默认 fixtures/weather_v1.csv，用于第二份数据")
    ap.add_argument("--agent-scope", choices=("guided", "free"), default="guided",
                    help="范围策略；默认 guided")
    ap.add_argument("--agent-max-tokens", type=int, default=0,
                    help="Agent 累计 token 上限；0=不限制")
    ap.add_argument("--mock", action="store_true",
                    help="离线脚本客户端：只验证指标管线，不代表模型能力")
    ap.add_argument("--no-replay", action="store_true", help="跳过重放，不算可重放率")
    ap.add_argument("--planner", choices=("rule", "llm"), default="rule",
                    help="计划层口径，默认 rule：两条模式共用同一份 12 项计划")
    ap.add_argument("--dry", action="store_true", help="只列用例")
    args = ap.parse_args(argv)

    cases = CASES
    if args.cases:
        keep = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in CASES if c["id"] in keep]
        if not cases:
            print("没有匹配的用例", file=sys.stderr)
            return 2
    if args.dry:
        print("共 %d 个天气用例：" % len(cases))
        for c in cases:
            print("  %-14s %s  需求=%s" % (
                c["id"], c["question"], "、".join(c["expect"]) or "（探针）"))
        return 0

    if args.planner == "rule":
        os.environ["MTSA_PLANNER"] = "rule"

    print("=== Agent 模式评估（T2.8） ===")
    print("来源：%s；计划层：%s；重放：%s" % (
        "mock（离线脚本）" if args.mock else "real（真实模型）",
        args.planner, "关闭" if args.no_replay else "开启"))

    client = None
    source = "real"
    model = ""
    if args.mock:
        source = "mock"
        model = "mock-scripted"
        client = lambda case: ScriptedClient(  # noqa: E731
            _scripted_tools(case.get("expect") or [], case.get("question") or ""))
    else:
        env_client = client_from_env()
        if env_client is None:
            print("未配置模型（MTSA_LLM_BASE_URL / MTSA_LLM_API_KEY），"
                  "不能产出真实模型结论；如只验证指标管线请加 --mock", file=sys.stderr)
            return 2
        # 每个用例独立客户端：usage 是客户端生命周期内的累计值，
        # 复用同一个实例会把 token 跨用例累加，单例成本就失去意义。
        client = lambda case: client_from_env()  # noqa: E731
        model = getattr(env_client, "model", "") or ""

    result = run_eval(cases=cases, out_root=args.out_root, client=client,
                      do_replay=not args.no_replay, source=source,
                      model=model, planner=args.planner, verbose=True,
                      input_path=args.input, scope=args.agent_scope,
                      max_tokens=args.agent_max_tokens)
    print()
    print(format_table(result))
    print()
    for k, v in result["summary"]["checks"].items():
        print("  %-22s %s" % (k, "通过" if v else "未通过"))
    for note in result["summary"]["checks_skipped"]:
        print("  跳过：%s" % note)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print()
    print("结果已写入：", args.out)
    return 0 if result["summary"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())