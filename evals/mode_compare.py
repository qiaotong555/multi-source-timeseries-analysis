# -*- coding: utf-8 -*-
"""T2.9 对照实验：pipeline vs agent（同一批天气用例、同一份规则计划）。

为什么固定在规则计划下对比
--------------------------
Agent 只负责分析层。若两条路径各自用 LLM 规划，差异里就混进了「计划层选了什么」，
无法归因到「分析层是否自主」。因此本实验强制 `MTSA_PLANNER=rule`：
两条模式共享同一份 12 项天气计划，差异只来自分析层。

口径
----
| 维度 | pipeline | agent |
| --- | --- | --- |
| 分析单元 | 固定 5 个（指标、频谱、检验、异常、变化点） | `step_count` |
| 规则计划覆盖率 | 12/12（固定全跑） | 已执行工具覆盖的分析项 / 12 |
| 需求覆盖率 | 1.0（全跑） | 覆盖该问题所需分析项的比例 |
| 结论 / 证据 | `conclusions` / `evidence` 条数 | 同左 |
| 耗时 | 进程内墙钟（含导入、清洗、图表、报告） | 同左 |
| token | 规则计划不调模型，记 0 | 服务端 usage；缺失记 None |

「pipeline 全跑因此覆盖率 1.0」不是优点而是代价：它与问题无关的分析也照跑。
对照的意义是把这份代价换算成时间、结论冗余和报告篇幅，而不是只比覆盖率。

真实 / Mock 必须分清
--------------------
`--mock` 用脚本客户端，只验证对照管线；结果 `source=mock`，不得当作模型结论。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from typing import Any, Callable, Optional
from unittest.mock import patch

from mtsa import engine
from mtsa.agent_runtime import AgentRunner
from mtsa.llm_client import client_from_env
from mtsa.models import AnalysisRequest

from evals.agent_eval import (CASES, RULE_PLAN_ITEMS, ScriptedClient, WEATHER_CSV,
                              _ensure_fixtures, _scripted_tools, metrics_from_result)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "fixtures")
DEFAULT_OUT = os.path.join(ROOT, "evals", "mode_compare_results.json")
DEFAULT_OUT_ROOT = os.path.join(ROOT, "artifacts", "mode_compare")

PIPELINE_ANALYSIS_UNITS = 5      # 指标、频谱、检验、异常、变化点，固定全跑


def _load_doc(out_dir: str) -> dict[str, Any]:
    with open(os.path.join(out_dir, "result.json"), encoding="utf-8") as f:
        return json.load(f)


def pipeline_metrics(doc: dict[str, Any], expect: Optional[list[str]] = None) -> dict[str, Any]:
    """从 pipeline 结果提取可对比指标。

    覆盖率按「计划即执行」记：固定流水线会跑完计划内所有分析项。
    这是设计事实，不是实测出来的优点；对照时看结论冗余与耗时。
    """
    data = doc.get("data") or {}
    plan = data.get("plan") or {}
    planned = set(plan.get("analyses") or [])
    covered = planned & set(RULE_PLAN_ITEMS)
    need = [e for e in (expect or []) if e]
    return {
        "status": doc.get("status"),
        "mode": "pipeline",
        "steps": PIPELINE_ANALYSIS_UNITS,
        "analysis_units": PIPELINE_ANALYSIS_UNITS,
        "conclusions": len(doc.get("conclusions") or []),
        "evidence": len(doc.get("evidence") or []),
        "metrics": len(data.get("metrics") or []),
        "scenario_metrics": len(data.get("scenario_metrics") or []),
        "tests": len(data.get("tests") or []),
        "spectrum": len(data.get("spectrum") or []),
        "anomalies": len(data.get("anomalies") or []),
        "change_points": len(data.get("change_points") or []),
        "rule_plan_coverage": round(len(covered) / len(RULE_PLAN_ITEMS), 4),
        "need": need,
        "need_coverage": 1.0 if need else None,
        "tokens": None,
        "token_basis": "规则计划不调用模型",
    }


def _runner_patch(client):
    def _build(ctx, question, env=None, max_steps=8, allowed_tools=None, scope_mode="guided", max_extra_steps=1):
        return AgentRunner(ctx, client, question, max_steps=max_steps,
                           allowed_tools=allowed_tools,
                           scope_mode=scope_mode, max_extra_steps=max_extra_steps)
    return patch.object(engine, "build_agent_runner", _build)


def run_one(case: dict[str, Any], mode: str, out_root: str,
            client_factory: Optional[Callable[[dict], Any]] = None) -> dict[str, Any]:
    """跑一个用例的一种模式；墙钟从进入引擎算起，两模式同口径。"""
    out_dir = os.path.join(out_root, f"{case['id']}_{mode}")
    req = AnalysisRequest(
        request_id=f"cmp_{case['id']}_{mode}",
        input_path=os.path.join(FIX, WEATHER_CSV),
        scene_id="weather",
        question=case["question"],
        output_dir=out_dir,
        mode=mode,
        agent_max_steps=8,
    )
    client = client_factory(case) if client_factory else None
    started = time.perf_counter()
    try:
        if mode == "agent" and client is not None:
            with _runner_patch(client):
                engine.run_analysis(req, verbose=False)
        else:
            engine.run_analysis(req, verbose=False)
    except Exception as exc:                      # noqa: BLE001
        return {"case": case["id"], "mode": mode, "status": "crash",
                "error": f"{type(exc).__name__}: {exc}", "wall_ms": None,
                "steps": 0, "conclusions": 0, "evidence": 0,
                "rule_plan_coverage": 0.0, "need_coverage": None,
                "need": case.get("expect") or [], "tokens": None}
    wall_ms = int((time.perf_counter() - started) * 1000)

    try:
        doc = _load_doc(out_dir)
    except (OSError, json.JSONDecodeError) as exc:
        return {"case": case["id"], "mode": mode, "status": "no_result",
                "error": str(exc), "wall_ms": wall_ms, "steps": 0,
                "conclusions": 0, "evidence": 0, "rule_plan_coverage": 0.0,
                "need_coverage": None, "need": case.get("expect") or [], "tokens": None}

    if mode == "agent":
        m = metrics_from_result(doc, case.get("expect") or [])
        # metrics_from_result 只算行为指标，报告体量要单独取
        m.update({"analysis_units": m["steps"],
                  "conclusions": len(doc.get("conclusions") or []),
                  "evidence": len(doc.get("evidence") or [])})
        data = doc.get("data") or {}
        m.update({
            "metrics": len(data.get("metrics") or []),
            "scenario_metrics": len(data.get("scenario_metrics") or []),
            "tests": len(data.get("tests") or []),
            "spectrum": len(data.get("spectrum") or []),
            "anomalies": len(data.get("anomalies") or []),
            "change_points": len(data.get("change_points") or []),
        })
    else:
        m = pipeline_metrics(doc, case.get("expect") or [])

    m.update({
        "case": case["id"],
        "mode": mode,
        "wall_ms": wall_ms,
        "token_basis": m.get("token_basis") or ("服务端 usage" if m.get("tokens") else "未返回 usage"),
    })
    return m


def compare(case_rows: list[dict[str, Any]], source: str, model: str,
            planner: str = "rule") -> dict[str, Any]:
    by_case: dict[str, dict[str, Any]] = {}
    for row in case_rows:
        by_case.setdefault(row["case"], {})[row["mode"]] = row

    pairs: list[dict[str, Any]] = []
    for case_id, modes in by_case.items():
        p, a = modes.get("pipeline"), modes.get("agent")
        if not p or not a:
            continue
        pairs.append({
            "case": case_id,
            "expect": p.get("need") or [],
            "pipeline": {k: p.get(k) for k in ("status", "steps", "conclusions", "evidence",
                                               "rule_plan_coverage", "need_coverage", "wall_ms",
                                               "metrics", "tests", "anomalies", "change_points")},
            "agent": {k: a.get(k) for k in ("status", "steps", "terminate_reason", "conclusions",
                                            "evidence", "rule_plan_coverage", "need_coverage",
                                            "wall_ms", "tokens")},
            "delta": {
                "conclusions": (a.get("conclusions") or 0) - (p.get("conclusions") or 0),
                "evidence": (a.get("evidence") or 0) - (p.get("evidence") or 0),
                "wall_ms": (a.get("wall_ms") or 0) - (p.get("wall_ms") or 0),
                "steps": (a.get("steps") or 0) - (p.get("steps") or 0),
            },
        })

    def _mean(rows: list[dict], key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.mean(vals), 2) if vals else None

    ps = [x["pipeline"] for x in pairs]
    ags = [x["agent"] for x in pairs]
    p_ok = sum(1 for r in ps if r.get("status") == "succeeded")
    a_ok = sum(1 for r in ags if r.get("status") == "succeeded")

    need_pairs = [(x["pipeline"].get("need_coverage"), x["agent"].get("need_coverage"))
                  for x in pairs if x["agent"].get("need_coverage") is not None]
    need_parity = all((a or 0) >= (p or 0) - 1e-9 for p, a in need_pairs) if need_pairs else None

    token_rows = [x["agent"]["tokens"] for x in pairs if x["agent"].get("tokens")]
    tokens_total = sum(int(t.get("total_tokens") or 0) for t in token_rows) if token_rows else None

    checks = {
        "pipeline 全部成功": p_ok == len(pairs) and bool(pairs),
        "agent 全部成功": a_ok == len(pairs) and bool(pairs),
        "agent 需求覆盖不低于 pipeline": bool(need_parity) if need_parity is not None else False,
    }

    return {
        "source": source,
        "model": model,
        "planner": planner,
        "n_cases": len(pairs),
        "pipeline": {
            "success": p_ok,
            "mean_steps": _mean(ps, "steps"),
            "mean_conclusions": _mean(ps, "conclusions"),
            "mean_evidence": _mean(ps, "evidence"),
            "mean_wall_ms": _mean(ps, "wall_ms"),
            "mean_rule_plan_coverage": _mean(ps, "rule_plan_coverage"),
            "mean_need_coverage": _mean([r for r in ps if r.get("need_coverage") is not None],
                                        "need_coverage"),
            "tokens_total": 0,
        },
        "agent": {
            "success": a_ok,
            "mean_steps": _mean(ags, "steps"),
            "mean_conclusions": _mean(ags, "conclusions"),
            "mean_evidence": _mean(ags, "evidence"),
            "mean_wall_ms": _mean(ags, "wall_ms"),
            "mean_rule_plan_coverage": _mean(ags, "rule_plan_coverage"),
            "mean_need_coverage": _mean([r for r in ags if r.get("need_coverage") is not None],
                                        "need_coverage"),
            "tokens_total": tokens_total,
            "tokens_avg_per_run": (round(tokens_total / len(token_rows), 1)
                                   if tokens_total is not None else None),
        },
        "deltas": {
            "mean_conclusions": round((_mean(ags, "conclusions") or 0)
                                      - (_mean(ps, "conclusions") or 0), 2),
            "mean_evidence": round((_mean(ags, "evidence") or 0)
                                   - (_mean(ps, "evidence") or 0), 2),
            "mean_wall_ms": round((_mean(ags, "wall_ms") or 0)
                                  - (_mean(ps, "wall_ms") or 0), 1),
            "mean_rule_plan_coverage": round((_mean(ags, "rule_plan_coverage") or 0)
                                             - (_mean(ps, "rule_plan_coverage") or 0), 4),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "cases": pairs,
        "scope_note": ("同批 9 个天气用例、同一份 12 项规则计划；覆盖率为「分析是否执行」，"
                       "不代表检出结果的正确性"),
    }


def run_compare(cases: Optional[list[dict[str, Any]]] = None,
                out_root: str = DEFAULT_OUT_ROOT,
                client_factory: Optional[Callable[[dict], Any]] = None,
                source: str = "real", model: str = "", planner: str = "rule",
                verbose: bool = True) -> dict[str, Any]:
    _ensure_fixtures()
    cases = cases if cases is not None else CASES
    rows: list[dict[str, Any]] = []
    for case in cases:
        p = run_one(case, "pipeline", out_root)
        a = run_one(case, "agent", out_root, client_factory=client_factory)
        rows.extend([p, a])
        if verbose:
            print("  %-14s pipeline %d 结论/%d ms | agent %d 结论/%d ms 步数=%s" % (
                case["id"], p.get("conclusions") or 0, p.get("wall_ms") or 0,
                a.get("conclusions") or 0, a.get("wall_ms") or 0, a.get("steps")))
    result = compare(rows, source=source, model=model, planner=planner)
    result["generated_at"] = datetime.now().isoformat(timespec="seconds")
    result["thresholds"] = {"pipeline_success": 1.0, "agent_success": 1.0,
                            "agent_need_parity": True}
    return result


def format_table(result: dict[str, Any]) -> str:
    lines = ["| 用例 | 模式 | 步数/单元 | 结论 | 证据 | 规则计划覆盖 | 需求覆盖 | 耗时(ms) | token |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in result["cases"]:
        for mode in ("pipeline", "agent"):
            r = c[mode]
            need = "—" if r.get("need_coverage") is None else "%.2f" % r["need_coverage"]
            tok = "0" if mode == "pipeline" else (
                "—" if not r.get("tokens") else str(r["tokens"].get("total_tokens")))
            lines.append("| %s | %s | %s | %s | %s | %.2f | %s | %s | %s |" % (
                c["case"], mode, r.get("steps"), r.get("conclusions"), r.get("evidence"),
                r.get("rule_plan_coverage") or 0.0, need, r.get("wall_ms"), tok))
    p, a, d = result["pipeline"], result["agent"], result["deltas"]
    lines.append("")
    lines.append("汇总：pipeline 平均 %.1f 结论 / %.1f 证据 / %.0f ms；"
                 "agent 平均 %.1f 结论 / %.1f 证据 / %.0f ms / %.2f 步" % (
                     p["mean_conclusions"] or 0, p["mean_evidence"] or 0, p["mean_wall_ms"] or 0,
                     a["mean_conclusions"] or 0, a["mean_evidence"] or 0,
                     a["mean_wall_ms"] or 0, a["mean_steps"] or 0))
    lines.append("差值（agent - pipeline）：结论 %+.1f，证据 %+.1f，耗时 %+.0f ms，"
                 "规则计划覆盖 %+.2f；agent token 合计 %s" % (
                     d["mean_conclusions"], d["mean_evidence"], d["mean_wall_ms"],
                     d["mean_rule_plan_coverage"],
                     "—" if a["tokens_total"] is None else a["tokens_total"]))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    # Windows 控制台默认 GBK，表格里的符号会导致 UnicodeEncodeError
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                             # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="pipeline / agent 对照实验（T2.9）")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--cases", default=None, help="只跑指定用例，逗号分隔")
    ap.add_argument("--mock", action="store_true", help="脚本客户端，只验证对照管线")
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
        print("共 %d 个天气用例（pipeline 与 agent 各跑一遍）：" % len(cases))
        for c in cases:
            print("  %-14s %s" % (c["id"], c["question"]))
        return 0

    os.environ["MTSA_PLANNER"] = "rule"           # 两条模式共享同一份规则计划
    source, model, factory = "real", "", None
    if args.mock:
        source, model = "mock", "mock-scripted"
        factory = lambda case: ScriptedClient(  # noqa: E731
            _scripted_tools(case.get("expect") or [], case.get("question") or ""))
    else:
        if client_from_env() is None:
            print("未配置模型，不能产出真实对照；只验证管线请加 --mock", file=sys.stderr)
            return 2
        model = client_from_env().model
        factory = lambda case: client_from_env()  # noqa: E731

    print("=== T2.9 对照实验：pipeline vs agent ===")
    print("来源：%s；计划层：rule；用例：%d" % (
        "mock（脚本）" if args.mock else "real（真实模型）", len(cases)))
    result = run_compare(cases=cases, out_root=args.out_root,
                         client_factory=factory, source=source, model=model)
    print()
    print(format_table(result))
    print()
    for k, v in result["checks"].items():
        print("  %-24s %s" % (k, "通过" if v else "未通过"))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print()
    print("结果已写入：", args.out)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())