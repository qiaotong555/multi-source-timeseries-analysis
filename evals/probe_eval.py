# -*- coding: utf-8 -*-
"""D. 探针与重复：6 个开放/风险/对比问题，重复运行看范围选择与低频抖动。

设计说明（与 `spec/archive/下一阶段任务规划.md` 的差异）
------------------------------------------------
任务卡写「guided/free 各 5 次 × 6 用例」= 60 次，同时又写「总运行 ≤ 30 次」。
两者不能同时满足。本轮优先满足验收真正要的 free 步数分布：
    free 6 用例 × 5 次 = 30 次
    guided 6 用例 × 1 次 = 6 次（只做完整性/范围下限检查，不当作分布）
共 36 次，略超 30 次，原因与代价在记录中写明。

运行：
    PYTHONPATH=src python -m evals.probe_eval --plan free:5,guided:1
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Optional

from .agent_eval import ScriptedClient, _scripted_tools, run_case
from .conclusion_common import ROOT
from mtsa.llm_client import client_from_env

DEFAULT_OUT = os.path.join(ROOT, "evals", "probe_eval_results.json")
DEFAULT_OUT_ROOT = os.path.join(ROOT, "artifacts", "d_probe_eval")

# 6 个探针：开放性、风险式、对比式各 2 个。expect 为空，不参与需求覆盖率。
PROBES: list[dict[str, Any]] = [
    {"id": "p_open_temp", "kind": "open",
     "question": "帮我看看这两年的温度数据有什么值得注意的地方"},
    {"id": "p_risk_outdoor", "kind": "risk",
     "question": "这两年的天气对户外作业有什么风险"},
    {"id": "p_open_hidden", "kind": "open",
     "question": "这份数据里有哪些不明显的异常或变化？"},
    {"id": "p_risk_night", "kind": "risk",
     "question": "如果要安排夜间施工，哪些天气条件需要重点防范？"},
    {"id": "p_compare_station", "kind": "compare",
     "question": "两个站点的温度和气压变化有什么不同？"},
    {"id": "p_compare_season", "kind": "compare",
     "question": "夏季和冬季的极端天气风险有什么差别？"},
]


def parse_plan(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        scope, _, n = part.partition(":")
        if scope not in ("guided", "free"):
            raise ValueError(f"未知 scope：{scope}")
        out[scope] = int(n)
    if not out:
        raise ValueError("--plan 不能为空")
    return out


def _mean(vals: list[float]) -> Optional[float]:
    return round(statistics.mean(vals), 4) if vals else None


def _median(vals: list[float]) -> Optional[float]:
    return round(statistics.median(vals), 4) if vals else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_case[(r["base_case"], r["scope"])].append(r)

    cases: list[dict[str, Any]] = []
    for (case, scope), rs in sorted(by_case.items()):
        steps = [int(r.get("steps") or 0) for r in rs]
        concl = [int(r.get("conclusions") or 0) for r in rs]
        walls = [float(r["elapsed_ms"]) for r in rs if r.get("elapsed_ms") is not None]
        tok = [int((r.get("tokens") or {}).get("total_tokens") or 0) for r in rs if r.get("tokens")]
        freq = Counter(steps)
        cases.append({
            "base_case": case, "scope": scope, "runs": len(rs),
            "succeeded": sum(1 for r in rs if r.get("status") == "succeeded"),
            "steps_min": min(steps), "steps_median": _median(steps), "steps_max": max(steps),
            "steps_freq": {str(k): v for k, v in sorted(freq.items())},
            "steps_range": max(steps) - min(steps),
            "conclusions_mean": _mean([float(x) for x in concl]),
            "conclusions_min": min(concl), "conclusions_max": max(concl),
            "wall_ms_mean": _mean(walls),
            "tokens_total": sum(tok) if tok else None,
            "tokens_mean": _mean([float(x) for x in tok]) if tok else None,
            "terminate": dict(Counter(r.get("terminate_reason") or "unknown" for r in rs)),
            "scope_met_rate": (round(sum(1 for r in rs if r.get("scope_met") is True) / len(rs), 4)
                               if any(r.get("scope_met") is not None for r in rs) else None),
        })

    by_scope: dict[str, Any] = {}
    for scope in sorted({r["scope"] for r in rows}):
        rs = [r for r in rows if r["scope"] == scope]
        ss = [c for c in cases if c["scope"] == scope]
        steps = [int(r.get("steps") or 0) for r in rs]
        by_scope[scope] = {
            "runs": len(rs),
            "cases": len(ss),
            "succeeded": sum(1 for r in rs if r.get("status") == "succeeded"),
            "avg_steps": _mean([float(x) for x in steps]),
            "steps_min": min(steps), "steps_max": max(steps),
            "steps_freq": {str(k): v for k, v in sorted(Counter(steps).items())},
            "steps_range_by_case": {c["base_case"]: c["steps_range"] for c in ss},
            "scope_met_rate": (round(sum(1 for r in rs if r.get("scope_met") is True) / len(rs), 4)
                               if any(r.get("scope_met") is not None for r in rs) else None),
            "tokens_total": sum(int((r.get("tokens") or {}).get("total_tokens") or 0)
                                for r in rs if r.get("tokens")) or None,
        }

    checks = {}
    free = [c for c in cases if c["scope"] == "free"]
    checks["free_cases_6"] = len(free) == 6
    checks["free_repeats_5"] = all(c["runs"] == 5 for c in free) if free else False
    checks["free_all_succeeded"] = all(c["succeeded"] == c["runs"] for c in free) if free else False
    checks["guided_all_succeeded"] = all(
        c["succeeded"] == c["runs"] for c in cases if c["scope"] == "guided")
    return {"cases": cases, "by_scope": by_scope, "checks": checks,
            "passed": all(checks.values())}


def format_table(result: dict[str, Any]) -> str:
    lines = ["| 探针 | scope | 次数 | 步数 min/中位/max | 频次 | 结论均值 | token 均值 | 终止原因 |",
             "|---|---|---|---|---|---|---|---|"]
    for c in result["summary"]["cases"]:
        lines.append("| %s | %s | %d | %s / %s / %s | %s | %s | %s | %s |" % (
            c["base_case"], c["scope"], c["runs"], c["steps_min"], c["steps_median"],
            c["steps_max"], c["steps_freq"], c["conclusions_mean"],
            "—" if c["tokens_mean"] is None else "%.0f" % c["tokens_mean"],
            c["terminate"]))
    return "\n".join(lines)


def run_probe_eval(plan: dict[str, int], out_root: str, input_path: Optional[str],
                   max_tokens: int, client_factory=None, verbose: bool = True) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for scope, repeats in plan.items():
        for repeat in range(1, repeats + 1):
            for base in PROBES:
                case = {**base, "id": f"{base['id']}__{scope}_r{repeat}"}
                client = client_factory(case) if callable(client_factory) else client_factory
                row = run_case(case, out_root, client=client, do_replay=False,
                               verbose=False, input_path=input_path, scope=scope,
                               max_tokens=max_tokens)
                row.update({"base_case": base["id"], "scope": scope, "repeat": repeat,
                            "kind": base.get("kind"), "question": base["question"]})
                rows.append(row)
                if verbose:
                    print("  %-28s %s r%d 步数=%s 结论=%s 终止=%s" % (
                        base["id"], scope, repeat, row.get("steps"),
                        row.get("conclusions"), row.get("terminate_reason")))
    summary = summarize(rows)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "mock" if client_factory is not None else "real",
        "model": getattr(client_factory, "model", "") if client_factory else "",
        "plan": plan,
        "probes": PROBES,
        "out_root": os.path.relpath(out_root, ROOT),
        "input_path": (os.path.relpath(input_path, ROOT) if input_path
                       else "fixtures/weather_v1.csv"),
        "max_tokens": max_tokens,
        "summary": summary,
        "cases": rows,
        "notes": [
            "探针没有 expect，需求覆盖率记为 null，不参与覆盖率平均。",
            "free 6×5=30 次用于步数分布；guided 6×1=6 次只做完整性检查。",
            "计划卡要求 guided/free 各 5 次（60 次）与总≤30 次互相矛盾；本轮优先 free 分布，总 36 次，已在记录中写明。",
        ],
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="D. 探针与重复")
    ap.add_argument("--plan", default="free:5,guided:1",
                    help="scope:重复次数，逗号分隔；默认 free:5,guided:1")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--input", default=None)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args(argv)

    if args.dry:
        print("探针 %d 个：" % len(PROBES))
        for p in PROBES:
            print("  %-20s [%s] %s" % (p["id"], p["kind"], p["question"]))
        print("计划：", parse_plan(args.plan))
        return 0

    plan = parse_plan(args.plan)
    if args.mock:
        factory = lambda case: ScriptedClient(_scripted_tools([], case["question"]))  # noqa: E731
        source = "mock"
    else:
        if client_from_env() is None:
            print("未配置模型（MTSA_LLM_BASE_URL / MTSA_LLM_API_KEY），不能跑真实探针；"
                  "如只验证管线请加 --mock", file=sys.stderr)
            return 2
        factory = lambda case: client_from_env()  # noqa: E731
        source = "real"

    print("=== D. 探针与重复 ===")
    print("来源：%s；计划：%s；max_tokens=%d" % (source, plan, args.max_tokens))
    result = run_probe_eval(plan, args.out_root, args.input, args.max_tokens,
                            client_factory=factory, verbose=True)
    result["source"] = source
    result["model"] = (os.environ.get("MTSA_LLM_MODEL", "") if source == "real"
                       else "mock-scripted")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print()
    print(format_table(result))
    print()
    for k, v in result["summary"]["checks"].items():
        print("  %-22s %s" % (k, "通过" if v else "未通过"))
    print("结果写入：", args.out)
    return 0 if result["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())