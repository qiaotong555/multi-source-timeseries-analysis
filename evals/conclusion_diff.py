# -*- coding: utf-8 -*-
"""B. 结论差异溯源：把 agent 比 pipeline 少掉的结论逐类归因。

只比较同一用例、同一份规则计划下的两份 `result.json`。每条「少了 N 条」必须
落入以下三类之一，否则记 unattributed，并按验收要求把比例压到 ≤ 5%：

1. tool_not_run：该类别对应的工具在 agent 轨迹里根本没执行；
2. no_detection：工具执行了，但异常/变化点结果为空；
3. plan_or_parameter_diff：工具执行了且不是空结果，但结论条数仍不同
   （计划列范围、参数或收尾层裁剪差异）。

运行：
    PYTHONPATH=src python -m evals.conclusion_diff
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Optional

from .conclusion_common import (
    CONCL_CATEGORY_TO_TOOL, DEFAULT_COMPARE_ROOT, ROOT, conclusion_counter,
    discover_result_dirs, executed_tools, load_json, sha1_file,
)

DEFAULT_OUT = os.path.join(ROOT, "evals", "conclusion_diff_results.json")
REASONS = ("tool_not_run", "no_detection", "plan_or_parameter_diff")


def _tool_for(category: str) -> Optional[str]:
    return CONCL_CATEGORY_TO_TOOL.get(category)


def attribute_case(case: str, p_doc: dict[str, Any], a_doc: dict[str, Any]) -> dict[str, Any]:
    p_counts = conclusion_counter(p_doc)
    a_counts = conclusion_counter(a_doc)
    a_tools = set(executed_tools(a_doc, "agent"))
    a_anom = ((a_doc.get("data") or {}).get("anomalies") or [])
    a_cp = ((a_doc.get("data") or {}).get("change_points") or [])
    p_metrics = len(((p_doc.get("data") or {}).get("metrics") or []))
    a_metrics = len(((a_doc.get("data") or {}).get("metrics") or []))
    p_spectrum = len(((p_doc.get("data") or {}).get("spectrum") or []))
    a_spectrum = len(((a_doc.get("data") or {}).get("spectrum") or []))

    rows: list[dict[str, Any]] = []
    for category in sorted(set(p_counts) | set(a_counts)):
        pc, ac = int(p_counts.get(category, 0)), int(a_counts.get(category, 0))
        missing, extra = max(0, pc - ac), max(0, ac - pc)
        reason = ""
        detail = ""
        tool = _tool_for(category)
        if missing:
            if category == "数据口径":
                reason, detail = "plan_or_parameter_diff", "预处理口径结论；两模式应一致，差异按参数/裁剪处理"
            elif tool and tool not in a_tools:
                reason, detail = "tool_not_run", f"agent 未执行 {tool}"
            elif category == "异常检测" and not a_anom:
                reason, detail = "no_detection", "detect_anomalies 已执行，但 anomalies 为空"
            elif category == "变化点" and not a_cp:
                reason, detail = "no_detection", "detect_change_points 已执行，但 change_points 为空"
            elif tool and tool in a_tools:
                reason = "plan_or_parameter_diff"
                if category == "周期":
                    detail = f"spectrum 条数 pipeline={p_spectrum} / agent={a_spectrum}"
                else:
                    detail = f"metrics 条数 pipeline={p_metrics} / agent={a_metrics}"
            else:
                reason, detail = "unattributed", "没有可对应的工具或检测结果"
        rows.append({
            "category": category, "pipeline": pc, "agent": ac,
            "missing": missing, "extra": extra,
            "tool": tool or "—", "reason": reason or ("agent_more" if extra else ""),
            "detail": detail,
        })

    total_missing = sum(r["missing"] for r in rows)
    unattributed = sum(r["missing"] for r in rows if r["reason"] == "unattributed")
    return {
        "case": case,
        "pipeline_total": sum(p_counts.values()),
        "agent_total": sum(a_counts.values()),
        "missing": total_missing,
        "extra": sum(r["extra"] for r in rows),
        "unattributed": unattributed,
        "executed_tools": sorted(a_tools),
        "categories": rows,
    }


def build_result(result_root: str = DEFAULT_COMPARE_ROOT) -> dict[str, Any]:
    items = discover_result_dirs(result_root)
    if not items:
        raise FileNotFoundError(
            f"没有在 {result_root} 找到结果目录；请先运行 evals.mode_compare")
    by: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    paths: dict[tuple[str, str], str] = {}
    for it in items:
        by[it["case"]][it["mode"]] = load_json(it["path"])
        paths[(it["case"], it["mode"])] = it["path"]

    cases: list[dict[str, Any]] = []
    for case in sorted(by):
        if "pipeline" not in by[case] or "agent" not in by[case]:
            continue
        row = attribute_case(case, by[case]["pipeline"], by[case]["agent"])
        row["pipeline_path"] = os.path.relpath(paths[(case, "pipeline")], ROOT)
        row["agent_path"] = os.path.relpath(paths[(case, "agent")], ROOT)
        cases.append(row)

    category = defaultdict(lambda: {"pipeline": 0, "agent": 0, "missing": 0,
                                    "tool_not_run": 0, "no_detection": 0,
                                    "plan_or_parameter_diff": 0, "unattributed": 0,
                                    "extra": 0})
    reasons = Counter()
    for c in cases:
        for r in c["categories"]:
            x = category[r["category"]]
            x["pipeline"] += r["pipeline"]
            x["agent"] += r["agent"]
            x["missing"] += r["missing"]
            x["extra"] += r["extra"]
            if r["missing"] and r["reason"] in x:
                x[r["reason"]] += r["missing"]
            if r["missing"]:
                reasons[r["reason"]] += r["missing"]

    total_p = sum(c["pipeline_total"] for c in cases)
    total_a = sum(c["agent_total"] for c in cases)
    total_missing = sum(c["missing"] for c in cases)
    unattributed = sum(c["unattributed"] for c in cases)
    reason_total = sum(reasons.values())
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "existing_results",
        "result_root": os.path.relpath(result_root, ROOT),
        "n_cases": len(cases),
        "summary": {
            "pipeline_total": total_p,
            "agent_total": total_a,
            "missing_total": total_missing,
            "mean_missing_per_case": round(total_missing / len(cases), 4) if cases else None,
            "extra_total": sum(c["extra"] for c in cases),
            "reasons": dict(reasons),
            "reason_total": reason_total,
            "unattributed": unattributed,
            "unattributed_rate": round(unattributed / total_missing, 4) if total_missing else 0.0,
            "attribution_complete": (unattributed / total_missing <= 0.05) if total_missing else True,
        },
        "category_matrix": dict(sorted(category.items())),
        "cases": cases,
        "notes": [
            "13 个用例共用同一份 fixture；总量是 13 次重复测量，不是 13 份独立数据。",
            "tool_not_run 表示 agent 没有执行该类结论对应的工具，不代表工具算错。",
            "no_detection 只用于异常/变化点；工具执行但结果为空。",
            "plan_or_parameter_diff 表示工具已执行且非空，但条数仍不同；需结合 metrics/spectrum 条数看。",
        ],
    }


def format_table(result: dict[str, Any]) -> str:
    lines = ["| 类别 | pipeline | agent | 差值 | 主要归因 | 无法归因 |",
             "|---|---|---|---|---|---|"]
    for cat, x in result["category_matrix"].items():
        diff = x["agent"] - x["pipeline"]
        reasons = {k: x[k] for k in REASONS if x[k]}
        main = "、".join("%s %d" % (k, v) for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])) or "—"
        lines.append("| %s | %d | %d | %+d | %s | %d |" % (
            cat, x["pipeline"], x["agent"], diff, main, x["unattributed"]))
    s = result["summary"]
    lines.append("")
    lines.append("总差异 %d 条，无法归因 %d 条（%.2f%%）；平均每用例少 %.2f 条。" % (
        s["missing_total"], s["unattributed"], 100 * s["unattributed_rate"],
        s["mean_missing_per_case"] or 0.0))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="B. 结论差异溯源")
    ap.add_argument("--result-root", default=DEFAULT_COMPARE_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    result = build_result(args.result_root)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    if not args.quiet:
        print("=== B. 结论差异溯源 ===")
        print(format_table(result))
        print("结果写入：", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())