# -*- coding: utf-8 -*-
"""股票场景 Agent 评估（E4）。

为什么单独一个评估器
--------------------
`evals/agent_eval.py` 评的是天气场景的 4 项能力（成功率 / 需求覆盖率 /
规则计划覆盖率 / 可重放率），口径与分母都是天气专用的（规则计划 12 项）。
本脚本**不重复**那 4 项，只补股票场景真正新增的风险：

> 工具没跑时，结论有没有被写成「未发现」？

股票场景的结论分两层：场景层（复权/收益/回撤/回测/量价）随 `compute_metrics`
恒算，与工具选择无关；分析层（异常/变化点）必须按 `ctx.tool_ran` 门控。
门控写反了，报告就会把「没查」说成「查过且干净」——这正是项目硬约束禁止的。

指标口径（先写标准，后跑数据）
------------------------------
| 指标 | 定义 | 判定 |
| --- | --- | --- |
| 运行成功率 | `status=succeeded` 的比例 | 1.0 |
| 门控一致率 | 对 anomaly / changepoint 逐项比对「工具是否执行」与「结论是否含未执行」 | 1.0 |
| 场景层完整率 | 收益风险 / 回测 / 复权校验 / 量价关系 四类结论齐全的比例 | 1.0 |
| 需求覆盖率 | 问题关键词要求的分析项被工具覆盖的比例 | 1.0 |
| 平均步数 | `step_count` 算术平均 | 只记录 |
| 终止原因分布 | `terminate_reason` 计数 | 只记录 |

「未执行」的判定是文本匹配，容易被措辞变化打穿，因此断言写在**结论存在性**上：
跑了工具就不允许出现「未执行X检测」，没跑就必须出现。改措辞必须同步改这里，
这条约束写进 `doc/评估设计.md`。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import mtsa.engine as engine  # noqa: E402
from mtsa.models import AnalysisRequest  # noqa: E402

STOCK_CSV = os.path.join(ROOT, "fixtures", "stock_v1.csv")
OUT_JSON = os.path.join(HERE, "agent_stock_results.json")

# 场景层四类结论：只有 compute_metrics 跑过才可能出现，缺一条说明场景层没走通
SCENARIO_CATEGORIES = ("收益风险", "回测", "复权校验", "量价关系")

# 分析层：类别 → 对应工具名
ANALYSIS_GATES = {"异常检测": "detect_anomalies", "变化点": "detect_change_points"}

CASES = [
    {
        "id": "s_return",
        "question": "这只股票的收益、回撤和量价关系如何？",
        "need": ("compute_metrics",),
    },
    {
        "id": "s_anomaly",
        "question": "有没有异常交易日？跳空多不多？",
        "need": ("compute_metrics", "detect_anomalies"),
    },
    {
        "id": "s_changepoint",
        "question": "行情在什么时候发生了转折？",
        "need": ("compute_metrics", "detect_change_points"),
    },
    {
        "id": "s_mixed",
        "question": "收益怎么样？价格序列平稳吗？有没有异常？",
        "need": ("compute_metrics", "run_stat_tests", "detect_anomalies"),
    },
]


def _run(case: dict, out_root: str) -> dict:
    out = os.path.join(out_root, case["id"])
    req = AnalysisRequest(
        request_id="eval_" + case["id"],
        input_path=STOCK_CSV,
        scene_id="stock",
        question=case["question"],
        output_dir=out,
        mode="agent",
        planner="rule",          # 固定计划，差异只来自分析层
        agent_max_steps=8,
    )
    res = engine.run_analysis(req, verbose=False)
    agent = (res.data or {}).get("agent") or {}
    cats = [c.category for c in res.conclusions]
    texts = {c.category: c.text for c in res.conclusions}
    return {
        "id": case["id"],
        "question": case["question"],
        "status": res.status,
        "error": (res.error.code if res.error else None),
        "steps": agent.get("step_count"),
        "terminate_reason": agent.get("terminate_reason"),
        "tools_run": agent.get("tools_run") or [],
        "allowed_tools": agent.get("allowed_tools") or [],
        "tool_coverage": (agent.get("coverage") or {}).get("tool_coverage"),
        "analysis_coverage": (agent.get("coverage") or {}).get("analysis_coverage"),
        "categories": cats,
        "texts": texts,
        "need": list(case["need"]),
    }


def _gate_ok(row: dict) -> tuple[bool, list[str]]:
    """门控一致性：跑了工具就不许写「未执行」，没跑就必须写「未执行」。"""
    problems = []
    for cat, tool in ANALYSIS_GATES.items():
        ran = tool in row["tools_run"]
        text = row["texts"].get(cat, "")
        if not text:
            problems.append(f"{cat}：缺结论")
            continue
        said_not_run = "未执行" in text
        if ran and said_not_run:
            problems.append(f"{cat}：工具已执行，结论却写「未执行」")
        if (not ran) and (not said_not_run):
            problems.append(f"{cat}：工具未执行，结论未声明「未执行」")
    return (not problems), problems


def evaluate(out_root: str, cases: list[dict] | None = None) -> dict:
    rows = []
    for case in (cases or CASES):
        row = _run(case, out_root)
        ok, problems = _gate_ok(row)
        row["gate_ok"] = ok
        row["problems"] = problems
        row["scenario_layer_ok"] = all(c in row["categories"] for c in SCENARIO_CATEGORIES)
        row["need_coverage"] = (
            sum(1 for t in row["need"] if t in row["tools_run"]) / len(row["need"])
            if row["need"] else 1.0
        )
        rows.append(row)

    ok_rows = [r for r in rows if r["status"] == "succeeded"]
    summary = {
        "runs": len(rows),
        "succeeded": len(ok_rows),
        "success_rate": round(len(ok_rows) / len(rows), 4) if rows else 0.0,
        "gate_consistency": round(sum(1 for r in rows if r["gate_ok"]) / len(rows), 4) if rows else 0.0,
        "scenario_layer_rate": round(
            sum(1 for r in rows if r["scenario_layer_ok"]) / len(rows), 4) if rows else 0.0,
        "need_coverage": round(
            sum(r["need_coverage"] for r in rows) / len(rows), 4) if rows else 0.0,
        "avg_steps": round(statistics.mean([r["steps"] or 0 for r in ok_rows]), 4) if ok_rows else None,
        "terminate_reasons": {},
    }
    for r in rows:
        key = r["terminate_reason"] or "none"
        summary["terminate_reasons"][key] = summary["terminate_reasons"].get(key, 0) + 1

    thresholds = {
        "success_rate": (">=", 1.0),
        "gate_consistency": (">=", 1.0),
        "scenario_layer_rate": (">=", 1.0),
        "need_coverage": (">=", 1.0),
    }
    checks = []
    for name, (op, thr) in thresholds.items():
        val = summary[name]
        ok = val >= thr if op == ">=" else val == thr
        checks.append({"name": name, "value": val, "op": op, "threshold": thr, "ok": ok})
    summary["checks"] = checks
    summary["ok"] = all(c["ok"] for c in checks)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scene": "stock",
        "mode": "agent",
        "planner": "rule",
        "source": "real",
        "input": os.path.relpath(STOCK_CSV, ROOT),
        "summary": summary,
        "cases": rows,
    }


def render_markdown(doc: dict) -> str:
    s = doc["summary"]
    lines = ["# 股票 Agent 评估（E4）", "",
             f"> 生成：{doc['generated_at']}　场景：`stock`　模式：`agent`　计划器：`rule`", "",
             "## 汇总", "",
             "| 指标 | 实测 | 阈值 | 结论 |", "| --- | --- | --- | --- |"]
    for c in s["checks"]:
        lines.append(f"| {c['name']} | {c['value']} | {c['op']} {c['threshold']} | "
                     f"{'PASS' if c['ok'] else 'FAIL'} |")
    lines += ["", f"平均步数 {s['avg_steps']}；终止原因 {s['terminate_reasons']}", "",
              "## 逐例", "",
              "| 用例 | 状态 | 步数 | 已执行工具 | 门控 | 问题 |",
              "| --- | --- | --- | --- | --- | --- |"]
    for r in doc["cases"]:
        lines.append(f"| {r['id']} | {r['status']} | {r['steps']} | "
                     f"{'、'.join(r['tools_run']) or '—'} | "
                     f"{'一致' if r['gate_ok'] else '**不一致**'} | "
                     f"{'；'.join(r['problems']) or '—'} |")
    lines += ["", "口径：「门控一致」要求 —— 工具执行过，结论不得写「未执行」；"
                  "工具没执行，结论必须写「未执行」。两条都查，防止把没查说成没发现。", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=os.path.join(ROOT, "artifacts", "agent_stock_eval"))
    ap.add_argument("--only", default="", help="逗号分隔的用例 id")
    args = ap.parse_args(argv)

    cases = CASES
    if args.only:
        keep = {x.strip() for x in args.only.split(",") if x.strip()}
        cases = [c for c in CASES if c["id"] in keep]
        if not cases:
            print("没有匹配的用例：%s" % args.only)
            return 2

    os.makedirs(args.out_root, exist_ok=True)
    doc = evaluate(args.out_root, cases)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    md_path = os.path.splitext(OUT_JSON)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(doc))

    for c in doc["summary"]["checks"]:
        print("%-22s %-8s %s %s" % (c["name"], c["value"], c["op"], c["threshold"]),
              "PASS" if c["ok"] else "FAIL")
    print()
    print("结果：%s" % OUT_JSON)
    return 0 if doc["summary"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
