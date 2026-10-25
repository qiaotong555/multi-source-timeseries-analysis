# -*- coding: utf-8 -*-
"""评估表补充三项指标（C2）：结构化输出有效率 / 证据支持率 / 费用。

口径写死在这里，避免每次重新解释：

1. **结构化输出有效率**＝有明确 `status` 的结果数 ÷ 运行总数。
   「有明确 status」等价于「result.json 能被解析成 AnalysisResult」——解析不了的
   运行在评估里根本不会有 status。它不是「业务成功」，失败运行也算一次有效结构化输出。
2. **证据支持率**＝结论中 `evidence_ids` 非空且全部能在 evidence 里解析到的比例。
   与 `evals/real_data_eval.py::evidence_coverage` 同一口径（结构可追溯，不是语义支持）。
3. **费用**＝按 `prices/` 的价目表把 token 换算成 CNY 的**估算**。
   历史运行用的是 `deepseek-chat`，本项目没有它的价目表；这里按 `deepseek-flash`
   高峰价折算，只用于量级参考，不能当账单。

只读 `evals/*.json` 的历史记录，不重跑任何分析。
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"
OUT_JSON = EVALS / "table_metrics.json"
PRICE_TABLE = ROOT / "prices" / "deepseek-flash.peak.json"

sys.path.insert(0, str(ROOT / "src"))
from mtsa import cost  # noqa: E402


def _load(name: str):
    path = EVALS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _status_counts(statuses: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in statuses:
        out[s] = out.get(s, 0) + 1
    return dict(sorted(out.items()))


def collect() -> dict:
    rows: list[dict] = []

    # ---- A. agent_eval：13 次 agent 运行，带 token 明细 ----
    a = _load("agent_eval_results.json")
    if a:
        cases = a.get("cases") or []
        statuses = [str(c.get("status")) for c in cases]
        rows.append({
            "source": "agent_eval",
            "runs": len(cases),
            "status": _status_counts(statuses),
            "structured_valid": sum(1 for s in statuses if s and s != "None"),
        })

    # ---- B. mode_compare：13 用例 × 2 模式 ----
    b = _load("mode_compare_results.json")
    if b:
        cases = b.get("cases") or []
        statuses, cons, evid = [], 0, 0
        for case in cases:
            for mode in ("pipeline", "agent"):
                block = case.get(mode) or {}
                if not block:
                    continue
                statuses.append(str(block.get("status")))
                cons += int(block.get("conclusions") or 0)
                evid += int(block.get("evidence") or 0)
        rows.append({
            "source": "mode_compare",
            "runs": len(statuses),
            "status": _status_counts(statuses),
            "structured_valid": sum(1 for s in statuses if s and s != "None"),
            "conclusions": cons, "supported": evid,
        })

    # ---- C. real_data：80 次真实数据运行 ----
    c = _load("real_data_results.json")
    if c:
        matrix = c.get("matrix") or {}
        statuses = [str(v) for v in matrix.values()]
        rows.append({
            "source": "real_data",
            "runs": len(statuses),
            "status": _status_counts(statuses),
            "structured_valid": sum(1 for s in statuses if s and s not in ("None", "missing")),
            "evidence_coverage": {
                "pipeline": (c.get("macro") or {}).get("by_mode.pipeline.evidence_coverage"),
                "agent": (c.get("macro") or {}).get("by_mode.agent.evidence_coverage"),
            },
        })

    # ---- D. llm_eval：规划层 JSON 合法率（单列，不与运行级混算）----
    d = _load("llm_eval_results.json")
    planner = None
    if d:
        s = d.get("summary") or {}
        planner = {"n_cases": s.get("n_cases"), "n_valid": s.get("n_valid"),
                   "valid_rate": s.get("valid_rate"),
                   "fallback_rate": s.get("fallback_rate")}

    total_runs = sum(r["runs"] for r in rows)
    total_valid = sum(r["structured_valid"] for r in rows)

    # ---- 费用：按 agent_eval 的 token 明细估算 ----
    fee = None
    if a and PRICE_TABLE.exists():
        table = cost.load_price_table(str(PRICE_TABLE))
        totals, per_run = 0.0, []
        for case in a.get("cases") or []:
            tokens = case.get("tokens")
            if not tokens:
                continue
            try:
                item = cost.estimate_cost(tokens, table)
            except cost.CostError:
                continue
            totals += item["total"]
            per_run.append(item["total"])
        if per_run:
            fee = {
                "runs": len(per_run),
                "total_cny": round(totals, 4),
                "mean_cny": round(totals / len(per_run), 4),
                "price_table": str(PRICE_TABLE.relative_to(ROOT)),
                "tier": table.tier, "effective_date": table.effective_date,
                "estimated": True,
                "caveat": ("历史运行模型为 deepseek-chat，项目没有它的价目表；"
                           "按 deepseek-flash 高峰价折算，仅作量级参考，不是账单"),
            }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "definitions": {
            "structured_output_valid_rate": "有明确 status 的结果数 / 运行总数（解析失败不会有 status）",
            "evidence_support_rate": "结论中 evidence_ids 非空且全部可解析的比例",
            "cost_cny": "按 prices/ 价目表把 token 折算成 CNY 的估算，不是账单",
        },
        "structured_output_valid_rate": {
            "valid": total_valid, "runs": total_runs,
            "rate": round(total_valid / total_runs, 4) if total_runs else None,
        },
        "evidence_support_rate": {
            "mode_compare_pooled": (
                round(rows[1]["supported"] / rows[1]["conclusions"], 4)
                if len(rows) > 1 and rows[1].get("conclusions") else None),
            "real_data_by_mode": rows[-1].get("evidence_coverage") if rows and rows[-1]["source"] == "real_data" else None,
        },
        "cost": fee,
        "planner_json": planner,
        "by_source": rows,
    }


def main() -> int:
    doc = collect()
    OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("| 指标 | 值 | 口径 |")
    print("| --- | --- | --- |")
    s = doc["structured_output_valid_rate"]
    print(f"| 结构化输出有效率 | {s['valid']}/{s['runs']} = {s['rate']:.4f} | 有明确 status 的结果数 / 运行总数 |")
    e = doc["evidence_support_rate"]
    mc = e.get("mode_compare_pooled")
    rd = e.get("real_data_by_mode") or {}
    print(f"| 证据支持率 | 对照实验 {mc if mc is not None else '—'}；"
          f"真实数据 pipeline {rd.get('pipeline')} / agent {rd.get('agent')} | "
          "结论 evidence_ids 非空且全部可解析 |")
    if doc["cost"]:
        f = doc["cost"]
        print(f"| 费用（估算） | 13 次共 {f['total_cny']} CNY，均值 {f['mean_cny']} CNY/次 | "
              f"{f['price_table']}（{f['tier']}，{f['effective_date']}） |")
    if doc["planner_json"]:
        p = doc["planner_json"]
        print(f"| 规划层 JSON 合法率 | {p['n_valid']}/{p['n_cases']} = {p['valid_rate']} | "
              "单独统计，不与运行级混算 |")
    print(f"\n写出：{OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
