# -*- coding: utf-8 -*-
"""T2.11 输出稳定性评估：同一问题重复跑，步数到底会不会漂。

为什么单独成脚本
----------------
T2.8 只跑一遍，报的是「平均步数」；平均值掩盖了「同题两次跑出 3 步和 5 步」。
稳定性是可复现性的一部分：如果同一输入两次运行的分析范围不同，报告之间就
无法逐条对比，审计也失去意义。

本脚本对每个用例在指定模式下重复 N 次，统计：
- 步数 min / max / 极差；
- 结论数 min / max / 极差；
- 需求覆盖率最小值；
- 终止原因的分布。

验收口径（写进代码，不靠事后解释）
----------------------------------
1. guided 模式下，每个用例的步数极差 ≤ 1；
2. 所有非探针用例的需求覆盖率 = 1.00；
3. 终止原因不得出现 `scope_unmet`。

用法：
    python -m evals.stability_eval --repeats 3
    python -m evals.stability_eval --repeats 3 --scopes guided,free
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Optional

import mtsa.engine as engine
from mtsa.models import AnalysisRequest

from evals.agent_eval import CASES, WEATHER_CSV, _ensure_fixtures, metrics_from_result

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "fixtures")
DEFAULT_OUT = os.path.join(ROOT, "evals", "stability_results.json")
DEFAULT_OUT_ROOT = os.path.join(ROOT, "artifacts", "stability")

# 4 个有代表性的用例：单一需求、多需求、窄问题、探针。
DEFAULT_CASES = ("wx_trend", "wx_stationary", "wx_anomaly", "wx_normal")

MAX_STEP_RANGE = 1


def _load_doc(out_dir: str) -> dict[str, Any]:
    """读产物目录下的 result.json；缺失或损坏由调用方转成失败记录。"""
    with open(os.path.join(out_dir, "result.json"), encoding="utf-8") as fh:
        return json.load(fh)


def run_repeat(case: dict[str, Any], scope: str, repeat: int, out_root: str) -> dict[str, Any]:
    """跑一次并抽取稳定性需要的字段；异常不外抛，记为 crash。"""
    out_dir = os.path.join(out_root, f"{case['id']}_{scope}_r{repeat}")
    req = AnalysisRequest(
        request_id=f"stab_{case['id']}_{scope}_{repeat}",
        input_path=os.path.join(FIX, WEATHER_CSV),
        scene_id="weather",
        question=case["question"],
        output_dir=out_dir,
        mode="agent",
        agent_scope=scope,
    )
    started = time.perf_counter()
    try:
        engine.run_analysis(req, verbose=False)
    except Exception as exc:                     # noqa: BLE001
        return {"case": case["id"], "scope": scope, "repeat": repeat,
                "status": "crash", "error": f"{type(exc).__name__}: {exc}",
                "steps": None, "terminate_reason": None, "conclusions": None,
                "need_coverage": None, "wall_ms": None, "tokens": None}
    wall_ms = int((time.perf_counter() - started) * 1000)
    try:
        doc = _load_doc(out_dir)
    except (OSError, json.JSONDecodeError) as exc:
        return {"case": case["id"], "scope": scope, "repeat": repeat,
                "status": "no_result", "error": str(exc), "steps": None,
                "terminate_reason": None, "conclusions": None,
                "need_coverage": None, "wall_ms": wall_ms, "tokens": None}

    m = metrics_from_result(doc, case.get("expect") or [])
    agent = (doc.get("data") or {}).get("agent") or {}
    return {
        "case": case["id"], "scope": scope, "repeat": repeat,
        "status": doc.get("status"),
        "steps": m["steps"],
        "terminate_reason": m["terminate_reason"],
        "conclusions": len(doc.get("conclusions") or []),
        "need_coverage": m["need_coverage"],
        "need_missing": m["need_missing"],
        "scope_met": (agent.get("scope") or {}).get("met"),
        "extra_used": (agent.get("scope") or {}).get("extra_used"),
        "wall_ms": wall_ms,
        "tokens": m["tokens"],
    }


def summarize(rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    """按用例汇总极差与验收结论；纯函数，便于离线复核。"""
    cases: list[dict[str, Any]] = []
    for case_id in dict.fromkeys(r["case"] for r in rows if r["scope"] == scope):
        sub = [r for r in rows if r["scope"] == scope and r["case"] == case_id]
        steps = [r["steps"] for r in sub if isinstance(r.get("steps"), int)]
        concl = [r["conclusions"] for r in sub if isinstance(r.get("conclusions"), int)]
        needs = [r["need_coverage"] for r in sub if r.get("need_coverage") is not None]
        reasons: dict[str, int] = {}
        for r in sub:
            key = str(r.get("terminate_reason"))
            reasons[key] = reasons.get(key, 0) + 1
        step_range = (max(steps) - min(steps)) if steps else None
        cases.append({
            "case": case_id,
            "n": len(sub),
            "steps": steps,
            "steps_min": min(steps) if steps else None,
            "steps_max": max(steps) if steps else None,
            "steps_range": step_range,
            "conclusions": concl,
            "conclusions_range": (max(concl) - min(concl)) if concl else None,
            "need_coverage_min": min(needs) if needs else None,
            "terminate_reasons": reasons,
            "all_succeeded": all(r["status"] == "succeeded" for r in sub),
        })
    checks = []
    for c in cases:
        ok_step = c["steps_range"] is not None and c["steps_range"] <= MAX_STEP_RANGE
        ok_need = c["need_coverage_min"] in (None, 1.0)
        ok_reason = "scope_unmet" not in c["terminate_reasons"]
        checks.append({"case": c["case"], "steps_range_ok": ok_step,
                       "need_coverage_ok": ok_need,
                       "scope_unmet_free": ok_reason,
                       "passed": bool(ok_step and ok_need and ok_reason
                                      and c["all_succeeded"])})
    return {"scope": scope, "n_runs": sum(c["n"] for c in cases),
            "max_step_range": MAX_STEP_RANGE, "cases": cases,
            "checks": checks, "passed": all(c["passed"] for c in checks) if checks else False}


def format_table(result: dict[str, Any]) -> str:
    lines = ["| 范围 | 用例 | 步数（每次） | 极差 | 结论数（每次） | 极差 | 需求覆盖最小值 | 终止原因 | 判定 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for scope in result["scopes"]:
        for c in scope["cases"]:
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                scope["scope"], c["case"],
                "/".join(str(s) for s in c["steps"]), c["steps_range"],
                "/".join(str(s) for s in c["conclusions"]), c["conclusions_range"],
                "—" if c["need_coverage_min"] is None else "%.2f" % c["need_coverage_min"],
                "、".join(f"{k}×{v}" for k, v in c["terminate_reasons"].items()),
                "通过" if next(x for x in scope["checks"] if x["case"] == c["case"])["passed"]
                else "未通过"))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="T2.11 输出稳定性评估")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cases", default=",".join(DEFAULT_CASES))
    ap.add_argument("--scopes", default="guided,free")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    args = ap.parse_args(argv)

    from mtsa.llm_client import client_from_env
    env_client = client_from_env()
    if env_client is None:
        print("未配置模型（MTSA_LLM_BASE_URL / MTSA_LLM_API_KEY），无法产出真实结论",
              file=__import__("sys").stderr)
        return 2

    _ensure_fixtures()
    keep = [c.strip() for c in args.cases.split(",") if c.strip()]
    cases = [c for c in CASES if c["id"] in keep]
    scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]

    rows: list[dict[str, Any]] = []
    for scope in scopes:
        for case in cases:
            for i in range(1, args.repeats + 1):
                row = run_repeat(case, scope, i, args.out_root)
                rows.append(row)
                print("  %-8s %-14s r%d  steps=%s  concl=%s  %s" % (
                    scope, case["id"], i, row.get("steps"), row.get("conclusions"),
                    row.get("terminate_reason")))

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": getattr(env_client, "model", "") or "",
        "repeats": args.repeats,
        "cases": keep,
        "max_step_range": MAX_STEP_RANGE,
        "rows": rows,
        "scopes": [summarize(rows, s) for s in scopes],
    }
    result["ok"] = all(s["passed"] for s in result["scopes"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print()
    print(format_table(result))
    print()
    print("汇总：%s；明细 %s" % ("全部通过" if result["ok"] else "存在未通过项", args.out))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
