# -*- coding: utf-8 -*-
"""证据时效检查：每个评估产物是否「覆盖了当前代码」。

背景：2026-11-03 发现 `evals/eval_results.json` 停在 10-25，而 11-02/11-03 改过变化点判据，
「结论级变化点F1」两项还停留在 0.5（修完是 1.0）。产物本身没写错，只是**没人知道它过期了**。
README 的状态表逐行引用了这些文件，于是整张表都在悄悄失真。

判据刻意保守：**产物文件的 mtime 必须晚于它依赖的源码的最后修改时间**。
保守意味着会有假阳性（改了与某指标无关的代码也会被标过期），但不会漏掉「代码变了、证据没跟」。

用法：
    python scripts/evidence_freshness.py            # 打印表格，有过期项则退出码 1
    python scripts/evidence_freshness.py --quiet    # 只返回退出码
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC = "src/mtsa/**/*.py"
EVALS = "evals/*.py"

# (证据文件, 它依赖的源码 glob, 一句话说明)
# 只列 README 与文档实际引用的产物；未列出的不判，避免把一次性实验产物也算进来。
EVIDENCE: list[tuple[str, tuple[str, ...], str]] = [
    ("evals/test_logs/summary.json", (SRC, "tests/*.py"), "工程基线（测试汇总）"),
    ("evals/robustness_results.json", (SRC, EVALS), "鲁棒性 37/37"),
    ("evals/conclusion_gate_v2_results.json", (SRC, EVALS), "结论门禁 24/24"),
    ("evals/eval_results.json", (SRC, EVALS), "22 项规则指标"),
    ("evals/real_data_results.json", (SRC, EVALS), "真实数据 14 项验收"),
    ("evals/d3_results.json", (SRC, EVALS), "数据扩容 D3"),
    ("evals/forecast_results.json", (SRC, EVALS), "预测 F3–F6"),
    ("evals/mode_compare_results.json", (SRC, "src/mtsa/agent*.py", EVALS), "Agent 对照 13 用例"),
    ("evals/llm_eval_results.json", ("src/mtsa/planner.py", "src/mtsa/scenarios/*.py",
                                     "evals/llm_eval.py"), "LLM 计划指标"),
    ("evals/agent_eval_results.json", ("src/mtsa/agent_runtime.py", "src/mtsa/tools.py",
                                       "src/mtsa/scenarios/*.py"), "Agent 指标"),
    ("evals/holdout_results.json", (SRC, "evals/holdout_eval.py"), "holdout 三折"),
    ("evals/stability_results.json", (SRC, "evals/stability_eval.py"), "稳定性"),
    ("evals/probe_eval_results.json", (SRC, "evals/probe_eval.py"), "探针"),
    ("evals/conclusion_eval_results.json", (SRC, "evals/conclusion_eval.py"), "结论评估"),
    ("evals/benchmark_results.json", (SRC, "evals/benchmark.py"), "规模与性能"),
]


def _max_mtime(patterns: tuple[str, ...], root: str = ROOT) -> tuple[float, str]:
    """给定 glob 集合里最新的 mtime 与文件名。"""
    best, who = 0.0, "—"
    for pat in patterns:
        for path in glob.glob(os.path.join(root, pat), recursive=True):
            try:
                mt = os.path.getmtime(path)
            except OSError:
                continue
            if mt > best:
                best, who = mt, os.path.relpath(path, root).replace(os.sep, "/")
    return best, who


def _fmt(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "—"


def check(evidence: list[tuple[str, tuple[str, ...], str]] = EVIDENCE,
          root: str = ROOT, tol_hours: float = 0.0) -> list[dict]:
    rows = []
    for rel, patterns, label in evidence:
        path = os.path.join(root, rel)
        src_mt, src_who = _max_mtime(patterns, root)
        if not os.path.exists(path):
            rows.append({"file": rel, "label": label, "status": "missing",
                         "mtime": 0.0, "source_mtime": src_mt, "source": src_who,
                         "lag_hours": None})
            continue
        mt = os.path.getmtime(path)
        # 容差：刚生成完又顺手改了一行源码（或被编辑器重写）时，分钟级落后不该报红。
        # 默认 0（严格）；要用宽松口径时显式传，别把默认值调松。
        stale = mt < src_mt - tol_hours * 3600.0
        rows.append({
            "file": rel, "label": label,
            "status": "stale" if stale else "fresh",
            "mtime": mt, "source_mtime": src_mt, "source": src_who,
            "lag_hours": round((src_mt - mt) / 3600.0, 1) if stale else 0.0,
        })
    return rows


def render(rows: list[dict]) -> str:
    out = ["| 证据 | 用途 | 生成时间 | 源码最新 | 判定 |",
           "| --- | --- | --- | --- | --- |"]
    for r in rows:
        if r["status"] == "missing":
            verdict = "**缺失**"
        elif r["status"] == "stale":
            verdict = "**过期**（源码新 %.1f h）" % r["lag_hours"]
        else:
            verdict = "新鲜"
        out.append("| `%s` | %s | %s | %s | %s |" % (
            r["file"], r["label"], _fmt(r["mtime"]), _fmt(r["source_mtime"]), verdict))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="检查评估产物是否覆盖当前代码")
    ap.add_argument("--quiet", action="store_true", help="只返回退出码，不打印表格")
    ap.add_argument("--tol-hours", type=float, default=0.0,
                    help="容忍多少小时以内的落后（默认 0，严格）")
    args = ap.parse_args(argv)

    rows = check(tol_hours=args.tol_hours)
    if not args.quiet:
        print(render(rows))
        stale = [r for r in rows if r["status"] == "stale"]
        missing = [r for r in rows if r["status"] == "missing"]
        print()
        print("新鲜 %d / 过期 %d / 缺失 %d（共 %d）" % (
            len(rows) - len(stale) - len(missing), len(stale), len(missing), len(rows)))
        if stale:
            print("过期清单：")
            for r in stale:
                print("  - %s（%s，落后 %.1f h）" % (r["file"], r["label"], r["lag_hours"]))
        print()
        print("判据：产物 mtime 必须晚于其依赖源码的最后修改时间。"
              "保守口径——改到无关代码也会被标过期，但不会漏掉「代码变了、证据没跟」。")
    return 0 if not any(r["status"] in ("stale", "missing") for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
