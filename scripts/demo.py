# -*- coding: utf-8 -*-
"""一键演示：同一份数据、同一个问题，分别跑 pipeline 与 agent。

用法：
    python scripts/demo.py
    python scripts/demo.py --mode pipeline
    python scripts/demo.py --mode agent --input fixtures/weather_eval_v2.csv
    python scripts/demo.py --question "分析温度的趋势、季节性和异常点"

设计边界：
- 只调用既有 `run_analysis`，不改变任何分析逻辑或阈值；
- 缺模型时 `--mode both` 明确跳过 agent，`--mode agent` 直接失败；
- 结果写到 `artifacts/demo/`（可被 .gitignore 忽略），摘要同时落 JSON 与 Markdown。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from typing import Any, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from mtsa.engine import run_analysis  # noqa: E402
from mtsa.models import AnalysisRequest, DEFAULT_SEED  # noqa: E402

DEFAULT_INPUT = os.path.join(ROOT, "fixtures", "weather_v1.csv")
DEFAULT_QUESTION = "分析温度的趋势、季节性和异常点，并检测变化点"
DEFAULT_OUT = os.path.join(ROOT, "artifacts", "demo")
MODEL_KEYS = ("MTSA_LLM_BASE_URL", "MTSA_LLM_API_KEY", "MTSA_LLM_MODEL")
CATEGORIES = ("趋势", "季节性", "异常检测", "变化点")


def has_model() -> bool:
    return all(os.environ.get(k) for k in MODEL_KEYS)


def _clip(text: Any, n: int = 120) -> str:
    raw = str(text or "").replace("\n", " ").strip()
    return raw if len(raw) <= n else raw[: n - 1] + "…"


def summarize_result(doc: dict[str, Any], mode: str, elapsed_ms: int,
                     result_path: str, report_path: Optional[str]) -> dict[str, Any]:
    """把一个 result.json 压成演示摘要（纯函数，便于单测）。"""
    data = doc.get("data") or {}
    prof = data.get("profile") or {}
    ta = prof.get("time_axis") or {}
    quality = data.get("quality") or {}
    issues = quality.get("issues") or []
    conclusions = doc.get("conclusions") or []
    evidence = doc.get("evidence") or []
    ev_ids = {e.get("evidence_id") for e in evidence}
    covered = sum(1 for c in conclusions
                  if c.get("evidence_ids") and all(i in ev_ids for i in c["evidence_ids"]))
    anomalies = data.get("anomalies") or []
    cps = data.get("change_points") or []
    by_series = Counter(a.get("column") for a in anomalies if a.get("column"))
    agent = data.get("agent") or {}

    samples = {}
    for cat in CATEGORIES:
        items = [c for c in conclusions if (c.get("category") or "") == cat]
        samples[cat] = [{"text": _clip(c.get("text"), 110),
                         "evidence_ids": (c.get("evidence_ids") or [])[:3]}
                        for c in items[:2]]

    return {
        "mode": mode,
        "status": doc.get("status"),
        "run_id": doc.get("run_id"),
        "elapsed_ms": elapsed_ms,
        "rows": prof.get("row_count"),
        "columns": prof.get("column_count"),
        "time_start": ta.get("start"),
        "time_end": ta.get("end"),
        "median_interval_seconds": ta.get("median_interval_seconds"),
        "quality_issues": len(issues),
        "quality_codes": dict(Counter(i.get("code") for i in issues if i.get("code"))),
        "conclusions": len(conclusions),
        "evidence": len(evidence),
        "evidence_coverage": (round(covered / len(conclusions), 4)
                              if conclusions else None),
        "anomalies": len(anomalies),
        "anomaly_series": len(by_series),
        "anomalies_by_series": dict(by_series.most_common(12)),
        "change_points": len(cps),
        "change_point_series": sorted({c.get("column") for c in cps if c.get("column")}),
        "change_point_samples": [
            {"column": c.get("column"), "timestamp": c.get("timestamp"),
             "kind": c.get("kind"), "statistic": c.get("statistic")}
            for c in cps[:5]],
        "conclusion_samples": samples,
        "agent": ({"steps": agent.get("step_count"),
                   "terminate_reason": agent.get("terminate_reason"),
                   "tools_run": agent.get("tools_run") or [],
                   "missing_tools": ((agent.get("coverage") or {}).get("missing_tools") or []),
                   "max_steps": agent.get("max_steps"),
                   "max_tokens": agent.get("max_tokens"),
                   "tokens": agent.get("tokens"),
                   "elapsed_ms": agent.get("elapsed_ms")}
                  if mode == "agent" else None),
        "warnings": list(doc.get("warnings") or [])[:8],
        "error": ((doc.get("error") or {}).get("code")
                  if doc.get("error") else None),
        "result_path": result_path,
        "report_path": report_path,
    }


def _fmt(v: Any) -> str:
    return "—" if v is None else str(v)


def _display_path(path: str) -> str:
    """仓库内路径显示为相对路径（正斜杠），仓库外保留绝对路径。"""
    try:
        rel = os.path.relpath(path, ROOT)
    except ValueError:
        return path
    if rel.startswith(".."):
        return path
    return rel.replace(os.sep, "/")


def _codes(codes: dict[str, int]) -> str:
    return "、".join("%s×%d" % (k, v) for k, v in codes.items()) or "无"


def _seconds(v: Any) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return str(int(f)) if f.is_integer() else "%.1f" % f


def _comparison_table(summary: dict[str, Any]) -> list[str]:
    modes = [m["mode"] for m in summary["runs"]]
    lines = ["## 结果对比", "",
             "| 项 | " + " | ".join(modes) + " |",
             "| --- |" + " --- |" * len(modes)]

    def row(label, fn):
        vals = []
        for r in summary["runs"]:
            if r.get("skipped"):
                vals.append("跳过")
            else:
                vals.append(_fmt(fn(r)))
        lines.append("| %s | %s |" % (label, " | ".join(vals)))

    row("状态", lambda r: r["status"])
    row("结论数", lambda r: r["conclusions"])
    row("证据覆盖", lambda r: ("%.2f" % r["evidence_coverage"]
                             if r["evidence_coverage"] is not None else "—"))
    row("异常点 / 系列", lambda r: "%s / %s" % (r["anomalies"], r["anomaly_series"]))
    row("变化点", lambda r: r["change_points"])
    row("耗时 ms", lambda r: r["elapsed_ms"])

    def agent_steps(r):
        if r["mode"] != "agent":
            return "—"
        a = r.get("agent") or {}
        return "%s 步（%s）" % (a.get("steps"), a.get("terminate_reason"))

    row("Agent 步数", agent_steps)
    row("Token", lambda r: (((r.get("agent") or {}).get("tokens") or {}).get("total_tokens")
                            if r["mode"] == "agent" else "—"))
    return lines


def render_md(summary: dict[str, Any]) -> str:
    lines = [
        "# 多源时序分析 Agent · 一键演示",
        "",
        "生成时间：%s" % summary["generated_at"],
        "输入：`%s`" % summary["input"],
        "问题：%s" % summary["question"],
        "场景：%s　计划：%s　模型：%s" % (
            summary["scene"], summary["planner"], summary.get("model") or "未调用"),
        "",
    ] + _comparison_table(summary)

    for r in summary["runs"]:
        lines += ["", "## %s" % r["mode"], ""]
        if r.get("skipped"):
            lines.append("跳过：%s" % r.get("reason"))
            continue
        if r["status"] != "succeeded":
            lines.append("失败：`%s`（详见 result.json）" % (r.get("error") or "UNKNOWN"))
            lines += ["", "- result.json：`%s`" % r["result_path"]]
            continue
        lines += [
            "- 数据：%s 行 × %s 列；%s → %s；中位采样 %s 秒" % (
                r["rows"], r["columns"], r["time_start"], r["time_end"],
                _seconds(r["median_interval_seconds"])),
            "- 质量：%s 个问题（%s）" % (r["quality_issues"], _codes(r["quality_codes"])),
            "- 结论：%s 条；证据：%s 条；证据覆盖 %s" % (
                r["conclusions"], r["evidence"],
                "—" if r["evidence_coverage"] is None else "%.2f" % r["evidence_coverage"]),
            "- 异常：%s 个点，覆盖 %s 个系列；变化点：%s 个" % (
                r["anomalies"], r["anomaly_series"], r["change_points"]),
        ]
        if r.get("agent"):
            a = r["agent"]
            lines.append("- Agent：%s 步；终止 `%s`；已执行 %s；未执行 %s" % (
                a.get("steps"), a.get("terminate_reason"),
                ", ".join(a.get("tools_run") or []) or "—",
                ", ".join(a.get("missing_tools") or []) or "无"))
        for cat in CATEGORIES:
            items = (r.get("conclusion_samples") or {}).get(cat) or []
            if items:
                lines.append("- %s：%s" % (cat, items[0]["text"]))
        if r.get("change_point_samples"):
            cps = r["change_point_samples"][:3]
            lines.append("- 变化点示例：%s" % "；".join(
                "%s %s" % (c.get("column"), c.get("timestamp")) for c in cps))
        lines += ["", "产物：", "",
                  "- `%s`" % r["result_path"],
                  "- `%s`" % r["report_path"] if r.get("report_path") else "- （无 report.md）"]

    lines += ["", "## 复现", "", "```bash", summary["command"], "```", "",
              "> demo 只做汇总展示，不改任何检测阈值；指标定义见 `doc/评估设计.md`。", ""]
    return "\n".join(lines)


def _run_one(mode: str, args: argparse.Namespace, out_root: str,
             stamp: str) -> dict[str, Any]:
    out_dir = os.path.join(out_root, mode)
    os.makedirs(out_dir, exist_ok=True)
    started = time.perf_counter()
    res = run_analysis(AnalysisRequest(
        request_id="demo_%s_%s" % (mode, stamp),
        input_path=os.path.abspath(args.input),
        question=args.question,
        scene_id=args.scene,
        mode=mode,
        seed=args.seed,
        output_dir=out_dir,
        agent_scope="guided",
        agent_max_steps=args.max_steps,
        agent_max_tokens=(args.max_tokens if mode == "agent" else 0),
    ), verbose=False)
    elapsed = int((time.perf_counter() - started) * 1000)
    result_path = os.path.join(out_dir, "result.json")
    report_path = os.path.join(out_dir, "report.md")
    if os.path.exists(result_path):
        with open(result_path, encoding="utf-8") as f:
            doc = json.load(f)
    else:
        doc = {"status": getattr(res, "status", "failed"),
               "run_id": getattr(res, "run_id", None),
               "error": {"code": "NO_RESULT_JSON"}}
    summary = summarize_result(
        doc, mode, elapsed, os.path.relpath(result_path, ROOT).replace(os.sep, "/"),
        (os.path.relpath(report_path, ROOT).replace(os.sep, "/")
         if os.path.exists(report_path) else None))
    summary["elapsed_total_ms"] = elapsed
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="多源时序分析 Agent 一键演示")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument("--scene", default="weather", choices=("weather", "stock"))
    ap.add_argument("--mode", default="both", choices=("pipeline", "agent", "both"))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8000,
                    help="agent 累计 token 上限；0 表示不限制")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    if not os.path.exists(args.input):
        print("[DEMO_INPUT_MISSING] 找不到输入文件：%s" % args.input, file=sys.stderr)
        return 3
    if args.scene != "weather" and args.mode in ("agent", "both"):
        print("[DEMO_SCENE_UNSUPPORTED] Agent 当前只支持 weather；"
              "股票请用 --mode pipeline", file=sys.stderr)
        return 2

    requested = ["pipeline", "agent"] if args.mode == "both" else [args.mode]
    runs: list[dict[str, Any]] = []
    skipped_agent = None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.out, exist_ok=True)

    for mode in requested:
        if mode == "agent" and not has_model():
            reason = "缺少 %s" % "/".join(k for k in MODEL_KEYS if not os.environ.get(k))
            if args.mode == "agent":
                print("[DEMO_AGENT_UNAVAILABLE] %s；请配置后重试，或改用 --mode pipeline"
                      % reason, file=sys.stderr)
                return 2
            skipped_agent = reason
            print("[跳过 agent] %s" % reason)
            runs.append({"mode": "agent", "skipped": True, "reason": reason})
            continue
        print("运行 %s ..." % mode)
        runs.append(_run_one(mode, args, args.out, stamp))

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input": _display_path(os.path.abspath(args.input)),
        "question": args.question,
        "scene": args.scene,
        "planner": os.environ.get("MTSA_PLANNER", "rule"),
        "model": os.environ.get("MTSA_LLM_MODEL") if any(
            (r.get("agent") and not r.get("skipped")) for r in runs) else None,
        "command": "python scripts/demo.py --input %s --question \"%s\" --mode %s%s" % (
            _display_path(os.path.abspath(args.input)), args.question, args.mode,
            (" --out %s" % _display_path(os.path.abspath(args.out)))
            if os.path.abspath(args.out) != os.path.abspath(DEFAULT_OUT) else ""),
        "runs": runs,
        "agent_skipped_reason": skipped_agent,
    }

    json_path = os.path.join(args.out, "demo_summary.json")
    md_path = os.path.join(args.out, "demo_summary.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    with open(md_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(render_md(summary))

    print()
    print("\n".join(_comparison_table(summary)))
    print()
    print("演示完成：%s" % ("、".join(
        ("%s=%s" % (r["mode"], "跳过" if r.get("skipped") else r.get("status")))
        for r in runs)))
    print("摘要：%s" % md_path)
    print("数据：%s" % json_path)

    if any((not r.get("skipped")) and r.get("status") != "succeeded" for r in runs):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
