# -*- coding: utf-8 -*-
"""方案 2：独立数据上的结论级门禁。

在 weather_eval_v2 上跑 4 个「全量分析」问题 × pipeline/agent。
点级异常、变化点、趋势与季节沿用 conclusion_eval 的口径；异常事件改为按标签
event_id 合并真值、按 3 小时合并检出，再套用运行前冻结的阈值。

阈值不是从这份数据调的：见 spec/当前开发任务.md「方案 2」任务书。

运行：
    PYTHONPATH=src python -m evals.conclusion_gate_eval
    PYTHONPATH=src python -m evals.conclusion_gate_eval --pipeline-only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from typing import Any, Optional

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for _p in (SRC, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evals.conclusion_eval import evaluate_run, summarize  # noqa: E402
from evals.conclusion_common import (  # noqa: E402
    load_labels, load_truth, match_events, merge_events, series_key,
    truth_anomalies, truth_change_points,
)
from evals.split_eval import rule_planner_env  # noqa: E402
from mtsa.engine import run_analysis  # noqa: E402
from mtsa.models import AnalysisRequest  # noqa: E402

FIX = os.path.join(ROOT, "fixtures")
DATA = os.path.join(FIX, "weather_eval_v2.csv")
LABELS = os.path.join(FIX, "labels_weather_eval_v2.csv")
TRUTH = os.path.join(FIX, "truth_weather_eval_v2.json")
OUT_ROOT = os.path.join(ROOT, "artifacts", "gate_v2")
OUT_JSON = os.path.join(ROOT, "evals", "conclusion_gate_v2_results.json")
OUT_MD = os.path.join(ROOT, "evals", "conclusion_gate_v2_report.md")

CASES = [
    {"id": "v2_full_1", "question": "分析温度的趋势、季节性和异常点，并检测变化点",
     "expect": ["trend", "seasonal", "anomaly", "changepoint"]},
    {"id": "v2_full_2", "question": "请全面检查这份天气数据的趋势、周期、异常与结构变化",
     "expect": ["trend", "spectrum", "anomaly", "changepoint"]},
    {"id": "v2_full_3", "question": "找出温度湿度的异常、气压的突变，并说明季节规律",
     "expect": ["anomaly", "changepoint", "seasonal"]},
    {"id": "v2_full_4", "question": "做一次完整分析：趋势、季节性、异常点、变化点",
     "expect": ["trend", "seasonal", "anomaly", "changepoint"]},
]

# 运行前冻结。不得在看到 v2 结果后调整。
THRESHOLDS = {
    "anomaly_event_recall": 0.80,
    "anomaly_event_f1": 0.40,
    "changepoint_f1": 0.50,
    "trend_accuracy": 0.80,
    "seasonal_accuracy": 0.80,
    "need_coverage": 0.90,
    "evidence_coverage": 1.00,
    "success_rate": 0.90,
}
AGENT_MARGIN = 0.10
EVENT_GAP_SECONDS = 3 * 3600
SCORED_RATE_MIN = 0.90  # 附加护栏：准确率必须建立在足够的已评分样本上


def _truth_events(labels):
    """按 event_id 把真值异常点合并成事件。

    点异常每个 event_id 只有一个点；持续段一个 event_id 有 8 个点。
    不能把每个真值点都当成一个事件，否则 8 小时持续段会被算成 8 个
    独立事件，事件口径与检出侧不一致。
    """
    groups = {}
    for x in labels:
        if x.get("kind") != "anomaly":
            continue
        eid = x.get("event_id") or x.get("timestamp")
        key = (series_key(x.get("column", ""), x.get("station")), str(eid))
        groups.setdefault(key, []).append(str(x["timestamp"]))
    out = []
    for (series, eid), stamps in groups.items():
        ts = sorted(pd.Timestamp(t.replace(" ", "T")) for t in stamps)
        out.append({"series": series, "event_id": eid,
                    "start": ts[0].isoformat(), "end": ts[-1].isoformat(),
                    "count": len(ts)})
    return out


def _detected_events(doc, gap_seconds=EVENT_GAP_SECONDS):
    data = doc.get("data") or {}
    pts = []
    for x in data.get("anomalies") or []:
        col = x.get("column")
        ts = x.get("timestamp")
        if col and ts:
            pts.append({"series": series_key(col, x.get("station")), "timestamp": ts})
    return merge_events(pts, gap_seconds)


def _event_metrics(rows, labels, gap_seconds=EVENT_GAP_SECONDS):
    """按每个用例重算事件级 P/R/F1，未执行的用例计入召回分母。

    rows 需要带 `_detected_events`（由主流程从 result.json 提取）。
    """
    truth = _truth_events(labels)
    n_truth = len(truth)
    scored = [r for r in rows if r["anomaly"]["state"] == "scored"]
    not_run = len(rows) - len(scored)

    tp = fp = fn = 0
    per_run = []
    for r in scored:
        m = match_events(r.get("_detected_events") or [], truth)
        tp += int(m["tp"])
        fp += int(m["fp"])
        fn += int(m["fn"])
        per_run.append({"case": r["case"], "detected": m["detected"],
                        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                        "precision": m["precision"],
                        "recall": m["recall"], "f1": m["f1"]})

    unseen = not_run * n_truth
    recall_adj = _ratio(tp, tp + fn + unseen)
    precision = _ratio(tp, tp + fp)
    return {
        "truth_per_run": n_truth,
        "scored_runs": len(scored),
        "not_run_runs": not_run,
        "not_run_truth_instances": unseen,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall_adj,
        "f1": _f1(precision, recall_adj),
        "per_run": per_run,
    }


def _rel(path: str) -> str:
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def _has_model() -> bool:
    return all(os.environ.get(k) for k in ("MTSA_LLM_BASE_URL", "MTSA_LLM_API_KEY", "MTSA_LLM_MODEL"))


def _evidence_coverage(doc: dict[str, Any]) -> float:
    cs = doc.get("conclusions") or []
    ids = {e.get("evidence_id") for e in (doc.get("evidence") or [])}
    if not cs:
        return 0.0
    ok = sum(1 for c in cs if c.get("evidence_ids") and all(i in ids for i in c["evidence_ids"]))
    return round(ok / len(cs), 4)


def _run_one(case: dict[str, Any], mode: str, max_tokens: int) -> dict[str, Any]:
    out = os.path.join(OUT_ROOT, f"{case['id']}_{mode}")
    os.makedirs(out, exist_ok=True)
    result_path = os.path.join(out, "result.json")
    if os.path.exists(result_path):
        os.remove(result_path)
    req = AnalysisRequest(
        request_id=f"gate_{case['id']}_{mode}", input_path=DATA, scene_id="weather",
        question=case["question"], mode=mode, seed=20260927, output_dir=out,
        agent_scope="guided", agent_max_tokens=(max_tokens if mode == "agent" else 0),
    )
    with rule_planner_env("rule"):
        res = run_analysis(req, verbose=False)
    if not os.path.exists(result_path):
        raise RuntimeError(f"{case['id']}/{mode}: 没有 result.json（status={res.status}）")
    with open(result_path, encoding="utf-8") as f:
        doc = json.load(f)
    doc["_elapsed_ms"] = (res.data.get("agent") or {}).get("elapsed_ms") if mode == "agent" else None
    return doc


def _adjust(by_mode: dict[str, Any], rows: list[dict[str, Any]],
            labels: list[dict[str, Any]]) -> None:
    """用真实事件口径重算异常事件指标，并把「未执行」计入召回分母。"""
    for mode in ("pipeline", "agent"):
        m = by_mode[mode]
        rs = [r for r in rows if r["mode"] == mode]
        a = m["anomaly"]
        ev = _event_metrics(rs, labels)
        a["event"] = {k: ev[k] for k in (
            "truth_per_run", "scored_runs", "not_run_runs",
            "not_run_truth_instances", "tp", "fp", "fn",
            "precision", "recall", "f1", "per_run")}
        a["event_recall_adjusted"] = ev["recall"]
        a["event_f1_adjusted"] = ev["f1"]
        a["event_ci"] = {
            "precision": _wilson(ev["tp"], ev["tp"] + ev["fp"]),
            "recall": _wilson(ev["tp"], ev["tp"] + ev["fn"] + ev["not_run_truth_instances"]),
        }

        c = m["changepoint"]
        cp_per_run = (int(c["point"].get("truth") or 0) // c["scored_runs"]
                      if c["scored_runs"] else 0)
        c["recall_adjusted"] = _ratio(
            c["point"].get("tp"),
            (c["point"].get("tp") or 0) + (c["point"].get("fn") or 0)
            + cp_per_run * c["not_run_runs"],
        )
        c["f1_adjusted"] = _f1(c["point"].get("precision"), c["recall_adjusted"])
        c["ci"] = {
            "precision": _wilson(c["point"].get("tp") or 0,
                                 (c["point"].get("tp") or 0) + (c["point"].get("fp") or 0)),
            "recall": _wilson(c["point"].get("tp") or 0,
                              (c["point"].get("tp") or 0) + (c["point"].get("fn") or 0)
                              + cp_per_run * c["not_run_runs"]),
        }

        for key in ("trend", "seasonal"):
            x = m[key]
            total = sum(x["states"].values())
            x["scored_rate"] = _ratio(x["scored"], total)
            x["accuracy_ci"] = _wilson(x["correct"], x["scored"])

        cn = m["conclusion_need"]
        cn["coverage_ci"] = _wilson(cn["covered"], cn["mapped"])


def _wilson(k: int, n: int, z: float = 1.96) -> Optional[dict[str, Any]]:
    """二项比例的 Wilson 95% 置信区间；n=0 返回 None，不猜。"""
    k, n = int(k or 0), int(n or 0)
    if n <= 0:
        return None
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"k": k, "n": n, "value": round(p, 4),
            "lo": round(max(0.0, center - half), 4),
            "hi": round(min(1.0, center + half), 4)}


def _ratio(num, den) -> Optional[float]:
    if den in (0, None) or num is None:
        return None
    return round(float(num) / float(den), 4)


def _f1(p, r) -> Optional[float]:
    if p is None or r is None or (p + r) == 0:
        return None
    return round(2 * p * r / (p + r), 4)


def _mean(xs: list[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 4) if xs else None


def _check(name: str, value: Optional[float], threshold: float, op: str = ">=") -> dict[str, Any]:
    if value is None:
        ok = False
    elif op == ">=":
        ok = value >= threshold
    else:
        ok = value <= threshold
    return {"name": name, "value": value, "threshold": threshold, "op": op, "ok": ok}


def _gate(by_mode: dict[str, Any], evidence: dict[str, float]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for mode in ("pipeline", "agent"):
        m = by_mode[mode]
        checks += [
            _check(f"{mode}.anomaly_event_recall", m["anomaly"]["event_recall_adjusted"],
                   THRESHOLDS["anomaly_event_recall"]),
            _check(f"{mode}.anomaly_event_f1", m["anomaly"]["event_f1_adjusted"],
                   THRESHOLDS["anomaly_event_f1"]),
            _check(f"{mode}.changepoint_f1", m["changepoint"]["f1_adjusted"],
                   THRESHOLDS["changepoint_f1"]),
            _check(f"{mode}.trend_accuracy", m["trend"]["accuracy"],
                   THRESHOLDS["trend_accuracy"]),
            _check(f"{mode}.seasonal_accuracy", m["seasonal"]["accuracy"],
                   THRESHOLDS["seasonal_accuracy"]),
            _check(f"{mode}.trend_scored_rate", m["trend"]["scored_rate"], SCORED_RATE_MIN),
            _check(f"{mode}.seasonal_scored_rate", m["seasonal"]["scored_rate"], SCORED_RATE_MIN),
            _check(f"{mode}.need_coverage", m["conclusion_need"]["coverage"],
                   THRESHOLDS["need_coverage"]),
            _check(f"{mode}.evidence_coverage", evidence.get(mode),
                   THRESHOLDS["evidence_coverage"]),
            _check(f"{mode}.success_rate",
                   _ratio(m["succeeded"], m["runs"]), THRESHOLDS["success_rate"]),
        ]
    p, a = by_mode["pipeline"], by_mode["agent"]
    for key, label in (
        (("anomaly", "event_f1_adjusted"), "anomaly_event_f1"),
        (("changepoint", "f1_adjusted"), "changepoint_f1"),
        (("trend", "accuracy"), "trend_accuracy"),
        (("seasonal", "accuracy"), "seasonal_accuracy"),
    ):
        pv = p[key[0]].get(key[1])
        av = a[key[0]].get(key[1])
        threshold = (pv - AGENT_MARGIN) if pv is not None else None
        checks.append({
            "name": f"agent_non_inferior.{label}", "value": av,
            "threshold": threshold, "op": ">=", "ok": av is not None and threshold is not None and av >= threshold,
        })
    return {"checks": checks, "passed": all(c["ok"] for c in checks)}


def _dataset_spec(labels: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    kinds = {}
    for x in labels:
        kinds[x.get("kind")] = kinds.get(x.get("kind"), 0) + 1
    return {
        "rows": int(len(df)),
        "columns": [c for c in df.columns if c not in ("timestamp", "station")],
        "stations": sorted(str(x) for x in df["station"].dropna().unique()),
        "start": str(pd.to_datetime(df["timestamp"]).min()),
        "end": str(pd.to_datetime(df["timestamp"]).max()),
        "label_kinds": kinds,
        "anomaly_events": len(_truth_events(labels)),
    }


def _fmt(v, ci=None) -> str:
    if v is None:
        return "—"
    out = "%.4f" % v
    if ci:
        out += " [%.3f, %.3f]" % (ci["lo"], ci["hi"])
    return out


def _fmt_metric(m: dict[str, Any]) -> str:
    if not m or m.get("precision") is None:
        return "—"
    return "%.3f / %.3f / %.3f" % (m["precision"], m["recall"], m["f1"])


def render_md(doc: dict[str, Any]) -> str:
    spec = doc.get("dataset_spec") or {}
    kinds = spec.get("label_kinds") or {}
    lines = [
        "# 方案 2：独立结论门禁报告",
        "",
        "生成时间：%s" % doc["generated_at"],
        "数据：`%s`（SHA1 `%s`）" % (doc["dataset"], doc["dataset_sha1"]),
        "标签：`%s`（SHA1 `%s`）" % (doc["labels"], doc["labels_sha1"]),
        "真值：`%s`（SHA1 `%s`）" % (doc["truth"], doc["truth_sha1"]),
        "阈值：运行前冻结；agent 非劣于 pipeline − %.2f。" % AGENT_MARGIN,
        "",
        "## 数据规格",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        "| 行数 | %s |" % spec.get("rows"),
        "| 站点 | %s |" % ", ".join(spec.get("stations") or []),
        "| 列 | %s |" % ", ".join(spec.get("columns") or []),
        "| 时间范围 | %s → %s |" % (spec.get("start"), spec.get("end")),
        "| 异常点 / 异常事件 | %s / %s |" % (kinds.get("anomaly"), spec.get("anomaly_events")),
        "| 变化点 | %s |" % kinds.get("level_shift_start"),
        "| 缺失缺口标签 | %s |" % kinds.get("missing_gap_start"),
        "",
        "> v2 是独立于旧 fixture 的新数据，但生成器在构造阶段经过修订"
        "（修复缺失遮蔽、去掉变化点叠加、把注入幅度调整到可检测区间）。"
        "因此本报告验证的是**固定版 v2** 上的表现，不能宣称是完全未看过的隐藏盲测。"
        "检测算法与阈值均未因本数据修改。",
        "",
        "## 结论",
        "",
        "**%s**" % ("通过" if doc["passed"] else "未通过"),
        "",
    ]
    failed = [c for c in doc["checks"]["checks"] if not c["ok"]]
    if failed:
        lines += ["未通过项：", ""]
        for c in failed:
            lines.append("- `%s`：实际 %s，要求 %s %s" % (
                c["name"], "—" if c["value"] is None else "%.4f" % c["value"],
                c["op"], c["threshold"]))
        lines.append("")
    if doc["checks"].get("reason"):
        lines += ["> %s" % doc["checks"]["reason"], ""]

    ties = [c for c in doc["checks"]["checks"] if c["value"] is not None
            and c["threshold"] is not None and c["threshold"] < 1.0
            and abs(c["value"] - c["threshold"]) < 1e-9]
    notes = []
    if ties:
        notes.append("压线通过（无余量）：%s。" % "、".join(
            "%s = %.4f（阈值 %.2f）" % (c["name"], c["value"], c["threshold"]) for c in ties))
    if "agent" in doc["modes"]:
        agents = [r for r in doc["runs"] if r["mode"] == "agent"]
        terms = Counter(r.get("terminate_reason") for r in agents)
        notes.append("agent 终止原因：%s。" % "，".join(
            "%s %d 次" % (k, v) for k, v in sorted(terms.items())))
        if any(k == "budget_exhausted" for k in terms):
            notes.append("`budget_exhausted` 指 guided 模式下限之外的**追加步数**用尽，"
                         "不是 token 预算耗尽（见 `rejects[].kind=scope_exceeded` 与 `tokens`）。")
        misses = Counter(t for r in agents for t in (r.get("missing_tools") or []))
        if misses:
            notes.append("agent 未执行工具：%s。结论级指标相同不等于工具覆盖相同，"
                         "应结合 `data.agent.coverage` 阅读。" % "，".join(
                             "%s %d 次" % (k, v) for k, v in sorted(misses.items())))
    if notes:
        lines += ["读数注意：", ""] + ["- %s" % n for n in notes] + [""]

    modes = doc["modes"]
    lines += ["## 分模式指标", "",
              "| 指标 | " + " | ".join(modes) + " |",
              "| --- |" + " --- |" * len(modes)]

    def per_mode(fn):
        return [fn(doc["summary"]["by_mode"][m]) for m in modes]

    rows = [
        ("异常事件 Recall（未执行计入分母）",
         per_mode(lambda m: m["anomaly"]["event_recall_adjusted"]),
         per_mode(lambda m: (m["anomaly"].get("event_ci") or {}).get("recall"))),
        ("异常事件 Precision",
         per_mode(lambda m: m["anomaly"]["event"].get("precision")),
         per_mode(lambda m: (m["anomaly"].get("event_ci") or {}).get("precision"))),
        ("异常事件 F1（未执行计入分母）",
         per_mode(lambda m: m["anomaly"]["event_f1_adjusted"]), [None] * len(modes)),
        ("异常点级 F1（参考，无门槛）",
         per_mode(lambda m: m["anomaly"]["point"].get("f1")), [None] * len(modes)),
        ("变化点 F1（未执行计入分母）",
         per_mode(lambda m: m["changepoint"]["f1_adjusted"]), [None] * len(modes)),
        ("趋势正确率", per_mode(lambda m: m["trend"]["accuracy"]),
         per_mode(lambda m: m["trend"].get("accuracy_ci"))),
        ("季节正确率", per_mode(lambda m: m["seasonal"]["accuracy"]),
         per_mode(lambda m: m["seasonal"].get("accuracy_ci"))),
        ("趋势已评分率", per_mode(lambda m: m["trend"]["scored_rate"]), [None] * len(modes)),
        ("季节已评分率", per_mode(lambda m: m["seasonal"]["scored_rate"]), [None] * len(modes)),
        ("结论需求覆盖率", per_mode(lambda m: m["conclusion_need"]["coverage"]),
         per_mode(lambda m: m["conclusion_need"].get("coverage_ci"))),
        ("证据覆盖率", [doc["evidence_coverage"].get(m) for m in modes],
         [doc["evidence_coverage_ci"].get(m) for m in modes]),
        ("成功率", per_mode(lambda m: m["success_rate"]),
         per_mode(lambda m: m.get("success_rate_ci"))),
    ]
    for name, values, cis in rows:
        lines.append("| %s | %s |" % (
            name, " | ".join(_fmt(v, c) for v, c in zip(values, cis))))
    lines += ["", "置信区间为 Wilson 95%；F1 由 Precision/Recall 区间端点不自洽，故只给点估计。", ""]

    lines += ["## 逐用例", "",
              "| 用例 | 模式 | 状态 | 步数 | 终止 | 结论数 | 证据覆盖 | 需求覆盖 | 异常点 TP/FP/FN | 异常事件 TP/FP/FN | 变化点 TP/FP/FN |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in doc["runs"]:
        def trio(m):
            if not m:
                return "—"
            return "%s/%s/%s" % (m.get("tp"), m.get("fp"), m.get("fn"))
        lines.append("| %s | %s | %s | %s | %s | %s | %.3f | %s | %s | %s | %s |" % (
            r["case"], r["mode"], r.get("status"), r.get("steps") or "—",
            r.get("terminate_reason") or "—", r.get("conclusion_count") or 0,
            r.get("evidence_coverage") or 0.0,
            "—" if r.get("need_coverage") is None else "%.2f" % r["need_coverage"],
            trio(r.get("anomaly_point")), trio(r.get("anomaly_event")),
            trio(r.get("changepoint_point"))))

    lines += ["", "## 阈值检查", "", "| 检查 | 实际 | 阈值 | 结果 |", "| --- | --- | --- | --- |"]
    for c in doc["checks"]["checks"]:
        lines.append("| %s | %s | %s %s | %s |" % (
            c["name"], "—" if c["value"] is None else "%.4f" % c["value"],
            c["op"], c["threshold"], "PASS" if c["ok"] else "FAIL"))
    lines += ["", "## 复现", "", "```bash",
              "PYTHONPATH=src python -m evals.conclusion_gate_eval",
              "PYTHONPATH=src python -m evals.conclusion_gate_eval --pipeline-only",
              "```", ""]
    return "\n".join(lines)


def _output_paths(pipeline_only: bool, out: Optional[str],
                  md: Optional[str]) -> tuple[str, str]:
    """解析输出路径。

    pipeline-only 只跑一种模式，检查项为空，不构成两模式门禁；
    它的默认产物必须与正式 24 项结果分开，否则一次诊断运行会覆盖证据。
    """
    if pipeline_only:
        return (out or os.path.join(ROOT, "evals",
                                    "conclusion_gate_v2_pipeline_only.json"),
                md or os.path.join(ROOT, "evals",
                                   "conclusion_gate_v2_pipeline_only_report.md"))
    return out or OUT_JSON, md or OUT_MD


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-only", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args(argv)

    from mtsa.io_loaders import sha1_of_file
    labels = load_labels(LABELS)
    truth = load_truth(TRUTH)
    truth_a = truth_anomalies(labels)
    truth_cp = truth_change_points(labels)
    modes = ["pipeline"] if args.pipeline_only else ["pipeline", "agent"]
    if "agent" in modes and not _has_model():
        print("缺少 MTSA_LLM_BASE_URL / MTSA_LLM_API_KEY / MTSA_LLM_MODEL；"
              "如需离线运行请加 --pipeline-only", file=sys.stderr)
        return 2

    out_json, out_md = _output_paths(args.pipeline_only, args.out, args.md)

    os.makedirs(OUT_ROOT, exist_ok=True)
    rows = []
    run_rows = []
    for case in CASES:
        for mode in modes:
            print("运行 %s / %s ..." % (case["id"], mode))
            doc = _run_one(case, mode, args.max_tokens)
            ev = evaluate_run(doc, case["id"], mode, case["question"], case["expect"], labels, truth)
            ev["evidence_coverage"] = _evidence_coverage(doc)
            ev["steps"] = (doc.get("data", {}).get("agent") or {}).get("step_count")
            ev["terminate_reason"] = (doc.get("data", {}).get("agent") or {}).get("terminate_reason")
            ev["need_coverage"] = (ev.get("conclusion_need") or {}).get("coverage")
            ev["_detected_events"] = _detected_events(doc)
            rows.append(ev)
            run_rows.append({
                "case": case["id"], "mode": mode, "status": ev.get("status"),
                "steps": ev.get("steps"), "terminate_reason": ev.get("terminate_reason"),
                "conclusion_count": ev.get("conclusion_count"),
                "evidence_coverage": ev.get("evidence_coverage"),
                "need_coverage": ev.get("need_coverage"),
                "anomaly_point": (ev.get("anomaly") or {}).get("point"),
                "changepoint_point": (ev.get("changepoint") or {}).get("point"),
                "missing_tools": (((doc.get("data") or {}).get("agent") or {})
                                  .get("coverage") or {}).get("missing_tools"),
            })
            print("  status=%s 结论=%s 证据覆盖=%.2f" % (
                ev.get("status"), ev.get("conclusion_count"), ev.get("evidence_coverage") or 0.0))

    summary = summarize(rows, truth_a, truth_cp, 2 * 3600)
    _adjust(summary["by_mode"], rows, labels)
    ev_by_case = {}
    for mode in modes:
        for item in summary["by_mode"][mode]["anomaly"]["event"]["per_run"]:
            ev_by_case[(item["case"], mode)] = item
    for r in run_rows:
        r["anomaly_event"] = ev_by_case.get((r["case"], r["mode"]))
    evidence = {
        mode: _mean([r["evidence_coverage"] for r in rows if r["mode"] == mode])
        for mode in modes
    }
    evidence_ci = {}
    for mode in modes:
        m = summary["by_mode"][mode]
        rs = [r for r in rows if r["mode"] == mode]
        m["evidence_coverage"] = evidence[mode]
        evidence_ci[mode] = _wilson(
            sum(1 for r in rs if (r["evidence_coverage"] or 0.0) >= 1.0), len(rs))
        m["success_rate"] = _ratio(m["succeeded"], m["runs"])
        m["success_rate_ci"] = _wilson(m["succeeded"], m["runs"])
    gate = _gate(summary["by_mode"], evidence) if len(modes) == 2 else {
        "checks": [], "passed": False,
        "reason": "pipeline-only 模式不构成两模式门禁",
    }
    doc = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": _rel(DATA),
        "dataset_sha1": sha1_of_file(DATA),
        "labels": _rel(LABELS),
        "labels_sha1": sha1_of_file(LABELS),
        "truth": _rel(TRUTH),
        "truth_sha1": sha1_of_file(TRUTH),
        "dataset_spec": _dataset_spec(labels),
        "truth_events": len(_truth_events(labels)),
        "model": os.environ.get("MTSA_LLM_MODEL") if "agent" in modes else None,
        "modes": modes,
        "thresholds": THRESHOLDS,
        "agent_margin": AGENT_MARGIN,
        "summary": summary,
        "evidence_coverage": evidence,
        "evidence_coverage_ci": evidence_ci,
        "runs": run_rows,
        "checks": gate,
        "passed": bool(gate["passed"]),
        "partial": not gate["checks"],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2, default=str)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(render_md(doc))
    print()
    if not gate["checks"]:
        print("门禁：PARTIAL（%s）" % (gate.get("reason") or "检查项为空"))
        print("结果：%s" % out_json)
        print("报告：%s" % out_md)
        print("提示：这是诊断产物，不构成门禁，也不会覆盖 conclusion_gate_v2_results.json。")
        return 0
    print("门禁：%s（%d/%d 项检查通过）" % (
        "PASS" if doc["passed"] else "FAIL",
        sum(1 for c in gate["checks"] if c["ok"]), len(gate["checks"])))
    print("结果：%s" % out_json)
    print("报告：%s" % out_md)
    return 0 if doc["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
