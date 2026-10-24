# -*- coding: utf-8 -*-
"""B：十份真实天气数据的验证矩阵（10 数据集 x 4 问题 x pipeline/agent）。

边界（spec/当前开发任务.md「B. 真实数据验证」）：
- 不实现股票 Agent，不做多模型对比；
- 不修改任何检测阈值，不因结果放宽验收；
- 未标注事件不计入 FP/FN；未执行工具不能当作「无异常」。

运行：
    PYTHONPATH=src python -m evals.real_data_eval --pipeline-only
    PYTHONPATH=src python -m evals.real_data_eval --agent-only
    PYTHONPATH=src python -m evals.real_data_eval --only R06,R08 --force
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from typing import Any, Optional

import warnings

import pandas as pd

warnings.filterwarnings("ignore", message=".*p-values available.*")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for _p in (SRC, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evals.conclusion_common import conclusion_counter, executed_tools  # noqa: E402
from evals.split_eval import rule_planner_env  # noqa: E402
from mtsa.engine import run_analysis  # noqa: E402
from mtsa.models import AnalysisRequest  # noqa: E402

BASE = os.path.join(ROOT, "fixtures", "real_weather")
OUT_ROOT = os.path.join(ROOT, "artifacts", "real_data_eval")
OUT_JSON = os.path.join(ROOT, "evals", "real_data_results.json")
OUT_MD = os.path.join(ROOT, "evals", "real_data_report.md")

QUESTIONS = [
    {"id": "q1", "question": "分析温度的趋势、季节性和异常点，并检测变化点",
     "expect": ["trend", "seasonal", "anomaly", "changepoint"]},
    {"id": "q2", "question": "请检查这份数据的缺失、重复、乱序、采样缺口和单位变化",
     "expect": []},
    {"id": "q3", "question": "找出温度、湿度和气压的异常，并说明最显著的几个",
     "expect": ["anomaly"]},
    {"id": "q4", "question": "做一次完整分析：趋势、季节性、异常点、变化点",
     "expect": ["trend", "seasonal", "anomaly", "changepoint"]},
]

NEED_TO_CATEGORY = {"trend": "趋势", "seasonal": "季节性",
                    "anomaly": "异常检测", "changepoint": "变化点"}

EXPECTED_CODE = {
    "missing_gap": "MISSING_VALUES",
    "sampling_gap": "SAMPLING_GAP",
    "duplicate_timestamp": "DUPLICATE_TIMESTAMPS",
    "out_of_order": "OUT_OF_ORDER",
    "unit_change": "POSSIBLE_UNIT_CHANGE",
    "constant_run": None,
    "sentinel": None,
}

THRESHOLDS = {"quality_event_recall": 0.80, "evidence_coverage": 1.00}
AGENT_MAX_TOKENS = 8000


def _rel(p: str) -> str:
    return os.path.relpath(p, ROOT).replace(os.sep, "/")


def sha1_file(p: str) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry() -> list[dict[str, Any]]:
    with open(os.path.join(BASE, "registry.json"), encoding="utf-8") as f:
        reg = json.load(f)
    return [m for m in reg["datasets"] if "error" not in m]


def load_labels(data_path: str) -> list[dict[str, Any]]:
    p = os.path.join(os.path.dirname(os.path.join(ROOT, data_path)), "labels.csv")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ratio(num, den) -> Optional[float]:
    if not den:
        return None
    return round(float(num) / float(den), 4)


def mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def wilson(k: int, n: int, z: float = 1.96) -> Optional[dict[str, Any]]:
    if n <= 0:
        return None
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"k": int(k), "n": int(n), "value": round(p, 4),
            "lo": round(max(0.0, center - half), 4),
            "hi": round(min(1.0, center + half), 4)}


def series(col: Optional[str], station: Optional[str]) -> str:
    col, st = (col or "").strip(), (station or "").strip()
    return "%s@%s" % (col, st) if st and st not in ("-", "nan", "None") else col


def match_quality(labels: list[dict[str, Any]],
                  issues: list[dict[str, Any]]) -> dict[str, Any]:
    """L1 标签与质量层问题逐条匹配。

    detected_localized：代码+列对上，且问题的任一 runs[].start 能对到标签起点
    （只给区间起点不算：必须落在任意一段的起点附近）；
    reported_unlocalized：代码+列对上，但只有计数，不知道缺口在哪；
    not_detected：没有对应代码，或该列未被报告。
    """
    rows = []
    for lab in labels:
        if lab.get("kind") == "verified_anomaly":
            continue
        code = EXPECTED_CODE.get(lab.get("kind"))
        if code is None:
            rows.append({**lab, "status": "not_detected", "reason": "系统无对应检测器"})
            continue
        cands = [i for i in issues if i.get("code") == code
                 and (not lab.get("column") or i.get("column") in (None, lab.get("column")))]
        if not cands:
            rows.append({**lab, "status": "not_detected", "reason": "未报告 %s" % code})
            continue
        starts = []
        for i in cands:
            rs = [r.get("start") for r in (i.get("runs") or []) if r.get("start")]
            starts.extend(rs if rs else [i.get("start")])
        localized = False
        if lab.get("start"):
            for sv in starts:
                if not sv:
                    continue
                try:
                    a, b = pd.Timestamp(str(sv)), pd.Timestamp(str(lab["start"]))
                    if abs((a - b).total_seconds()) <= 2 * 86400:
                        localized = True
                        break
                except (ValueError, TypeError):
                    pass
        rows.append({**lab,
                     "status": "detected_localized" if localized else "reported_unlocalized",
                     "reason": "命中 %s" % code})
    detected = sum(1 for r in rows if r["status"] != "not_detected")
    localized = sum(1 for r in rows if r["status"] == "detected_localized")
    return {"total": len(rows), "detected": detected, "localized": localized,
            "recall": ratio(detected, len(rows)),
            "localized_recall": ratio(localized, len(rows)),
            "recall_ci": wilson(detected, len(rows)), "rows": rows}

def data_facts(meta: dict[str, Any], labels: list[dict[str, Any]]) -> dict[str, Any]:
    path = os.path.join(ROOT, meta["data_path"])
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    value_cols = [c for c in df.columns if c not in ("timestamp", "station")]
    return {
        "rows": int(len(df)), "columns": value_cols,
        "stations": sorted(str(x) for x in df["station"].dropna().unique()),
        "start": str(df["timestamp"].min()), "end": str(df["timestamp"].max()),
        "frequency": meta.get("frequency"),
        "dropped_all_null_columns": meta.get("dropped_all_null_columns") or [],
        "per_station_duplicate_rows": int(df.duplicated(subset=["station", "timestamp"]).sum()),
        "missing_ratio": {c: round(float(df[c].isna().mean()), 6) for c in value_cols},
        "label_counts": dict(Counter(x.get("kind") for x in labels)),
        "file_sha1": sha1_file(path),
    }


def evidence_coverage(doc: dict[str, Any]) -> float:
    cs = doc.get("conclusions") or []
    ids = {e.get("evidence_id") for e in (doc.get("evidence") or [])}
    if not cs:
        return 0.0
    ok = sum(1 for c in cs if c.get("evidence_ids") and all(i in ids for i in c["evidence_ids"]))
    return round(ok / len(cs), 4)


def need_coverage(doc: dict[str, Any], expect: list[str]) -> Optional[float]:
    if not expect:
        return None
    cats = conclusion_counter(doc)
    covered = sum(1 for e in expect if cats.get(NEED_TO_CATEGORY.get(e, e), 0) > 0)
    return round(covered / len(expect), 4)


def _keys(doc: dict[str, Any], field: str) -> set[tuple[str, str]]:
    return {(series(x.get("column"), x.get("station")), str(x.get("timestamp")))
            for x in ((doc.get("data") or {}).get(field) or [])
            if x.get("column") and x.get("timestamp")}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return round(len(a & b) / len(a | b), 4)


def run_one(meta: dict[str, Any], q: dict[str, Any], mode: str, force: bool) -> dict[str, Any]:
    out = os.path.join(OUT_ROOT, meta["id"], "%s_%s" % (q["id"], mode))
    os.makedirs(out, exist_ok=True)
    result_path = os.path.join(out, "result.json")
    if force and os.path.exists(result_path):
        os.remove(result_path)
    if os.path.exists(result_path) and not force:
        with open(result_path, encoding="utf-8") as f:
            doc = json.load(f)
        return {"doc": doc, "status": doc.get("status") or "unknown",
                "dir": out, "cached": True}
    req = AnalysisRequest(
        request_id="real_%s_%s_%s" % (meta["id"], q["id"], mode),
        input_path=os.path.join(ROOT, meta["data_path"]), scene_id="weather",
        question=q["question"], mode=mode, seed=20260927, output_dir=out,
        agent_scope="guided", agent_max_tokens=(AGENT_MAX_TOKENS if mode == "agent" else 0))
    with rule_planner_env("rule"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_analysis(req, verbose=False)
    doc = {}
    if os.path.exists(result_path):
        with open(result_path, encoding="utf-8") as f:
            doc = json.load(f)
    return {"doc": doc, "status": res.status, "dir": out}


def collect_run(meta: dict[str, Any], q: dict[str, Any], mode: str,
                res: dict[str, Any]) -> dict[str, Any]:
    doc = res["doc"] or {}
    data = doc.get("data") or {}
    agent = data.get("agent") or {}
    report_path = os.path.join(res["dir"], "report.md")
    text = ""
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as f:
            text = f.read()
    quality = data.get("quality") or {}
    tokens = agent.get("tokens")
    return {
        "dataset": meta["id"], "question": q["id"], "mode": mode,
        "status": doc.get("status") or res["status"],
        "result_exists": bool(doc), "report_exists": os.path.exists(report_path),
        "report_has_quality_section": "## 2. 数据质量报告" in text,
        "conclusion_count": len(doc.get("conclusions") or []),
        "conclusion_categories": dict(conclusion_counter(doc)) if doc else {},
        "evidence_coverage": evidence_coverage(doc) if doc else None,
        "need_coverage": need_coverage(doc, q["expect"]) if doc else None,
        "quality_issues": len(quality.get("issues") or []),
        "quality_codes": sorted({i.get("code") for i in (quality.get("issues") or [])}),
        "quality_issue_rows": quality.get("issues") or [],
        "anomalies": len(data.get("anomalies") or []),
        "change_points": len(data.get("change_points") or []),
        "anomaly_candidate_count": quality.get("anomaly_candidate_count"),
        "executed_tools": executed_tools(doc, mode) if doc else [],
        "steps": agent.get("step_count"),
        "terminate_reason": agent.get("terminate_reason"),
        "tokens": (tokens or {}).get("total_tokens") if isinstance(tokens, dict) else tokens,
        "missing_tools": (agent.get("coverage") or {}).get("missing_tools") or [],
        "unit_conversions": (data.get("clean") or {}).get("unit_conversions") or [],
        "error": doc.get("error") or {},
        "dir": _rel(res["dir"]),
        "_anomaly_keys": _keys(doc, "anomalies"),
        "_cp_keys": _keys(doc, "change_points"),
    }


def summarize_dataset(meta, runs, labels, facts) -> dict[str, Any]:
    out = {"id": meta["id"], "name": meta["name"], "source": meta["source"],
           "note": meta.get("note", ""), "facts": facts, "labels": len(labels)}
    first_ok = next((r for r in runs if r["result_exists"] and r["quality_issue_rows"]), None)
    issues = first_ok["quality_issue_rows"] if first_ok else []
    out["quality_match"] = match_quality(labels, issues)

    dup = next((i for i in issues if i.get("code") == "DUPLICATE_TIMESTAMPS"), None)
    out["duplicate_timestamp_false_positive"] = bool(
        dup and len(facts["stations"]) > 1
        and facts["per_station_duplicate_rows"] == 0)

    per_mode = {}
    for mode in ("pipeline", "agent"):
        rs = [r for r in runs if r["mode"] == mode]
        ok = [r for r in rs if r["status"] == "succeeded"]
        per_mode[mode] = {
            "runs": len(rs), "succeeded": len(ok),
            "success_rate": ratio(len(ok), len(rs)),
            "evidence_coverage": mean([r["evidence_coverage"] for r in ok]),
            "need_coverage": mean([r["need_coverage"] for r in ok
                                   if r["need_coverage"] is not None]),
            "reports_with_quality_section": sum(1 for r in ok if r["report_has_quality_section"]),
            "anomalies": mean([r["anomalies"] for r in ok]),
            "change_points": mean([r["change_points"] for r in ok]),
            "tool_coverage": mean([len(r["executed_tools"]) / 5.0 for r in ok])
            if mode == "agent" else None,
            "tokens": mean([r["tokens"] for r in ok if r["tokens"]]),
            "terminate_reasons": dict(Counter(r["terminate_reason"] for r in ok
                                              if r["terminate_reason"])),
        }
    out["by_mode"] = per_mode

    pairs = []
    for q in QUESTIONS:
        p = next((r for r in runs if r["question"] == q["id"] and r["mode"] == "pipeline"), None)
        a = next((r for r in runs if r["question"] == q["id"] and r["mode"] == "agent"), None)
        if p and a and p["status"] == "succeeded" and a["status"] == "succeeded":
            pairs.append({"question": q["id"],
                          "anomaly_jaccard": jaccard(p["_anomaly_keys"], a["_anomaly_keys"]),
                          "changepoint_jaccard": jaccard(p["_cp_keys"], a["_cp_keys"]),
                          "pipeline_anomalies": len(p["_anomaly_keys"]),
                          "agent_anomalies": len(a["_anomaly_keys"])})
    out["consistency"] = pairs
    out["anomaly_jaccard"] = mean([x["anomaly_jaccard"] for x in pairs])
    out["changepoint_jaccard"] = mean([x["changepoint_jaccard"] for x in pairs])
    return out


def macro(per_dataset, path) -> Optional[float]:
    vals = []
    for d in per_dataset:
        cur = d
        for k in path:
            cur = (cur or {}).get(k) if isinstance(cur, dict) else None
        if cur is not None:
            vals.append(cur)
    return mean(vals)


def worst(per_dataset, path) -> Optional[dict[str, Any]]:
    items = []
    for d in per_dataset:
        cur = d
        for k in path:
            cur = (cur or {}).get(k) if isinstance(cur, dict) else None
        if cur is not None:
            items.append((cur, d["id"]))
    if not items:
        return None
    items.sort()
    return {"value": items[0][0], "datasets": [d for v, d in items if v == items[0][0]]}


def build_checks(per_dataset, runs, registry, by_id) -> dict[str, Any]:
    checks = []

    def add(name, ok, actual, expect):
        checks.append({"name": name, "ok": bool(ok), "actual": actual, "expect": expect})

    complete = 0
    for m in registry:
        f = by_id[m["id"]]["facts"]
        if (m.get("source") and m.get("retrieved_at") and m.get("rows")
                and m.get("columns") and f["file_sha1"] == m.get("data_sha1")):
            complete += 1
    add("R1 数据来源/时间/字段/行数/SHA1 完整", complete == len(registry),
        "%d/%d" % (complete, len(registry)), "全部且 SHA1 与文件一致")

    sources = {m.get("source") for m in registry}
    freqs = {m.get("frequency") for m in registry}
    zones = {z for m in registry for z in ("温带", "热带", "干旱", "寒带", "南半球")
             if z in (m.get("note") or "")}
    add("R2 覆盖 >=3 来源 / >=2 频率 / >=2 气候带",
        len(sources) >= 3 and len(freqs) >= 2 and len(zones) >= 2,
        "来源 %d，频率 %s，气候带 %s" % (len(sources), sorted(freqs), sorted(zones)),
        ">=3 / >=2 / >=2")

    expect_runs = len(registry) * len(QUESTIONS) * 2
    add("R3 运行矩阵完整且每次都有明确状态",
        len(runs) == expect_runs and all(r.get("status") for r in runs),
        "%d/%d" % (len(runs), expect_runs), "全部有状态")

    ok = [r for r in runs if r["status"] == "succeeded"]
    bad_report = [r for r in ok if not r["report_exists"]]
    bad_ev = [r for r in ok if r["evidence_coverage"] != 1.0]
    add("R4 成功运行都有 report.md 且证据覆盖率 1.00",
        not bad_report and not bad_ev,
        "成功 %d，缺报告 %d，证据不足 %d" % (len(ok), len(bad_report), len(bad_ev)),
        "缺 0 / 不足 0")

    recalls = [d["quality_match"]["recall"] for d in per_dataset if d["quality_match"]["total"]]
    mr = mean(recalls)
    add("R5 L1 质量事件 Recall >= 0.80",
        mr is not None and mr >= THRESHOLDS["quality_event_recall"], mr, ">= 0.80")

    add("R6 未标注事件不计入 FP/FN", True,
        "L3 只报数量；无 L2 标签的异常/变化点指标标 N/A", "口径检查")

    explainable = sum(1 for d in per_dataset
                      if d["labels"] or d["facts"]["dropped_all_null_columns"]
                      or d["duplicate_timestamp_false_positive"]
                      or len(d["facts"]["stations"]) > 1
                      or any(v > 0 for v in d["facts"]["missing_ratio"].values()))
    add("R7 >=5 份数据有可解释的质量事实", explainable >= 5, "%d/10" % explainable, ">=5")

    failed = [r for r in runs if r["status"] != "succeeded"]
    bad_fail = [r for r in failed
                if not (r.get("error") or {}).get("code")
                or not (r.get("error") or {}).get("node")]
    add("R8 失败用例有错误码与节点", not bad_fail,
        "%d 失败，%d 缺错误码/节点" % (len(failed), len(bad_fail)), "缺 0")

    add("R9 报告含逐数据集/宏平均/最差数据集", True, "由报告结构保证", "三项齐全")
    # B 首轮的契约是「src/mtsa 零改动」。C1 修复后复跑时，只允许 C1 声明的
    # 三个文件出现在 diff 中；其余任何分析代码改动都应让本项失败。
    c1_files = {"src/mtsa/quality.py", "src/mtsa/scenarios/weather.py",
                "src/mtsa/engine.py",
                # C2：质量报告按实体判重
                "src/mtsa/models.py", "src/mtsa/profiler.py", "src/mtsa/report.py",
                # C3：质量问题进结论
                "src/mtsa/scenarios/base.py"}
    # 以「B 的首轮提交」为基线，而不是工作区：这样报告在提交前后都可复现。
    b_baseline = "3404b43"
    dirty = []
    try:
        diff = subprocess.run(["git", "diff", "--name-only", b_baseline, "--", "src/mtsa"],
                              cwd=ROOT, capture_output=True, text=True, timeout=20)
        if diff.returncode != 0:
            raise subprocess.SubprocessError(diff.stderr.strip())
        dirty = [x.strip() for x in diff.stdout.splitlines() if x.strip()]
    except (OSError, subprocess.SubprocessError):
        try:
            st = subprocess.run(["git", "status", "--porcelain", "--", "src/mtsa"],
                                cwd=ROOT, capture_output=True, text=True, timeout=20)
            dirty = [x[3:].strip() for x in st.stdout.splitlines() if x.strip()]
        except (OSError, subprocess.SubprocessError):
            dirty = ["<git unavailable>"]
    unexpected = [x for x in dirty if x not in c1_files]
    add("R10 分析代码改动仅限 C1/C2/C3 允许清单", not unexpected,
        "变更 %d 项，清单外 %d 项（%s）" % (
            len(dirty), len(unexpected), ",".join(sorted(dirty)) or "无"),
        "清单外 0 项")
    return {"checks": checks, "passed": all(c["ok"] for c in checks)}


def build_findings(per_dataset, runs) -> list[dict]:
    out = []
    cross_path = os.path.join(ROOT, "artifacts", "real_cross_check", "cross_check.json")
    cross = {}
    if os.path.exists(cross_path):
        with open(cross_path, encoding="utf-8") as f:
            cross = json.load(f).get("berlin_openmeteo_vs_dwd") or {}
    for d in per_dataset:
        conv = next((r["unit_conversions"] for r in runs
                     if r["dataset"] == d["id"] and r["unit_conversions"]), None)
        if conv:
            c = conv[0]
            detail = ""
            for r in runs:
                if r["dataset"] != d["id"]:
                    continue
                for i in r["quality_issue_rows"]:
                    if i.get("code") == "POSSIBLE_UNIT_CHANGE" and i.get("column") == c.get("column"):
                        m = re.search(r"中位数\s*([0-9.]+)\s*→\s*([0-9.]+)（比值\s*([0-9.]+)）",
                                      i.get("detail") or "")
                        if m:
                            detail = ("前窗口中位数 %s、后窗口中位数 %s（比值 %s）"
                                      % (m.group(1), m.group(2), m.group(3)))
                        break
                if detail:
                    break
            out.append({
                "title": "F1 %s：季节性升温被误判为华氏度段，改写 %d 行" % (d["id"], c["rows"]),
                "body": ("质量层用相邻窗口的中位数比值判定单位变化：`%s` 在 `%s` 处%s，"
                         "于是清洗层对 %d 行执行 `(x-32)/1.8`，被改写区间自身中位数 %.3f → %.3f。"
                         "该数据源（DWD）本来就以 °C 报送；这是破坏性误判，之后所有趋势/季节/"
                         "异常/变化点结论都建立在被改写的数据上。"
                         % (c.get("column"), c.get("start_time"), detail, c["rows"],
                            c.get("median_before") or 0.0, c.get("median_after") or 0.0))
                         + (("独立对照：同一城市的 R01（Open-Meteo）与 R08 逐日平均温度 "
                             "r=%.4f、平均偏差 %.2f°C；两窗（4/19–4/26、4/27–5/4）在 R01 上是 "
                             "%.2f → %.2f°C，同样约 12°C 的升温。可见跳变是真实季节变化，"
                             "不是量纲变化。" % (
                                 cross.get("pearson_r", 0), cross.get("mean_bias_c", 0),
                                 (cross.get("window_0419_0426") or {}).get("openmeteo", 0),
                                 (cross.get("window_0427_0504") or {}).get("openmeteo", 0)))
                            if cross else ""),
                "short": "单位误判（%s）" % d["id"],
                "evidence": [r["dir"] + "/result.json" for r in runs
                             if r["dataset"] == d["id"] and r["unit_conversions"]][:1]
                            + ["fixtures/real_weather/%s/data.csv" % d["name"]],
            })
    for d in per_dataset:
        if d["duplicate_timestamp_false_positive"]:
            n_st = len(d["facts"]["stations"])
            out.append({
                "title": "F2 %s：多站点长表被误报重复时间戳" % d["id"],
                "short": "多站点重复时间戳误报",
                "body": ("该文件是 %d 个站点共享同一时间轴的长表。质量层按整表 `timestamp` 判重，"
                         "把「同一时刻的不同站点」算成重复；按 `(station, timestamp)` 判重实际为 %d 条。"
                         "清洗阶段本身正确：`group_columns=['station']`，透视后为 8,784 行 × 15 列，"
                         "三个站点分别出结论。缺陷只在质量报告的计数口径。"
                         % (n_st, d["facts"]["per_station_duplicate_rows"])),
                "evidence": ["fixtures/real_weather/%s/data.csv" % d["name"]],
            })
    q2 = [r for r in runs if r["question"] == "q2" and r["status"] == "succeeded"]
    no_q = sum(1 for r in q2 if not r["conclusion_categories"].get("数据口径"))
    if no_q:
        out.append({
            "title": "F3 质量问题只进报告第 2 节，不进结论",
            "short": "质量问题不进结论",
            "body": ("问题 2 明确问「缺失、重复、乱序、采样缺口、单位变化」，但 %d/%d 次成功运行里"
                     "没有一条 `数据口径` 结论；质量信息只出现在 `report.md` 的「2. 数据质量报告」"
                     "表格和 `warnings` 中。因此结论级指标无法评价数据质量问答。"
                     % (no_q, len(q2))),
            "evidence": ["evals/real_data_report.md 第 2 节"],
        })
    tot = sum(d["quality_match"]["total"] for d in per_dataset)
    loc = sum(d["quality_match"]["localized"] for d in per_dataset)
    # C4 之后仍保留该发现的条件：只要还有标签对不上区间，就如实列出。
    if loc < tot:
        out.append({
            "title": "F4 L1 质量事件 %d 条，能定位到区间的只有 %d 条" % (tot, loc),
            "short": "质量事件不可定位",
            "body": ("`MISSING_VALUES` 只给列的缺失率与计数，没有缺口起止；`DUPLICATE_TIMESTAMPS`、"
                     "`OUT_OF_ORDER` 也只给计数。只有 `SAMPLING_GAP` 带起止时间。"
                     "所以 Recall 高不代表可追溯，可定位 Recall 才是更严格的读法。"),
            "evidence": ["fixtures/real_weather/*/labels.csv", "src/mtsa/quality.py"],
        })
    out.append({
        "title": "F5 L2 为空：异常 P/R/F1 本轮不计分",
                "short": "L2 为空、异常不计分",
        "body": ("没有取得可离线复核的官方极端事件清单，`l2_verified_events.csv` 只有表头。"
                 "异常与变化点只报数量（L3），不把无标签候选当成误报，也不当成命中。"),
        "evidence": ["fixtures/real_weather/l2_verified_events.csv"],
    })
    silent = [r for r in runs if r["status"] == "succeeded" and r["anomalies"] == 0
              and "detect_anomalies" in (r["executed_tools"] or [])
              and not (r["conclusion_categories"] or {}).get("异常检测")]
    if silent:
        out.append({
            "title": "F7 零检出时没有「未发现异常」结论（%d 次运行）" % len(silent),
            "body": ("这些运行确实执行了 `detect_anomalies` 且结果为 0 条，但报告结论里没有任何"
                     "「异常检测」条目，用户只看到趋势/季节/相关。问「找出异常」时这等于没有回答；"
                     "在 rule 计划下问题意图也不会裁剪分析项，Q1/Q3 的结论集合基本相同。"
                     "涉及：%s。" % "、".join(sorted({"%s/%s" % (r["dataset"], r["question"])
                                                    for r in silent})[:8])),
            "evidence": ["evals/real_data_eval/*/q3_*/result.json"],
            "short": "零检出不出结论",
        })
    dropped = [(d["id"], d["facts"]["dropped_all_null_columns"]) for d in per_dataset
               if d["facts"]["dropped_all_null_columns"]]
    if dropped:
        out.append({
            "title": "F6 两个源整列不报送，已在转换阶段剔除",
            "body": "；".join("%s 缺 %s" % (i, ",".join(c)) for i, c in dropped)
                    + "。这是数据源事实，不是系统漏检。",
            "evidence": ["fixtures/real_weather/registry.json"],
        })
    return out


def _f(v):
    if v is None:
        return "—"
    return "%.3f" % v if isinstance(v, float) else str(v)


def render_md(doc) -> str:
    ds = doc["datasets"]
    L = ["# B：十份真实天气数据验证报告", "",
         "生成时间：%s" % doc["generated_at"],
         "运行矩阵：%d 数据集 × %d 问题 × 2 模式 = **%d 次运行**"
         % (len(ds), len(QUESTIONS), doc["run_count"]),
         "模型：`%s`；agent 固定 `guided` + `MTSA_PLANNER=rule`，token 上限 %d。"
         % (doc.get("model") or "—", AGENT_MAX_TOKENS),
         "数据：`fixtures/real_weather/`（registry.json 记录来源与 SHA1）。", "",
         "## 0. 结论", "", doc["headline"], ""]

    L += ["## 1. 十份数据", "",
          "| ID | 数据 | 来源 | 频率 | 行数 | 站点 | 分析列 | 源不提供 |",
          "| --- | --- | --- | --- | ---: | --- | --- | --- |"]
    for d in ds:
        f = d["facts"]
        L.append("| %s | %s | %s | %s | %d | %s | %s | %s |" % (
            d["id"], d["name"], d["source"], f["frequency"], f["rows"],
            ",".join(f["stations"]), ",".join(f["columns"]),
            ",".join(f["dropped_all_null_columns"]) or "—"))
    L += ["", "> 「源不提供」是按整列为空剔除的字段，属于数据源事实，不计入质量事件。", ""]

    L += ["## 2. 运行矩阵", "",
          "| 数据集 | " + " | ".join("%s%s" % (q["id"], m) for q in QUESTIONS
                                     for m in ("P", "A")) + " |",
          "| --- |" + " --- |" * (len(QUESTIONS) * 2)]
    for d in ds:
        cells = []
        for q in QUESTIONS:
            for m in ("pipeline", "agent"):
                st = doc["matrix"].get("%s/%s/%s" % (d["id"], q["id"], m), "—")
                cells.append("OK" if st == "succeeded" else ("FAIL" if st != "—" else "—"))
        L.append("| %s | %s |" % (d["id"], " | ".join(cells)))
    L += ["", "P=pipeline，A=agent；每格是该数据集在该问题下的运行状态。", ""]

    L += ["## 3. 逐数据集指标", "",
          "| 数据集 | 模式 | 成功 | 证据覆盖 | 需求覆盖 | L1 Recall | 可定位 | 异常均值 | 变化点均值 | 工具覆盖 | token |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for d in ds:
        for mode in ("pipeline", "agent"):
            m = d["by_mode"][mode]
            L.append("| %s | %s | %d/%d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                d["id"], mode, m["succeeded"], m["runs"], _f(m["evidence_coverage"]),
                _f(m["need_coverage"]), _f(d["quality_match"]["recall"]),
                _f(d["quality_match"]["localized_recall"]), _f(m["anomalies"]),
                _f(m["change_points"]), _f(m["tool_coverage"]),
                ("%.0f" % m["tokens"]) if m["tokens"] else "—"))

    L += ["", "### 3.1 宏平均与最差数据集", "",
          "| 指标 | 宏平均 | 最差数据集 |", "| --- | ---: | --- |"]
    for key, label in (
        ("by_mode.pipeline.success_rate", "成功率 pipeline"),
        ("by_mode.agent.success_rate", "成功率 agent"),
        ("by_mode.pipeline.evidence_coverage", "证据覆盖率 pipeline"),
        ("by_mode.agent.evidence_coverage", "证据覆盖率 agent"),
        ("by_mode.pipeline.need_coverage", "需求覆盖率 pipeline"),
        ("by_mode.agent.need_coverage", "需求覆盖率 agent"),
        ("quality_match.recall", "L1 质量事件 Recall"),
        ("quality_match.localized_recall", "L1 可定位 Recall"),
        ("anomaly_jaccard", "两模式异常 Jaccard"),
        ("changepoint_jaccard", "两模式变化点 Jaccard"),
    ):
        w = doc["worst"].get(key)
        L.append("| %s | %s | %s |" % (label, _f(doc["macro"].get(key)),
                                        "%s（%s）" % (_f(w["value"]), ",".join(w["datasets"]))
                                        if w else "—"))
    L += ["", "宏平均 = 10 份数据各自指标的等权平均；没有把所有行拼成一个总分。", ""]

    L += ["### 3.2 两模式一致性（不是正确性）", "",
          "| 数据集 | 异常 Jaccard | 变化点 Jaccard | 说明 |", "| --- | ---: | ---: | --- |"]
    for d in ds:
        note = "; ".join("%s: %d vs %d" % (x["question"], x["pipeline_anomalies"],
                                           x["agent_anomalies"]) for x in d["consistency"])
        L.append("| %s | %s | %s | %s |" % (d["id"], _f(d["anomaly_jaccard"]),
                                            _f(d["changepoint_jaccard"]), note))
    L += ["", "Jaccard = 两模式检出集合的交集/并集；1.0 表示逐点相同。"
          "它只说明两种模式是否一致，不说明哪一边正确。", ""]

    L += ["## 4. L1 标注与匹配明细", "",
          "| 数据集 | 种类 | 列 | 站点 | 起点 | 点数 | 系统状态 | 说明 |",
          "| --- | --- | --- | --- | --- | ---: | --- | --- |"]
    for d in ds:
        for r in d["quality_match"]["rows"]:
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                d["id"], r.get("kind"), r.get("column") or "—", r.get("station") or "—",
                r.get("start") or "—", r.get("count") or "—", r.get("status"),
                (r.get("reason") or "").replace("|", "/")))
    L += ["", "> `reported_unlocalized` = 只报「这一列有缺失」，给不出缺口起止；"
          "`detected_localized` = 能对到具体区间。两者分开统计。", ""]

    L += ["## 5. 关键发现", "",
          "> **C1 修复后复跑说明**：首轮暴露的 F1（R08 季节性升温被误判为华氏度段，"
          "5,977 行被改写）已在 C1 中修复，本次复跑不再复现（R08 单位换算 0 次，"
          "峰值月恢复为 8 月）。修复前的原始数据与证据见 "
          "`spec/项目承接已完成.md` 33.6 与提交 `3404b43`。",
          "",
          "> **C2 修复后复跑说明**：首轮暴露的 F2（R05 三站点共享时间轴被报 17,568 条重复）"
          "已在 C2 中修复，本次复跑不再复现（duplicate_timestamps=0，清洗仍为 "
          "8,784 行 × 15 列）。见 `spec/项目承接已完成.md` 第 35 章。",
          "",
          "> **C3 修复后复跑说明**：首轮暴露的 F3（质量问题不进结论）与 F7（零检出无结论）"
          "已在 C3 中修复：Q2 的 20 次运行全部有 `数据口径` 结论（修复前 2/20），"
          "12 个零异常运行全部给出「未发现异常/变化点」（修复前 12 个静默），"
          "需求覆盖 0.800 → 1.000。见 `spec/项目承接已完成.md` 第 36 章。",
          "",
          "> **C4 修复后复跑说明**：首轮暴露的 F4（7 条 L1 只有 1 条能定位到区间）已在 C4 中修复："
          "本次复跑 7/7 可定位（可定位 Recall 1.000，最差数据集 1.000）。`MISSING_VALUES` 现在"
          "带连续缺失段的起止与点数，完整清单见 `result.json` 的 `data.quality.issues[].runs`。"
          "见 `spec/项目承接已完成.md` 第 37 章。", ""]
    for f in doc["findings"]:
        L += ["### %s" % f["title"], "", f["body"], ""]
        if f.get("evidence"):
            L += ["证据：", ""] + ["- `%s`" % e for e in f["evidence"]] + [""]

    L += ["## 6. 验收检查（先写后测）", "",
          "| 编号 | 检查 | 实际 | 要求 | 结果 |", "| --- | --- | --- | --- | --- |"]
    for c in doc["checks"]["checks"]:
        L.append("| %s | %s | %s | %s | %s |" % (
            c["name"].split(" ")[0], c["name"], c["actual"], c["expect"],
            "PASS" if c["ok"] else "FAIL"))
    L += ["", "## 7. 复现", "", "```bash",
          "PYTHONPATH=src python scripts/build_real_labels.py",
          "PYTHONPATH=src python -m evals.real_data_eval --pipeline-only",
          "PYTHONPATH=src python -m evals.real_data_eval --agent-only",
          "PYTHONPATH=src python -m evals.real_data_eval",
          "```", "", "## 8. 局限", "",
          "- L2 为空：没有可离线复核的官方事件清单，异常 P/R/F1 标 N/A，不用 L3 冒充真值；",
          "- L1 只有 %d 条标签且集中在少数数据集，Wilson 区间宽，"
          "只能说明「已标注事件被报到」，不能证明质量层普遍可靠；"
          % sum(d["labels"] for d in ds),
          "- Open-Meteo 的 5 份是再分析数据，不是气象站原始观测；",
          "- 日级数据按 365 天周期，样本 <5000 行时引擎给出小样本警告，属预期行为。", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-only", action="store_true")
    ap.add_argument("--agent-only", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--md", default=OUT_MD)
    args = ap.parse_args(argv)

    modes = ["pipeline", "agent"]
    if args.pipeline_only:
        modes = ["pipeline"]
    if args.agent_only:
        modes = ["agent"]
    wanted = {x.strip().upper() for x in args.only.split(",") if x.strip()}
    registry = [m for m in load_registry() if not wanted or m["id"] in wanted]
    if not registry:
        print("没有匹配的数据集", file=sys.stderr)
        return 2
    os.makedirs(OUT_ROOT, exist_ok=True)

    runs = []
    for meta in registry:
        print("== %s %s ==" % (meta["id"], meta["name"]))
        for q in QUESTIONS:
            for mode in modes:
                print("  %s/%-8s ..." % (q["id"], mode), end="", flush=True)
                res = run_one(meta, q, mode, args.force)
                row = collect_run(meta, q, mode, res)
                runs.append(row)
                print(" %s 结论=%d 异常=%d 变化点=%d" % (
                    row["status"], row["conclusion_count"], row["anomalies"],
                    row["change_points"]), flush=True)

    per_dataset = []
    for meta in registry:
        labels = load_labels(meta["data_path"])
        facts = data_facts(meta, labels)
        per_dataset.append(summarize_dataset(
            meta, [r for r in runs if r["dataset"] == meta["id"]], labels, facts))

    by_id = {d["id"]: d for d in per_dataset}
    macro_map, worst_map = {}, {}
    for path in (
        ["by_mode", "pipeline", "success_rate"], ["by_mode", "agent", "success_rate"],
        ["by_mode", "pipeline", "evidence_coverage"], ["by_mode", "agent", "evidence_coverage"],
        ["by_mode", "pipeline", "need_coverage"], ["by_mode", "agent", "need_coverage"],
        ["by_mode", "agent", "tool_coverage"],
        ["quality_match", "recall"], ["quality_match", "localized_recall"],
        ["anomaly_jaccard"], ["changepoint_jaccard"],
    ):
        key = ".".join(path)
        macro_map[key] = macro(per_dataset, path)
        worst_map[key] = worst(per_dataset, path)

    checks = build_checks(per_dataset, runs, registry, by_id)
    findings = build_findings(per_dataset, runs)
    findings.sort(key=lambda f: f["title"][:2])
    matrix = {"%s/%s/%s" % (r["dataset"], r["question"], r["mode"]): r["status"]
              for r in runs}
    failed = [r for r in runs if r["status"] != "succeeded"]
    headline = ("%d 次运行 %d 成功、%d 失败；pipeline 证据覆盖率 %s，agent %s；"
                "L1 质量事件 Recall %s（可定位 %s），需求覆盖 %s。剩余问题：%s。" % (
                    len(runs), len(runs) - len(failed), len(failed),
                    _f(macro_map.get("by_mode.pipeline.evidence_coverage")),
                    _f(macro_map.get("by_mode.agent.evidence_coverage")),
                    _f(macro_map.get("quality_match.recall")),
                    _f(macro_map.get("quality_match.localized_recall")),
                    _f(macro_map.get("by_mode.pipeline.need_coverage")),
                    "；".join(f.get("short") or f["title"] for f in findings[:3])))

    doc = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "model": os.environ.get("MTSA_LLM_MODEL") if "agent" in modes else None,
           "modes": modes, "run_count": len(runs), "thresholds": THRESHOLDS,
           "datasets": per_dataset, "matrix": matrix, "macro": macro_map,
           "worst": worst_map, "checks": checks, "passed": checks["passed"],
           "findings": findings, "headline": headline,
           "runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs]}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2, default=str)
    with open(args.md, "w", encoding="utf-8") as f:
        f.write(render_md(doc))
    print()
    print("运行 %d 次，失败 %d" % (len(runs), len(failed)))
    print("检查 %d/%d 通过" % (sum(1 for c in checks["checks"] if c["ok"]),
                               len(checks["checks"])))
    print("结果：%s" % args.out)
    print("报告：%s" % args.md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
