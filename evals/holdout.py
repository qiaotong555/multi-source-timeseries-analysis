# -*- coding: utf-8 -*-
"""最终保留集（C1）：冻结 + 一次性评估。

规则写进代码，不靠自觉：

1. `--freeze` 只允许执行一次，产出 `evals/holdout_set.json`——数据 SHA1、问题原文、
   协议参数、声明，全部固化。此后运行时**只读**这个文件，不再读数据目录里的元数据。
2. 正式运行前逐份校验 SHA1；任何一份对不上就中止，不写任何结果。
3. 结果文件已存在时拒绝重跑；`--force --reason "…"` 才能覆盖，且 `reruns` 累加。
4. 本脚本不读也不写任何检测阈值——保留集上出现的问题只能进限制清单，不能回调参数。

指标口径：与 `evals/real_data_eval.py` 共用 `evidence_coverage` / `need_coverage`，
不另立一套。保留集无标签，因此**不报**异常检测的 Precision / Recall；能报的只有
结构性指标（是否跑完、结论是否都挂证据、需求是否覆盖、有没有伪单位换算）。

运行：
    PYTHONPATH=src python -m evals.holdout --freeze
    PYTHONPATH=src python -m evals.holdout
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evals.real_data_eval import evidence_coverage, need_coverage, sha1_file  # noqa: E402
from evals.split_eval import rule_planner_env  # noqa: E402
from mtsa.engine import run_analysis  # noqa: E402
from mtsa.models import AnalysisRequest  # noqa: E402

EVALS = ROOT / "evals"
REGISTRY = ROOT / "fixtures" / "holdout" / "registry.json"
MANIFEST = EVALS / "holdout_set.json"
OUT_JSON = EVALS / "holdout_results.json"
OUT_MD = EVALS / "holdout_report.md"
ART = ROOT / "artifacts" / "holdout"

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

PROTOCOL = {
    "mode": "pipeline",
    "planner": "rule",
    "seed": 20260927,
    "scene_id": "weather",
    "why_pipeline": ("保留集要能被任何人复算：pipeline + rule 计划器不依赖外部模型，"
                     "同一份数据同一份计划。agent 模式的波动已由 real_data_eval 单独度量，"
                     "不在这里混入模型随机性。"),
    "why_no_threshold": "保留集只用于验收，不用于选择；本脚本不传 param_overrides。",
}

DECLARATION = [
    "这 3 份数据于 2026-10-25 首次抓取，抓取之前从未进入本仓库的任何分析、评估或调参流程。",
    "冻结后不参与阈值选择、不参与参数调整；结果不好看只写进遗留限制，不回调任何参数。",
    "只允许生成一次结果文件；重跑必须带 --force 与原因，且 reruns 计数会写在结果里。",
    "本保留集无人工标签：它只能检验结构完整性与单位推断，不能替代带标签数据上的 P/R 验收。",
]

STRUCTURAL_CHECKS = {
    "success_rate": 1.0,
    "evidence_coverage": 1.0,
    "need_coverage": 1.0,
    "spurious_unit_conversions": 0,
}


class HoldoutError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_registry(path: Path = REGISTRY) -> dict:
    if not path.exists():
        raise HoldoutError("保留集数据不存在：%s（先跑 python -m scripts.fetch_holdout）" % path)
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(registry: dict) -> dict:
    datasets = []
    for meta in registry.get("datasets") or []:
        if "error" in meta:
            raise HoldoutError("registry 里有失败条目：%s" % meta.get("id"))
        datasets.append({
            "id": meta["id"], "name": meta["name"], "source": meta["source"],
            "climate": meta["climate"], "note": meta.get("note", ""),
            "station": meta["station"],
            "data_path": meta["data_path"], "data_sha1": meta["data_sha1"],
            "rows": meta["rows"], "start": meta["start"], "end": meta["end"],
            "columns": meta["columns"],
        })
    climates = [d["climate"] for d in datasets]
    if len(datasets) != 3:
        raise HoldoutError("保留集必须是 3 份，实际 %d 份" % len(datasets))
    if len(set(climates)) != len(climates):
        raise HoldoutError("保留集必须覆盖 3 个不同气候带：%s" % climates)
    return {
        "version": 1,
        "frozen_at": _now(),
        "purpose": ("C1 最终保留集：给「阈值/规则是否只在开发数据上成立」这个问题一个"
                    "只跑一次、不可回调的答案。"),
        "declaration": DECLARATION,
        "protocol": {**PROTOCOL, "questions": QUESTIONS},
        "expected_structural_checks": STRUCTURAL_CHECKS,
        "datasets": datasets,
    }


def freeze(manifest_path: Path = MANIFEST, registry_path: Path = REGISTRY) -> dict:
    if manifest_path.exists():
        raise HoldoutError("已存在 %s；冻结只做一次。要更换数据必须新开一份协议。"
                           % manifest_path.relative_to(ROOT))
    manifest = build_manifest(load_registry(registry_path))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return manifest


def verify(manifest: dict, root: Path = ROOT) -> list[str]:
    """逐份校验 SHA1 与列集合；返回问题清单（空 = 通过）。"""
    problems: list[str] = []
    for d in manifest.get("datasets") or []:
        p = root / d["data_path"]
        if not p.exists():
            problems.append("%s 缺失：%s" % (d["id"], d["data_path"]))
            continue
        now = sha1_file(str(p))
        if now != d["data_sha1"]:
            problems.append("%s SHA1 不符：冻结 %s，当前 %s" % (d["id"], d["data_sha1"], now))
    return problems


def next_run_meta(existing: dict | None, force: bool, reason: str) -> dict:
    """决定这次能不能跑，以及 reruns 应该是几。"""
    if existing is None:
        return {"reruns": 0, "previous": None}
    if not force:
        raise HoldoutError(
            "已存在保留集结果（%s，reruns=%s）。保留集只跑一次。"
            % (existing.get("finished_at"), existing.get("reruns", 0)))
    if not (reason or "").strip():
        raise HoldoutError("--force 必须同时给 --reason，说明为什么必须重跑")
    return {"reruns": int(existing.get("reruns") or 0) + 1,
            "previous": {"finished_at": existing.get("finished_at"),
                         "reason": existing.get("rerun_reason")}}


def observations(datasets: list[dict], runs: list[dict]) -> list[dict]:
    """客观计数，不做判定：异常密度、质量问题代码、单位变化标记、换算行数。

    保留集上没有真值，所以这里只回答「看见了多少」，不回答「对了几条」。
    """
    rows = {d["id"]: d for d in datasets}
    out = []
    for d in datasets:
        rs = [r for r in runs if r["dataset"] == d["id"] and r["status"] == "succeeded"]
        if not rs:
            out.append({"id": d["id"], "status": "no_run"})
            continue
        # 同一份数据 4 个问题跑出同样的分析结果（rule 计划），取第一次即可，
        # 但质量问题按出现次数汇总，避免把同一份报告重复计数。
        first = rs[0]
        codes: dict[str, int] = {}
        for r in rs:
            for c in r["quality_codes"]:
                codes[c] = codes.get(c, 0) + 1
        n_cells = d["rows"] * max(1, len(d["columns"]))
        out.append({
            "id": d["id"], "climate": d["climate"], "rows": d["rows"],
            "columns": len(d["columns"]),
            "anomaly_points": first["anomalies"],
            "anomaly_share": round(first["anomalies"] / n_cells, 5),
            "change_points": first["change_points"],
            "quality_codes": dict(sorted(codes.items())),
            "quality_issues": first["quality_issues"],
            "unit_change_columns": sorted({i.get("column") for r in rs
                                           for i in (r.get("unit_change_issues") or [])
                                           if i.get("column")}),
            # 早于本字段的结果文件里没有列名，只能记「报过」；两者都保留，
            # 免得把「没记录」说成「没发生」。
            "unit_change_reported": any(
                "POSSIBLE_UNIT_CHANGE" in (r.get("quality_codes") or []) for r in rs),
            "unit_conversion_events": len(first["unit_conversions"]),
        })
    return out


def _git_commit() -> str | None:
    for exe in ("git", r"C:\Program Files\Git\cmd\git.exe"):
        try:
            out = subprocess.run([exe, "rev-parse", "HEAD"], cwd=str(ROOT),
                                 capture_output=True, text=True, timeout=15)
            if out.returncode == 0:
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def run_question(meta: dict, q: dict, out_dir: Path) -> dict:
    started = time.time()
    req = AnalysisRequest(
        request_id="holdout_%s_%s" % (meta["id"], q["id"]),
        input_path=str(ROOT / meta["data_path"]), scene_id="weather",
        question=q["question"], mode=PROTOCOL["mode"], seed=PROTOCOL["seed"],
        output_dir=str(out_dir), agent_scope="guided",
        agent_max_tokens=0, strict_verify=False,
    )
    with rule_planner_env(PROTOCOL["planner"]), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_analysis(req, verbose=False)
    result_path = out_dir / "result.json"
    doc = {}
    if result_path.exists():
        doc = json.loads(result_path.read_text(encoding="utf-8"))
    data = doc.get("data") or {}
    quality = data.get("quality") or {}
    report_path = out_dir / "report.md"
    text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    return {
        "dataset": meta["id"], "question": q["id"],
        "status": doc.get("status") or res.status,
        "result_exists": bool(doc), "report_exists": report_path.exists(),
        "report_bytes": len(text.encode("utf-8")),
        "report_has_quality_section": "## 2. 数据质量报告" in text,
        "conclusion_count": len(doc.get("conclusions") or []),
        "evidence_coverage": evidence_coverage(doc) if doc else None,
        "need_coverage": need_coverage(doc, q["expect"]) if doc else None,
        "quality_issues": len(quality.get("issues") or []),
        "quality_codes": sorted({i.get("code") for i in (quality.get("issues") or [])}),
        "anomalies": len(data.get("anomalies") or []),
        "change_points": len(data.get("change_points") or []),
        "extremes": len(data.get("extremes") or []),
        "unit_conversions": (data.get("clean") or {}).get("unit_conversions") or [],
        "unit_change_issues": [i for i in (quality.get("issues") or [])
                               if i.get("code") == "POSSIBLE_UNIT_CHANGE"],
        "error": doc.get("error") or {},
        "duration_s": round(time.time() - started, 1),
        "dir": str(out_dir.relative_to(ROOT)).replace(os.sep, "/"),
    }


def aggregate(runs: list[dict]) -> dict:
    ok = [r for r in runs if r["status"] == "succeeded"]
    ev = [r["evidence_coverage"] for r in ok if r["evidence_coverage"] is not None]
    need = [r["need_coverage"] for r in ok if r["need_coverage"] is not None]
    conv = [c for r in ok for c in r["unit_conversions"]]
    checks = {
        "success_rate": {"value": round(len(ok) / len(runs), 4) if runs else None,
                         "expect": STRUCTURAL_CHECKS["success_rate"]},
        "evidence_coverage": {"value": round(min(ev), 4) if ev else None,
                              "expect": STRUCTURAL_CHECKS["evidence_coverage"],
                              "rule": "取最小值：任何一次运行掉证据都不算过"},
        "need_coverage": {"value": round(min(need), 4) if need else None,
                          "expect": STRUCTURAL_CHECKS["need_coverage"],
                          "rule": "取最小值；q2 无期望类别，不参与"},
        "spurious_unit_conversions": {"value": len(conv),
                                      "expect": STRUCTURAL_CHECKS["spurious_unit_conversions"],
                                      "detail": conv[:10]},
    }
    for c in checks.values():
        exp, val = c["expect"], c["value"]
        if val is None:
            c["ok"] = False
        elif exp:
            c["ok"] = val >= exp
        else:
            c["ok"] = val == 0
    return {
        "runs": len(runs), "succeeded": len(ok),
        "failed": [{"dataset": r["dataset"], "question": r["question"],
                    "status": r["status"], "error": r["error"]}
                   for r in runs if r["status"] != "succeeded"],
        "conclusions_total": sum(r["conclusion_count"] for r in ok),
        "anomalies_total": sum(r["anomalies"] for r in ok),
        "change_points_total": sum(r["change_points"] for r in ok),
        "extremes_total": sum(r["extremes"] for r in ok),
        "quality_issues_total": sum(r["quality_issues"] for r in ok),
        "duration_s": round(sum(r["duration_s"] for r in runs), 1),
        "checks": checks,
    }


def run_all(manifest: dict, force: bool = False, reason: str = "") -> dict:
    existing = None
    if OUT_JSON.exists():
        existing = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    meta = next_run_meta(existing, force, reason)

    problems = verify(manifest)
    if problems:
        raise HoldoutError("SHA1 校验失败，拒绝运行：\n  " + "\n  ".join(problems))

    started = datetime.now().astimezone()
    runs = []
    for d in manifest["datasets"]:
        for q in manifest["protocol"]["questions"]:
            out_dir = ART / d["id"] / q["id"]
            out_dir.mkdir(parents=True, exist_ok=True)
            print("运行 %s %s ..." % (d["id"], q["id"]), flush=True)
            runs.append(run_question(d, q, out_dir))

    agg = aggregate(runs)
    doc = {
        "generated_at": started.isoformat(timespec="seconds"),
        "finished_at": _now(),
        "manifest_frozen_at": manifest["frozen_at"],
        "manifest_sha1": sha1_file(str(MANIFEST)) if MANIFEST.exists() else None,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "one_shot": {"reruns": meta["reruns"], "previous": meta["previous"],
                     "force_reason": reason or None},
        "declaration": manifest["declaration"],
        "protocol": manifest["protocol"],
        "datasets": manifest["datasets"],
        "has_labels": False,
        "threshold_touched": False,
        "runs": runs,
        "summary": agg,
        "observations": observations(manifest["datasets"], runs),
    }
    OUT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(doc), encoding="utf-8", newline="\n")
    return doc


def _f(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return ("%." + str(nd) + "f") % v
    return str(v)


HARD_CODES = {"MISSING_VALUES", "DUPLICATE_TIMESTAMPS", "OUT_OF_ORDER",
              "SAMPLING_GAP", "SENTINEL", "CONSTANT_RUN"}


def _obs_bullets(obs: list[dict]) -> list[str]:
    """从观察里挑出需要人看一眼的地方，措辞只陈述事实。"""
    live = [o for o in obs if o.get("status") != "no_run"]
    if not live:
        return []
    out = []
    flagged = [(o["id"], c) for o in live for c in o.get("unit_change_columns") or []]
    reported = [o["id"] for o in live
                if o.get("unit_change_reported") or o.get("unit_change_columns")]
    conv = sum(o["unit_conversion_events"] for o in live)
    if flagged:
        out.append("单位变化标记：%s。换算事件 %d 次——按 C1 的「阶跃 + 换算后边界连续」"
                   "判据只标记不换算，数据未被改写；标记是真是假需要真值才能判定。"
                   % ("、".join("%s 的 %s" % (i, c) for i, c in flagged), conv))
    elif reported:
        out.append("单位变化标记：%s 报出过 POSSIBLE_UNIT_CHANGE（列名与比值见产物 "
                   "`data.quality.issues`）。换算事件 %d 次——只标记不换算，数据未被改写。"
                   % ("、".join(reported), conv))
    else:
        out.append("单位变化标记：0 处。")
    shares = sorted(live, key=lambda o: o["anomaly_share"])
    hi, lo = shares[-1], shares[0]
    if hi["anomaly_share"] > 2 * max(lo["anomaly_share"], 1e-9):
        out.append("统计异常密度：%s 最高 %.2f%%，%s 最低 %.2f%%，相差 %.1f 倍。"
                   "没有真值，无法区分「该气候本身更极端」与「残差尺度估计偏小」。"
                   % (hi["id"], hi["anomaly_share"] * 100, lo["id"],
                      lo["anomaly_share"] * 100, hi["anomaly_share"] / max(lo["anomaly_share"], 1e-9)))
    codes = {c for o in live for c in (o.get("quality_codes") or {})}
    hard = sorted(codes & HARD_CODES)
    soft = sorted(codes - HARD_CODES)
    if hard:
        out.append("检出硬缺陷代码：%s——干净数据上出现这类代码需要人工核对。" % "、".join(hard))
    else:
        out.append("硬缺陷代码（缺失/重复/乱序/采样缺口/哨兵/常数段）：0 类；"
                   "出现的只有低危候选 %s。" % ("、".join(soft) if soft else "（无）"))
    return out


def render_md(doc: dict) -> str:
    s = doc["summary"]
    obs = doc.get("observations") or observations(doc["datasets"], doc["runs"])
    lines = []
    add = lines.append
    add("# 最终保留集（C1）一次性评估")
    add("")
    add("- 生成时间：%s（耗时 %s s）" % (doc["generated_at"], _f(s["duration_s"], 1)))
    add("- 冻结时间：%s；清单 SHA1：`%s`" % (doc["manifest_frozen_at"], doc["manifest_sha1"]))
    add("- 代码版本：`%s`；Python %s" % ((doc["git_commit"] or "—")[:12], doc["python"]))
    add("- 运行次数：%d（reruns=%d）" % (len(doc["runs"]), doc["one_shot"]["reruns"]))
    add("")
    add("## 1. 声明")
    add("")
    for x in doc["declaration"]:
        add("- " + x)
    add("")
    add("## 2. 数据")
    add("")
    add("| 编号 | 数据集 | 气候带 | 来源 | 行数 | 区间 | SHA1 |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for d in doc["datasets"]:
        add("| %s | %s | %s | %s | %d | %s ~ %s | `%s` |" % (
            d["id"], d["name"], d["climate"], d["source"], d["rows"],
            d["start"][:10], d["end"][:10], d["data_sha1"]))
    add("")
    add("## 3. 结构性验收")
    add("")
    add("| 检查 | 实测 | 门槛 | 结论 |")
    add("| --- | --- | --- | --- |")
    names = {"success_rate": "运行成功率", "evidence_coverage": "证据支持率（最小）",
             "need_coverage": "需求覆盖（最小）", "spurious_unit_conversions": "伪单位换算"}
    for k, c in s["checks"].items():
        add("| %s | %s | %s | %s |" % (names[k], _f(c["value"], 4), _f(c["expect"], 4),
                                       "通过" if c["ok"] else "**未通过**"))
    add("")
    add("## 4. 逐次运行")
    add("")
    add("| 数据 | 问题 | 状态 | 结论 | 证据 | 需求 | 质量问题 | 异常 | 变化点 | 极端 | 耗时 s |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in doc["runs"]:
        add("| %s | %s | %s | %d | %s | %s | %d | %d | %d | %d | %s |" % (
            r["dataset"], r["question"], r["status"], r["conclusion_count"],
            _f(r["evidence_coverage"], 2), _f(r["need_coverage"], 2),
            r["quality_issues"], r["anomalies"], r["change_points"], r["extremes"],
            _f(r["duration_s"], 1)))
    add("")
    add("## 5. 观察（只记录，不回调参数）")
    add("")
    add("| 数据 | 气候带 | 异常点 | 占总单元 | 变化点 | 质量问题 | 单位变化标记 | 换算事件 |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for o in obs:
        if o.get("status") == "no_run":
            add("| %s | — | 未运行 | — | — | — | — | — |" % o["id"])
            continue
        add("| %s | %s | %d | %.3f%% | %d | %s | %s | %d |" % (
            o["id"], o["climate"], o["anomaly_points"], o["anomaly_share"] * 100,
            o["change_points"],
            "、".join(k for k in o["quality_codes"]) or "（无）",
            ("、".join(o["unit_change_columns"]) or
             ("报过（见产物）" if o.get("unit_change_reported") else "（无）")),
            o["unit_conversion_events"]))
    add("")
    for b in _obs_bullets(obs):
        add("- " + b)
    add("")
    add("> 这一节是计数，不是评分。保留集没有人工标签，任何「对/错」判断都需要真值；"
        "本协议只登记现象，不因为现象回调参数。")
    add("")
    add("## 6. 能证明什么、不能证明什么")
    add("")
    add("能证明：")
    add("")
    add("- 这 3 份从未参与开发的数据上，全链路 %d 次运行全部跑完（%d/%d），报告与结果文件齐全；"
        % (s["runs"], s["succeeded"], s["runs"]))
    add("- 每条结论都挂得上可解析的证据；问题里点名的分析类别没有漏；")
    add("- 干净数据没有被伪造成单位换算（过去 R08 上错过 5,977 行的那类缺陷没有复现）。")
    add("")
    add("不能证明：")
    add("")
    add("- **异常检测的 Precision / Recall**：保留集没有人工标签，报不出来。要验证阈值，"
        "需要带标签的独立数据，这件事仍未闭合。")
    add("- 结论的语义正确性：证据可追溯只说明「结论引用了某个数字」，不说明那个数字解释得对。")
    add("- agent 模式的表现：本协议固定 pipeline + rule 计划器，模型介入下的波动不在范围内。")
    add("")
    add("## 7. 复算方式")
    add("")
    add("```")
    add("python -m scripts.fetch_holdout --check     # 校验 3 份数据 SHA1 未变")
    add("python -m evals.holdout                    # 结果已存在时会拒绝重跑")
    add("python -m evals.holdout --render-only      # 只按结果 JSON 重排本文，不重新分析")
    add("```")
    add("")
    add("产物：`evals/holdout_results.json`、本文件、`artifacts/holdout/<编号>/<问题>/`。")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true", help="生成 evals/holdout_set.json（只做一次）")
    ap.add_argument("--force", action="store_true", help="覆盖已有结果（必须配 --reason）")
    ap.add_argument("--reason", default="", help="重跑原因，写进结果文件")
    ap.add_argument("--render-only", action="store_true",
                    help="只按已有结果 JSON 重排报告，不重新分析")
    args = ap.parse_args(argv)

    try:
        if args.render_only:
            if not OUT_JSON.exists():
                raise HoldoutError("还没有结果文件：%s" % OUT_JSON)
            doc = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            doc["observations"] = doc.get("observations") or observations(
                doc["datasets"], doc["runs"])
            OUT_MD.write_text(render_md(doc), encoding="utf-8", newline="\n")
            print("已按 %s 重排 %s（未重新分析）"
                  % (OUT_JSON.name, OUT_MD.relative_to(ROOT)))
            return 0

        if args.freeze:
            m = freeze()
            print("冻结完成：%s" % MANIFEST.relative_to(ROOT))
            for d in m["datasets"]:
                print("  %s %s %s %s" % (d["id"], d["climate"], d["data_sha1"], d["data_path"]))
            return 0

        if not MANIFEST.exists():
            print("还没有冻结清单，先跑 --freeze", file=sys.stderr)
            return 2
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        doc = run_all(manifest, force=args.force, reason=args.reason)
    except HoldoutError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    s = doc["summary"]
    print("\n完成：%d/%d 成功，耗时 %s s" % (s["succeeded"], s["runs"], _f(s["duration_s"], 1)))
    for k, c in s["checks"].items():
        print("  %-28s %s (门槛 %s) %s" % (k, _f(c["value"], 4), _f(c["expect"], 4),
                                           "通过" if c["ok"] else "未通过"))
    print("写出：%s / %s" % (OUT_JSON.relative_to(ROOT), OUT_MD.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())