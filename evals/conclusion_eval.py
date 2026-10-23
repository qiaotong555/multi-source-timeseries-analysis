# -*- coding: utf-8 -*-
"""A. 结论级评估：用真值标签给 agent 与 pipeline 的结论打分。

为什么需要它
------------
T2.8/T2.9 的指标回答的是「工具有没有执行、跑了几步」，不回答「结论对不对」。
本脚本只读 `artifacts/t212_compare/<case>_<mode>/result.json`，不改任何检测阈值。

口径
----
1. 异常：点级严格同系列同时间；事件级按「同系列、间隔 ≤ 3 小时」合并后按区间重叠匹配。
2. 变化点：同系列、时间差 ≤ 2 个采样点。
3. 趋势/季节性：标签未覆盖，使用 `fixtures/truth_weather_v1.json` 中由生成器注入的真值；
   只评 temperature@STA_A / temperature@STA_B，其他列标 unlabeled，不计分。
4. 数据质量：只报「检测到 / 报告了但无法定位 / 未检测到」，不把无定位信息的问题硬算成 FP。
5. 工具未执行：记 `not_run`，不记 0 分，也不计入 P/R/F1 分母；另报「因未执行而未看的真值」。

运行
----
PYTHONPATH=src python -m evals.conclusion_eval
PYTHONPATH=src python -m evals.conclusion_eval --result-root artifacts/t212_compare
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Optional

from .conclusion_common import (
    CONCL_CATEGORY_TO_TOOL, DEFAULT_COMPARE_ROOT, DEFAULT_LABELS, DEFAULT_TRUTH,
    NEED_TO_CATEGORY, ROOT, UNLABELED_NEEDS, conclusion_counter,
    discover_result_dirs, executed_tools, load_json, load_labels, load_truth,
    match_events, match_points, median_interval_seconds, merge_events,
    month_distance, parse_seasonal, parse_trend, point_metrics, series_key,
    sha1_file, truth_anomalies, truth_change_points, truth_quality_events,
)

DEFAULT_OUT = os.path.join(ROOT, "evals", "conclusion_eval_results.json")


def _safe_mean(values: list[float]) -> Optional[float]:
    return round(statistics.mean(values), 4) if values else None


def _pct(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def _truth_conclusions(conclusions: list[dict[str, Any]], category: str) -> dict[str, dict[str, Any]]:
    """按 series 保存该类别的解析结果；同一 series 多条时保留第一条并标注重复。"""
    out: dict[str, dict[str, Any]] = {}
    for c in conclusions:
        if (c.get("category") or "") != category:
            continue
        parsed = parse_trend(c.get("text", "")) if category == "趋势" else parse_seasonal(c.get("text", ""))
        series = parsed.get("series")
        if not series:
            continue
        if series not in out:
            out[series] = parsed
        else:
            out[series].setdefault("duplicates", 0)
            out[series]["duplicates"] += 1
    return out


def _trend_status(parsed: Optional[dict[str, Any]], truth: dict[str, Any],
                  tool_ran: bool) -> dict[str, Any]:
    if not tool_ran:
        return {"state": "not_run", "correct": None,
                "reason": "compute_metrics 未执行"}
    if parsed is None:
        return {"state": "missing", "correct": None,
                "reason": "工具已执行，但没有该序列的趋势结论"}
    if not parsed.get("parsed"):
        return {"state": "unlabeled", "correct": None,
                "reason": "结论文本无法解析出方向，不猜"}
    expected = truth.get("trend_direction")
    if expected not in ("up", "down"):
        return {"state": "unlabeled", "correct": None,
                "reason": "该数据集没有趋势方向真值"}
    actual = parsed.get("direction")
    correct = (actual == expected)
    return {"state": "scored", "correct": correct,
            "actual_direction": actual, "expected_direction": expected,
            "ols_slope": parsed.get("ols_slope"),
            "significant": parsed.get("significant"),
            "reason": "" if correct else "方向与注入真值不一致"}


def _seasonal_status(parsed: Optional[dict[str, Any]], truth: dict[str, Any],
                     tool_ran: bool) -> dict[str, Any]:
    if not tool_ran:
        return {"state": "not_run", "correct": None,
                "reason": "compute_metrics 未执行"}
    if truth.get("seasonal_available") is False or truth.get("annual_amplitude_peak_to_trough") is None:
        return {"state": "unlabeled", "correct": None,
                "reason": "短样本不足以估计年周期，真值不设季节性"}
    if parsed is None:
        return {"state": "missing", "correct": None,
                "reason": "工具已执行，但没有该序列的季节性结论"}
    if not parsed.get("parsed"):
        return {"state": "unlabeled", "correct": None,
                "reason": "结论文本无法解析出振幅/峰值月/谷值月，不猜"}
    tol_amp = 3.0
    tol_month = 1
    amp_ok = abs(float(parsed["amplitude"]) - float(truth["annual_amplitude_peak_to_trough"])) <= tol_amp
    peak_ok = month_distance(parsed["peak_month"], truth["peak_month"]) <= tol_month
    trough_ok = month_distance(parsed["trough_month"], truth["trough_month"]) <= tol_month
    correct = bool(amp_ok and peak_ok and trough_ok)
    return {"state": "scored", "correct": correct,
            "amplitude": parsed["amplitude"],
            "expected_amplitude": truth["annual_amplitude_peak_to_trough"],
            "peak_month": parsed["peak_month"], "expected_peak_month": truth["peak_month"],
            "trough_month": parsed["trough_month"], "expected_trough_month": truth["trough_month"],
            "amplitude_ok": amp_ok, "peak_ok": peak_ok, "trough_ok": trough_ok,
            "reason": "" if correct else "振幅/峰值月/谷值月至少一项超出容差"}


def _quality_status(doc: dict[str, Any], events: list[dict[str, Any]],
                    interval_s: float) -> list[dict[str, Any]]:
    issues = (((doc.get("data") or {}).get("quality") or {}).get("issues") or [])
    codes = {i.get("code") for i in issues}
    out: list[dict[str, Any]] = []
    for ev in events:
        kind = ev.get("kind")
        status = "not_detected"
        detail = ""
        if kind == "unit_change_start":
            hits = [i for i in issues if i.get("code") == "POSSIBLE_UNIT_CHANGE"
                    and i.get("group") == ev.get("station")]
            if hits:
                status = "detected"
                detail = "POSSIBLE_UNIT_CHANGE 已定位到站点与起始时间"
            elif "POSSIBLE_UNIT_CHANGE" in codes:
                status = "reported_unlocalized"
                detail = "报告了单位变化，但无法定位到该站点/时刻"
        elif kind == "missing_gap_start":
            if "MISSING_VALUES" in codes:
                status = "reported_unlocalized"
                detail = "MISSING_VALUES 只有聚合计数，没有缺口起点"
        elif kind == "sampling_gap_start":
            if "SAMPLING_GAP" in codes:
                status = "reported_unlocalized"
                detail = "SAMPLING_GAP 有区间，但没有站点/逐条对应"
        elif kind == "dst_shift":
            if "SAMPLING_GAP" in codes:
                status = "detected"
                detail = "DST 跳变被识别为采样缺口（起点为跳变前最后一个采样点）"
        elif kind == "out_of_order_block":
            if "OUT_OF_ORDER" in codes:
                status = "reported_unlocalized"
                detail = "OUT_OF_ORDER 只有计数，没有乱序块起点"
        out.append({**ev, "status": status, "detail": detail})
    return out


def _series_alias(truth_keys: Any) -> dict[str, str]:
    """单站点结果的结论用裸列名（temperature），真值用 column@station。

    只在某列只有一个真值站点时建立别名；多站点时裸列名有歧义，不猜。
    """
    by_col: dict[str, list[str]] = defaultdict(list)
    for key in set(truth_keys):
        if "@" in key:
            col = key.split("@", 1)[0]
            by_col[col].append(key)
    return {col: keys[0] for col, keys in by_col.items() if len(keys) == 1}


def _canon(series: Optional[str], alias: dict[str, str]) -> Optional[str]:
    return alias.get(series, series) if series else series


def evaluate_run(doc: dict[str, Any], case: str, mode: str, question: str,
                 expect: list[str], labels: list[dict[str, Any]],
                 truth: dict[str, Any]) -> dict[str, Any]:
    tools = executed_tools(doc, mode)
    data = doc.get("data") or {}
    interval = median_interval_seconds(doc)
    anom_truth = truth_anomalies(labels)
    cp_truth = truth_change_points(labels)
    alias_keys = list((truth.get("series") or {}).keys()) + [k for k, _ in (anom_truth | cp_truth)]
    alias = _series_alias(alias_keys)

    detected_a = {(_canon(x.get("column"), alias), x.get("timestamp"))
                  for x in (data.get("anomalies") or []) if x.get("column") and x.get("timestamp")}
    detected_cp = {(_canon(x.get("column"), alias), x.get("timestamp"))
                   for x in (data.get("change_points") or []) if x.get("column") and x.get("timestamp")}

    if "detect_anomalies" in tools:
        point = match_points(detected_a, anom_truth, 0.0)
        det_events = merge_events([{"series": s, "timestamp": t} for s, t in detected_a])
        truth_events = [{"series": s, "start": t, "end": t} for s, t in anom_truth]
        event = match_events(det_events, truth_events)
        anomaly = {"state": "scored", "point": point, "event": event,
                   "detected_events": len(det_events)}
    else:
        anomaly = {"state": "not_run", "point": None, "event": None,
                   "reason": "detect_anomalies 未执行，不能把未看当成没异常"}

    if "detect_change_points" in tools:
        cp_point = match_points(detected_cp, cp_truth, 2.0 * interval)
        changepoint = {"state": "scored", "point": cp_point,
                       "tolerance_seconds": round(2.0 * interval, 3)}
    else:
        changepoint = {"state": "not_run", "point": None,
                       "reason": "detect_change_points 未执行"}

    conclusions = doc.get("conclusions") or []
    trend_by = {(_canon(k, alias) or k): v
                for k, v in _truth_conclusions(conclusions, "趋势").items()}
    seasonal_by = {(_canon(k, alias) or k): v
                   for k, v in _truth_conclusions(conclusions, "季节性").items()}
    truth_series = truth.get("series") or {}
    tool_ran = "compute_metrics" in tools

    trend_details = []
    for series, t in truth_series.items():
        st = _trend_status(trend_by.get(series), t, tool_ran)
        trend_details.append({"series": series, **st})
    seasonal_details = []
    for series, t in truth_series.items():
        st = _seasonal_status(seasonal_by.get(series), t, tool_ran)
        seasonal_details.append({"series": series, **st})

    # 非真值列的同类结论：不计分，只报数量。
    trend_unlabeled_series = sorted(set(trend_by) - set(truth_series))
    seasonal_unlabeled_series = sorted(set(seasonal_by) - set(truth_series))

    categories = conclusion_counter(doc)
    covered, missing, unlabeled_needs = [], [], []
    for need in expect or []:
        cat = NEED_TO_CATEGORY.get(need)
        if cat is None:
            unlabeled_needs.append(need)
        elif categories.get(cat, 0) > 0:
            covered.append(need)
        else:
            missing.append(need)

    return {
        "case": case, "mode": mode, "status": doc.get("status"),
        "question": question, "expect": expect,
        "executed_tools": tools,
        "conclusion_count": len(conclusions),
        "conclusion_categories": dict(categories),
        "anomaly": anomaly,
        "changepoint": changepoint,
        "trend": {"details": trend_details,
                  "unlabeled_series": trend_unlabeled_series},
        "seasonal": {"details": seasonal_details,
                     "unlabeled_series": seasonal_unlabeled_series},
        "quality": _quality_status(doc, truth_quality_events(labels), interval),
        "conclusion_need": {
            "covered": covered, "missing": missing,
            "unlabeled": unlabeled_needs,
            "mapped": len(covered) + len(missing),
            "coverage": _pct(len(covered), len(covered) + len(missing)),
        },
    }


def _merge_metric_dicts(items: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("tp", "fp", "fn", "detected", "truth")
    out = {k: sum(int(x.get(k) or 0) for x in items) for k in keys}
    p, r = _pct(out["tp"], out["tp"] + out["fp"]), _pct(out["tp"], out["tp"] + out["fn"])
    f1 = (round(2 * p * r / (p + r), 4) if p is not None and r is not None and p + r else None)
    return {**out, "precision": p, "recall": r, "f1": f1}


def summarize(rows: list[dict[str, Any]], truth_a: set[tuple[str, str]],
              truth_cp: set[tuple[str, str]], cp_tol: float) -> dict[str, Any]:
    by_mode: dict[str, Any] = {}
    for mode in ("pipeline", "agent"):
        rs = [r for r in rows if r["mode"] == mode]
        anom_rows = [r for r in rs if r["anomaly"]["state"] == "scored"]
        cp_rows = [r for r in rs if r["changepoint"]["state"] == "scored"]
        point_items = [r["anomaly"]["point"] for r in anom_rows]
        event_items = [r["anomaly"]["event"] for r in anom_rows]
        cp_items = [r["changepoint"]["point"] for r in cp_rows]
        truth_per_run = int(point_items[0]["truth"]) if point_items else len(truth_a)
        cp_truth_per_run = int(cp_items[0]["truth"]) if cp_items else len(truth_cp)

        trend_details = [d for r in rs for d in r["trend"]["details"]]
        seasonal_details = [d for r in rs for d in r["seasonal"]["details"]]
        trend_scored = [d for d in trend_details if d["state"] == "scored"]
        seasonal_scored = [d for d in seasonal_details if d["state"] == "scored"]

        need_rows = [r for r in rs if r.get("expect")]
        mapped_rows = [r for r in need_rows if r["conclusion_need"]["mapped"] > 0]
        mapped = sum(r["conclusion_need"]["mapped"] for r in need_rows)
        covered = sum(len(r["conclusion_need"]["covered"]) for r in need_rows)
        missing = sum(len(r["conclusion_need"]["missing"]) for r in need_rows)
        unlabeled = sum(len(r["conclusion_need"]["unlabeled"]) for r in need_rows)

        by_mode[mode] = {
            "runs": len(rs),
            "succeeded": sum(1 for r in rs if r.get("status") == "succeeded"),
            "anomaly": {
                "scored_runs": len(anom_rows),
                "not_run_runs": len(rs) - len(anom_rows),
                "truth_per_run": truth_per_run,
                "not_run_truth_instances": truth_per_run * (len(rs) - len(anom_rows)),
                "point": _merge_metric_dicts(point_items),
                "event": _merge_metric_dicts(event_items),
                "mean_precision": _safe_mean([x["precision"] for x in point_items if x.get("precision") is not None]),
                "mean_recall": _safe_mean([x["recall"] for x in point_items if x.get("recall") is not None]),
            },
            "changepoint": {
                "scored_runs": len(cp_rows),
                "not_run_runs": len(rs) - len(cp_rows),
                "truth_per_run": cp_truth_per_run,
                "not_run_truth_instances": cp_truth_per_run * (len(rs) - len(cp_rows)),
                "point": _merge_metric_dicts(cp_items),
                "mean_precision": _safe_mean([x["precision"] for x in cp_items if x.get("precision") is not None]),
                "mean_recall": _safe_mean([x["recall"] for x in cp_items if x.get("recall") is not None]),
            },
            "trend": {
                "states": dict(Counter(d["state"] for d in trend_details)),
                "scored": len(trend_scored),
                "correct": sum(1 for d in trend_scored if d["correct"]),
                "incorrect": sum(1 for d in trend_scored if not d["correct"]),
                "accuracy": _pct(sum(1 for d in trend_scored if d["correct"]), len(trend_scored)),
                "unlabeled_series": sorted({s for r in rs for s in r["trend"]["unlabeled_series"]}),
            },
            "seasonal": {
                "states": dict(Counter(d["state"] for d in seasonal_details)),
                "scored": len(seasonal_scored),
                "correct": sum(1 for d in seasonal_scored if d["correct"]),
                "incorrect": sum(1 for d in seasonal_scored if not d["correct"]),
                "accuracy": _pct(sum(1 for d in seasonal_scored if d["correct"]), len(seasonal_scored)),
                "unlabeled_series": sorted({s for r in rs for s in r["seasonal"]["unlabeled_series"]}),
            },
            "conclusion_need": {
                "runs_with_expect": len(need_rows),
                "runs_with_mapped_need": len(mapped_rows),
                "mapped": mapped, "covered": covered, "missing": missing,
                "unlabeled": unlabeled,
                "coverage": _pct(covered, mapped),
                "all_covered_rate": _pct(sum(1 for r in mapped_rows if not r["conclusion_need"]["missing"]), len(mapped_rows)),
            },
            "quality": dict(Counter(
                q["status"] for r in rs for q in r["quality"]
            )),
        }

    # 跨用例去重后的检测能力：避免把同一份 fixture 重复 13 次当成 13 倍证据。
    distinct: dict[str, Any] = {}
    for mode in ("pipeline", "agent"):
        rs_all = [r for r in rows if r["mode"] == mode]
        rs = [r for r in rs_all if r["anomaly"]["state"] == "scored"]
        det_a = set()
        det_cp = set()
        for r in rs:
            # 行里不存全量集合，从原始结果重新读会重复 IO；这里用事件计数近似不了，
            # 因此由 build_result 把集合放进 _private 字段，输出前再删掉。
            det_a |= r.get("_detected_anomalies", set())
            det_cp |= r.get("_detected_change_points", set())
        trend_by_series: dict[str, str] = {}
        seasonal_by_series: dict[str, str] = {}
        for r in rs_all:
            for d in r["trend"]["details"]:
                if d["state"] == "scored":
                    if d["correct"]:
                        trend_by_series.setdefault(d["series"], "correct")
                    else:
                        trend_by_series[d["series"]] = "incorrect"
            for d in r["seasonal"]["details"]:
                if d["state"] == "scored":
                    if d["correct"]:
                        seasonal_by_series.setdefault(d["series"], "correct")
                    else:
                        seasonal_by_series[d["series"]] = "incorrect"
        distinct[mode] = {
            "anomaly_point": _merge_metric_dicts([match_points(det_a, truth_a)]),
            "changepoint_point": _merge_metric_dicts([match_points(det_cp, truth_cp, cp_tol)]),
            "anomaly_hit": len(det_a & truth_a),
            "changepoint_hit": len(det_cp & truth_cp),
            "trend_by_series": trend_by_series,
            "seasonal_by_series": seasonal_by_series,
        }

    has_both = (by_mode["pipeline"]["runs"] > 0 and by_mode["agent"]["runs"] > 0)
    return {"by_mode": by_mode, "distinct": distinct,
            "agent_vs_pipeline": (_compare_modes(by_mode, distinct) if has_both
                                  else {"available": False,
                                        "reason": "缺少一种模式的结果，无法做两模式对照"})}


def _compare_modes(by_mode: dict[str, Any], distinct: dict[str, Any]) -> dict[str, Any]:
    p, a = by_mode["pipeline"], by_mode["agent"]
    p_a, a_a = p["anomaly"], a["anomaly"]
    p_c, a_c = p["changepoint"], a["changepoint"]
    truth_n = (int(p_a["point"].get("truth") or 0) // p_a["scored_runs"]
               if p_a["scored_runs"] else 0)
    cp_n = (int(p_c["point"].get("truth") or 0) // p_c["scored_runs"]
            if p_c["scored_runs"] else 0)
    return {
        "anomaly": {
            "pipeline_scored_runs": p_a["scored_runs"],
            "agent_scored_runs": a_a["scored_runs"],
            "agent_not_run_runs": a_a["not_run_runs"],
            "pipeline_tp_instances": p_a["point"].get("tp"),
            "agent_tp_instances": a_a["point"].get("tp"),
            "missed_instances_due_to_not_run": truth_n * a_a["not_run_runs"],
            "pipeline_fp_instances": p_a["point"].get("fp"),
            "agent_fp_instances": a_a["point"].get("fp"),
            "distinct_truth": truth_n,
            "pipeline_distinct_hit": distinct["pipeline"]["anomaly_hit"],
            "agent_distinct_hit": distinct["agent"]["anomaly_hit"],
            "agent_distinct_missed": truth_n - distinct["agent"]["anomaly_hit"],
        },
        "changepoint": {
            "pipeline_scored_runs": p_c["scored_runs"],
            "agent_scored_runs": a_c["scored_runs"],
            "agent_not_run_runs": a_c["not_run_runs"],
            "pipeline_tp_instances": p_c["point"].get("tp"),
            "agent_tp_instances": a_c["point"].get("tp"),
            "missed_instances_due_to_not_run": cp_n * a_c["not_run_runs"],
            "distinct_truth": cp_n,
            "pipeline_distinct_hit": distinct["pipeline"]["changepoint_hit"],
            "agent_distinct_hit": distinct["agent"]["changepoint_hit"],
            "agent_distinct_missed": cp_n - distinct["agent"]["changepoint_hit"],
        },
        "trend": {
            "pipeline_accuracy": p["trend"]["accuracy"],
            "agent_accuracy": a["trend"]["accuracy"],
        },
        "seasonal": {
            "pipeline_accuracy": p["seasonal"]["accuracy"],
            "agent_accuracy": a["seasonal"]["accuracy"],
        },
    }


def build_result(result_root: str = DEFAULT_COMPARE_ROOT,
                 labels_path: str = DEFAULT_LABELS,
                 truth_path: str = DEFAULT_TRUTH) -> dict[str, Any]:
    from .agent_eval import CASES

    labels = load_labels(labels_path)
    truth = load_truth(truth_path)
    case_meta = {c["id"]: c for c in CASES}
    items = discover_result_dirs(result_root)
    if not items:
        raise FileNotFoundError(
            f"没有在 {result_root} 找到 <case>_<mode>/result.json；"
            "请先运行 PYTHONPATH=src python -m evals.mode_compare")

    rows = []
    for item in items:
        doc = load_json(item["path"])
        meta = case_meta.get(item["case"], {"question": "", "expect": []})
        row = evaluate_run(doc, item["case"], item["mode"],
                           meta.get("question", ""), list(meta.get("expect") or []),
                           labels, truth)
        # 去重统计需要全量集合；写入 JSON 前会删除下划线私有字段。
        row["_detected_anomalies"] = {(x.get("column"), x.get("timestamp"))
                                      for x in ((doc.get("data") or {}).get("anomalies") or [])
                                      if x.get("column") and x.get("timestamp")}
        row["_detected_change_points"] = {(x.get("column"), x.get("timestamp"))
                                          for x in ((doc.get("data") or {}).get("change_points") or [])
                                          if x.get("column") and x.get("timestamp")}
        row["_path"] = item["path"]
        rows.append(row)

    truth_a = truth_anomalies(labels)
    truth_cp = truth_change_points(labels)
    cp_tol = 2.0 * median_interval_seconds(load_json(items[0]["path"]))
    summary = summarize(rows, truth_a, truth_cp, cp_tol)
    cases = []
    for r in rows:
        clean = {k: v for k, v in r.items() if not k.startswith("_")}
        clean["result_path"] = os.path.relpath(r["_path"], ROOT)
        cases.append(clean)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "existing_results",
        "result_root": os.path.relpath(result_root, ROOT),
        "labels": {"path": os.path.relpath(labels_path, ROOT), "sha1": sha1_file(labels_path)},
        "truth": {"path": os.path.relpath(truth_path, ROOT), "sha1": sha1_file(truth_path)},
        "summary": summary,
        "cases": cases,
        "notes": [
            "13 个用例共用同一份 weather_v1.csv；跨用例累加是重复测量，不是 13 倍独立证据。",
            "P/R/F1 只在对应工具实际执行的用例上计算；未执行记 not_run，不记 0。",
            "点级异常按严格同一时刻匹配；事件级按同系列间隔≤3小时合并后按区间重叠匹配。",
            "变化点容差为 2 个采样点；异常点不做时间容差。",
            "趋势/季节性只评 temperature@STA_A/STA_B；其他列无真值，标 unlabeled。",
            "数据质量事件只报 detected/reported_unlocalized/not_detected，不把无定位信息硬算成 FP。",
        ],
    }


def format_table(result: dict[str, Any]) -> str:
    lines = ["| 模式 | 异常点级 P/R/F1 | 异常事件级 P/R/F1 | 变化点 P/R/F1 | 趋势正确 | 季节正确 | 结论需求覆盖 |",
             "|---|---|---|---|---|---|---|"]
    for mode in ("pipeline", "agent"):
        m = result["summary"]["by_mode"][mode]
        a, e, c = m["anomaly"]["point"], m["anomaly"]["event"], m["changepoint"]["point"]

        def fmt(x: dict[str, Any]) -> str:
            if not x or x.get("precision") is None:
                return "—"
            return "%.3f / %.3f / %.3f" % (x["precision"], x["recall"], x["f1"])

        lines.append("| %s | %s | %s | %s | %d/%d | %d/%d | %s |" % (
            mode, fmt(a), fmt(e), fmt(c),
            m["trend"]["correct"], m["trend"]["scored"],
            m["seasonal"]["correct"], m["seasonal"]["scored"],
            "—" if m["conclusion_need"]["coverage"] is None
            else "%.2f" % m["conclusion_need"]["coverage"]))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="A. 结论级评估（只读 result.json）")
    ap.add_argument("--result-root", default=DEFAULT_COMPARE_ROOT)
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--truth", default=DEFAULT_TRUTH)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    result = build_result(args.result_root, args.labels, args.truth)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    if not args.quiet:
        print("=== A. 结论级评估 ===")
        print(format_table(result))
        av = result["summary"]["agent_vs_pipeline"]
        print()
        if av.get("available") is False:
            print("两模式对照：跳过（%s）" % av.get("reason"))
        else:
            print("异常：pipeline %d 次检出、agent %d 次；agent 未执行 %d 个用例，"
                  "因此未看 %d 个真值实例；去重后 agent 漏 %d 个真值事件。" % (
                      av["anomaly"]["pipeline_tp_instances"], av["anomaly"]["agent_tp_instances"],
                      av["anomaly"]["agent_not_run_runs"],
                      av["anomaly"]["missed_instances_due_to_not_run"],
                      av["anomaly"]["agent_distinct_missed"]))
        print("结果写入：", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())