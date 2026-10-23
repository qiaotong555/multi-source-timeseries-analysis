# -*- coding: utf-8 -*-
"""结论级评估共用函数：标签、真值、结果读取、事件匹配与结论文本解析。

这里只放纯函数和只读加载逻辑，不运行引擎、不调用模型。这样结论级评估可以
直接复核历史 `result.json`，也能在单测里用手写小样本验证口径。
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Iterable, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "fixtures")
DEFAULT_COMPARE_ROOT = os.path.join(ROOT, "artifacts", "t212_compare")
DEFAULT_LABELS = os.path.join(FIX, "labels_weather_v1.csv")
DEFAULT_TRUTH = os.path.join(FIX, "truth_weather_v1.json")

WEATHER_TOOLS = (
    "compute_metrics",
    "analyze_spectrum",
    "run_stat_tests",
    "detect_anomalies",
    "detect_change_points",
)

# 结论类别 -> 产出该类别的工具。用于差异溯源，不用于判定正确性。
CONCL_CATEGORY_TO_TOOL = {
    "趋势": "compute_metrics",
    "季节性": "compute_metrics",
    "相关性": "compute_metrics",
    "滞后关系": "compute_metrics",
    "周期": "analyze_spectrum",
    "异常检测": "detect_anomalies",
    "变化点": "detect_change_points",
}

# 问题期望项 -> 结论类别。没有专门结论类别的期望项标 unlabeled，不猜。
NEED_TO_CATEGORY = {
    "trend": "趋势",
    "seasonal": "季节性",
    "corr": "相关性",
    "lag": "滞后关系",
    "spectrum": "周期",
    "anomaly": "异常检测",
    "changepoint": "变化点",
}

UNLABELED_NEEDS = {"extreme", "stationarity", "autocorr", "normality", "test"}


def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_labels(path: str = DEFAULT_LABELS) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_truth(path: str = DEFAULT_TRUTH) -> dict[str, Any]:
    return load_json(path)


def series_key(column: str, station: Optional[str]) -> str:
    """把标签里的 (column, station) 统一成结果里的 `column@station` 形式。"""
    station = (station or "").strip()
    if station and station not in ("—", "-", "nan", "None"):
        return f"{column}@{station}"
    return column


def truth_anomalies(labels: Iterable[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(series_key(x["column"], x.get("station")), x["timestamp"])
            for x in labels if x.get("kind") == "anomaly"}


def truth_change_points(labels: Iterable[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(series_key(x["column"], x.get("station")), x["timestamp"])
            for x in labels if x.get("kind") == "level_shift_start"}


def truth_quality_events(labels: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for x in labels:
        if x.get("kind") == "anomaly" or x.get("kind") == "level_shift_start":
            continue
        out.append({
            "kind": x.get("kind"),
            "column": x.get("column"),
            "station": x.get("station"),
            "timestamp": x.get("timestamp"),
            "series": series_key(x.get("column", ""), x.get("station")),
        })
    return out


def discover_result_dirs(root: str = DEFAULT_COMPARE_ROOT) -> list[dict[str, str]]:
    """发现 `<case>_<mode>/result.json` 或 `<case>/<mode>/result.json`。

    返回按 case、mode 排序的列表。调用方不需要知道目录命名细节。
    """
    found: list[dict[str, str]] = []
    if not os.path.isdir(root):
        return found
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        for mode in ("pipeline", "agent"):
            rp = os.path.join(p, "result.json")
            if name.endswith("_" + mode) and os.path.isfile(rp):
                found.append({"case": name[: -len(mode) - 1], "mode": mode, "path": rp})
        if name.endswith("_pipeline") or name.endswith("_agent"):
            continue
        for mode in ("pipeline", "agent"):
            rp = os.path.join(p, mode, "result.json")
            if os.path.isfile(rp):
                found.append({"case": name, "mode": mode, "path": rp})
    return found


def executed_tools(doc: dict[str, Any], mode: str) -> list[str]:
    if mode == "pipeline":
        # 固定流水线的分析层五个单元总是执行；这里只用于差异归因。
        return list(WEATHER_TOOLS)
    agent = ((doc.get("data") or {}).get("agent") or {})
    cov = agent.get("coverage") or {}
    tools = list(cov.get("executed_tools") or [])
    if tools:
        return tools
    states = agent.get("tool_states") or {}
    return [t for t, st in states.items() if st in ("ok", "skipped")]


def conclusion_counter(doc: dict[str, Any]) -> Counter:
    return Counter(c.get("category") or "未分类" for c in (doc.get("conclusions") or []))


def parse_series(text: str) -> Optional[str]:
    m = re.match(r"^([^：]+)：", (text or "").strip())
    return m.group(1).strip() if m else None


def parse_trend(text: str) -> dict[str, Any]:
    """解析趋势结论。解析不了就返回 parsed=False，调用方按 unlabeled 处理。"""
    out: dict[str, Any] = {"series": parse_series(text), "parsed": False,
                           "significant": None, "direction": None,
                           "ols_slope": None}
    t = text or ""
    if "未发现显著单调趋势" in t:
        out["significant"] = False
        out["direction"] = "none"
        out["parsed"] = True
    elif "存在显著单调趋势" in t:
        out["significant"] = True
    m = re.search(r"方向\s*(上升|下降)", t)
    if m:
        out["direction"] = "up" if m.group(1) == "上升" else "down"
        out["parsed"] = True
    m = re.search(r"OLS\s*斜率\s*([+-]?[0-9]+(?:\.[0-9]+)?)", t)
    if m:
        try:
            out["ols_slope"] = float(m.group(1))
        except ValueError:
            pass
    if out["direction"] is None:
        out["parsed"] = False
    return out


def parse_seasonal(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"series": parse_series(text), "parsed": False,
                           "amplitude": None, "peak_month": None,
                           "trough_month": None}
    t = text or ""
    m = re.search(r"月均振幅\s*([0-9]+(?:\.[0-9]+)?)", t)
    if m:
        out["amplitude"] = float(m.group(1))
    m = re.search(r"峰值月\s*([0-9]{1,2})\s*月", t)
    if m:
        out["peak_month"] = int(m.group(1))
    m = re.search(r"谷值月\s*([0-9]{1,2})\s*月", t)
    if m:
        out["trough_month"] = int(m.group(1))
    out["parsed"] = (out["amplitude"] is not None
                     and out["peak_month"] is not None
                     and out["trough_month"] is not None)
    return out


def _ratio(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def _f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None or (precision + recall) == 0:
        return None
    return round(2 * precision * recall / (precision + recall), 4)


def point_metrics(detected: set[tuple[str, str]], truth: set[tuple[str, str]]) -> dict[str, Any]:
    """集合级 TP/FP/FN；要求调用方已经按时间容差完成匹配。"""
    tp = len(detected & truth)
    fp = len(detected - truth)
    fn = len(truth - detected)
    p, r = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "detected": len(detected),
            "truth": len(truth), "precision": p, "recall": r, "f1": _f1(p, r)}


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def match_points(detected: set[tuple[str, str]], truth: set[tuple[str, str]],
                 tolerance_seconds: float = 0.0) -> dict[str, Any]:
    """同系列、时间差 ≤ 容差即命中；每个真值最多匹配一个检出。"""
    used: set[tuple[str, str]] = set()
    for d in sorted(detected):
        dt = _parse_ts(d[1])
        cands = []
        for t in truth:
            if t[0] != d[0] or t in used:
                continue
            delta = abs((_parse_ts(t[1]) - dt).total_seconds())
            if delta <= tolerance_seconds:
                cands.append((delta, t))
        if cands:
            cands.sort(key=lambda x: (x[0], x[1][1]))
            used.add(cands[0][1])
    tp = len(used)
    fp = len(detected) - tp
    fn = len(truth) - tp
    p, r = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "detected": len(detected),
            "truth": len(truth), "precision": p, "recall": r, "f1": _f1(p, r)}


def merge_events(items: Iterable[dict[str, Any]], gap_seconds: float = 3 * 3600) -> list[dict[str, Any]]:
    """按 series 合并相邻时间点；默认间隔 ≤ 3 小时算同一事件。"""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for x in items:
        series = x.get("series") or x.get("column")
        if not series:
            continue
        grouped[series].append(x)
    out: list[dict[str, Any]] = []
    for series, xs in grouped.items():
        xs = sorted(xs, key=lambda x: x["timestamp"])
        cur: list[dict[str, Any]] = []
        for x in xs:
            t = _parse_ts(x["timestamp"])
            if cur and (t - _parse_ts(cur[-1]["timestamp"])).total_seconds() > gap_seconds:
                out.append({"series": series, "start": cur[0]["timestamp"],
                            "end": cur[-1]["timestamp"], "count": len(cur),
                            "timestamps": [y["timestamp"] for y in cur]})
                cur = []
            cur.append(x)
        if cur:
            out.append({"series": series, "start": cur[0]["timestamp"],
                        "end": cur[-1]["timestamp"], "count": len(cur),
                        "timestamps": [y["timestamp"] for y in cur]})
    return out


def match_events(detected: list[dict[str, Any]], truth: list[dict[str, Any]],
                 tolerance_seconds: float = 0.0) -> dict[str, Any]:
    """事件区间有交集即命中；每个检出事件最多匹配一个真值事件。"""
    used: set[int] = set()
    matched = 0
    for d in sorted(detected, key=lambda x: (x["series"], x["start"])):
        ds, de = _parse_ts(d["start"]), _parse_ts(d["end"])
        cands = []
        for i, t in enumerate(truth):
            if i in used or t["series"] != d["series"]:
                continue
            ts, te = _parse_ts(t["start"]), _parse_ts(t["end"])
            if ds <= te + timedelta(seconds=tolerance_seconds) and ts <= de + timedelta(seconds=tolerance_seconds):
                cands.append((abs((ts - ds).total_seconds()), i))
        if cands:
            cands.sort()
            used.add(cands[0][1])
            matched += 1
    tp = matched
    fp = len(detected) - tp
    fn = len(truth) - tp
    p, r = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "detected": len(detected),
            "truth": len(truth), "precision": p, "recall": r, "f1": _f1(p, r)}


def median_interval_seconds(doc: dict[str, Any]) -> float:
    ta = (((doc.get("data") or {}).get("profile") or {}).get("time_axis") or {})
    v = ta.get("median_interval_seconds")
    try:
        return float(v) if v else 3600.0
    except (TypeError, ValueError):
        return 3600.0


def month_distance(a: int, b: int) -> int:
    d = abs(int(a) - int(b)) % 12
    return min(d, 12 - d)


def result_by_case_mode(root: str) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for item in discover_result_dirs(root):
        out[(item["case"], item["mode"])] = item
    return out