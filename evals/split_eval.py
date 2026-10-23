# -*- coding: utf-8 -*-
"""评估集划分：dev / holdout 前向滚动验证。

解决的问题
----------
原先只有一套带标签数据，异常检测阈值在同一套数据上选定又报告，属于自证。
本模块把时间序列切成「前段 dev」与「紧随其后的 holdout」：

- 阈值只在 dev 上扫描选定；
- 选中的阈值在 holdout 上端到端重跑，只报告 holdout 的结果；
- 同时给出 dev 与 holdout 的差距，差距过大说明阈值不稳定。

切分必须按**时间**而不能随机：时序数据存在自相关与季节结构，随机切分会让
holdout 样本与 dev 样本在时间上交错，等价于把答案泄漏给调参环节。

滚动的意义
----------
单次切分的 holdout 结果仍可能是运气。采用三折前向滚动（dev 逐折扩大、holdout
紧随其后），若三折选出的阈值一致且 holdout 指标接近，才说明阈值稳健。

已知限制
--------
三折的 holdout 段存在重叠，不构成完全独立的重复实验。异常真值来自评估专用集
`weather_eval_v1`（每折 holdout 约 12 个），与演示集 `weather_v1` 分离。
"""
from __future__ import annotations

import contextlib
import json
import os
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

# 评估专用集：异常密度高于演示集，避免每折 holdout 只有三五条真值
EVAL_DATASET = "weather_eval_v1.csv"
EVAL_LABELS = "labels_weather_eval_v1.csv"

# 三折前向滚动：dev 逐折扩大，holdout 紧随其后、长度固定为 20%
FOLDS = [
    {"name": "fold1", "dev_ratio": 0.60, "holdout_len": 0.20},
    {"name": "fold2", "dev_ratio": 0.70, "holdout_len": 0.20},
    {"name": "fold3", "dev_ratio": 0.80, "holdout_len": 0.20},
]

# dev 上的阈值扫描网格。上界放到 12：若阈值仍选在边界，说明在当前数据上
# 「越大越好」，应作为数据特性写进结论，而不是把边界值当作最优解。
THRESHOLD_GRID = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]

MIN_PRECISION_TARGET = 0.70
MIN_RECALL_TARGET = 0.70
# 容差取 0.20 而非 0.10：每折 holdout 约 12 个真值，单个样本的波动就是 1/12≈0.083，
# 0.10 的容差只容得下一条样本的差异，任何正常抽样波动都会被判成「不稳定」。
MAX_GAP_TOLERANCE = 0.20
MIN_TRUTH_FOR_JUDGEMENT = 5
MIN_VALID_FOLDS = 2


# ---------------- 计划层固定 ----------------

@contextlib.contextmanager
def rule_planner_env(mode: str = "rule"):
    """在评估期间固定计划层，退出时恢复原值。

    评估必须可复算。若沿用 MTSA_PLANNER=auto 且配置了模型，计划层会按问题
    裁剪分析项（例如「极端事件」不会选 anomaly），holdout 端到端运行就不再执行
    被评估的检测器，指标会静默变成 0。LLM 计划能力由 evals.llm_eval 单独评估，
    不进入阈值指标的复算链。
    """
    old = os.environ.get("MTSA_PLANNER")
    os.environ["MTSA_PLANNER"] = mode
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("MTSA_PLANNER", None)
        else:
            os.environ["MTSA_PLANNER"] = old


def holdout_plan_problem(analyses, required=("anomaly",)):
    """holdout 计划缺少被评估的分析项时返回原因，否则返回 None。

    这是「静默 0 分」的护栏：检测器没执行与检测器全漏必须区分开。
    """
    have = set(analyses or ())
    missing = [a for a in required if a not in have]
    if missing:
        return (f"holdout 计划未包含 {missing}，被评估的检测器没有执行；"
                "本次指标无意义，记为 harness_error 而不是 0 分")
    return None


# ---------------- 切分 ----------------

def split_by_time(frame: pd.DataFrame, time_column: str,
                  start_ratio: float, end_ratio: float) -> tuple[pd.DataFrame, dict]:
    """按唯一时间点切分区间 [start_ratio, end_ratio)。

    以「唯一时间点」而非行号定位：多实体共享时间轴时，同一时刻可能有多行，
    按行号切会把同一时刻的记录劈到两侧，破坏实体完整性。
    """
    ts = pd.to_datetime(frame[time_column])
    uniq = np.sort(pd.unique(ts.dropna()))
    n = len(uniq)
    if n == 0:
        raise ValueError("时间列为空，无法切分")
    i0 = int(np.floor(n * start_ratio))
    i1 = int(np.floor(n * end_ratio))
    i0 = min(max(i0, 0), n - 1)
    i1 = min(max(i1, i0 + 1), n)
    lo = uniq[i0]
    hi = uniq[i1] if i1 < n else None

    mask = ts >= lo
    if hi is not None:
        mask &= ts < hi
    part = frame.loc[mask].copy()
    info = {
        "start": str(pd.Timestamp(lo)),
        "end": str(pd.Timestamp(hi)) if hi is not None else None,
        "rows": int(len(part)),
        "unique_timepoints": int(i1 - i0),
    }
    return part, info


def split_labels(labels: pd.DataFrame, start: str, end: Optional[str]) -> pd.DataFrame:
    """按时间区间过滤真值标签（与数据切分口径一致）。"""
    if labels.empty:
        return labels
    ts = pd.to_datetime(labels["timestamp"])
    mask = ts >= pd.Timestamp(start)
    if end is not None:
        mask &= ts < pd.Timestamp(end)
    return labels.loc[mask].copy()


# ---------------- 真值比对 ----------------

def _anomaly_pairs(anomalies: Iterable[Any]) -> list[tuple[str, str]]:
    """统一成 (列名, 时间戳字符串) 二元组，兼容字典与对象两种形态。"""
    out = []
    for a in anomalies:
        if isinstance(a, dict):
            col, ts = a["column"], a["timestamp"]
        else:
            col, ts = a.column, a.timestamp
        out.append((str(col), str(pd.Timestamp(ts))))
    return out


def _truth_pairs(labels: pd.DataFrame) -> list[tuple[str, str]]:
    out = []
    for _, r in labels[labels["kind"] == "anomaly"].iterrows():
        out.append((f"{r['column']}@{r['station']}", str(pd.Timestamp(r["timestamp"]))))
    return out


def anomaly_pr(anomalies: Iterable[Any], labels: pd.DataFrame) -> dict:
    """计算点级精确率与召回率。

    分母只统计有真值的列：未注入异常的列（如降雨）上的检出不计入假阳性，
    否则会把「极端值检测」这类合理检出误判成误报。
    """
    truth_pairs = _truth_pairs(labels)
    truth_by_col: dict[str, set] = {}
    for col, ts in truth_pairs:
        truth_by_col.setdefault(col, set()).add(ts)

    got_by_col: dict[str, set] = {}
    for col, ts in _anomaly_pairs(anomalies):
        if col in truth_by_col:
            got_by_col.setdefault(col, set()).add(ts)

    tp = fp = fn = 0
    for col, ts_set in truth_by_col.items():
        got = got_by_col.get(col, set())
        tp += len(ts_set & got)
        fp += len(got - ts_set)
        fn += len(ts_set - got)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "truth_count": tp + fn,
        "detected_count": tp + fp,
    }


def _merge_events(pairs: Iterable[tuple[str, str]], max_gap_seconds: float) -> list[tuple]:
    """把同一列的相邻检出点合并为事件，返回 [(列, 起, 止)]。"""
    by_col: dict[str, list] = {}
    for col, ts in pairs:
        by_col.setdefault(col, []).append(pd.Timestamp(ts))
    events = []
    for col, lst in by_col.items():
        lst.sort()
        start = prev = lst[0]
        for t in lst[1:]:
            if (t - prev).total_seconds() <= max_gap_seconds:
                prev = t
            else:
                events.append((col, start, prev))
                start = prev = t
        events.append((col, start, prev))
    return events


def event_pr(anomalies: Iterable[Any], labels: pd.DataFrame,
             max_gap_seconds: float = 3 * 3600) -> dict:
    """事件级精确率与召回率，并按异常类型分别给出召回。

    连续多点的偏离属于**一个**异常事件。若按点比对，一段 8 小时的异常会被判成
    8 个误报，指标会随异常持续时间线性失真。事件级比对才是运维视角下的正确口径。
    """
    anom = labels[labels["kind"] == "anomaly"]
    truth: list[dict] = []
    if anom.empty:
        return {"precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "fn": 0,
                "truth_events": 0, "detected_events": 0, "recall_by_type": {}}

    if "event_id" in anom.columns:
        for (col, st, _eid), g in anom.groupby(["column", "station", "event_id"], sort=False):
            ts = pd.to_datetime(g["timestamp"])
            atype = g["anomaly_type"].iloc[0] if "anomaly_type" in g.columns else "point"
            truth.append({"key": f"{col}@{st}", "start": ts.min(), "end": ts.max(),
                          "type": str(atype)})
    else:
        for (col, st), g in anom.groupby(["column", "station"], sort=False):
            cols = pd.to_datetime(g["timestamp"]).sort_values()
            start = prev = cols.iloc[0]
            for t in cols.iloc[1:]:
                if (t - prev).total_seconds() <= max_gap_seconds:
                    prev = t
                else:
                    truth.append({"key": f"{col}@{st}", "start": start, "end": prev, "type": "point"})
                    start = prev = t
            truth.append({"key": f"{col}@{st}", "start": start, "end": prev, "type": "point"})

    det = _merge_events(_anomaly_pairs(anomalies), max_gap_seconds)
    # 只比较有真值的列，与点级口径保持一致：未注入异常的列（如降雨）上的合理检出
    # 不应被算成假阳性事件，否则事件级精确率会被系统性压低。
    truth_cols = {t["key"] for t in truth}
    det = [d for d in det if d[0] in truth_cols]

    used: set[int] = set()
    by_type: dict[str, dict[str, int]] = {}
    tp = 0
    for t in truth:
        bucket = by_type.setdefault(t["type"], {"tp": 0, "fn": 0})
        hit = None
        for i, d in enumerate(det):
            if i in used or d[0] != t["key"]:
                continue
            if d[1] <= t["end"] and d[2] >= t["start"]:   # 时间区间有交集
                hit = i
                break
        if hit is None:
            bucket["fn"] += 1
        else:
            used.add(hit)
            bucket["tp"] += 1
            tp += 1

    fp = len(det) - len(used)
    fn = len(truth) - tp
    recall_by_type = {
        k: round(v["tp"] / (v["tp"] + v["fn"]), 4) if (v["tp"] + v["fn"]) else 0.0
        for k, v in by_type.items()
    }
    return {
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
        "truth_events": len(truth),
        "detected_events": len(det),
        "recall_by_type": recall_by_type,
        "truth_by_type": {k: v["tp"] + v["fn"] for k, v in by_type.items()},
    }



# ---------------- 阈值选择 ----------------

def _f1(precision: float, recall: float) -> float:
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    """二项比例的 Wilson 置信下界。

    真值样本只有几条时，点估计 1.0 会严重高估可靠性；给出下界才能如实表达
    「全检出」究竟意味着什么（4 个样本全中，95% 下界约 0.51）。
    """
    if n <= 0:
        return 0.0
    phat = k / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5)
    return round(max(0.0, (centre - margin) / denom), 4)


def select_threshold(scan: list[dict], min_precision: float = MIN_PRECISION_TARGET) -> dict:
    """在 dev 扫描结果上选出阈值。

    规则（顺序固定，避免「看 holdout 再回调」）：
    1. 在精确率达标且召回大于零的候选中，取召回最高者；
    2. 平票取阈值更大者（更保守，减少误报）；
    3. 若无候选达标，退化为 F1 最大者，同样平票取更大阈值。
    """
    if not scan:
        raise ValueError("扫描结果为空，无法选择阈值")

    ok = [r for r in scan if r["precision"] >= min_precision and r["recall"] > 0]
    if ok:
        best = max(ok, key=lambda r: (r["recall"], r["threshold"]))
        rule = f"precision>={min_precision} 前提下召回最高"
    else:
        best = max(scan, key=lambda r: (_f1(r["precision"], r["recall"]), r["threshold"]))
        rule = "无候选满足精确率约束，退化为 F1 最大"
    return {"threshold": best["threshold"], "rule": rule,
            "dev_precision": best["precision"], "dev_recall": best["recall"]}


# ---------------- dev 扫描（轻量路径） ----------------

def prepare_clean(csv_path: str, scene_id: str, question: str):
    """只跑到清洗为止，供阈值扫描复用（不含图表与报告，避免重复渲染）。"""
    from mtsa import io_loaders, preprocess, profiler, quality
    from mtsa.scenarios import stock as _s  # noqa: F401
    from mtsa.scenarios import weather as _w  # noqa: F401
    from mtsa.scenarios.base import get_adapter

    raw = io_loaders.load_dataset(csv_path)
    profile, parsed = profiler.profile_dataset(raw)
    qrep = quality.check_quality(raw, profile, parsed)
    adapter = get_adapter(scene_id)
    plan = adapter.plan(question, profile, qrep, raw.frame)
    clean = preprocess.clean_dataset(raw, profile, parsed, plan)
    return clean, plan


def scan_thresholds(clean, plan, labels: pd.DataFrame,
                    grid: Optional[list[float]] = None) -> list[dict]:
    """在给定数据段上逐个阈值计算 P/R。"""
    from mtsa import anomaly

    rows = []
    for thr in (grid or THRESHOLD_GRID):
        plan.parameters["anomaly_threshold"] = thr
        pts = anomaly.detect_anomalies(clean, plan)
        pr = anomaly_pr(pts, labels)
        rows.append({
            "threshold": float(thr),
            "precision": pr["precision"],
            "recall": pr["recall"],
            "f1": round(_f1(pr["precision"], pr["recall"]), 4),
            "tp": pr["tp"], "fp": pr["fp"], "fn": pr["fn"],
            "detected_count": pr["detected_count"],
        })
    return rows


# ---------------- 主流程 ----------------

def run_split_eval(
    base_csv: str,
    labels_csv: str,
    scene_id: str,
    question: str,
    out_root: str,
    folds: Optional[list[dict]] = None,
    verbose: bool = False,
    planner_mode: str = "rule",
) -> dict:
    """执行三折前向滚动评估，返回结构化结果。

    planner_mode 默认 rule：阈值指标必须只依赖本地确定性代码。若用 LLM 计划，
    问题措辞会改变分析项集合（例如不选 anomaly），指标随模型版本漂移。
    """
    from mtsa.engine import run_analysis
    from mtsa.models import AnalysisRequest

    frame = pd.read_csv(base_csv)
    labels_all = pd.read_csv(labels_csv)
    time_col = "timestamp" if "timestamp" in frame.columns else frame.columns[0]

    fold_reports = []
    with rule_planner_env(planner_mode):
        for fold in (folds or FOLDS):
            name = fold["name"]
            dev_start, dev_end = 0.0, fold["dev_ratio"]
            hold_start, hold_end = fold["dev_ratio"], min(1.0, fold["dev_ratio"] + fold["holdout_len"])

            fold_dir = os.path.join(out_root, name)
            os.makedirs(fold_dir, exist_ok=True)

            dev_df, dev_info = split_by_time(frame, time_col, dev_start, dev_end)
            hold_df, hold_info = split_by_time(frame, time_col, hold_start, hold_end)
            dev_labels = split_labels(labels_all, dev_info["start"], dev_info["end"])
            hold_labels = split_labels(labels_all, hold_info["start"], hold_info["end"])

            dev_csv = os.path.join(fold_dir, "dev.csv")
            hold_csv = os.path.join(fold_dir, "holdout.csv")
            dev_df.to_csv(dev_csv, index=False, encoding="utf-8-sig")
            hold_df.to_csv(hold_csv, index=False, encoding="utf-8-sig")

            # 1) 只在 dev 上扫描并选定阈值
            clean_dev, plan_dev = prepare_clean(dev_csv, scene_id, question)
            scan = scan_thresholds(clean_dev, plan_dev, dev_labels)
            choice = select_threshold(scan)

            # 2) 用选中的阈值在 holdout 上端到端重跑（不被 dev 结果参看）
            req = AnalysisRequest(
                request_id=f"split_{name}", input_path=hold_csv, scene_id=scene_id,
                question=question, seed=20260927,
                param_overrides={"anomaly_threshold": choice["threshold"]},
                output_dir=os.path.join(fold_dir, "holdout_run"),
            )
            res = run_analysis(req, verbose=False)
            if res.status != "succeeded":
                fold_reports.append({
                    "fold": name, "status": res.status,
                    "error": res.error.model_dump() if res.error else None,
                    "dev": dev_info, "holdout": hold_info,
                })
                continue

            plan = res.data.get("plan") or {}
            problem = holdout_plan_problem(plan.get("analyses"))
            if problem:
                # 检测器没执行，不能按 0 分计入指标：那是评估链路的错误，不是模型/算法的失败。
                fold_reports.append({
                    "fold": name, "status": "harness_error", "error": problem,
                    "plan_source": plan.get("plan_source"),
                    "plan_analyses": plan.get("analyses") or [],
                    "dev": dev_info, "holdout": hold_info,
                })
                if verbose:
                    print(f"  {name}: harness_error —— {problem}")
                continue

            pr_hold = anomaly_pr(res.data["anomalies"], hold_labels)
            ev_hold = event_pr(res.data["anomalies"], hold_labels)
            type_recall = ev_hold.get("recall_by_type", {})

            fold_reports.append({
                "fold": name,
                "status": "succeeded",
                "plan_source": plan.get("plan_source"),
                "plan_analyses": plan.get("analyses") or [],
                "dev": dev_info, "holdout": hold_info,
                "dev_truth_count": int((dev_labels["kind"] == "anomaly").sum()),
                "holdout_truth_count": pr_hold["truth_count"],
                "scan": scan,
                "selected_threshold": choice["threshold"],
                "selection_rule": choice["rule"],
                "dev_precision": choice["dev_precision"],
                "dev_recall": choice["dev_recall"],
                "holdout_precision": pr_hold["precision"],
                "holdout_recall": pr_hold["recall"],
                "holdout_tp": pr_hold["tp"], "holdout_fp": pr_hold["fp"], "holdout_fn": pr_hold["fn"],
                # 差距取有方向的「退化量」= dev − holdout。holdout 反而更好时为正数以外的
                # 负值，说明没有过拟合，不应判为不稳定。
                "degradation_precision": round(choice["dev_precision"] - pr_hold["precision"], 4),
                "degradation_recall": round(choice["dev_recall"] - pr_hold["recall"], 4),
                "gap_precision": round(abs(choice["dev_precision"] - pr_hold["precision"]), 4),
                "gap_recall": round(abs(choice["dev_recall"] - pr_hold["recall"]), 4),
                "holdout_precision_lower95": wilson_lower(pr_hold["tp"], pr_hold["tp"] + pr_hold["fp"]),
                "holdout_recall_lower95": wilson_lower(pr_hold["tp"], pr_hold["tp"] + pr_hold["fn"]),
                "holdout_event_precision": ev_hold["precision"],
                "holdout_event_recall": ev_hold["recall"],
                "holdout_event_truth": ev_hold["truth_events"],
                "holdout_event_detected": ev_hold["detected_events"],
                "holdout_recall_by_type": type_recall,
                "holdout_truth_by_type": ev_hold.get("truth_by_type", {}),
                "insufficient_truth": pr_hold["truth_count"] < MIN_TRUTH_FOR_JUDGEMENT,
            })
            if verbose:
                print(f"  {name}: dev阈值={choice['threshold']} "
                      f"dev P/R={choice['dev_precision']}/{choice['dev_recall']} "
                      f"hold P/R={pr_hold['precision']}/{pr_hold['recall']} "
                      f"事件 P/R={ev_hold['precision']}/{ev_hold['recall']} "
                      f"分类型={type_recall} truth={pr_hold['truth_count']}")

    return summarize(fold_reports)


def summarize(fold_reports: list[dict]) -> dict:
    """汇总三折结果并给出通过判定。"""
    valid = [f for f in fold_reports if f.get("status") == "succeeded" and not f.get("insufficient_truth")]
    skipped = [f["fold"] for f in fold_reports
               if f.get("status") != "succeeded" or f.get("insufficient_truth")]

    if not valid:
        return {"folds": fold_reports, "summary": {
            "passed": False, "reason": "没有可用于判定的折（全部失败或真值样本不足）",
            "skipped_folds": skipped}}

    mean_p = float(np.mean([f["holdout_precision"] for f in valid]))
    mean_r = float(np.mean([f["holdout_recall"] for f in valid]))
    mean_ev_r = float(np.mean([f.get("holdout_event_recall", 0.0) for f in valid]))
    collective = [f["holdout_recall_by_type"].get("collective", 0.0) for f in valid
                  if "collective" in f.get("holdout_recall_by_type", {})]
    mean_coll_r = float(np.mean(collective)) if collective else None
    max_deg_p = float(max(f.get("degradation_precision", 0.0) for f in valid))
    max_deg_r = float(max(f.get("degradation_recall", 0.0) for f in valid))
    thresholds = sorted({f["selected_threshold"] for f in valid})
    truth_total = int(sum(f.get("holdout_truth_count", 0) for f in valid))

    checks = {
        "holdout_precision>=0.70": mean_p >= MIN_PRECISION_TARGET,
        "holdout_recall>=0.70": mean_r >= MIN_RECALL_TARGET,
        "event_recall>=0.70": mean_ev_r >= MIN_RECALL_TARGET,
        "precision_degradation<=0.20": max_deg_p <= MAX_GAP_TOLERANCE,
        "recall_degradation<=0.20": max_deg_r <= MAX_GAP_TOLERANCE,
        f"valid_folds>={MIN_VALID_FOLDS}": len(valid) >= MIN_VALID_FOLDS,
    }
    if mean_coll_r is not None:
        checks["collective_recall>=0.70"] = mean_coll_r >= MIN_RECALL_TARGET
    # 有效折不足时，P/R 的点估计不可用于验收：判为不可判定，而不是「通过」。
    # 宁可承认证据不足，也不能让几条样本撑起的 1.0 变成验收结论。
    reasons = []
    if len(valid) < MIN_VALID_FOLDS:
        reasons.append(f"有效折数 {len(valid)} < {MIN_VALID_FOLDS}")
    if truth_total < MIN_TRUTH_FOR_JUDGEMENT:
        reasons.append(f"有效真值合计 {truth_total} < {MIN_TRUTH_FOR_JUDGEMENT}")
    inconclusive = bool(reasons)
    return {
        "folds": fold_reports,
        "summary": {
            "passed": all(checks.values()) and not inconclusive,
            "inconclusive": inconclusive,
            "checks": checks,
            "mean_holdout_precision": round(mean_p, 4),
            "mean_holdout_recall": round(mean_r, 4),
            "mean_holdout_event_recall": round(mean_ev_r, 4),
            "mean_collective_recall": (round(mean_coll_r, 4) if mean_coll_r is not None else None),
            "max_degradation_precision": round(max_deg_p, 4),
            "max_degradation_recall": round(max_deg_r, 4),
            "gap_tolerance": MAX_GAP_TOLERANCE,
            "selected_thresholds": thresholds,
            "threshold_stable": len(thresholds) == 1,
            "threshold_at_grid_edge": any(t in (THRESHOLD_GRID[0], THRESHOLD_GRID[-1]) for t in thresholds),
            "valid_folds": [f["fold"] for f in valid],
            "skipped_folds": skipped,
            "valid_truth_total": truth_total,
            "reason": ("；".join(reasons) + "，证据不足以判定") if inconclusive else None,
            "note": "holdout 段存在重叠，不构成完全独立的重复实验；"
                    "每折真值样本有限，应结合置信下界解读",
        },
    }


def main() -> int:
    """独立运行：便于单独复核切分结果。"""
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.join(root, "src") not in sys.path:
        sys.path.insert(0, os.path.join(root, "src"))
    res = run_split_eval(
        base_csv=os.path.join(root, "fixtures", EVAL_DATASET),
        labels_csv=os.path.join(root, "fixtures", EVAL_LABELS),
        scene_id="weather",
        question="分析温度的趋势、季节性与极端事件",
        out_root=os.path.join(root, "artifacts", "eval_split"),
        verbose=True,
    )
    print(json.dumps(res["summary"], ensure_ascii=False, indent=2))
    return 0 if res["summary"].get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
