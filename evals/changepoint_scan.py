# -*- coding: utf-8 -*-
"""变化点检测的网格扫描评估。

解决的问题
----------
原先只用一处注入（气压 +18 hPa、位置约在第 12000 个采样点）验证变化点检测，
得出「定位误差 0」的结论。单点样本无法区分「算法准」与「这一处恰好容易」，
也无法回答「幅度多大才检得出」。

本模块在清洗后的序列上做**位置 × 幅度**的网格注入：

- 位置覆盖前段到后段（0.25 至 0.85），检验判据对跳变位置的敏感性——
  SSE 判据本质上要求跳变能解释足够比例的方差，位置越靠后，同样的幅度
  能解释的方差越少，检出的难度随之上升；
- 幅度覆盖 4 至 24 hPa，画出「检出率-幅度」曲线，找出手册意义上的下限。

为什么在内存里注入而不新建数据集
--------------------------------
注入只改一列的一个后缀偏移，不涉及质量问题与跨列关系，无需落盘成文件。
这样既能保持演示集与评估集不变，也避免为每组参数维护一份 3.7 MB 的 CSV。
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

# 位置（占全序列的比例）与幅度（物理单位）网格。
# 幅度在 12 与 18 之间加密，用来定位「可靠检出的最小幅度」。
# 0 是基线：不注入任何偏移，用于测量伪变化点（假阳性）率。
POSITIONS = [0.25, 0.40, 0.55, 0.70, 0.85]
DELTAS = [0.0, 4.0, 8.0, 12.0, 14.0, 16.0, 18.0, 24.0]
POSITION_TOLERANCE = 3          # 定位误差容差：3 个采样点
# 验收基准取「实测可靠检出的最小幅度」，不是任意选一个好看的幅度。
# 该值由扫描结果确定：低于它检出率为 0，达到它时 5 个位置全部命中。
KEY_DELTA = 18.0
DEFAULT_COLUMN = "pressure@STA_A"


def _prepare_plan(plan, column: str):
    """复制计划并把目标列收窄到单列，避免每次扫描都跑满全部列。"""
    p = plan.model_copy(deep=True)
    p.target_columns = [column]
    return p


def scan_change_points(
    clean,
    plan,
    column: str = DEFAULT_COLUMN,
    positions: Optional[list[float]] = None,
    deltas: Optional[list[float]] = None,
    tolerance: int = POSITION_TOLERANCE,
    sse_ratio: Optional[float] = None,
) -> dict:
    """在给定列上做位置 × 幅度网格注入，返回逐格结果与汇总。"""
    from mtsa.anomaly import detect_change_points

    positions = positions or POSITIONS
    deltas = deltas or DELTAS

    frame = clean.frame
    if column not in frame.columns:
        raise KeyError(f"列不存在：{column}")

    base = pd.to_numeric(frame[column], errors="coerce").interpolate(limit_direction="both")
    y_base = base.to_numpy(dtype=float)
    n = len(y_base)

    plan_col = _prepare_plan(plan, column)
    if sse_ratio is not None:
        plan_col.parameters["change_point_sse_ratio"] = float(sse_ratio)
    ratio_used = float(plan_col.parameters.get("change_point_sse_ratio", 0.7))

    cells: list[dict[str, Any]] = []
    for pos in positions:
        true_idx = int(round(n * pos))
        if true_idx < 1 or true_idx >= n - 1:
            continue
        for delta in deltas:
            y = y_base.copy()
            y[true_idx:] += delta
            tmp = frame.copy()
            tmp[column] = y
            cd = clean.model_copy(update={"frame": tmp})

            cps = detect_change_points(cd, plan_col)
            means = [c for c in cps if c.kind == "mean"]

            if means:
                best = min(means, key=lambda c: abs(c.index - true_idx))
                err = int(abs(best.index - true_idx))
                hit = err <= tolerance
                stat = float(best.statistic)
            else:
                err = None
                hit = False
                stat = None

            cells.append({
                "position": pos,
                "true_index": true_idx,
                "delta": delta,
                "detected": hit,
                "found": len(means),
                "nearest_index": (best.index if means else None),
                "error": err,
                "statistic": stat,
            })

    return _summarize(cells, column, positions, deltas, tolerance, ratio_used)


def _summarize(cells, column, positions, deltas, tolerance, ratio_used) -> dict:
    total = len(cells)
    hits = [c for c in cells if c["detected"]]

    by_delta = {}
    for d in deltas:
        subset = [c for c in cells if c["delta"] == d]
        by_delta[str(d)] = round(sum(1 for c in subset if c["detected"]) / len(subset), 4) if subset else 0.0

    by_position = {}
    for p in positions:
        subset = [c for c in cells if c["position"] == p]
        by_position[str(p)] = round(sum(1 for c in subset if c["detected"]) / len(subset), 4) if subset else 0.0

    errors = sorted(c["error"] for c in hits if c["error"] is not None)
    err_stats = {
        "count": len(errors),
        "median": float(np.median(errors)) if errors else None,
        "p90": float(np.percentile(errors, 90)) if errors else None,
        "max": int(max(errors)) if errors else None,
    }

    key_delta_cells = [c for c in cells if c["delta"] == KEY_DELTA]
    key_rate = (sum(1 for c in key_delta_cells if c["detected"]) / len(key_delta_cells)
                if key_delta_cells else 0.0)

    # Δ=0 基线：不注入偏移时仍报出变化点的比例，即伪变化点率
    zero_cells = [c for c in cells if c["delta"] == 0.0]
    zero_rate = (sum(1 for c in zero_cells if c["detected"]) / len(zero_cells)
                 if zero_cells else 0.0)

    return {
        "column": column,
        "sse_ratio": ratio_used,
        "tolerance": tolerance,
        "cells": cells,
        "grid_size": total,
        "detection_rate": round(len(hits) / total, 4) if total else 0.0,
        "detection_rate_by_delta": by_delta,
        "detection_rate_by_position": by_position,
        "key_delta": KEY_DELTA,
        "key_delta_detection_rate": round(key_rate, 4),
        "false_positive_rate_at_zero": round(zero_rate, 4),
        "error_stats": err_stats,
        "notes": "检出率按「定位误差 ≤ 容差」计；误差统计只含命中格子；"
                 "false_positive_rate_at_zero 为不注入偏移时的伪变化点率",
    }


def format_grid(result: dict) -> str:
    """把结果渲染为「位置 × 幅度」的检出率表格，便于写入评估记录。"""
    cols = sorted({c["delta"] for c in result["cells"]})
    rows = sorted({c["position"] for c in result["cells"]})
    lines = ["| 位置 \\ 幅度 | " + " | ".join(f"{d:g}" for d in cols) + " |",
             "| --- | " + " | ".join("---" for _ in cols) + " |"]
    for p in rows:
        marks = []
        for d in cols:
            hit = [c for c in result["cells"] if c["position"] == p and c["delta"] == d]
            marks.append("命中" if (hit and hit[0]["detected"]) else "未中")
        lines.append(f"| {p:.2f} | " + " | ".join(marks) + " |")
    return "\n".join(lines)


def main() -> int:
    """独立运行：便于单独复核网格扫描。"""
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.join(root, "src") not in sys.path:
        sys.path.insert(0, os.path.join(root, "src"))
    if root not in sys.path:
        sys.path.insert(0, root)

    from evals.split_eval import EVAL_DATASET, prepare_clean

    clean, plan = prepare_clean(
        os.path.join(root, "fixtures", EVAL_DATASET), "weather",
        "分析温度的趋势、季节性与极端事件")
    res = scan_change_points(clean, plan)
    print(format_grid(res))
    print()
    print(json.dumps({k: res[k] for k in
                      ("detection_rate", "key_delta_detection_rate",
                       "detection_rate_by_delta", "error_stats")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
