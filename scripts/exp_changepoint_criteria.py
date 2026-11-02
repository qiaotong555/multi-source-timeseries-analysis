# -*- coding: utf-8 -*-
"""A1 变化点判据对照实验（只读 src，不改产线）。

要回答的问题
------------
v2 门禁的变化点 F1 = 0.50，恰好压在阈值上。先看清它由什么构成：

- 召回 1.000（8/8），精确 0.333（8/24）——**不是漏检，是过检**。
- 每次运行固定报 6 个点：pressure@STA_A 的 3 个均值点、pressure@STA_B 的 1 个、
  rainfall 两个站的方差点。真值只有 2 个（STA_A 的 3-01、STA_B 的 7-19）。

因此本实验不追求「更灵敏」，而是量化「哪些判据能在不丢召回的前提下少报」。
候选判据全部在脚本内实现，产线代码不动。

四条候选
--------
1. `local_stat`：沿用现有 SSE 递归，但用**局部分段**的中位数与稳健 σ 重新判显著，
   而不是用全序列前后缀（现有实现的嵌套分割点拿的是全局量，统计量描述的是主导跳变）。
2. `bic`：二分割接受条件改为 BIC 改善，不再用「SSE 降到 0.7 倍」这种无量纲比例。
3. `pettitt`：非参数位置检验，p 值阈值控制。
4. `no_var`：只去掉方差类变化点（零膨胀的降水列把「雨季开始」读成方差突变）。

灵敏度护栏
----------
precision 修好但灵敏度垮了等于没修。每个候选都要过 `changepoint_scan` 的
位置 × 幅度网格（5 位置 × 8 幅度），记录检出率、Δ=18 命中率、Δ=0 伪报率。

运行
----
    PYTHONPATH=src python scripts/exp_changepoint_criteria.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mtsa.quality import robust_sigma                    # noqa: E402
from mtsa.anomaly import MIN_SEGMENT, _best_split       # noqa: E402

DATASET = os.path.join(ROOT, "fixtures", "weather_eval_v2.csv")
LABELS = os.path.join(ROOT, "fixtures", "labels_weather_eval_v2.csv")
QUESTION = "分析温度的趋势、季节性和异常点，并检测变化点"
OUT = os.path.join(ROOT, "artifacts", "exp_changepoint_criteria.json")

# 门禁的匹配容差：2 小时（合成数据为小时采样，等价 2 个采样点）
MATCH_TOL = 2 * 3600
# 网格扫描沿用 changepoint_scan 的设定，保证与历史结果可比
POSITIONS = [0.25, 0.40, 0.55, 0.70, 0.85]
DELTAS = [0.0, 4.0, 8.0, 12.0, 14.0, 16.0, 18.0, 24.0]
GRID_TOL = 3
KEY_DELTA = 18.0


# ------------------------------------------------------------------ 工具
def _sse(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    return float(np.sum((y - y.mean()) ** 2))


def _segment_bounds(y: np.ndarray, min_len: int, max_depth: int,
                    ratio: float) -> list[tuple[int, int, int]]:
    """复现现有 `_segment`，但记录每个分割点所属的**父分段**。

    返回 [(k, parent_lo, parent_hi), ...]，k 是全局下标。
    现有实现只回传 k，导致后续算 before/after 时只能用全序列，无法判断
    这个点在自己那段里到底跳了多少。
    """
    out: list[tuple[int, int, int]] = []

    def rec(lo: int, hi: int, depth: int) -> None:
        if depth <= 0:
            return
        seg = y[lo:hi]
        k_rel, sse = _best_split(seg, min_len)
        if k_rel is None or not np.isfinite(sse):
            return
        sse_parent = _sse(seg)
        if sse_parent <= 0 or sse > ratio * sse_parent:
            return
        k = lo + k_rel
        out.append((k, lo, hi))
        rec(lo, k, depth - 1)
        rec(k, hi, depth - 1)

    rec(0, len(y), max_depth)
    return sorted(out)


def _sigma_diff(seg: np.ndarray) -> float:
    """用一阶差分的 MAD 估计**高频噪声**尺度。

    为什么不用 `robust_sigma(seg)`：分段内常含缓慢漂移（v2 的压力在 7300 点里
    从 1080 漂到 1067）。漂移会被 MAD 当成散布，σ 从 2.4 虚高到 8.2，
    一个 18 hPa 的真跳变于是只算 2σ，被误判成噪声。
    差分把漂移与台阶都消掉（台阶只贡献 1 个差分、被 MAD 忽略），剩下的就是噪声。
    """
    if len(seg) < 4:
        return 0.0
    d = np.diff(seg)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    if mad <= 0:
        return 0.0
    return 1.4826 * mad / math.sqrt(2.0)


def _sigma_sides(y: np.ndarray, k: int, lo: int, hi: int, min_len: int) -> float:
    """切点两侧**各自去中位数**后的合并 MAD。

    这是唯一同时躲开三个陷阱的尺度：
    - 不用全序列 MAD：会被同列其它台阶撑大（STA_A 全长 σ=6.85，真实噪声 2.4）；
    - 不用父分段 MAD：会被段内缓慢漂移撑大（[1440,8760] σ=5.83，18 hPa 只算 2σ）；
    - 不用差分 MAD：这套合成数据太平滑，差分 σ 只有 0.66，等于不设门槛。

    两侧各自减中位数后合并，台阶被消掉、漂移被局限在半个窗口内，
    剩下的是局部散布——正是「这个跳变是不是噪声」该比的分母。
    """
    h = min(max(2 * min_len, 24), max(4, (hi - lo) // 4))
    left = y[max(lo, k - h):k]
    right = y[k:min(hi, k + h)]
    if len(left) < 8 or len(right) < 8:
        return 0.0
    dev = np.concatenate([left - np.median(left), right - np.median(right)])
    mad = float(np.median(np.abs(dev)))
    if mad <= 0:
        return 0.0
    return 1.4826 * mad


def _local(y: np.ndarray, k: int, lo: int, hi: int,
           sigma_mode: str = "parent", min_len: int = 12) -> dict[str, Any]:
    """分割点在自己父分段里的局部量。

    `before` / `after` 取**切点两侧紧邻窗口**的中位数，不是前缀/后缀中位数：
    后者会被段内漂移稀释（v2 压力在 7300 点里漂 13 hPa，18 hPa 的台阶只量出 9.97），
    也会被同列其它台阶带偏。水平突变的定义就是「紧邻两侧的水平差」。
    """
    h = min(max(2 * min_len, 24), max(4, (hi - lo) // 4))
    left_win = y[max(lo, k - h):k]
    right_win = y[k:min(hi, k + h)]
    left, right, seg = y[lo:k], y[k:hi], y[lo:hi]
    before = float(np.median(left_win)) if len(left_win) else float("nan")
    after = float(np.median(right_win)) if len(right_win) else float("nan")
    jump = abs(after - before)
    if sigma_mode == "sides":
        sigma = _sigma_sides(y, k, lo, hi, min_len) or robust_sigma(pd.Series(seg))
    elif sigma_mode == "diff":
        sigma = _sigma_diff(seg) or robust_sigma(pd.Series(seg))
    else:
        sigma = robust_sigma(pd.Series(seg))
    sse_parent = _sse(seg)
    sse_split = _sse(left) + _sse(right)
    # 生产实现用**全序列**前后缀中位数，两个口径都记，便于解释差异
    before_g = float(np.median(y[:k])) if k else float("nan")
    after_g = float(np.median(y[k:])) if k < len(y) else float("nan")
    jump_global = abs(after_g - before_g)
    return {
        "before": before, "after": after, "jump": jump, "sigma": sigma,
        "jump_global": jump_global,
        "stat": (jump / sigma) if sigma > 0 else float("nan"),
        "stat_global": (jump_global / sigma) if sigma > 0 else float("nan"),
        "n": len(seg),
        "rss_ratio": (sse_split / sse_parent) if sse_parent > 0 else 1.0,
        "gain": (1.0 - sse_split / sse_parent) if sse_parent > 0 else 0.0,
    }


def _pettitt_k(y: np.ndarray) -> tuple[int, float]:
    """Pettitt 位置检验：返回 (最可能的分割点相对下标, 近似 p 值)。

    统计量 U_t = Σ_{i≤t} Σ_{j>t} sgn(y_i - y_j)，用秩的累加和做 O(n) 计算。
    近似 p = 2·exp(-6K²/(n³+n²))，只对独立样本成立；这里的数据有强自相关，
    所以它天然偏灵敏——本实验会把这个偏差量出来，而不是假设它不存在。
    """
    n = len(y)
    if n < 4:
        return 0, 1.0
    ranks = pd.Series(y).rank(method="average").to_numpy()
    u = np.cumsum(ranks) - np.arange(1, n + 1) * (n + 1) / 2.0
    k = int(np.argmax(np.abs(u)))
    stat = float(np.max(np.abs(u)))
    p = 2.0 * math.exp(-6.0 * stat ** 2 / (n ** 3 + n ** 2))
    return k + 1, min(1.0, p)


def _segment_by_criterion(y: np.ndarray, min_len: int, max_depth: int,
                          accept: Callable[[np.ndarray, int], bool]) -> list[int]:
    """二分割：每段先找 SSE 最优切点，再用 accept 决定是否接受。"""
    out: list[int] = []

    def rec(lo: int, hi: int, depth: int) -> None:
        if depth <= 0 or hi - lo < 2 * min_len:
            return
        k_rel, _ = _best_split(y[lo:hi], min_len)
        if k_rel is None:
            return
        k = lo + k_rel
        if not accept(y[lo:hi], k_rel):
            return
        out.append(k)
        rec(lo, k, depth - 1)
        rec(k, hi, depth - 1)

    rec(0, len(y), max_depth)
    return sorted(out)


def _variance_point(y: np.ndarray, w: int) -> tuple[int, float] | None:
    """现有实现的方差突变：滚动标准差的单点最大跳变，需超过基线 2 倍。"""
    roll = pd.Series(y).rolling(w, min_periods=w).std()
    d = roll.diff().abs()
    base = float(roll.median())
    if not (base > 0) or not d.notna().any():
        return None
    j = int(d.idxmax())
    if float(d.loc[j]) > 2.0 * base:
        return j, float(d.loc[j] / base)
    return None


# ------------------------------------------------------------------ 候选判据
def _detect(variant: str, y: np.ndarray, depth: int = 2, ratio: float = 0.7,
            k_stat: float = 3.0, bic_lambda: float = 1.0,
            pettitt_alpha: float = 0.01,
            sigma_mode: str = "parent") -> list[dict[str, Any]]:
    """对单列跑一个候选判据，返回 [{index, kind, stat_local, ...}]。"""
    n = len(y)
    min_len = max(MIN_SEGMENT, n // 40)
    out: list[dict[str, Any]] = []

    if variant in ("current", "no_var", "local_stat"):
        for k, lo, hi in _segment_bounds(y, min_len, depth, ratio):
            loc = _local(y, k, lo, hi, sigma_mode, min_len)
            if variant == "local_stat" and not (
                    np.isfinite(loc["stat"]) and loc["stat"] >= k_stat):
                continue
            out.append({"index": k, "kind": "mean", "parent": [lo, hi], **loc})
    elif variant == "bic":
        def accept(seg: np.ndarray, k: int) -> bool:
            nn = len(seg)
            s0 = _sse(seg)
            s1 = _sse(seg[:k]) + _sse(seg[k:])
            if s0 <= 0 or s1 <= 0:
                return False
            # 均值突变：参数 2 → 3（均值各一 + 共用方差），多 1 个参数
            return nn * math.log(s1 / s0) < -bic_lambda * math.log(nn)

        for k in _segment_by_criterion(y, min_len, depth, accept):
            loc = _local(y, k, 0, n, sigma_mode, min_len)
            out.append({"index": k, "kind": "mean", "parent": [0, n], **loc})
    elif variant == "pettitt":
        def accept(seg: np.ndarray, k: int) -> bool:
            return _pettitt_k(seg)[1] < pettitt_alpha

        for k in _segment_by_criterion(y, min_len, depth, accept):
            loc = _local(y, k, 0, n, sigma_mode, min_len)
            out.append({"index": k, "kind": "mean", "parent": [0, n], **loc})
    else:
        raise ValueError(variant)

    if variant not in ("no_var", "bic", "pettitt"):
        w = max(12, n // 40)
        var = _variance_point(y, w)
        if var is not None:
            out.append({"index": int(var[0]), "kind": "variance", "stat": var[1]})
    return out


# ------------------------------------------------------------------ 打分
def _truth() -> list[tuple[str, pd.Timestamp]]:
    labels = pd.read_csv(LABELS, encoding="utf-8-sig")
    out = []
    for _, r in labels[labels["kind"] == "level_shift_start"].iterrows():
        out.append(("%s@%s" % (r["column"], r["station"]), pd.Timestamp(r["timestamp"])))
    return out


def _score(dets: list[dict[str, Any]], truth: list[tuple[str, pd.Timestamp]],
           index: pd.Index, columns: list[str]) -> dict[str, Any]:
    """按门禁口径（同列 + 时间在 ±2h 内、1 对 1）算 tp/fp/fn。"""
    pool = list(truth)
    used: set[int] = set()
    tp = 0
    for d in dets:
        ts = pd.Timestamp(index[d["index"]])
        best, best_gap = None, None
        for i, (series, t) in enumerate(pool):
            if i in used or series != d["column"]:
                continue
            gap = abs((t - ts).total_seconds())
            if gap <= MATCH_TOL and (best_gap is None or gap < best_gap):
                best, best_gap = i, gap
        if best is not None:
            used.add(best)
            tp += 1
    fp, fn = len(dets) - tp, len(pool) - tp
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else (0.0 if prec is not None or rec is not None else None)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def _grid(variant: str, y_base: np.ndarray, **kw) -> dict[str, Any]:
    """位置 × 幅度网格：灵敏度护栏。"""
    n = len(y_base)
    cells = []
    for pos in POSITIONS:
        true_idx = int(round(n * pos))
        for delta in DELTAS:
            y = y_base.copy()
            y[true_idx:] += delta
            means = [d for d in _detect(variant, y, **kw) if d["kind"] == "mean"]
            if means:
                best = min(means, key=lambda d: abs(d["index"] - true_idx))
                hit = abs(best["index"] - true_idx) <= GRID_TOL
            else:
                hit = False
            cells.append({"position": pos, "delta": delta, "detected": hit})
    by_delta = {str(d): round(sum(1 for c in cells if c["delta"] == d and c["detected"])
                              / sum(1 for c in cells if c["delta"] == d), 4)
                for d in DELTAS}
    zero = [c for c in cells if c["delta"] == 0.0]
    return {
        "detection_rate": round(sum(1 for c in cells if c["detected"]) / len(cells), 4),
        "by_delta": by_delta,
        "key_delta_rate": round(sum(1 for c in cells if c["delta"] == KEY_DELTA and c["detected"])
                                / sum(1 for c in cells if c["delta"] == KEY_DELTA), 4),
        "false_positive_rate_at_zero": round(sum(1 for c in zero if c["detected"]) / len(zero), 4),
    }


def main() -> int:
    from evals.split_eval import prepare_clean

    clean, plan = prepare_clean(DATASET, "weather", QUESTION)
    frame = clean.frame
    cols = [c for c in plan.target_columns if c in frame.columns]
    truth = _truth()
    print("数据集 %d 行；目标列 %s" % (len(frame), cols))
    print("真值变化点 %d 个：%s" % (len(truth), [(s, str(t)) for s, t in truth]))

    series = {}
    for col in cols:
        s = pd.to_numeric(frame[col], errors="coerce").interpolate(limit_direction="both")
        series[col] = s.to_numpy(dtype=float)

    variants = [
        ("current", {}),
        ("no_var", {}),
        ("local_sides_k2", {"variant": "local_stat", "k_stat": 2.0, "sigma_mode": "sides"}),
        ("local_sides_k25", {"variant": "local_stat", "k_stat": 2.5, "sigma_mode": "sides"}),
        ("local_sides_k3", {"variant": "local_stat", "k_stat": 3.0, "sigma_mode": "sides"}),
        ("local_sides_k4", {"variant": "local_stat", "k_stat": 4.0, "sigma_mode": "sides"}),
        ("local_sides_k5", {"variant": "local_stat", "k_stat": 5.0, "sigma_mode": "sides"}),
        ("bic_l2", {"variant": "bic", "bic_lambda": 2.0}),
        ("pettitt_001", {"variant": "pettitt", "pettitt_alpha": 0.001}),
    ]

    results = []
    for name, kw in variants:
        kw = dict(kw)
        impl = kw.pop("variant", name)
        t0 = time.time()
        dets: list[dict[str, Any]] = []
        for col in cols:
            for d in _detect(impl, series[col], **kw):
                dets.append({"column": col, **d})
        score = _score(dets, truth, frame.index, cols)
        # 灵敏度只看默认列（与历史 changepoint_scan 同列，保证可比）
        grid_col = "pressure@STA_A" if "pressure@STA_A" in series else cols[0]
        grid = _grid(impl, series[grid_col], **kw)
        results.append({
            "variant": name, "impl": impl, "params": {k: v for k, v in kw.items()},
            "detections": [{k: v for k, v in d.items() if k != "parent"} for d in dets],
            "count": len(dets), "score": score, "grid": grid,
            "grid_column": grid_col, "seconds": round(time.time() - t0, 2),
        })
        print("\n== %s  检出 %d 个  tp=%d fp=%d fn=%d  P=%.4f R=%.4f F1=%.4f  "
              "网格: 总体=%.2f Δ18=%.2f Δ8=%.2f Δ4=%.2f 伪报=%.2f (%.1fs)" % (
                  name, len(dets), score["tp"], score["fp"], score["fn"],
                  score["precision"] or 0, score["recall"] or 0, score["f1"] or 0,
                  grid["detection_rate"], grid["key_delta_rate"],
                  grid["by_delta"].get("8.0", 0), grid["by_delta"].get("4.0", 0),
                  grid["false_positive_rate_at_zero"], results[-1]["seconds"]))
        for d in dets:
            mark = "TP" if any(abs((t - pd.Timestamp(frame.index[d["index"]])).total_seconds()) <= MATCH_TOL
                               and s == d["column"] for s, t in truth) else "  "
            print("   %s %-16s %-8s idx=%-6d stat_local=%-8s gain=%-7s" % (
                mark, d["column"], d["kind"], d["index"],
                ("%.2f" % d["stat"]) if d.get("stat") == d.get("stat") else "—",
                ("%.4f" % d["gain"]) if "gain" in d else "—"))

    doc = {
        "dataset": os.path.relpath(DATASET, ROOT).replace(os.sep, "/"),
        "rows": len(frame), "columns": cols, "grid_column": results[0]["grid_column"],
        "truth": [{"series": s, "timestamp": str(t)} for s, t in truth],
        "tolerance_seconds": MATCH_TOL,
        "baseline_f1": 0.5, "threshold_f1": 0.5, "target_f1": 0.55,
        "results": results,
        "notes": "判据全部在脚本内实现；current 用于验证复现口径（应与门禁一致：6 检出 / tp=2）",
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2, default=str)
    print("\n结果：%s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
