# -*- coding: utf-8 -*-
"""A2 对照实验：变化点候选生成器（SSE 二分 vs 窗口中位数扫描）。

为什么换候选生成器
------------------
A1 之后过检修好了，但暴露了更根本的问题：**17 份真实天气数据上，均值突变只检出 4 个，
而且全在 1.3–2.1σ**。不是门槛太严，是候选生成器根本提不出候选：

SSE 二分要求「切开后残差平方和降到 0.7 倍」。在季节主导的序列上，年周期（振幅 20 °C）
本身就是最大的方差来源，一个几 hPa / 几 °C 的台阶解释不了 30%，于是连候选都进不来。

A2 的设想：候选不该由「方差解释比例」产生，而应该由「切点两侧的水平差」产生——
这与 A1 的接受判据（局部信噪比）本来就是同一件事，只是现在把它用在**扫描**上，
而不是用在**递归二分的结果**上。

| 项 | A（现状） | B（本实验） |
| --- | --- | --- |
| 候选 | SSE 二分递归，比例 0.7，深度 2 | 切点两侧各 h 点的中位数差，取局部极大值，按差值排序、最小间隔 min_len、最多 top_k 个 |
| 接受 | 局部信噪比 ≥ 3 | 同左（同一个函数） |
| 上限 | 每列 2^depth−1 = 3 | top_k |

要看三件事：v2 的 F1 不掉、网格灵敏度不掉（甚至更好）、真实数据能不能真的检出东西。
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from mtsa import anomaly                                  # noqa: E402
from mtsa import io_loaders, preprocess, profiler, quality  # noqa: E402
from mtsa.scenarios import weather as _w                  # noqa: F401,E402
from mtsa.scenarios.base import get_adapter               # noqa: E402

DATASET = os.path.join(ROOT, "fixtures", "weather_eval_v2.csv")
LABELS = os.path.join(ROOT, "fixtures", "labels_weather_eval_v2.csv")
REG = os.path.join(ROOT, "fixtures", "real_weather", "registry.json")
OUT = os.path.join(ROOT, "artifacts", "exp_changepoint_scan_gen.json")
QUESTION = "分析温度与气压的趋势、异常与结构变化"

POSITIONS = [0.25, 0.40, 0.55, 0.70, 0.85]
DELTAS = [0.0, 4.0, 8.0, 12.0, 14.0, 16.0, 18.0, 24.0]
GRID_TOL = 3
KEY_DELTA = 18.0
MATCH_TOL = 2 * 3600


# ------------------------------------------------------------------ B：窗口中位数扫描
def _window_medians(y: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """返回 (左窗中位数, 右窗中位数)，分别覆盖 [k-h, k) 与 [k, k+h)。"""
    s = pd.Series(y)
    need = max(8, h // 2)
    left = s.rolling(h, min_periods=need).median().shift(1).to_numpy()
    right = s[::-1].rolling(h, min_periods=need).median()[::-1].to_numpy()
    return left, right


def scan_candidates(y: np.ndarray, h: int, min_len: int, top_k: int) -> list[int]:
    """按「两侧水平差」的局部极大值取候选，按差值降序、最小间隔 min_len。"""
    left, right = _window_medians(y, h)
    diff = np.abs(right - left)
    n = len(diff)
    if n < 2 * min_len:
        return []
    prev = np.r_[np.nan, diff[:-1]]
    nxt = np.r_[diff[1:], np.nan]
    with np.errstate(invalid="ignore"):
        peak = np.isfinite(diff) & (diff >= prev) & (diff >= nxt) & (diff > 0)
    idx = np.flatnonzero(peak)
    if idx.size == 0:
        return []
    # 端点附近窗口不完整，剔掉
    idx = idx[(idx >= min_len) & (idx <= n - min_len)]
    if idx.size == 0:
        return []
    order = idx[np.argsort(-diff[idx])]
    picked: list[int] = []
    for k in order:
        if all(abs(int(k) - p) >= min_len for p in picked):
            picked.append(int(k))
            if len(picked) >= top_k:
                break
    return picked


def detect_scan(y: np.ndarray, phase: np.ndarray | None, period: int,
                min_len: int, h: int, top_k: int = 3,
                min_snr: float = 3.0) -> list[dict[str, Any]]:
    """B 方案：扫描提候选 + 局部信噪比接受，最多输出 top_k 个。"""
    out: list[dict[str, Any]] = []
    for k in scan_candidates(y, h, min_len, max(top_k * 3, 6)):
        loc = anomaly._local_snr(y, phase, period, k, 0, len(y), h)
        if loc is None or not (loc["snr"] >= min_snr):
            continue
        out.append({"index": int(k), "kind": "mean", **loc})
        if len(out) >= top_k:
            break
    return out


def detect_coarse_to_fine(y: np.ndarray, phase: np.ndarray | None, period: int,
                          min_len: int, h: int, top_k: int = 3,
                          min_snr: float = 3.0) -> list[dict[str, Any]]:
    """C 方案：粗扫定位「哪里有水平差」，SSE 在局部段内定到哪一点。

    为什么必须两段：
    - 「两侧中位数差」在台阶附近是一段**平台**（宽约 ±h/2），峰顶只是数值噪声，
      直接拿它当位置会偏几十到几百个点（实测 v2 偏 51 / 173 小时，F1=0）；
    - SSE 在**局部段**上定位很准，但把整条序列交给它时，预算被季节分界吃掉
      （实测 R01 注入 12 °C，SSE 首选 2841 而非 4392，台阶连候选都不是）。

    粗扫步长取 h/2，与平台的宽度同量级，保证不漏；每列最多 top_k 个输出。
    """
    n = len(y)
    step = max(h // 2, 8)
    hot: list[tuple[float, int]] = []
    for k in range(min_len, n - min_len, step):
        loc = anomaly._local_snr(y, phase, period, k, 0, n, h)
        if loc and np.isfinite(loc["snr"]) and loc["snr"] >= min_snr:
            hot.append((loc["snr"], k))
    if not hot:
        return []

    # 相邻热点并成区间，区间内用 SSE 精确定位
    hot.sort(key=lambda x: x[1])
    regions: list[tuple[int, int, float]] = []
    for snr, k in hot:
        if regions and k - regions[-1][1] <= h:
            lo, _, best = regions[-1]
            regions[-1] = (lo, k, max(best, snr))
        else:
            regions.append((k, k, snr))

    out: list[dict[str, Any]] = []
    for lo, hi, _ in regions:
        a, b = max(0, lo - h), min(n, hi + h)
        if b - a < 2 * min_len:
            continue
        k_rel, _ = anomaly._best_split(y[a:b], min_len)
        if k_rel is None:
            continue
        k = a + k_rel
        loc = anomaly._local_snr(y, phase, period, k, 0, n, h)
        if loc is None or not np.isfinite(loc["snr"]) or loc["snr"] < min_snr:
            continue
        out.append({"index": int(k), "kind": "mean", **loc})
    out.sort(key=lambda d: -d["snr"])
    picked: list[dict[str, Any]] = []
    for d in out:
        if all(abs(d["index"] - p["index"]) >= min_len for p in picked):
            picked.append(d)
        if len(picked) >= top_k:
            break
    return picked


# ------------------------------------------------------------------ 打分
def _truth() -> list[tuple[str, pd.Timestamp]]:
    labels = pd.read_csv(LABELS, encoding="utf-8-sig")
    return [("%s@%s" % (r["column"], r["station"]), pd.Timestamp(r["timestamp"]))
            for _, r in labels[labels["kind"] == "level_shift_start"].iterrows()]


def _score(dets: list[dict[str, Any]], truth: list[tuple[str, pd.Timestamp]],
           index: pd.Index) -> dict[str, Any]:
    used: set[int] = set()
    tp = 0
    for d in dets:
        ts = pd.Timestamp(index[d["index"]])
        best, gap = None, None
        for i, (series, t) in enumerate(truth):
            if i in used or series != d["column"]:
                continue
            g = abs((t - ts).total_seconds())
            if g <= MATCH_TOL and (gap is None or g < gap):
                best, gap = i, g
        if best is not None:
            used.add(best)
            tp += 1
    fp, fn = len(dets) - tp, len(truth) - tp
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def _grid(detector, y_base: np.ndarray, phase, period, min_len, h, top_k) -> dict:
    cells = []
    for pos in POSITIONS:
        t = int(round(len(y_base) * pos))
        for delta in DELTAS:
            y = y_base.copy()
            y[t:] += delta
            means = detector(y, phase, period, min_len, h, top_k)
            if means:
                best = min(means, key=lambda d: abs(d["index"] - t))
                hit = abs(best["index"] - t) <= GRID_TOL
            else:
                hit = False
            cells.append({"position": pos, "delta": delta, "detected": hit})
    by_delta = {str(d): round(sum(1 for c in cells if c["delta"] == d and c["detected"])
                              / sum(1 for c in cells if c["delta"] == d), 4)
                for d in DELTAS}
    zero = [c for c in cells if c["delta"] == 0.0]
    return {"detection_rate": round(sum(1 for c in cells if c["detected"]) / len(cells), 4),
            "by_delta": by_delta,
            "key_delta_rate": by_delta.get(str(KEY_DELTA), 0.0),
            "false_positive_rate_at_zero": round(sum(1 for c in zero if c["detected"]) / len(zero), 4)}


def _detector_a(y, phase, period, min_len, h, top_k):
    """A 方案：生产现状（SSE 候选 + 信噪比接受），用 _segment 等价复现。"""
    out = []
    for k, lo, hi in anomaly._segment(y, min_len, 2, 0.7):
        loc = anomaly._local_snr(y, phase, period, k, lo, hi, h)
        if loc is None or not (loc["snr"] >= 3.0):
            continue
        out.append({"index": int(k), "kind": "mean", **loc})
    return out


def _detector_fallback(y, phase, period, min_len, h, top_k):
    """D 方案：先跑 SSE；只有它一个都没接受时，才启用粗扫兜底。"""
    strict = _detector_a(y, phase, period, min_len, h, top_k)
    if strict:
        return strict
    return detect_coarse_to_fine(y, phase, period, min_len, h, top_k)


def main() -> int:
    from evals.split_eval import prepare_clean

    clean, plan = prepare_clean(DATASET, "weather", QUESTION)
    frame = clean.frame
    cols = [c for c in plan.target_columns if c in frame.columns]
    period = int(plan.parameters.get("seasonal_period") or 0)
    n = len(frame)
    min_len = max(anomaly.MIN_SEGMENT, n // 40)
    h = min(max(2 * min_len, anomaly.SNR_WINDOW_PERIODS * max(period, 1), 24), max(8, n // 4))
    phase = anomaly._seasonal_phase(frame.index, period) if period >= 2 else None
    truth = _truth()
    print("v2：%d 行；周期=%d；min_len=%d；h=%d；真值 %d 个" % (n, period, min_len, h, len(truth)))

    series = {c: pd.to_numeric(frame[c], errors="coerce").interpolate(limit_direction="both")
              .to_numpy(dtype=float) for c in cols}

    doc: dict[str, Any] = {"v2": {"rows": n, "period": period, "min_len": min_len, "h": h,
                                 "truth": [{"series": s, "timestamp": str(t)} for s, t in truth]},
                           "variants": {}}

    for name, det, top_k in (("A_sse", _detector_a, 3),
                             ("C_coarse_k3", detect_coarse_to_fine, 3),
                             ("D_fallback", _detector_fallback, 3)):
        t0 = time.time()
        dets = []
        for col in cols:
            for d in det(series[col], phase, period, min_len, h, top_k):
                dets.append({"column": col, **d})
        score = _score(dets, truth, frame.index)
        grid = _grid(det, series["pressure@STA_A"], phase, period, min_len, h, top_k)
        secs = time.time() - t0
        doc["variants"][name] = {
            "detections": [{k: v for k, v in d.items() if k != "parent"} for d in dets],
            "count": len(dets), "score": score, "grid": grid, "seconds": round(secs, 2),
        }
        print("\n== %-11s 检出 %d   tp=%d fp=%d fn=%d  P=%.3f R=%.2f F1=%.4f  "
              "网格 总体=%.2f Δ18=%.2f Δ12=%.2f Δ8=%.2f 伪报=%.2f  (%.1fs)" % (
                  name, len(dets), score["tp"], score["fp"], score["fn"],
                  score["precision"] or 0, score["recall"] or 0, score["f1"],
                  grid["detection_rate"], grid["key_delta_rate"],
                  grid["by_delta"].get("12.0", 0), grid["by_delta"].get("8.0", 0),
                  grid["false_positive_rate_at_zero"], secs))
        for d in dets:
            print("   %-16s idx=%-6d snr=%-7.2f jump=%-8.2f" % (
                d["column"], d["index"], d["snr"], d["jump"]))

    # ---------------------------------------------------------------- 真实数据
    reg = json.load(open(REG, encoding="utf-8"))
    rows = []
    print("\n真实数据（A 生产现状 vs B 扫描）：")
    for meta in reg["datasets"]:
        path = os.path.join(ROOT, meta["data_path"])
        if not os.path.exists(path):
            continue
        raw = io_loaders.load_dataset(path)
        prof, parsed = profiler.profile_dataset(raw)
        q = quality.check_quality(raw, prof, parsed)
        pl = get_adapter("weather").plan(QUESTION, prof, q, raw.frame)
        pl.scene_confidence = 1.0
        cl = preprocess.clean_dataset(raw, prof, parsed, pl)
        nn = len(cl.frame)
        pr = int(pl.parameters.get("seasonal_period") or 0)
        ml = max(anomaly.MIN_SEGMENT, nn // 40)
        hh = min(max(2 * ml, anomaly.SNR_WINDOW_PERIODS * max(pr, 1), 24), max(8, nn // 4))
        ph = anomaly._seasonal_phase(cl.frame.index, pr) if pr >= 2 else None
        per_ds = []
        for col in pl.target_columns:
            if col not in cl.frame.columns:
                continue
            s = pd.to_numeric(cl.frame[col], errors="coerce").interpolate(limit_direction="both")
            y = s.to_numpy(dtype=float)
            ta = time.time()
            a = _detector_a(y, ph, pr, ml, hh, 3)
            tb = time.time()
            cc = detect_coarse_to_fine(y, ph, pr, ml, hh, 3)
            dd = _detector_fallback(y, ph, pr, ml, hh, 3)
            per_ds.append({"column": col, "a": [d["index"] for d in a],
                           "c": [{"index": d["index"], "snr": round(d["snr"], 2),
                                  "jump": round(d["jump"], 2)} for d in cc],
                           "d": [{"index": d["index"], "snr": round(d["snr"], 2),
                                  "jump": round(d["jump"], 2)} for d in dd],
                           "t_a": round(tb - ta, 2), "t_c": round(time.time() - tb, 2)})
        rows.append({"id": meta["id"], "rows": nn, "period": pr, "columns": per_ds})
        print("  %-4s 行=%-6d A=%d  C=%d  D=%d   A耗时=%.2fs C耗时=%.2fs" % (
            meta["id"], nn, sum(len(c["a"]) for c in per_ds),
            sum(len(c["c"]) for c in per_ds), sum(len(c["d"]) for c in per_ds),
            sum(c["t_a"] for c in per_ds), sum(c["t_c"] for c in per_ds)))
        for c in per_ds:
            if c["c"]:
                print("       %-18s C: %s" % (c["column"], c["c"]))
    doc["real"] = rows
    doc["real_totals"] = {
        "a": sum(len(c["a"]) for r in rows for c in r["columns"]),
        "c": sum(len(c["c"]) for r in rows for c in r["columns"]),
        "d": sum(len(c["d"]) for r in rows for c in r["columns"]),
    }
    print("\n真实数据合计：A=%d 个，C=%d 个，D=%d 个" % (
        doc["real_totals"]["a"], doc["real_totals"]["c"], doc["real_totals"]["d"]))

    # ---------------------------------------------------------------- 注入台阶能否检出
    print("\n注入台阶（+8 / +12）能否检出（误差=个采样点）：")
    for rid in ("R01", "R10", "R03", "R06"):
        if rid not in {m["id"]: m for m in reg["datasets"]}:
            continue
        meta = {m["id"]: m for m in reg["datasets"]}[rid]
        raw = io_loaders.load_dataset(os.path.join(ROOT, meta["data_path"]))
        prof, parsed = profiler.profile_dataset(raw)
        q = quality.check_quality(raw, prof, parsed)
        pl = get_adapter("weather").plan(QUESTION, prof, q, raw.frame)
        pl.scene_confidence = 1.0
        cl = preprocess.clean_dataset(raw, prof, parsed, pl)
        nn = len(cl.frame)
        pr = int(pl.parameters.get("seasonal_period") or 0)
        ml = max(anomaly.MIN_SEGMENT, nn // 40)
        hh = min(max(2 * ml, anomaly.SNR_WINDOW_PERIODS * max(pr, 1), 24), max(8, nn // 4))
        ph = anomaly._seasonal_phase(cl.frame.index, pr) if pr >= 2 else None
        base = pd.to_numeric(cl.frame["temperature"], errors="coerce").interpolate(
            limit_direction="both").to_numpy(dtype=float)
        k0 = nn // 2
        cells = []
        for delta in (8.0, 12.0):
            y2 = base.copy()
            y2[k0:] += delta
            row = []
            for tag, fn in (("A", _detector_a), ("C", detect_coarse_to_fine),
                            ("D", _detector_fallback)):
                got = fn(y2, ph, pr, ml, hh, 3)
                if not got:
                    row.append("%s:未检出" % tag)
                else:
                    best = min(got, key=lambda d: abs(d["index"] - k0))
                    err = best["index"] - k0
                    row.append("%s:%+d%s" % (tag, err, "" if abs(err) <= 3 else "!"))
            cells.append("Δ%g[%s]" % (delta, " ".join(row)))
        print("  %-4s 行=%-6d %s" % (rid, nn, "  ".join(cells)))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2, default=str)
    print("结果：%s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
