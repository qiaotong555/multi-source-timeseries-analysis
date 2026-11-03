# -*- coding: utf-8 -*-
"""A3 方差突变通道的实测证据（只读 src，不改产线）。

产出三组数字，对应 `doc/A3方差通道.md` 的第二、三、四节：

1. **量纲**：纯方差注入（后半段残差 ×2、均值不变）后，原判据的「跳变 / 基线」是多少。
   判据是 `max|Δ滚动标准差| > 2 × 滚动标准差中位数`，而 σ→2σ 时跳变≈基线，
   比值恒≈1.0 —— 真变化永远够不到 2.0。
2. **显著性**：对「对数标准差比」做标签置换（不依赖分布形状），比较
   ① v2 降水的两个点（数据本身是 iid，无方差变化）② 注入的 σ×3 变化 ③ 真实数据的点。
3. **残差方案**：改成「去季节残差的相邻窗标准差比」并纠正定位偏移后，
   未注入与注入两种情形的表现。

运行
----
    PYTHONPATH=src python scripts/exp_variance_channel.py
"""
from __future__ import annotations

import json
import os
import sys
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

V2 = os.path.join(ROOT, "fixtures", "weather_eval_v2.csv")
REG = os.path.join(ROOT, "fixtures", "real_weather", "registry.json")
OUT = os.path.join(ROOT, "artifacts", "exp_variance_channel.json")
QUESTION = "分析温度与气压的趋势、异常与结构变化"
PERIOD = 24
COLS = ("temperature@STA_A", "humidity@STA_A", "pressure@STA_A",
        "wind_speed@STA_A", "rainfall@STA_A")


def _old_criterion(y: np.ndarray) -> dict[str, Any]:
    """原判据：滚动标准差的单步跳变 > 2 × 中位数。返回位置与比值。"""
    n = len(y)
    w = max(12, n // 40)
    roll = pd.Series(y).rolling(w, min_periods=w).std()
    d = roll.diff().abs()
    base = float(roll.median())
    if not (base > 0) or not d.notna().any():
        return {"index": None, "ratio": None, "fires": False}
    j = int(d.idxmax())
    ratio = float(d.loc[j]) / base
    return {"index": j, "ratio": round(ratio, 4), "fires": ratio > 2.0}


def _perm_p(y: np.ndarray, k: int, w: int, n_perm: int = 400,
            seed: int = 7) -> tuple[float | None, float | None]:
    """两侧窗口标准差比的置换 p 值（双尾，统计量取 |log σ比|）。"""
    l, r = y[max(0, k - w):k], y[k:k + w]
    if len(l) < 8 or len(r) < 8:
        return None, None
    sl, sr = float(np.std(l, ddof=1)), float(np.std(r, ddof=1))
    if sl <= 0 or sr <= 0:
        return None, None
    obs = abs(np.log(sr / sl))
    pool = np.concatenate([l, r])
    n1 = len(l)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(pool)
        sa, sb = float(np.std(perm[:n1], ddof=1)), float(np.std(perm[n1:], ddof=1))
        if sa > 0 and sb > 0 and abs(np.log(sb / sa)) >= obs:
            cnt += 1
    return round(sr / sl, 4), round((1 + cnt) / (1 + n_perm), 4)


def _residual_ratio(y: np.ndarray, index: pd.Index, w: int,
                    min_nonzero: int = 100) -> dict[str, Any]:
    """残差相邻窗标准差比 + 可估性守卫（两侧窗口都要有足够非零点）。"""
    n = len(y)
    s = pd.Series(y, index=index)
    resid = anomaly._residual(s, PERIOD, w)
    roll = resid.rolling(w, min_periods=w).std()
    nz = (s != 0).rolling(w, min_periods=w).sum().to_numpy()
    lr = np.log(roll.to_numpy())
    d = np.full(n, np.nan)
    d[w:] = np.abs(lr[w:] - lr[:-w])
    ok = np.zeros(n, dtype=bool)
    ok[2 * w:] = (nz[w:-w] >= min_nonzero) & (nz[:-2 * w] >= min_nonzero)
    d = np.where(ok, d, np.nan)
    if not np.isfinite(d).any():
        return {"index": None, "ratio": None, "estimable": False}
    t = int(np.nanargmax(d))
    return {"index": t - w, "ratio": round(float(np.exp(d[t])), 4), "estimable": True}


def _clean_v2():
    from evals.split_eval import prepare_clean
    return prepare_clean(V2, "weather", "x")


def main() -> int:
    clean, _ = _clean_v2()
    frame = clean.frame
    n = len(frame)
    w = max(12, n // 40)
    doc: dict[str, Any] = {"rows": n, "window": w, "sections": {}}

    # ---- 1. 量纲：纯方差注入
    print("== 1. 纯方差注入（后半段残差 ×2，均值不变）后原判据的比值")
    inj = []
    half = n // 2
    for col in COLS:
        y = pd.to_numeric(frame[col], errors="coerce").interpolate(
            limit_direction="both").to_numpy(dtype=float)
        mid = float(np.mean(y[half:]))
        y2 = y.copy()
        y2[half:] = mid + (y[half:] - mid) * 2.0
        base = _old_criterion(y)
        got = _old_criterion(y2)
        err = None if got["index"] is None else got["index"] - half
        inj.append({"column": col, "ratio": got["ratio"], "index": got["index"],
                    "error": err, "fires": got["fires"],
                    "uninjected_ratio": base["ratio"]})
        print("   %-18s 跳变/基线=%-7s 误差=%-6s 判定=%s" % (
            col, got["ratio"], err if err is not None else "—",
            "命中" if (got["fires"] and err is not None and abs(err) <= 3) else "未命中"))
    doc["sections"]["scale_injection"] = inj

    # ---- 2. 置换 p
    print("\n== 2. 标准差比的置换 p（400 次）")
    perm = []
    for col, k in (("rainfall@STA_A", 4271), ("rainfall@STA_B", 6928)):
        y = pd.to_numeric(frame[col], errors="coerce").interpolate(
            limit_direction="both").to_numpy(dtype=float)
        ratio, p = _perm_p(y, k, w)
        perm.append({"case": "v2 " + col, "index": k, "ratio": ratio, "p": p,
                     "note": "v2 降水为 iid，无方差变化"})
        print("   %-26s σ比=%-7s p=%-7s" % ("v2 " + col, ratio, p))
    y = pd.to_numeric(frame["rainfall@STA_A"], errors="coerce").interpolate(
        limit_direction="both").to_numpy(dtype=float).copy()
    for mult in (2.0, 3.0):
        yy = y.copy()
        yy[6000:] *= mult
        ratio, p = _perm_p(yy, 6000, w)
        perm.append({"case": "v2 rainfall 注入 σ×%g" % mult, "index": 6000,
                     "ratio": ratio, "p": p, "note": "真变化"})
        print("   %-26s σ比=%-7s p=%-7s" % ("v2 注入 σ×%g" % mult, ratio, p))

    reg = json.load(open(REG, encoding="utf-8"))
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
        for col in pl.target_columns:
            if col not in cl.frame.columns:
                continue
            yv = pd.to_numeric(cl.frame[col], errors="coerce").interpolate(
                limit_direction="both").to_numpy(dtype=float)
            got = _old_criterion(yv)
            if not got["fires"]:
                continue
            ratio, p = _perm_p(yv, got["index"], w)
            perm.append({"case": "%s %s" % (meta["id"], col), "index": got["index"],
                         "ratio": ratio, "p": p, "note": "原判据给出的真实数据点"})
            print("   %-26s σ比=%-7s p=%-7s  (原判据给出的点)" % (
                "%s %s" % (meta["id"], col), ratio, p))
    doc["sections"]["permutation"] = perm

    # ---- 3. 残差相邻窗比
    print("\n== 3. 去季节残差的相邻窗标准差比 + 可估性守卫")
    resid = []
    for col in COLS:
        y = pd.to_numeric(frame[col], errors="coerce").interpolate(
            limit_direction="both").to_numpy(dtype=float)
        got = _residual_ratio(y, frame.index, w)
        resid.append({"column": col, "injected": False, **got})
        print("   %-18s 未注入：位置=%-7s σ比=%-7s 可估=%s" % (
            col, got["index"] if got["index"] is not None else "—",
            got["ratio"], got["estimable"]))
    for col in COLS:
        y = pd.to_numeric(frame[col], errors="coerce").interpolate(
            limit_direction="both").to_numpy(dtype=float)
        mid = float(np.mean(y[half:]))
        y2 = y.copy()
        y2[half:] = mid + (y[half:] - mid) * 2.0
        got = _residual_ratio(y2, frame.index, w)
        err = None if got["index"] is None else got["index"] - half
        resid.append({"column": col, "injected": True, "error": err, **got})
        print("   %-18s 注入 σ×2：位置=%-7s 误差=%-7s σ比=%-7s" % (
            col, got["index"] if got["index"] is not None else "—",
            err if err is not None else "—", got["ratio"]))
    doc["sections"]["residual_ratio"] = resid

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2, default=str)
    print("\n结果：%s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
