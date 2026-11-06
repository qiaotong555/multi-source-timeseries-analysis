# -*- coding: utf-8 -*-
"""评估指标的区间估计：按**事件**做 cluster bootstrap，不按点。

为什么不能按点重采样：逐小时异常点高度自相关，一个持续 8 小时的事件里 8 个点
几乎同时被命中或漏掉；按点独立重采样会把有效样本量当成「点数」，
算出的区间比真实窄一个量级。

做法：
- 真值侧按（列 + 间隔 ≤ 3 小时）聚成事件，每个事件记录「点数 n」与「命中数 hit」；
- 检出侧同理聚成假阳性簇，每个簇记录点数；
- 每次重采样分别对事件集合与假阳性簇集合有放回抽样，
  recall = Σhit / Σn，precision = Σhit / (Σhit + Σfp)。

区间取 percentile bootstrap（默认 2000 次，2.5%/97.5%）。
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

GAP_SECONDS = 3 * 3600
DEFAULT_BOOT = 2000
DEFAULT_SEED = 20261106


def _group(by_col: dict[str, set], gap_seconds: float) -> list[dict]:
    """把每列的时间戳集合按间隔聚成事件，保留原始时间戳以便统计命中数。"""
    out: list[dict] = []
    for col, stamps in by_col.items():
        ordered = sorted(stamps)
        if not ordered:
            continue
        group = [ordered[0]]
        for ts in ordered[1:]:
            if (ts - group[-1]).total_seconds() > gap_seconds:
                out.append({"column": col, "stamps": group})
                group = []
            group.append(ts)
        out.append({"column": col, "stamps": group})
    return out


def _pairs_by_col(pairs: Iterable[tuple[str, str]]) -> dict[str, set]:
    by_col: dict[str, set] = {}
    for col, ts in pairs or []:
        by_col.setdefault(col, set()).add(pd.Timestamp(ts))
    return by_col


def clusters(pairs: Iterable[tuple[str, str]],
             gap_seconds: float = GAP_SECONDS) -> list[dict]:
    """假阳性簇：每簇只关心点数。"""
    out = []
    for ev in _group(_pairs_by_col(pairs), gap_seconds):
        out.append({
            "column": ev["column"],
            "start": ev["stamps"][0],
            "end": ev["stamps"][-1],
            "size": len(ev["stamps"]),
        })
    return out


def truth_events(tp_pairs: Iterable[tuple[str, str]],
                 fn_pairs: Iterable[tuple[str, str]],
                 gap_seconds: float = GAP_SECONDS) -> list[dict]:
    """真值事件：size 为事件内全部真值点数，hit 为其中被判对的数量。"""
    tp_set = {(c, pd.Timestamp(t)) for c, t in tp_pairs or []}
    by_col = _pairs_by_col(tp_pairs)
    for col, stamps in _pairs_by_col(fn_pairs).items():
        by_col.setdefault(col, set()).update(stamps)
    out = []
    for ev in _group(by_col, gap_seconds):
        hit = sum(1 for ts in ev["stamps"] if (ev["column"], ts) in tp_set)
        out.append({
            "column": ev["column"],
            "start": ev["stamps"][0],
            "end": ev["stamps"][-1],
            "size": len(ev["stamps"]),
            "hit": hit,
        })
    return out


def cluster_bootstrap(events: list[dict], fp_clusters: list[dict],
                      n_boot: int = DEFAULT_BOOT, seed: int = DEFAULT_SEED,
                      alpha: float = 0.05) -> dict:
    """对事件与假阳性簇分别有放回抽样，返回 precision / recall 的百分位区间。"""
    if not events:
        return {"precision": None, "recall": None,
                "n_events": 0, "n_fp_clusters": len(fp_clusters), "n_boot": n_boot}

    n_sizes = np.array([e["size"] for e in events], dtype=float)
    n_hits = np.array([e["hit"] for e in events], dtype=float)
    fp_sizes = np.array([c["size"] for c in fp_clusters], dtype=float)

    rng = np.random.default_rng(seed)
    recalls = np.empty(n_boot)
    precisions = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(events), len(events))
        hit = float(n_hits[pick].sum())
        n = float(n_sizes[pick].sum())
        if fp_sizes.size:
            fpick = rng.integers(0, fp_sizes.size, fp_sizes.size)
            fp = float(fp_sizes[fpick].sum())
        else:
            fp = 0.0
        recalls[i] = hit / n if n else 0.0
        precisions[i] = hit / (hit + fp) if (hit + fp) else 0.0

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "precision": (round(float(np.percentile(precisions, lo)), 4),
                      round(float(np.percentile(precisions, hi)), 4)),
        "recall": (round(float(np.percentile(recalls, lo)), 4),
                   round(float(np.percentile(recalls, hi)), 4)),
        "n_events": len(events),
        "n_fp_clusters": len(fp_clusters),
        "n_boot": n_boot,
    }


def pr_interval(tp_pairs, fp_pairs, fn_pairs, n_boot: int = DEFAULT_BOOT,
                seed: int = DEFAULT_SEED) -> dict:
    """一步到位：由点对集合算出 precision / recall 的 95% 区间。"""
    return cluster_bootstrap(truth_events(tp_pairs, fn_pairs),
                             clusters(fp_pairs), n_boot=n_boot, seed=seed)


def merge_pairs(*groups) -> list[tuple[str, str]]:
    """把多折的 (列, 时间) 点对合并去重（多折 holdout 互不重叠，合并不产生重复）."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for g in groups:
        for col, ts in g or []:
            key = (col, str(ts))
            if key not in seen:
                seen.add(key)
                out.append((col, ts))
    return out
