# -*- coding: utf-8 -*-
"""区间估计测试：按事件重采样，不按点。"""
from __future__ import annotations

import pandas as pd

from evals import interval as I


def _events(hits: list[int], sizes: list[int], day0: int = 1):
    """构造互不相邻的事件点对：每个事件 size 个连续小时，间隔一天。"""
    tp, fn = [], []
    for i, (hit, size) in enumerate(zip(hits, sizes)):
        base = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
        for h in range(size):
            pair = ("t", str(base + pd.Timedelta(hours=h)))
            (tp if h < hit else fn).append(pair)
    return tp, fn


def _fp_clusters(count: int, size: int = 2, day0: int = 100):
    out = []
    for i in range(count):
        base = pd.Timestamp("2024-06-01") + pd.Timedelta(days=i)
        for h in range(size):
            out.append(("t", str(base + pd.Timedelta(hours=h))))
    return out


def test_gap_merges_consecutive_points_into_one_event():
    """同列、间隔 ≤3 小时的点算同一个事件，不能当独立样本。"""
    pairs = [("t", "2024-01-01 00:00:00"), ("t", "2024-01-01 01:00:00"),
             ("t", "2024-01-01 03:00:00")]
    assert len(I.clusters(pairs)) == 1
    # 隔到 4 小时以上就断开
    pairs2 = pairs + [("t", "2024-01-01 08:00:00")]
    assert len(I.clusters(pairs2)) == 2


def test_truth_events_track_partial_hits():
    """一个事件内部分命中时，hit 与 size 要分开记——按点算会把它们摊平。"""
    tp, fn = _events([2, 2, 1], [2, 2, 2])
    ev = I.truth_events(tp, fn)
    assert len(ev) == 3
    assert sorted((e["size"], e["hit"]) for e in ev) == [(2, 1), (2, 2), (2, 2)]


def test_ci_contains_point_estimate():
    tp, fn = _events([2] * 8 + [0] * 2, [2] * 10)
    r = I.pr_interval(tp, _fp_clusters(5), fn)
    p_point = 16 / (16 + 10)
    r_point = 16 / 20
    assert r["precision"][0] <= p_point <= r["precision"][1]
    assert r["recall"][0] <= r_point <= r["recall"][1]
    assert r["n_events"] == 10 and r["n_fp_clusters"] == 5


def test_fewer_events_gives_wider_interval():
    """事件少 → 区间宽。按点重采样会让这条失效（点数不变时区间也不变）。"""
    tp_many, fn_many = _events([2] * 8 + [0] * 2, [2] * 10)
    wide = I.pr_interval(tp_many, _fp_clusters(5), fn_many)
    tp_few, fn_few = _events([2, 2, 0], [2, 2, 2])
    narrow = I.pr_interval(tp_few, _fp_clusters(5), fn_few)
    w_many = wide["recall"][1] - wide["recall"][0]
    w_few = narrow["recall"][1] - narrow["recall"][0]
    assert w_few > w_many


def test_no_events_returns_none():
    assert I.pr_interval([], [], [])["precision"] is None


def test_bootstrap_is_deterministic():
    tp, fn = _events([2] * 6 + [0] * 4, [2] * 10)
    a = I.pr_interval(tp, _fp_clusters(4), fn)
    b = I.pr_interval(tp, _fp_clusters(4), fn)
    assert a == b


def test_merge_pairs_dedupes_across_folds():
    a = [("t", "2024-01-01 00:00:00")]
    b = [("t", "2024-01-01 00:00:00"), ("t", "2024-02-01 00:00:00")]
    assert len(I.merge_pairs(a, b)) == 2
