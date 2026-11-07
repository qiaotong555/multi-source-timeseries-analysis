# -*- coding: utf-8 -*-
"""实验：要多少条真值事件，精确率的区间下界才站得上 0.70？

背景（#36）：三折合并口径 P = 0.7542，按事件 bootstrap 的 95% 区间 [0.663, 0.818]，
下界落在 0.70 门线之下。问题不是「门线该不该改」，而是「这个判定要多少样本才稳」。

做法
----
拿现行评估的真实事件池（每个真值事件的点数与命中数、每个假阳性簇的点数）当总体，
按「N 个事件 + 与之成比例的假阳性簇」重复抽样，每一抽再用同样的 cluster bootstrap
算 95% 区间，统计：
  - 区间宽度的中位数；
  - 区间下界 ≥ 0.70 的比例（判定稳健性）；
  - 点估计 ≥ 0.70 的比例（现行判据的通过率）。

假设：新增事件与现有事件同分布（命中率 0.75、事件规模 2–4 点），假阳性簇与事件数的比例
维持现状（6/33）。这是外推，不是承诺——真实扩数据集若换了地区/季节，命中率会变。

用法：
    python scripts/exp_sample_size.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from evals import interval as I  # noqa: E402
from evals import split_eval as SE  # noqa: E402

TARGET = 0.70
EVENT_GRID = (33, 50, 75, 100, 150, 200, 300)
N_REPS = 300
N_BOOT = 400
SEED = 20261107


def pool_from_eval() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """跑一遍三折，取出事件池：真值事件的点数、命中数、假阳性簇点数。"""
    res = SE.run_split_eval(
        base_csv=os.path.join(ROOT, "fixtures", SE.EVAL_DATASET),
        labels_csv=os.path.join(ROOT, "fixtures", SE.EVAL_LABELS),
        scene_id="weather",
        question="识别天气数据中的异常点与极端事件",
        out_root=os.path.join(ROOT, "artifacts", "exp_sample_size"),
        verbose=False, keep_pairs=True)
    pairs = res["_pairs"]
    events = I.truth_events(pairs["tp"], pairs["fn"])
    fps = I.clusters(pairs["fp"])
    sizes = np.array([e["size"] for e in events], dtype=float)
    hits = np.array([e["hit"] for e in events], dtype=float)
    fp_sizes = np.array([c["size"] for c in fps], dtype=float)
    return sizes, hits, fp_sizes


def simulate(sizes, hits, fp_sizes, n_events: int, rng) -> tuple[float, bool, bool]:
    """抽 N 个事件（假阳性簇按现比例），返回（区间宽度, 下界是否达标, 点估计是否达标）。"""
    n_fp = max(1, int(round(n_events * fp_sizes.size / sizes.size)))
    idx = rng.integers(0, sizes.size, n_events)
    fidx = rng.integers(0, fp_sizes.size, n_fp)
    h = hits[idx]
    fp_pool = fp_sizes[fidx]
    fp = float(fp_pool.sum())

    # 内层 bootstrap：对抽到的这批簇再重采样，算区间（向量化，避免 N_BOOT 次 Python 循环）
    b = rng.integers(0, n_events, (N_BOOT, n_events))
    hh = h[b].sum(axis=1)
    ff = fp_pool[rng.integers(0, n_fp, (N_BOOT, n_fp))].sum(axis=1)
    boot = hh / (hh + ff)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    point = float(h.sum()) / (float(h.sum()) + fp)
    return float(hi - lo), bool(lo >= TARGET), bool(point >= TARGET)


def main() -> int:
    sizes, hits, fp_sizes = pool_from_eval()
    base_p = hits.sum() / (hits.sum() + fp_sizes.sum())
    print("事件池：%d 个真值事件（点数 %s），%d 个假阳性簇（点数 %s）" % (
        sizes.size, sorted(set(sizes.astype(int).tolist())),
        fp_sizes.size, sorted(set(fp_sizes.astype(int).tolist()))))
    print("当前合并口径 P = %.4f（事件命中率 %.3f，平均事件规模 %.1f 点）" % (
        base_p, hits.mean() / sizes.mean(), sizes.mean()))
    print("假阳性簇 / 事件数 = %.3f；假设扩样本时保持不变" % (fp_sizes.size / sizes.size))
    print()
    print("| 真值事件数 | 95% 区间宽度（中位） | 下界≥0.70 的比例 | 点估计≥0.70 的比例 |")
    print("|---|---|---|---|")

    rng = np.random.default_rng(SEED)
    for n in EVENT_GRID:
        widths, lo_ok, pt_ok = [], 0, 0
        for _ in range(N_REPS):
            w, lo_ok_i, pt_ok_i = simulate(sizes, hits, fp_sizes, n, rng)
            widths.append(w)
            lo_ok += int(lo_ok_i)
            pt_ok += int(pt_ok_i)
        print("| %d | %.3f | %.2f | %.2f |" % (
            n, float(np.median(widths)), lo_ok / N_REPS, pt_ok / N_REPS))
    print()
    print("注：区间宽度按合并口径的精确率算；下界达标比例是「换一批同分布样本仍能通过」的概率。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
