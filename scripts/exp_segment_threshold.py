# -*- coding: utf-8 -*-
"""实验：段级通道并入异常指标后，阈值与「指标 vs 新能力」的取舍。

方案 B（段检出并入 data.anomalies）的代价必须先量清楚：
标注集 `weather_eval_v1` 只有点和集体异常，没有「单点不超阈、整段持续偏移」这一类，
所以段通道在标注集上的新增检出**按定义就是 FP**——指标只会惩罚它，不会奖励它。

本脚本对每个段阈值跑一遍 dev/holdout 三折，报：
- 现有 22 项里异常相关的 4 项；
- 可行性实验里该阈值对应的注入召回（来自 doc/段级异常检测.md 的 ROC）。

用法：
    python scripts/exp_segment_threshold.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import mtsa.anomaly as A  # noqa: E402
from evals import split_eval as SE  # noqa: E402

# 可行性实验（w=24）实测的注入召回，阈值 → (3σ/24h, 4σ/24h)
RECALL_FROM_EXP = {
    2.6: (0.65, 0.85),
    3.0: (0.53, 0.75),
    3.4: (0.35, 0.65),
}


def main() -> int:
    base = os.path.join(ROOT, "fixtures", SE.EVAL_DATASET)
    labels = os.path.join(ROOT, "fixtures", SE.EVAL_LABELS)
    q = "识别天气数据中的异常点与极端事件"
    out_root = os.path.join(ROOT, "artifacts", "exp_segment_threshold")

    print("| 段阈值 | holdout P | holdout R | 事件级 R | 退化 P | 退化 R | 注入召回 3σ/24h | 4σ/24h | 通过 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for thr in (2.6, 3.0, 3.4, 4.0):
        A.SEGMENT_THRESHOLD = thr
        res = SE.run_split_eval(
            base_csv=base, labels_csv=labels, scene_id="weather", question=q,
            out_root=os.path.join(out_root, "t%s" % thr), verbose=False)
        s = res["summary"]
        r3, r4 = RECALL_FROM_EXP.get(thr, ("—", "—"))
        print("| %.1f | %.4f | %.4f | %.4f | %.4f | %.4f | %s | %s | %s |" % (
            thr, s["mean_holdout_precision"], s["mean_holdout_recall"],
            s["mean_holdout_event_recall"], s["max_degradation_precision"],
            s["max_degradation_recall"], r3, r4, "是" if s["passed"] else "否"))
    # 关掉段通道作为对照
    A.SEGMENT_THRESHOLD = 2.6
    print()
    print("对照：把参数 segment_anomaly 设为 False 时即回到纯点通道，指标应为 "
          "P 0.7540 / R 0.8697。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
