# -*- coding: utf-8 -*-
"""实验：三折 holdout 重叠 vs 不重叠，阈值与指标差多少。

背景：现行 `FOLDS` 是 dev_ratio 0.60 / 0.70 / 0.80，holdout 固定 20%，
三段 holdout 分别落在 [0.60,0.80)、[0.70,0.90)、[0.80,1.0)——**互相重叠**，
所以「三折都通过」并不等于「三次独立验证」。

不重叠方案：holdout 依次取 [0.40,0.60)、[0.60,0.80)、[0.80,1.0)，
dev 仍是「该折 holdout 之前的全部数据」（0.40 / 0.60 / 0.80），保持前向滚动。
代价是 fold1 的 dev 从 60% 降到 40%，阈值扫描的样本变少。

用法：
    python scripts/exp_split_nonoverlap.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from evals import split_eval  # noqa: E402

BASE = os.path.join(ROOT, "fixtures", split_eval.EVAL_DATASET)
LABELS = os.path.join(ROOT, "fixtures", split_eval.EVAL_LABELS)
OUT = os.path.join(ROOT, "artifacts", "exp_split_nonoverlap")
QUESTION = "识别天气数据中的异常点与极端事件"

OVERLAP = [
    {"name": "fold1", "dev_ratio": 0.60, "holdout_len": 0.20},
    {"name": "fold2", "dev_ratio": 0.70, "holdout_len": 0.20},
    {"name": "fold3", "dev_ratio": 0.80, "holdout_len": 0.20},
]
NONOVERLAP = [
    {"name": "fold1", "dev_ratio": 0.40, "holdout_len": 0.20},
    {"name": "fold2", "dev_ratio": 0.60, "holdout_len": 0.20},
    {"name": "fold3", "dev_ratio": 0.80, "holdout_len": 0.20},
]
# 折中方案：dev 保持 ≥60%（阈值扫描样本不缩水），把 holdout 从 20% 压到 13%，
# 让三段首尾相接、互不重叠。代价是每折 holdout 的真值变少。
COMPACT = [
    {"name": "fold1", "dev_ratio": 0.60, "holdout_len": 0.1333},
    {"name": "fold2", "dev_ratio": 0.7333, "holdout_len": 0.1333},
    {"name": "fold3", "dev_ratio": 0.8666, "holdout_len": 0.1334},
]


def run(name: str, folds: list[dict]) -> dict:
    print("=== %s ===" % name)
    res = split_eval.run_split_eval(
        base_csv=BASE, labels_csv=LABELS, scene_id="weather", question=QUESTION,
        out_root=os.path.join(OUT, name), folds=folds, verbose=True,
    )
    for f in res["folds"]:
        if f.get("status") != "succeeded":
            print("  %s: %s" % (f["fold"], f.get("status")))
            continue
        # holdout 的 end 为 None 表示切到序列末尾
        hold = f["holdout"]
        print("  %s dev=%.2f holdout=[%s, %s) 真值 %d 阈值 %s  "
              "P %.4f R %.4f 事件R %.4f" % (
                  f["fold"], folds[res["folds"].index(f)]["dev_ratio"],
                  hold["start"][:10], (hold["end"] or "序列末尾")[:10],
                  f["holdout_truth_count"], f["selected_threshold"],
                  f["holdout_precision"], f["holdout_recall"],
                  f["holdout_event_recall"]))
    s = res["summary"]
    print("  汇总：P %.4f R %.4f 事件R %.4f | 阈值 %s 稳定=%s | 通过=%s" % (
        s["mean_holdout_precision"], s["mean_holdout_recall"],
        s["mean_holdout_event_recall"], s["selected_thresholds"],
        s["threshold_stable"], s["passed"]))
    return res


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    a = run("overlap", OVERLAP)
    b = run("nonoverlap", NONOVERLAP)
    c = run("compact", COMPACT)
    with open(os.path.join(OUT, "compare.json"), "w", encoding="utf-8") as f:
        json.dump({"overlap": a, "nonoverlap": b, "compact": c}, f,
                  ensure_ascii=False, indent=2)

    print()
    print("| 指标 | 重叠（现行） | 不重叠 | 压缩折中 |")
    print("|---|---|---|---|")
    for k in ("mean_holdout_precision", "mean_holdout_recall",
              "mean_holdout_event_recall", "mean_collective_recall"):
        x, y, z = a["summary"].get(k), b["summary"].get(k), c["summary"].get(k)
        if x is None or y is None or z is None:
            continue
        print("| %s | %.4f | %.4f | %.4f |" % (k, x, y, z))
    print("| 选中阈值 | %s | %s | %s |" % (
        a["summary"]["selected_thresholds"], b["summary"]["selected_thresholds"],
        c["summary"]["selected_thresholds"]))
    print("| 阈值稳定 | %s | %s | %s |" % (
        a["summary"]["threshold_stable"], b["summary"]["threshold_stable"],
        c["summary"]["threshold_stable"]))
    print("| 每折真值 | %s | %s | %s |" % (
        [f.get("holdout_truth_count") for f in a["folds"]],
        [f.get("holdout_truth_count") for f in b["folds"]],
        [f.get("holdout_truth_count") for f in c["folds"]]))
    print("| 通过 | %s | %s | %s |" % (
        a["summary"]["passed"], b["summary"]["passed"], c["summary"]["passed"]))
    print()
    print("明细：", os.path.join(OUT, "compare.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
