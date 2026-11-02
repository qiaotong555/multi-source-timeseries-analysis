# -*- coding: utf-8 -*-
"""A1 影响面：变化点判据改动在 17 份真实数据上的前后对照。

`MIN_SNR = 0` 等价于旧行为（所有 SSE 候选都被接受），
`MIN_SNR = 3` 是新行为。两边用同一份清洗结果，只改门槛。

输出 artifacts/a1_changepoint_impact.json，并打印逐数据集计数。
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from mtsa import anomaly                                    # noqa: E402
from mtsa import io_loaders, preprocess, profiler, quality   # noqa: E402
from mtsa.scenarios import weather as _w                    # noqa: F401,E402
from mtsa.scenarios.base import get_adapter                 # noqa: E402

REG = os.path.join(ROOT, "fixtures", "real_weather", "registry.json")
OUT = os.path.join(ROOT, "artifacts", "a1_changepoint_impact.json")
QUESTION = "分析温度与气压的趋势、异常与结构变化"


def _run(clean, plan, snr: float) -> list:
    anomaly.MIN_SNR = snr
    try:
        return anomaly.detect_change_points(clean, plan)
    finally:
        anomaly.MIN_SNR = 3.0


def main() -> int:
    reg = json.load(open(REG, encoding="utf-8"))
    rows = []
    for meta in reg["datasets"]:
        path = os.path.join(ROOT, meta["data_path"])
        if not os.path.exists(path):
            print("跳过（缺文件）", meta["id"]); continue
        raw = io_loaders.load_dataset(path)
        prof, parsed = profiler.profile_dataset(raw)
        q = quality.check_quality(raw, prof, parsed)
        plan = get_adapter("weather").plan(QUESTION, prof, q, raw.frame)
        plan.scene_confidence = 1.0
        clean = preprocess.clean_dataset(raw, prof, parsed, plan)

        old = _run(clean, plan, 0.0)
        new = _run(clean, plan, 3.0)
        old_mean = [c for c in old if c.kind == "mean"]
        new_mean = [c for c in new if c.kind == "mean"]
        dropped = [(c.column, c.index) for c in old_mean
                   if (c.column, c.index) not in {(d.column, d.index) for d in new_mean}]
        rows.append({
            "id": meta["id"], "name": meta.get("name"), "rows": int(len(clean.frame)),
            "period": int(plan.parameters.get("seasonal_period") or 0),
            "old_total": len(old), "new_total": len(new),
            "old_mean": len(old_mean), "new_mean": len(new_mean),
            "old_var": len(old) - len(old_mean), "new_var": len(new) - len(new_mean),
            "dropped_mean": dropped,
            "new_points": [{"column": c.column, "kind": c.kind, "index": c.index,
                            "statistic": c.statistic, "confidence": c.confidence}
                           for c in new],
        })
        print("%-4s %-26s 行=%-7d 周期=%-3d 均值 %d→%d  方差 %d→%d  丢弃 %d" % (
            meta["id"], (meta.get("name") or "")[:26], len(clean.frame),
            rows[-1]["period"], len(old_mean), len(new_mean),
            rows[-1]["old_var"], rows[-1]["new_var"], len(dropped)))

    doc = {
        "question": QUESTION,
        "min_snr_new": 3.0,
        "datasets": rows,
        "totals": {
            "old_mean": sum(r["old_mean"] for r in rows),
            "new_mean": sum(r["new_mean"] for r in rows),
            "old_var": sum(r["old_var"] for r in rows),
            "new_var": sum(r["new_var"] for r in rows),
            "dropped_mean": sum(len(r["dropped_mean"]) for r in rows),
            "datasets_with_change": sum(1 for r in rows if r["old_mean"] != r["new_mean"]),
        },
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2, default=str)
    t = doc["totals"]
    print("\n合计：均值点 %d → %d（丢弃 %d），方差点 %d → %d；%d/%d 份数据有变化" % (
        t["old_mean"], t["new_mean"], t["dropped_mean"], t["old_var"], t["new_var"],
        t["datasets_with_change"], len(rows)))
    print("结果：%s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
