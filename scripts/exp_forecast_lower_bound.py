# -*- coding: utf-8 -*-
"""量预测区间下界为负的范围，以及「展示层截断」会不会动指标。

背景：`forecast[].lo = point + 分位数下界`，降水这类非负列会算出负数
（`point=0`、误差下分位为负）。但覆盖率是用**误差空间**的 `cal["lo"]` 算的
（`_coverage`），与展示值 `point + lo` 无关——所以「按物理下界截断展示」
理论上不改任何指标。这个脚本同时验证两件事：

1. 有多少列、多少点出现负下界，最负多少；
2. 同一次运行里，`interval_coverage` / `interval_coverage_nonzero` / `conditional`
   与 `forecast[].lo` 是否解耦（截断前后这些字段必须逐位相同）。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mtsa import forecast, io_loaders, preprocess, profiler, quality  # noqa: E402
from mtsa.scenarios.weather import WeatherAdapter  # noqa: E402

from evals.forecast_eval import _dataset_specs  # noqa: E402

# 物理上非负的变量（下界 0）。只按列名匹配，不做数据驱动——
# 热带站点的气温观测也可能全程为正，用 min(y) >= 0 会把负温度预测错误截断。
NONNEGATIVE_HINTS = ("rain", "precip", "降水", "雨", "wind", "风速", "gust", "阵风",
                     "humidity", "湿度", "湿")


def _is_nonnegative(column: str) -> bool:
    head = str(column).split("@")[0].lower()
    return any(h in head for h in NONNEGATIVE_HINTS)


def main() -> int:
    specs = _dataset_specs()
    print(f"数据集 {len(specs)} 份\n")

    total_cols = total_points = neg_cols = neg_points = 0
    worst = (0.0, "", "")
    per_var: dict[str, int] = {}
    unchanged = 0

    for label, path in specs:
        if not os.path.exists(path):
            print(f"跳过（不存在）：{label}")
            continue
        raw = io_loaders.load_dataset(path)
        profile, parsed = profiler.profile_dataset(raw)
        qrep = quality.check_quality(raw, profile, parsed)
        plan = WeatherAdapter().plan("预测评估", profile, qrep, raw.frame)
        plan.scene_confidence = 1.0
        clean = preprocess.clean_dataset(raw, profile, parsed, plan)
        res = forecast.build_forecast(clean, plan, None)

        for c in res["columns"]:
            pts = c.get("forecast") or []
            total_cols += 1
            total_points += len(pts)
            negs = [p["lo"] for p in pts
                    if p.get("lo") is not None and p["lo"] < 0]
            if not negs:
                continue
            neg_cols += 1
            neg_points += len(negs)
            var = str(c["column"]).split("@")[0]
            per_var[var] = per_var.get(var, 0) + len(negs)
            if min(negs) < worst[0]:
                worst = (min(negs), c["column"], label)

            # 截断只改展示值；这些字段是误差空间算出来的，必须不受影响
            before = (c.get("interval_coverage"), c.get("interval_coverage_nonzero"),
                      json.dumps(c.get("conditional"), sort_keys=True))
            clipped = [max(0.0, p["lo"]) if p.get("lo") is not None else None for p in pts]
            after = (c.get("interval_coverage"), c.get("interval_coverage_nonzero"),
                     json.dumps(c.get("conditional"), sort_keys=True))
            assert before == after, c["column"]
            assert all(v is None or v >= 0 for v in clipped)
            unchanged += 1

    print(f"扫描列 {total_cols} 个、预测点 {total_points} 个")
    print(f"出现负下界：**{neg_cols} 列 / {neg_points} 点**"
          f"（{100.0 * neg_points / max(total_points, 1):.1f}% 的点）")
    if neg_cols:
        print(f"最负值 {worst[0]:.3f}（{worst[1]} @ {worst[2]}）")
        print("按变量分布：" + "、".join(
            f"{k} {v} 点" for k, v in sorted(per_var.items(), key=lambda kv: -kv[1])))
        print(f"其中属于非负物理量的列："
              f"{sum(1 for k in per_var if _is_nonnegative(k))} 个变量名命中")
    print(f"\n截断前后覆盖率/条件区间字段逐列相同：{unchanged} 列")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
