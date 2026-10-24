# -*- coding: utf-8 -*-
"""F3/F4：预测的区间覆盖率与 MASE 横向评估。

口径：
- 数据：`fixtures/weather_eval_v2.csv` + `fixtures/real_weather/` 的 10 份公开真实数据；
- 预处理：与 engine 完全同一条链路（load → profile → quality → plan → clean），
  只跳过指标/检验/异常/作图——它们不改变 `clean` 与 `plan`，不影响预测输入；
- 指标：每列的回测 MASE（walk-forward，MASE<1 表示优于 seasonal naive 的
  「一步朴素」参照）、90% 区间覆盖率（只用最后 40% 的 origin，每个 origin 只向前取误差）；
- 门禁：F3 汇总覆盖率 ∈ [0.80, 0.98]；F4 选中方法 MASE ≤ 1.0 的列占比 ≥ 50%。

运行：
    PYTHONPATH=src python -m evals.forecast_eval
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for _p in (SRC, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mtsa import forecast, io_loaders, preprocess, profiler, quality  # noqa: E402
from mtsa.scenarios.weather import WeatherAdapter  # noqa: E402

FIX = os.path.join(ROOT, "fixtures")
V2 = os.path.join(FIX, "weather_eval_v2.csv")
REGISTRY = os.path.join(FIX, "real_weather", "registry.json")
OUT_JSON = os.path.join(ROOT, "evals", "forecast_results.json")
OUT_MD = os.path.join(ROOT, "evals", "forecast_report.md")

COV_LO, COV_HI = 0.80, 0.98
MASE_HIT_SHARE = 0.50


def _forecast_one(path: str, label: str) -> dict:
    raw = io_loaders.load_dataset(path)
    profile, parsed = profiler.profile_dataset(raw)
    qrep = quality.check_quality(raw, profile, parsed)
    plan = WeatherAdapter().plan("预测评估", profile, qrep, raw.frame)
    plan.scene_confidence = 1.0
    clean = preprocess.clean_dataset(raw, profile, parsed, plan)
    res = forecast.build_forecast(clean, plan, None)
    return {
        "id": label,
        "path": os.path.relpath(path, ROOT).replace(os.sep, "/"),
        "rows": int(len(clean.frame)),
        "period": res["period"],
        "horizon": res["horizon"],
        "columns": [
            {k: c.get(k) for k in ("column", "status", "selected", "selected_reason",
                                   "selected_mase", "baseline_mase", "beats_baseline",
                                   "interval_coverage", "interval_coverage_points",
                                   "origins_used", "points")}
            for c in res["columns"]
        ],
    }


def _dataset_specs() -> list[tuple[str, str]]:
    out = [("v2_fixture", V2)]
    with open(REGISTRY, encoding="utf-8") as f:
        reg = json.load(f)
    for d in reg["datasets"]:
        p = os.path.join(ROOT, d["data_path"])
        if os.path.exists(p):
            out.append((d["id"] + "_" + d["name"], p))
    return out


def _agg(runs: list[dict]) -> dict:
    cols = [c for r in runs for c in r["columns"] if c["status"] == "ok"]
    covs = [c["interval_coverage"] for c in cols if c["interval_coverage"] is not None]
    # 汇总覆盖率按点加权，而不是对列取平均：点数多的列不该和一列小样本等价
    hits = sum(c["interval_coverage"] * c["interval_coverage_points"]
               for c in cols if c["interval_coverage"] is not None)
    total = sum(c["interval_coverage_points"] for c in cols
                if c["interval_coverage"] is not None)
    mase_ok = [c for c in cols if c["selected_mase"] is not None
               and c["selected_mase"] <= 1.0]
    mase_def = [c for c in cols if c["selected_mase"] is not None]
    beats = [c for c in cols if c["beats_baseline"]]
    missing_reason = [c["column"] for c in cols if not (c["selected_reason"] or "").strip()]
    return {
        "columns_ok": len(cols),
        "columns_skipped": sum(1 for r in runs for c in r["columns"] if c["status"] != "ok"),
        "coverage_pooled": round(hits / total, 4) if total else None,
        "coverage_min": round(min(covs), 4) if covs else None,
        "coverage_max": round(max(covs), 4) if covs else None,
        "coverage_points": int(total),
        "coverage_out_of_band": [f"{c['column']}={c['interval_coverage']}"
                                 for c in cols
                                 if c["interval_coverage"] is not None
                                 and not (COV_LO <= c["interval_coverage"] <= COV_HI)],
        "mase_defined": len(mase_def),
        "mase_le_1": len(mase_ok),
        "mase_le_1_share": round(len(mase_ok) / len(cols), 4) if cols else None,
        "beats_baseline": len(beats),
        "beats_baseline_share": round(len(beats) / len(cols), 4) if cols else None,
        "missing_reason": missing_reason,
        "selected_counts": {k: sum(1 for c in cols if c["selected"] == k)
                            for k in set(c["selected"] for c in cols)},
    }


def render_md(doc: dict) -> str:
    s = doc["summary"]
    lines = [
        "# F：预测回测评估",
        "",
        "生成时间：%s" % doc["generated_at"],
        "数据：%d 份（v2 fixture + 10 份公开真实天气数据）" % len(doc["runs"]),
        "方法：seasonal_naive（基线）/ moving_average（窗=周期）/ drift（末值+平均增量）；"
        "按 walk-forward MASE 选型，候选须相对基线下降 ≥5% 才允许替换",
        "",
        "## 门禁",
        "",
        "| 项 | 实际 | 阈值 | 结果 |",
        "| --- | --- | --- | --- |",
        "| F3 汇总 90%% 区间覆盖率 | %s | [%.2f, %.2f] | %s |" % (
            s["coverage_pooled"], COV_LO, COV_HI,
            "PASS" if s["passed"]["F3"] else "FAIL"),
        "| F4 MASE≤1.0 的列占比 | %s（%d/%d） | ≥ %.2f | %s |" % (
            s["mase_le_1_share"], s["mase_le_1"], s["columns_ok"], MASE_HIT_SHARE,
            "PASS" if s["passed"]["F4"] else "FAIL"),
        "| F4 未优于基线的列都有选型说明 | %s | 0 缺失 | %s |" % (
            "是" if not s["missing_reason"] else "否：%s" % s["missing_reason"],
            "PASS" if s["passed"]["F4_note"] else "FAIL"),
        "",
        "**%s**" % ("全部通过" if doc["passed"] else "存在未通过项"),
        "",
        "## 汇总",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        "| 参与评估的列 | %d |" % s["columns_ok"],
        "| 样本不足未预测的列 | %d |" % s["columns_skipped"],
        "| 覆盖率（按点加权） | %s |" % s["coverage_pooled"],
        "| 覆盖率（逐列最小 / 最大） | %s / %s |" % (s["coverage_min"], s["coverage_max"]),
        "| 覆盖率参与点数 | %d |" % s["coverage_points"],
        "| 覆盖率落在 [%.2f, %.2f] 之外的列 | %s |" % (
            COV_LO, COV_HI, "、".join(s["coverage_out_of_band"]) or "无"),
        "| 选中方法 MASE≤1.0 | %d / %d = %s |" % (
            s["mase_le_1"], s["columns_ok"], s["mase_le_1_share"]),
        "| 选中方法优于基线 | %d / %d = %s |" % (
            s["beats_baseline"], s["columns_ok"], s["beats_baseline_share"]),
        "| 选中的方法 | %s |" % "、".join(
            "%s %d 列" % (k, v) for k, v in sorted(s["selected_counts"].items())),
        "",
        "## 逐数据集",
        "",
        "| 数据集 | 行数 | 周期 | horizon | 列 | 选中方法 | MASE | 基线 MASE | 覆盖率 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in doc["runs"]:
        for c in r["columns"]:
            if c["status"] != "ok":
                lines.append("| %s | %s | %s | %s | %s | 未预测 | — | — | — |" % (
                    r["id"], r["rows"], r["period"], r["horizon"], c["column"]))
                continue
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r["id"], r["rows"], r["period"], r["horizon"], c["column"],
                c["selected"], c["selected_mase"], c["baseline_mase"],
                c["interval_coverage"]))
    lines += [
        "",
        "## 读数注意",
        "",
        "- MASE<1 的含义是「误差小于 seasonal naive 的一步朴素参照」，"
        "不是「预测准确」；天气预报本身有不可约的随机分量，MASE≈1 在气温上很常见。",
        "- 覆盖率按点加权汇总，逐列偏离的少数列单独列在上表，不掩盖。",
        "- 未做多模型对比，也没有在评估集上调过任何阈值；三个方法在运行前就已固定。",
        "",
        "## 复现",
        "",
        "```bash",
        "PYTHONPATH=src python -m evals.forecast_eval",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    runs = []
    for label, path in _dataset_specs():
        print("评估 %s ..." % label)
        runs.append(_forecast_one(path, label))
    summary = _agg(runs)
    f3 = summary["coverage_pooled"] is not None and COV_LO <= summary["coverage_pooled"] <= COV_HI
    f4 = (summary["mase_le_1_share"] is not None
          and summary["mase_le_1_share"] >= MASE_HIT_SHARE)
    doc = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "params": {"methods": list(forecast.METHODS),
                   "origins": forecast.DEFAULT_ORIGINS,
                   "calib_window": forecast.CALIB_WINDOW,
                   "interval_level": forecast.INTERVAL_LEVEL,
                   "min_improvement": forecast.MIN_IMPROVEMENT},
        "thresholds": {"coverage": [COV_LO, COV_HI], "mase_le_1_share": MASE_HIT_SHARE},
        "runs": runs,
        "summary": summary,
        "passed": f3 and f4 and not summary["missing_reason"],
    }
    doc["summary"]["passed"] = {"F3": f3, "F4": f4, "F4_note": not summary["missing_reason"]}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2, default=str)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(render_md(doc))
    print()
    print("F3 coverage pooled = %s -> %s" % (summary["coverage_pooled"], "PASS" if f3 else "FAIL"))
    print("F4 MASE<=1 share = %s (%d/%d) -> %s" % (
        summary["mase_le_1_share"], summary["mase_le_1"], summary["columns_ok"],
        "PASS" if f4 else "FAIL"))
    print("结果：%s" % OUT_JSON)
    print("报告：%s" % OUT_MD)
    return 0 if doc["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
