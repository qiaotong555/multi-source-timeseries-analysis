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
import math
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for _p in (SRC, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mtsa import forecast, io_loaders, preprocess, prob_forecast, profiler, quality  # noqa: E402
from mtsa.scenarios.weather import WeatherAdapter  # noqa: E402

FIX = os.path.join(ROOT, "fixtures")
V2 = os.path.join(FIX, "weather_eval_v2.csv")
REGISTRY = os.path.join(FIX, "real_weather", "registry.json")
OUT_JSON = os.path.join(ROOT, "evals", "forecast_results.json")
OUT_MD = os.path.join(ROOT, "evals", "forecast_report.md")

COV_LO, COV_HI = 0.80, 0.98
MASE_HIT_SHARE = 0.50
# 逐列覆盖率是点估计，384 个点上 0.7995 与 0.8000 的差别没有统计意义。
# 所以出界判定用 Wilson 95% 置信区间：区间与 [COV_LO, COV_HI] 相交即不判出界。
WILSON_Z = 1.96
# 事件点数低于该值时 BSS 不可解释（「基本不报」也能赢过基线），标为未运行而不是失败。
MIN_EVENT_POINTS = prob_forecast.MIN_EVENTS_FOR_SKILL
# 条件区间（零膨胀列「真下雨时」的覆盖率）同样要求足够的非零观测才判定
MIN_NONZERO_POINTS = 100


def _wilson(hits: int, n: int, z: float = WILSON_Z) -> tuple:
    if n <= 0:
        return (None, None)
    p = hits / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


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
                                   "zero_share", "zero_inflated",
                                   "interval_coverage_nonzero",
                                   "interval_coverage_nonzero_points",
                                   "origins_used", "points")} |
            {"conditional": (None if not c.get("conditional") else
                              {k: c["conditional"].get(k) for k in ("level", "method",
                               "calib_window", "calib_points", "coverage",
                               "coverage_points")})}
            for c in res["columns"]
        ],
        "events": res.get("events", {}),
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


def _verdict(c: dict) -> dict:
    """逐列覆盖率判定。

    三种情况分开：
    - 零膨胀列：整体覆盖率衡量的是「猜没下雨猜对了吗」，不判；
    - 非零膨胀列：用 Wilson 95% 区间与 [COV_LO, COV_HI] 的关系判，不用点估计卡边界；
    - 没有覆盖率：标 no_coverage。
    """
    cov = c.get("interval_coverage")
    n = c.get("interval_coverage_points") or 0
    if cov is None or n <= 0:
        return {"verdict": "no_coverage", "ci": None}
    if c.get("zero_inflated"):
        return {"verdict": "not_applicable_zero_inflated", "ci": None,
                "nonzero_points": c.get("interval_coverage_nonzero_points")}
    hits = int(round(cov * n))
    lo, hi = _wilson(hits, n)
    out = (hi < COV_LO) or (lo > COV_HI)
    return {"verdict": "out_of_band" if out else "in_band",
            "ci": [round(lo, 4), round(hi, 4)], "hits": hits, "n": n}


def _agg(runs: list[dict]) -> dict:
    pairs = [(r["id"], c) for r in runs for c in r["columns"] if c["status"] == "ok"]
    cols = [c for _, c in pairs]
    for c in cols:
        c["coverage_verdict"] = _verdict(c)
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
    out_of_band = [(i, c) for i, c in pairs if c["coverage_verdict"]["verdict"] == "out_of_band"]
    nz = [(i, c) for i, c in pairs
          if c["coverage_verdict"]["verdict"] == "not_applicable_zero_inflated"]

    # 事件概率预报汇总：只有事件点数够的列才判 BSS，其余标「未判」
    ev_all = [(r["id"], ev) for r in runs for ev in r.get("events", {}).get("columns", [])]
    ev_ok = [(i, ev) for i, ev in ev_all if ev.get("status") == "ok"]
    ev_judged = [(i, ev) for i, ev in ev_ok
                 if (ev.get("event_points") or 0) >= MIN_EVENT_POINTS]
    ev_pass = None
    if ev_judged:
        ev_pass = all((ev.get("bss") is not None and ev["bss"] > 0) for _, ev in ev_judged)

    # 条件区间：零膨胀列在「真下雨时」的覆盖率，非零观测够多才判
    cond = [(i, c) for i, c in pairs
            if c.get("conditional") and (c["conditional"].get("coverage_points") or 0) >= MIN_NONZERO_POINTS]
    cond_pass = None
    if cond:
        cond_pass = all(COV_LO <= c["conditional"]["coverage"] <= COV_HI for _, c in cond)
    return {
        "columns_ok": len(cols),
        "columns_skipped": sum(1 for r in runs for c in r["columns"] if c["status"] != "ok"),
        "coverage_pooled": round(hits / total, 4) if total else None,
        "coverage_min": round(min(covs), 4) if covs else None,
        "coverage_max": round(max(covs), 4) if covs else None,
        "coverage_points": int(total),
        "coverage_out_of_band": [f"{c['column']}={c['interval_coverage']}"
                                 for _, c in out_of_band],
        "coverage_out_of_band_detail": [
            {"dataset": i, "column": c["column"], "coverage": c["interval_coverage"],
             "ci": c["coverage_verdict"]["ci"]} for i, c in out_of_band],
        "coverage_not_applicable": [
            {"dataset": i, "column": c["column"], "coverage": c["interval_coverage"],
             "zero_share": c.get("zero_share"),
             "nonzero_points": c.get("interval_coverage_nonzero_points")} for i, c in nz],
        "coverage_verdict_counts": {
            k: sum(1 for c in cols if c["coverage_verdict"]["verdict"] == k)
            for k in ("in_band", "out_of_band", "not_applicable_zero_inflated", "no_coverage")},
        "zero_inflated_columns": sum(1 for c in cols if c.get("zero_inflated")),
        "mase_defined": len(mase_def),
        "mase_le_1": len(mase_ok),
        "mase_le_1_share": round(len(mase_ok) / len(cols), 4) if cols else None,
        "beats_baseline": len(beats),
        "beats_baseline_share": round(len(beats) / len(cols), 4) if cols else None,
        "missing_reason": missing_reason,
        "selected_counts": {k: sum(1 for c in cols if c["selected"] == k)
                            for k in set(c["selected"] for c in cols)},
        "conditional": {
            "judged": len(cond),
            "min_nonzero_points": MIN_NONZERO_POINTS,
            "passed": cond_pass,
            "rows": [{"dataset": i, "column": c["column"],
                      "coverage": c["conditional"]["coverage"],
                      "points": c["conditional"]["coverage_points"],
                      "overall": c["interval_coverage_nonzero"]}
                     for i, c in cond],
        },
        "events": {
            "columns": len(ev_all),
            "evaluated": len(ev_ok),
            "judged": len(ev_judged),
            "min_event_points": MIN_EVENT_POINTS,
            "passed": ev_pass,
            "bss": [{"dataset": i, "column": ev["column"],
                     "event_points": ev.get("event_points"),
                     "observed_rate": ev.get("observed_rate"),
                     "bss": ev.get("bss"), "skill": ev.get("skill"),
                     "ece": ev.get("ece")}
                    for i, ev in ev_ok],
        },
    }


def render_md(doc: dict) -> str:
    s = doc["summary"]
    lines = [
        "# F：预测回测评估",
        "",
        "生成时间：%s" % doc["generated_at"],
        "数据：%d 份（v2 fixture + %d 份公开真实天气数据）" % (len(doc["runs"]), len(doc["runs"]) - 1),
        "方法：seasonal_naive（基线）/ moving_average（窗=周期）/ drift（末值+平均增量）；"
        "按 walk-forward MASE 选型，候选须相对基线下降 ≥5% 才允许替换",
        "区间：最近 %d 个 origin 的误差按 horizon 归一、分 %d 箱、取 conformal 有限样本修正分位"
        % (doc["params"]["calib_window"], forecast.CALIB_BINS),
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
        "| F6 零膨胀列条件区间覆盖率 ∈ [%.2f, %.2f]（非零点 ≥ %d） | %s | 每列在带内 | %s |" % (
            COV_LO, COV_HI, s["conditional"]["min_nonzero_points"],
            ("%d 列参与，%s" % (s["conditional"]["judged"],
                               "全部在带内" if s["conditional"]["passed"] else "存在带外列"))
            if s["conditional"]["passed"] is not None else "未运行（无非零点足够的列）",
            "PASS" if s["conditional"]["passed"] else (
                "FAIL" if s["conditional"]["passed"] is False else "未运行")),
        "| F5 零膨胀列事件概率预报 BSS > 0（事件点 ≥ %d） | %s | 每列 > 0 | %s |" % (
            s["events"]["min_event_points"],
            ("%d 列参与，全部为正" % s["events"]["judged"]) if s["events"]["passed"]
            else ("%d 列参与，存在非正 BSS" % s["events"]["judged"]
                  if s["events"]["judged"] else "未运行（无事件点足够的列）"),
            "PASS" if s["events"]["passed"] else (
                "FAIL" if s["events"]["passed"] is False else "未运行")),
        "",
        "**%s**" % ("全部通过" if doc["passed"] else "存在未通过项"),
        "",
        "## 汇总",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        "| 参与评估的列 | %d |" % s["columns_ok"],
        "| 样本不足未预测的列 | %d |" % s["columns_skipped"],
        "| 零膨胀列（零值 ≥ 50%%） | %d |" % s["zero_inflated_columns"],
        "| 覆盖率（按点加权） | %s |" % s["coverage_pooled"],
        "| 覆盖率（逐列最小 / 最大） | %s / %s |" % (s["coverage_min"], s["coverage_max"]),
        "| 覆盖率参与点数 | %d |" % s["coverage_points"],
        "| 判定分布 | 带内 %d、带外 %d、零膨胀不适用 %d |" % (
            s["coverage_verdict_counts"]["in_band"],
            s["coverage_verdict_counts"]["out_of_band"],
            s["coverage_verdict_counts"]["not_applicable_zero_inflated"]),
        "| 覆盖率带外（Wilson 95%% 区间与带不相交） | %s |" % (
            "、".join("%s=%.4f（CI %.3f–%.3f）" % (
                d["column"], d["coverage"], d["ci"][0], d["ci"][1])
                for d in s["coverage_out_of_band_detail"]) or "无"),
        "| 零膨胀不判区间覆盖率的列 | %s |" % (
            ("%d 列（全部为降水类）；整体覆盖率 %.3f ~ %.3f，非零点 %d ~ %d；逐列明细见 "
             "`coverage_not_applicable`" % (
                 len(s["coverage_not_applicable"]),
                 min(d["coverage"] for d in s["coverage_not_applicable"]),
                 max(d["coverage"] for d in s["coverage_not_applicable"]),
                 min(d["nonzero_points"] for d in s["coverage_not_applicable"]),
                 max(d["nonzero_points"] for d in s["coverage_not_applicable"])))
            if s["coverage_not_applicable"] else "无"),
        "| 选中方法 MASE≤1.0 | %d / %d = %s |" % (
            s["mase_le_1"], s["columns_ok"], s["mase_le_1_share"]),
        "| 选中方法优于基线 | %d / %d = %s |" % (
            s["beats_baseline"], s["columns_ok"], s["beats_baseline_share"]),
        "| 选中的方法 | %s |" % "、".join(
            "%s %d 列" % (k, v) for k, v in sorted(s["selected_counts"].items())),
        "",
        "## 条件区间（零膨胀列：真下雨时的覆盖率）",
        "",
        "整体覆盖率在这类列上不算判据，看「观测非零」时刻的覆盖率。"
        "`overall` 是用整体区间统计的（必然偏窄），`conditional` 是用只按非零误差标定的区间。",
        "",
        "| 数据集 | 列 | 非零点 | 整体区间覆盖率 | 条件区间覆盖率 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in s["conditional"]["rows"]:
        lines.append("| %s | %s | %s | %s | **%s** |" % (
            r["dataset"], r["column"], r["points"], r["overall"], r["coverage"]))
    if not s["conditional"]["rows"]:
        lines.append("| — | — | — | — | 无符合条件的列 |")
    lines += [
        "",
        "## 事件概率预报（零膨胀列）",
        "",
        "阈值 %.3g；主模型 = climatology 与 persistence_short 的等权平均；"
        "BSS 相对 climatology。事件点少于 %d 的列不判，因为「基本不报」也能把 BSS 抬高。"
        % (doc["params"]["event_threshold"], s["events"]["min_event_points"]),
        "",
        "| 数据集 | 列 | 事件点 | 实测发生率 | BSS | 等级 | ECE | 判定 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for e in s["events"]["bss"]:
        judged = (e["event_points"] or 0) >= s["events"]["min_event_points"]
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            e["dataset"], e["column"], e["event_points"], e["observed_rate"],
            e["bss"], e.get("skill"), e["ece"],
            "参与门禁" if judged else "未判（事件点不足）"))
    if not s["events"]["bss"]:
        lines.append("| — | — | — | — | — | — | — | 无零膨胀列 |")
    lines += [
        "",
        "## 逐数据集",
        "",
        "| 数据集 | 行数 | 周期 | horizon | 列 | 选中方法 | MASE | 基线 MASE | 覆盖率 | 零膨胀 | 判定 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in doc["runs"]:
        for c in r["columns"]:
            if c["status"] != "ok":
                lines.append("| %s | %s | %s | %s | %s | 未预测 | — | — | — | — | — |" % (
                    r["id"], r["rows"], r["period"], r["horizon"], c["column"]))
                continue
            v = c.get("coverage_verdict") or {}
            verdict = {"in_band": "带内", "out_of_band": "带外",
                       "not_applicable_zero_inflated": "零膨胀不适用",
                       "no_coverage": "无覆盖率"}.get(v.get("verdict"), "—")
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r["id"], r["rows"], r["period"], r["horizon"], c["column"],
                c["selected"], c["selected_mase"], c["baseline_mase"],
                c["interval_coverage"], "是" if c.get("zero_inflated") else "否", verdict))
    lines += [
        "",
        "## 读数注意",
        "",
        "- MASE<1 的含义是「误差小于 seasonal naive 的一步朴素参照」，"
        "不是「预测准确」；天气预报本身有不可约的随机分量，MASE≈1 在气温上很常见。",
        "- 覆盖率按点加权汇总，逐列偏离的少数列单独列在上表，不掩盖。",
        "- **零膨胀列不判区间覆盖率**：降水类列 79%–99% 的时刻观测为 0，"
        "「预测 0、实际也是 0」就能拿 0.99 的覆盖率；它衡量的是猜没下雨，不是报得准。"
        "这类列改看事件概率预报的 Brier / BSS / 可靠性。",
        "- 逐列出界用 Wilson 95% 置信区间判定，不用点估计卡边界：384 个点上的标准误约 1.5–2 个百分点，"
        "0.7995 与 0.8000 的差别没有统计意义。",
        "- 事件点数（`event_points`）少于 %d 时 BSS 不可解释——测试期几乎没发生事件时，"
        "「基本不报」的模型会赢过气候基线，那个 BSS 不能当能力。"
        % s["events"]["min_event_points"],
        "- 事件技能分档：BSS≤0 无技能；0–0.05 极弱（几乎无实用价值）；0.05–0.15 弱；"
        ">0.15 中等以上。F5 只要求「BSS>0」，通过不等于能用于业务预报。",
        "- **数据来源有重复**：`R05_three_cities` 的 `rainfall@SIN` 与 `R02_singapore` 是同一份"
        "新加坡数据，两列数字必然相同；算独立证据时必须按来源去重，不能当两列。",
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
    f5 = summary["events"]["passed"]          # None = 未运行（没有事件点足够的列）
    f6 = summary["conditional"]["passed"]     # None = 未运行（没有非零点够的列）
    doc = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "params": {"methods": list(forecast.METHODS),
                   "origins": forecast.DEFAULT_ORIGINS,
                   "calib_window": forecast.CALIB_WINDOW,
                   "calib_bins": forecast.CALIB_BINS,
                   "interval_level": forecast.INTERVAL_LEVEL,
                   "min_improvement": forecast.MIN_IMPROVEMENT,
                   "event_threshold": prob_forecast.EVENT_THRESHOLD,
                   "short_state": prob_forecast.SHORT_STATE},
        "thresholds": {"coverage": [COV_LO, COV_HI], "mase_le_1_share": MASE_HIT_SHARE,
                       "event_min_points": MIN_EVENT_POINTS,
                       "coverage_ci": "wilson95"},
        "runs": runs,
        "summary": summary,
        "passed": (f3 and f4 and not summary["missing_reason"]
                   and (f5 is not False) and (f6 is not False)),
    }
    doc["summary"]["passed"] = {"F3": f3, "F4": f4, "F4_note": not summary["missing_reason"],
                                "F5": f5, "F6": f6}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2, default=str)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(render_md(doc))
    print()
    print("F3 coverage pooled = %s -> %s" % (summary["coverage_pooled"], "PASS" if f3 else "FAIL"))
    print("F4 MASE<=1 share = %s (%d/%d) -> %s" % (
        summary["mase_le_1_share"], summary["mase_le_1"], summary["columns_ok"],
        "PASS" if f4 else "FAIL"))
    print("F5 event BSS = %s -> %s（参与 %d 列，事件点 ≥ %d）" % (
        "未运行" if f5 is None else f5, "PASS" if f5 else ("FAIL" if f5 is False else "未运行"),
        summary["events"]["judged"], MIN_EVENT_POINTS))
    print("F6 条件区间 = %s -> %s（参与 %d 列，非零点 >= %d）" % (
        "未运行" if f6 is None else f6, "PASS" if f6 else ("FAIL" if f6 is False else "未运行"),
        summary["conditional"]["judged"], MIN_NONZERO_POINTS))
    print("覆盖率判定：带内 %d、带外 %d、零膨胀不适用 %d" % (
        summary["coverage_verdict_counts"]["in_band"],
        summary["coverage_verdict_counts"]["out_of_band"],
        summary["coverage_verdict_counts"]["not_applicable_zero_inflated"]))
    print("结果：%s" % OUT_JSON)
    print("报告：%s" % OUT_MD)
    return 0 if doc["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
