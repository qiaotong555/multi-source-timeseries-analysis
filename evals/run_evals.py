# -*- coding: utf-8 -*-
"""执行批量评估并输出 evals/eval_results.json 与控制台摘要。

异常检测指标采用 dev / holdout 划分：阈值只在各折 dev 段上选定，
指标取各折 holdout 段平均；有效折不足时整体判为不可判定。

运行：
    python -m evals.run_evals          # 全部指标（含划分）
    python -m evals.split_eval         # 只复核划分结果
"""
from __future__ import annotations

import json
import math
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # evals/ -> 项目根
SRC = os.path.join(ROOT, "src")
for p in (SRC, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from mtsa.engine import run_analysis  # noqa: E402
from mtsa.io_loaders import sha1_of_file  # noqa: E402
from mtsa.models import AnalysisRequest  # noqa: E402
from mtsa import metrics as M  # noqa: E402
from mtsa.scenarios.stock import max_drawdown  # noqa: E402
from evals import split_eval  # noqa: E402
from evals import changepoint_scan  # noqa: E402
from evals import conclusion_cache  # noqa: E402

FIX = os.path.join(ROOT, "fixtures")
OUT = os.path.join(ROOT, "artifacts")
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

THRESHOLDS = {
    "数据类型识别率": ("=", 1.00),
    "时间字段识别率": ("=", 1.00),
    "采样异常检出率": (">=", 0.90),
    "方法选择准确率": (">=", 0.90),
    "关键指标复算误差": ("<=", 1e-6),
    "异常检测 Precision": (">=", 0.70),
    "异常检测 Recall": (">=", 0.70),
    "事件级异常召回率": (">=", 0.70),
    "集体异常召回率": (">=", 0.70),
    "dev与holdout精度退化": ("<=", 0.20),
    "dev与holdout召回退化": ("<=", 0.20),
    "划分有效折数": (">=", 2),
    "变化点检出率": (">=", 0.80),
    "变化点定位误差90分位": ("<=", 3),
    "变化点伪报率": ("<=", 0.20),
    "1e5行分析耗时": ("<=", 120.0),
    "1e5行峰值内存": ("<=", 1024.0),
    "图表字段映射准确率": ("=", 1.00),
    "重复运行一致率": ("=", 1.00),
    "计划来源可追溯": ("=", 1.00),
    "未来数据泄漏检出率": ("=", 1.00),
    "无证据结论数": ("=", 0),
}

# LLM 模式指标：仅在 llm_eval_results.json 存在时才纳入判定。
# 不放进 THRESHOLDS 的原因见 _load_llm_eval 的说明：依赖外部服务，
# 环境不具备时本就无法评估，不能因此把整套评估判为失败。
LLM_THRESHOLDS = {
    "LLM计划合法率": (">=", 0.90),
    "LLM回退率": ("<=", 0.10),
    "LLM分析项覆盖率": (">=", 0.60),
}

# Agent 指标：仅在 agent_eval_results.json 存在且来源为真实模型时纳入判定。
# mock 结果只能证明指标管线可用，不能当模型能力结论，故不纳入。
AGENT_THRESHOLDS = {
    "Agent成功率": (">=", 0.90),
    "Agent需求覆盖率": (">=", 0.80),
    "Agent可重放率": (">=", 0.95),
    "Agent工具失败率": ("<=", 0.10),
}


def _ensure_fixtures():
    from fixtures.gen_fixtures import main as gen_main
    if not os.path.exists(os.path.join(FIX, "weather_v1.csv")):
        gen_main()


def _run(name, scene, csv, question, out):
    req = AnalysisRequest(request_id=f"eval_{name}", input_path=os.path.join(FIX, csv),
                          scene_id=scene, question=question, seed=20260927,
                          output_dir=os.path.join(OUT, f"eval_{name}"))
    return run_analysis(req, verbose=False)


def _load_benchmark() -> dict:
    """读取性能基准缓存；缺失或损坏时如实标注，不伪造数字。"""
    path = os.path.join(EVAL_DIR, "benchmark_results.json")
    if not os.path.exists(path):
        return {"available": False,
                "reason": "benchmark_results.json 不存在，请先运行 python -m evals.benchmark"}
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception as exc:                      # noqa: BLE001
        return {"available": False, "reason": f"解析失败：{exc}"}

    scales = {s["label"]: s for s in d.get("scales", [])}
    pick = ("total_s", "peak_mb", "status")
    return {
        "available": True,
        "generated_at": d.get("generated_at"),
        "env": d.get("env"),
        "scale_1e4": {k: scales.get("1e4", {}).get(k) for k in pick},
        "scale_1e5": {k: scales.get("1e5", {}).get(k) for k in pick + ("timings",)},
        "scale_1e6_status": scales.get("1e6", {}).get("status"),
    }


def _load_llm_eval() -> dict:
    """读取 LLM 模式评估缓存；缺失标记「未运行」，不判失败。

    与性能基准的处理刻意不同：基准在本机一定跑得起来，缺了就是漏项，记 9999 判失败；
    LLM 评估依赖外部服务与密钥，环境不具备时本就无法进行，若判失败等于用
    「没配密钥」否定整个系统。因此这里用独立的「未运行」状态：
    既不冒充通过，也不误判为失败。
    """
    path = os.path.join(EVAL_DIR, "llm_eval_results.json")
    if not os.path.exists(path):
        return {"available": False,
                "reason": "llm_eval_results.json 不存在，请先运行 python -m evals.llm_eval"}
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception as exc:                      # noqa: BLE001
        return {"available": False, "reason": f"解析失败：{exc}"}

    s = d.get("summary") or {}
    keep = ("id", "scene", "question", "selected", "valid", "source",
            "tool_recall", "missing", "consistent")
    return {
        "available": True,
        "generated_at": d.get("generated_at"),
        "model": d.get("model"),
        "base_url": d.get("base_url"),
        "planner_version": d.get("planner_version"),
        "n_calls": d.get("n_calls"),
        "summary": s,
        "cases": [{k: c.get(k) for k in keep} for c in d.get("cases", [])],
    }


def _load_agent_eval() -> dict:
    """读取 Agent 模式评估缓存；缺失或 mock 都不纳入判定。

    与 llm_eval 同理：缺缓存是「未运行」，不是「未通过」；
    但 mock 结果必须单独标注，否则会把脚本客户端的能力当成真实模型能力。
    """
    path = os.path.join(EVAL_DIR, "agent_eval_results.json")
    if not os.path.exists(path):
        return {"available": False,
                "reason": "agent_eval_results.json 不存在，请先运行 python -m evals.agent_eval"}
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception as exc:                      # noqa: BLE001
        return {"available": False, "reason": f"解析失败：{exc}"}

    s = d.get("summary") or {}
    source = d.get("source") or s.get("source")
    keep = ("case", "status", "steps", "terminate_reason", "tool_coverage",
            "analysis_coverage", "rule_plan_coverage", "need_coverage",
            "tool_failure_rate", "guard_violations", "elapsed_ms", "replay_ok",
            "tokens")
    return {
        "available": True,
        "judged": source == "real",
        "source": source,
        "generated_at": d.get("generated_at"),
        "model": d.get("model"),
        "planner": d.get("planner"),
        "summary": s,
        "cases": [{k: c.get(k) for k in keep} for c in d.get("cases", [])],
    }


def _fmt_prf(d: dict) -> str:
    if not d or d.get("precision") is None:
        return "—"
    return "%.3f / %.3f / %.3f" % (d["precision"], d["recall"], d["f1"])


def evaluate() -> dict:
    # 评估复算必须固定规则计划：MTSA_PLANNER=auto 且配置了模型时，LLM 会按问题
    # 裁剪分析项（例如「极端事件」不选 anomaly），导致阈值指标随模型版本漂移，
    # 同一条命令两次结果不同。LLM 计划能力由 evals.llm_eval 单独评估。
    os.environ["MTSA_PLANNER"] = "rule"
    _ensure_fixtures()
    res: dict[str, object] = {}
    res["_planner"] = {
        "mode": "rule",
        "reason": "评估复算固定规则计划，避免 LLM 按问题裁剪分析项；LLM 计划由 evals.llm_eval 单独评估",
    }

    wx = _run("weather", "weather", "weather_v1.csv", "分析温度的趋势、季节性与极端事件", OUT)
    st = _run("stock", "stock", "stock_v1.csv", "计算收益风险指标并做 walk-forward 回测", OUT)
    lk = _run("leak", "stock", "stock_leak_v1.csv", "回测", OUT)

    # 1) 数据类型识别率 / 时间字段识别率
    prof_wx = wx.data["profile"]
    prof_st = st.data["profile"]
    res["数据类型识别率"] = 1.0 if (prof_wx["numeric_columns"] and prof_st["numeric_columns"]) else 0.0
    res["时间字段识别率"] = 1.0 if (prof_wx["time_axis"] and prof_st["time_axis"]) else 0.0

    # 2) 采样异常检出率：fixture 注入了缺口/乱序/重复
    codes = {i["code"] for i in wx.data["quality"]["issues"]}
    injected = {"SAMPLING_GAP", "OUT_OF_ORDER", "DUPLICATE_TIMESTAMPS", "MISSING_VALUES"}
    res["采样异常检出率"] = round(len(injected & codes) / len(injected), 4)

    # 3) 方法选择准确率：天气应含 spectrum/changepoint/corr，股票应含回测指标
    a_wx = set(wx.data["plan"]["analyses"])
    ok_wx = {"spectrum", "changepoint", "corr", "trend"} <= a_wx
    ok_st = any(r["metric_id"].startswith("stk.bt.") for r in st.data["scenario_metrics"])
    res["方法选择准确率"] = 1.0 if (ok_wx and ok_st) else 0.5

    # 4) 关键指标复算误差
    px = pd.read_csv(os.path.join(OUT, "eval_stock", "clean_data.csv"))
    close = pd.to_numeric(px["close"])
    expect_mdd, _, _ = max_drawdown(close.to_numpy() / float(close.iloc[0]))
    got_mdd = next(r["value"] for r in st.data["scenario_metrics"]
                   if r["metric_id"] == "stk.risk.max_drawdown")
    expect_cum = float(close.iloc[-1] / close.iloc[0] - 1.0)
    got_cum = next(r["value"] for r in st.data["scenario_metrics"]
                   if r["metric_id"] == "stk.ret.cumulative")
    errs = [abs(got_mdd - expect_mdd) / max(abs(expect_mdd), 1e-9),
            abs(got_cum - expect_cum) / max(abs(expect_cum), 1e-9)]
    res["关键指标复算误差"] = max(errs)

    # 5) 异常检测 Precision / Recall
    # 口径：阈值只在各折 dev 段上选定，指标取各折 holdout 段的平均。
    # 数据来自评估专用集 weather_eval_v1（异常密度高于演示集），演示集 weather_v1
    # 仅用于其余指标与交付演示，二者分离，避免「同一套数据调参又报告」。
    split_res = split_eval.run_split_eval(
        base_csv=os.path.join(FIX, split_eval.EVAL_DATASET),
        labels_csv=os.path.join(FIX, split_eval.EVAL_LABELS),
        scene_id="weather",
        question="分析温度的趋势、季节性与极端事件",
        out_root=os.path.join(OUT, "eval_split"),
        verbose=False,
        planner_mode="rule",
    )
    res["_split_eval"] = split_res
    s_sum = split_res["summary"]
    res["异常检测 Precision"] = s_sum.get("mean_holdout_precision", 0.0)
    res["异常检测 Recall"] = s_sum.get("mean_holdout_recall", 0.0)
    res["事件级异常召回率"] = s_sum.get("mean_holdout_event_recall", 0.0)
    res["集体异常召回率"] = s_sum.get("mean_collective_recall") or 0.0
    res["dev与holdout精度退化"] = s_sum.get("max_degradation_precision", 1.0)
    res["dev与holdout召回退化"] = s_sum.get("max_degradation_recall", 1.0)
    res["划分有效折数"] = len(s_sum.get("valid_folds", []))
    res["_anomaly_detail"] = {
        "metric_scope": "holdout 段平均（阈值在 dev 段选定），数据为 weather_eval_v1",
        "folds": [
            {k: f.get(k) for k in ("fold", "selected_threshold", "dev_precision", "dev_recall",
                                   "holdout_precision", "holdout_recall", "holdout_truth_count",
                                   "holdout_precision_lower95", "holdout_recall_lower95",
                                   "holdout_event_precision", "holdout_event_recall",
                                   "holdout_event_truth", "holdout_recall_by_type",
                                   "holdout_tp", "holdout_fp", "holdout_fn",
                                   "degradation_precision", "degradation_recall",
                                   "insufficient_truth")}
            for f in split_res["folds"]
        ],
        "summary": s_sum,
    }

    # 6) 变化点：位置 × 幅度网格扫描
    # 单点注入（原做法）只能回答「这一处能否检出」，无法区分「算法准」与「恰好好检」。
    # 改为 5 个位置 × 8 个幅度，同时给出检出率曲线、定位误差分布与零幅度伪报率。
    cp_clean, cp_plan = split_eval.prepare_clean(
        os.path.join(FIX, split_eval.EVAL_DATASET), "weather",
        "分析温度的趋势、季节性与极端事件")
    cp_res = changepoint_scan.scan_change_points(cp_clean, cp_plan)
    res["变化点检出率"] = cp_res["key_delta_detection_rate"]
    p90 = cp_res["error_stats"]["p90"]
    res["变化点定位误差90分位"] = float(p90) if p90 is not None else 999.0
    res["变化点伪报率"] = cp_res["false_positive_rate_at_zero"]
    res["_change_point_scan"] = cp_res

    # 7) 性能基准（读缓存结果）
    # 基准要跑八分钟，不放评估主流程；由 `python -m evals.benchmark` 单独产出。
    # 缓存缺失时记 0 分而不是跳过——「没测过」本身就不该算通过。
    res["_benchmark"] = _load_benchmark()
    scale_1e5 = res["_benchmark"].get("scale_1e5") or {}
    # 缓存缺失时记大值而非 0：0 会被「≤ 上限」判为通过，等于把「没测过」当成达标。
    res["1e5行分析耗时"] = float(scale_1e5.get("total_s") or 9999.0)
    res["1e5行峰值内存"] = float(scale_1e5.get("peak_mb") or 9999.0)

    # 8) LLM 模式评估（读缓存，缺失则不产生指标、不判失败）
    # 改造前所有指标都跑在规则模式下，等于没度量过模型这条路径。
    # 这里把「模型选择的工具是否合理」补进评估，口径为多用例真实调用。
    llm_eval = _load_llm_eval()
    res["_llm_eval"] = llm_eval
    if llm_eval.get("available"):
        s = llm_eval["summary"]
        res["LLM计划合法率"] = float(s.get("valid_rate") or 0.0)
        # 回退率缺失按 1.0 记（最差），不能用 0 冒充「没有回退」
        res["LLM回退率"] = float(s["fallback_rate"]) if s.get("fallback_rate") is not None else 1.0
        # 覆盖率为 None 表示没有任何带期望项的合法用例，记 0 而非 1
        res["LLM分析项覆盖率"] = float(s["mean_tool_recall"]) \
            if s.get("mean_tool_recall") is not None else 0.0

    # 8b) Agent 模式评估（读缓存，缺失或 mock 不产生指标、不判失败）
    agent_eval = _load_agent_eval()
    res["_agent_eval"] = agent_eval
    if agent_eval.get("judged"):
        sa = agent_eval["summary"]
        res["Agent成功率"] = float(sa.get("success_rate") or 0.0)
        if sa.get("mean_need_coverage") is not None:
            res["Agent需求覆盖率"] = float(sa["mean_need_coverage"])
        res["Agent规则计划覆盖率"] = float(sa.get("mean_rule_plan_coverage") or 0.0)
        if sa.get("replayable_rate") is not None:
            res["Agent可重放率"] = float(sa["replayable_rate"])
        res["Agent工具失败率"] = float(sa.get("mean_tool_failure_rate") or 0.0)
        res["Agent平均步数"] = float(sa.get("avg_steps") or 0.0)
        res["Agent越界拒绝总数"] = int(sa.get("guard_violations_total") or 0)

    # 8c) 结论级评估收口（A/B/C）：只报告，不加入 THRESHOLDS。
    # 缓存缺失时标「未运行」，不判失败；阈值须经单独标定后才能进入门禁。
    conclusion = conclusion_cache.load_all()
    res["_conclusion_eval"] = conclusion["conclusion_eval"]
    res["_conclusion_diff"] = conclusion["conclusion_diff"]
    res["_short_repro"] = conclusion["short_repro"]
    res["_short_agent"] = conclusion["short_agent"]
    res["_short_conclusion"] = conclusion["short_conclusion"]
    res.update(conclusion["flattened"])

    # 7) 图表字段映射准确率
    plan_cols = set(wx.data["plan"]["target_columns"])
    bad = [a for a in wx.artifacts
           if a.kind.startswith("figure") and a.fields and not set(a.fields) <= plan_cols]
    res["图表字段映射准确率"] = 1.0 if not bad else round(1 - len(bad) / max(len(wx.artifacts), 1), 4)

    # 8) 重复运行一致率
    st2 = _run("stock_again", "stock", "stock_v1.csv", "计算收益风险指标并做 walk-forward 回测", OUT)
    res["重复运行一致率"] = 1.0 if st.data == st2.data else 0.0

    # 8b) 计划来源可追溯
    # 计划层引入 LLM 后，必须能回答「这次计划是模型给的还是规则给的」。
    # 来源缺失或取值不在白名单，都说明审计链断了。
    valid_sources = {"rule", "llm", "cache", "mock"}
    plan_sources = {
        "weather": (wx.data or {}).get("plan", {}).get("plan_source"),
        "stock": (st.data or {}).get("plan", {}).get("plan_source"),
        "leak": (lk.data or {}).get("plan", {}).get("plan_source") if lk.status == "succeeded" else None,
    }
    known = {k: v for k, v in plan_sources.items() if v is not None}
    res["计划来源可追溯"] = 1.0 if (known and all(v in valid_sources for v in known.values())) else 0.0
    res["_plan_sources"] = plan_sources

    # 9) 未来数据泄漏检出率（必须为失败）
    res["未来数据泄漏检出率"] = 1.0 if (lk.status == "failed"
                                      and lk.error.code == "LOOKAHEAD_VIOLATION") else 0.0

    # 10) 无证据结论数
    res["无证据结论数"] = sum(1 for c in wx.conclusions + st.conclusions if not c.evidence_ids)

    res["_datasets"] = {
        "weather_v1": sha1_of_file(os.path.join(FIX, "weather_v1.csv")),
        "weather_eval_v1": sha1_of_file(os.path.join(FIX, "weather_eval_v1.csv")),
        "stock_v1": sha1_of_file(os.path.join(FIX, "stock_v1.csv")),
        "stock_leak_v1": sha1_of_file(os.path.join(FIX, "stock_leak_v1.csv")),
    }
    res["_status"] = {"weather": wx.status, "stock": st.status, "leak": lk.status}
    return res


def main() -> int:
    res = evaluate()
    # LLM 指标只在缓存存在时纳入判定：缺缓存是「未运行」，不是「未通过」
    thresholds = dict(THRESHOLDS)
    if res.get("_llm_eval", {}).get("available"):
        thresholds.update(LLM_THRESHOLDS)
    if res.get("_agent_eval", {}).get("judged"):
        # 只把本次实际产出的指标纳入判定（未跑重放就不判可重放率）
        thresholds.update({k: v for k, v in AGENT_THRESHOLDS.items() if k in res})
    lines = ["| 指标 | 实测值 | 阈值 | 结论 |", "|---|---|---|---|"]
    failed = 0
    for k, (op, thr) in thresholds.items():
        v = res[k]
        if op == "=":
            ok = (abs(v - thr) < 1e-12) if isinstance(v, float) else (v == thr)
        elif op == ">=":
            ok = v >= thr
        else:
            ok = v <= thr
        failed += 0 if ok else 1
        lines.append(f"| {k} | {v} | {op} {thr} | {'通过' if ok else '未通过'} |")
    out_md = "\n".join(lines)
    print(out_md)
    print()
    if "_split_eval" in res:
        print("=== 评估集划分（三折前向滚动）===")
        print("| 折 | dev 阈值 | dev P/R | holdout P/R | 召回下界95% | holdout 真值数 | 精度退化 |")
        print("|---|---|---|---|---|---|---|")
        for f in res["_split_eval"]["folds"]:
            if f.get("status") != "succeeded":
                print(f"| {f['fold']} | — | — | 运行失败 | — | — | — |")
                continue
            flag = "（样本不足，不参与判定）" if f.get("insufficient_truth") else ""
            print("| %s | %s | %.3f / %.3f | %.3f / %.3f | %.3f | %d%s | %+.3f |" % (
                f["fold"], f["selected_threshold"], f["dev_precision"], f["dev_recall"],
                f["holdout_precision"], f["holdout_recall"],
                f.get("holdout_recall_lower95", 0.0), f["holdout_truth_count"], flag,
                f.get("degradation_precision", 0.0)))
        print()
        print("划分汇总：", json.dumps(res["_split_eval"]["summary"], ensure_ascii=False))
    if "_change_point_scan" in res:
        print()
        print("=== 变化点网格扫描（位置 × 幅度，单位 hPa）===")
        print(changepoint_scan.format_grid(res["_change_point_scan"]))
        cp = res["_change_point_scan"]
        print("按幅度检出率：", json.dumps(cp["detection_rate_by_delta"], ensure_ascii=False))
        print("定位误差（命中格子）：", json.dumps(cp["error_stats"], ensure_ascii=False))
        print("零幅度伪报率：", cp["false_positive_rate_at_zero"])
    if "_benchmark" in res:
        print()
        print("=== 性能基准（缓存）===")
        b = res["_benchmark"]
        if not b.get("available"):
            print("不可用：", b.get("reason"))
        else:
            print("生成时间：", b.get("generated_at"), " 环境：", b.get("env"))
            print("| 规模 | 耗时(s) | 峰值内存(MB) | 状态 |")
            print("| --- | --- | --- | --- |")
            for label in ("1e4", "1e5"):
                s = b.get(f"scale_{label}") or {}
                print("| %s | %s | %s | %s |" % (label, s.get("total_s"), s.get("peak_mb"), s.get("status")))
            print("1e6 行状态：", b.get("scale_1e6_status"))
    if "_agent_eval" in res:
        a = res["_agent_eval"]
        print()
        print("=== Agent 模式评估（缓存）===")
        if not a.get("available"):
            print("不可用：", a.get("reason"))
        elif not a.get("judged"):
            print("来源为 %s，仅验证指标管线，不纳入判定" % a.get("source"))
        else:
            sa = a["summary"]
            print("模型：%s  计划层：%s  用例：%d" % (
                a.get("model"), a.get("planner"), sa.get("n_runs")))
            print("| 用例 | 步数 | 终止原因 | 需求覆盖 | 规则计划覆盖 | 工具失败 | 越界 | 重放 |")
            print("|---|---|---|---|---|---|---|---|")
            for c in a.get("cases", []):
                need = "—" if c.get("need_coverage") is None else "%.2f" % c["need_coverage"]
                rep = "—" if c.get("replay_ok") is None else ("一致" if c["replay_ok"] else "不一致")
                print("| %s | %s | %s | %s | %.2f | %s | %s | %s |" % (
                    c.get("case"), c.get("steps"), c.get("terminate_reason"), need,
                    c.get("rule_plan_coverage") or 0.0,
                    "—" if c.get("tool_failure_rate") is None
                    else "%.2f" % c["tool_failure_rate"],
                    c.get("guard_violations"), rep))
            print("汇总：平均步数 %s，需求覆盖率 %s，规则计划覆盖率 %s，可重放率 %s，"
                  "越界 %s 次，token %s" % (
                      sa.get("avg_steps"), sa.get("mean_need_coverage"),
                      sa.get("mean_rule_plan_coverage"), sa.get("replayable_rate"),
                      sa.get("guard_violations_total"), sa.get("tokens_total")))
    if "_conclusion_eval" in res:
        ce = res["_conclusion_eval"]
        print()
        print("=== 结论级评估（A，缓存；只报告，不纳入通过/失败）===")
        if not ce.get("available"):
            print("未运行：", ce.get("reason"))
        else:
            print("结果目录：%s  生成：%s" % (ce.get("result_root"), ce.get("generated_at")))
            print("| 模式 | 异常点 P/R/F1 | 异常事件 P/R/F1 | 变化点 P/R/F1 | 趋势 | 季节 | 结论需求覆盖 |")
            print("|---|---|---|---|---|---|---|")
            for mode in ("pipeline", "agent"):
                m = ce.get(mode) or {}
                print("| %s | %s | %s | %s | %s | %s | %s |" % (
                    mode, _fmt_prf(m.get("anomaly_point")), _fmt_prf(m.get("anomaly_event")),
                    _fmt_prf(m.get("changepoint")),
                    "—" if (m.get("trend") or {}).get("accuracy") is None
                    else "%.2f" % m["trend"]["accuracy"],
                    "—" if (m.get("seasonal") or {}).get("accuracy") is None
                    else "%.2f" % m["seasonal"]["accuracy"],
                    "—" if (m.get("conclusion_need") or {}).get("coverage") is None
                    else "%.2f" % m["conclusion_need"]["coverage"]))
            av = ce.get("agent_vs_pipeline") or {}
            if av.get("available") is not False and av.get("anomaly"):
                a = av["anomaly"]
                print("异常实例：pipeline TP=%s，agent TP=%s；agent 未执行 %s 个用例，"
                      "少看 %s 个真值实例；去重后漏 %s 个真值事件。" % (
                          a.get("pipeline_tp_instances"), a.get("agent_tp_instances"),
                          a.get("agent_not_run_runs"), a.get("missed_instances_due_to_not_run"),
                          a.get("agent_distinct_missed")))
        cd = res.get("_conclusion_diff") or {}
        print()
        print("=== 结论差异溯源（B，缓存；只报告）===")
        if not cd.get("available"):
            print("未运行：", cd.get("reason"))
        else:
            s2 = cd.get("summary") or {}
            print("pipeline %s / agent %s / 差 %s（平均 %.4f/用例）；无法归因 %s（%.2f%%）" % (
                s2.get("pipeline_total"), s2.get("agent_total"), s2.get("missing_total"),
                s2.get("mean_missing_per_case") or 0.0, s2.get("unattributed"),
                100 * (s2.get("unattributed_rate") or 0.0)))
            print("| 类别 | pipeline | agent | 差值 | 归因 |")
            print("|---|---|---|---|---|")
            for cat, x in (cd.get("category_matrix") or {}).items():
                if x.get("missing"):
                    reasons = []
                    for k in ("tool_not_run", "no_detection", "plan_or_parameter_diff", "unattributed"):
                        if x.get(k):
                            reasons.append("%s %s" % (k, x[k]))
                    print("| %s | %s | %s | %+d | %s |" % (
                        cat, x.get("pipeline"), x.get("agent"),
                        x.get("agent", 0) - x.get("pipeline", 0), "、".join(reasons) or "—"))
        sr = res.get("_short_repro") or {}
        sa = res.get("_short_agent") or {}
        sc = res.get("_short_conclusion") or {}
        print()
        print("=== 第二份数据（C，缓存；只报告）===")
        if not sr.get("available"):
            print("复现检查未运行：", sr.get("reason"))
        else:
            print("pipeline 两次语义一致：%s；差异字段：%s；结论数：%s" % (
                sr.get("identical"), sr.get("diffs"), sr.get("conclusions_count")))
        if not sa.get("available"):
            print("短数据 agent 未运行：", sa.get("reason"))
        elif not sa.get("judged"):
            print("短数据 agent 来源为 %s，仅验证管线，不纳入判定" % sa.get("source"))
        else:
            ss = sa.get("summary") or {}
            print("短数据 agent：成功 %s/%s，需求覆盖 %s，可重放 %s，token %s" % (
                ss.get("n_succeeded"), ss.get("n_runs"),
                ss.get("mean_need_coverage"), ss.get("replayable_rate"),
                ss.get("tokens_total")))
        if not sc.get("available"):
            print("短数据结论级复核未运行：", sc.get("reason"))
        else:
            sm = sc.get("agent") or {}
            print("短数据结论：需求覆盖 %s，异常点 F1 %s，趋势正确率 %s" % (
                (sm.get("conclusion_need") or {}).get("coverage"),
                (sm.get("anomaly_point") or {}).get("f1"),
                (sm.get("trend") or {}).get("accuracy")))

    if "_llm_eval" in res:
        print()
        llm = res["_llm_eval"]
        if not llm.get("available"):
            print("=== LLM 模式评估：未运行 ===")
            print("原因：", llm.get("reason"))
        else:
            print("=== LLM 模式评估（真实调用，读缓存）===")
            print("模型：%s  生成于：%s  调用次数：%s" % (
                llm.get("model"), llm.get("generated_at"), llm.get("n_calls")))
            print("| 用例 | 场景 | 合法 | 选中分析项 | 覆盖率 | 未覆盖 |")
            print("|---|---|---|---|---|---|")
            for c in llm["cases"]:
                cov = "—（探针）" if c.get("tool_recall") is None else "%.2f" % c["tool_recall"]
                print("| %s | %s | %s | %s | %s | %s |" % (
                    c["id"], c["scene"], "是" if c.get("valid") else "否",
                    "、".join(c.get("selected") or []) or "—", cov,
                    "、".join(c.get("missing") or []) or "无"))
            print("汇总：", json.dumps(llm["summary"], ensure_ascii=False))
    print("计划层：", json.dumps(res.get("_planner"), ensure_ascii=False))
    print("数据集版本：", json.dumps(res["_datasets"], ensure_ascii=False))
    print("运行状态：", json.dumps(res["_status"], ensure_ascii=False))
    with open(os.path.join(EVAL_DIR, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n未通过项：{failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
