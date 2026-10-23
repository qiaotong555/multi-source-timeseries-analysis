# -*- coding: utf-8 -*-
"""读取 A/B/C 的结论级缓存，供 `evals.run_evals` 统一收口。

原则与 `_load_llm_eval` / `_load_agent_eval` 一致：
- 缓存缺失或损坏 -> `available=False`，控制台显示「未运行」，不判失败；
- 缓存存在 -> 只报告，不加入 THRESHOLDS，直到阈值经过单独标定；
- 不重新跑引擎、不调用模型，避免把「评估收口」变成另一条昂贵路径。
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(name: str, producer: str) -> dict[str, Any]:
    path = os.path.join(EVAL_DIR, name)
    if not os.path.exists(path):
        return {"available": False,
                "reason": f"{name} 不存在，请先运行 {producer}"}
    try:
        with open(path, encoding="utf-8") as f:
            return {"available": True, "data": json.load(f), "path": path}
    except Exception as exc:                      # noqa: BLE001
        return {"available": False, "reason": f"{name} 解析失败：{exc}"}


def _pick_mode(m: dict[str, Any]) -> dict[str, Any]:
    m = m or {}
    anom = m.get("anomaly") or {}
    cp = m.get("changepoint") or {}
    return {
        "scored_runs": anom.get("scored_runs"),
        "not_run_runs": anom.get("not_run_runs"),
        "truth_per_run": anom.get("truth_per_run"),
        "not_run_truth_instances": anom.get("not_run_truth_instances"),
        "anomaly_point": anom.get("point") or {},
        "anomaly_event": anom.get("event") or {},
        "changepoint_scored_runs": cp.get("scored_runs"),
        "changepoint": cp.get("point") or {},
        "trend": m.get("trend") or {},
        "seasonal": m.get("seasonal") or {},
        "conclusion_need": m.get("conclusion_need") or {},
        "quality": m.get("quality") or {},
    }


def load_conclusion_eval(filename: str = "conclusion_eval_results.json") -> dict[str, Any]:
    got = _load(filename, "python -m evals.conclusion_eval")
    if not got["available"]:
        return got
    d = got["data"]
    s = d.get("summary") or {}
    by = s.get("by_mode") or {}
    if not by:
        return {"available": False, "reason": f"{filename} 缺少 summary.by_mode"}
    return {
        "available": True,
        "path": got["path"],
        "generated_at": d.get("generated_at"),
        "result_root": d.get("result_root"),
        "source": d.get("source"),
        "pipeline": _pick_mode(by.get("pipeline") or {}),
        "agent": _pick_mode(by.get("agent") or {}),
        "agent_vs_pipeline": s.get("agent_vs_pipeline") or {},
        "distinct": s.get("distinct") or {},
        "notes": d.get("notes") or [],
    }


def load_conclusion_diff() -> dict[str, Any]:
    got = _load("conclusion_diff_results.json", "python -m evals.conclusion_diff")
    if not got["available"]:
        return got
    d = got["data"]
    return {
        "available": True,
        "path": got["path"],
        "generated_at": d.get("generated_at"),
        "summary": d.get("summary") or {},
        "category_matrix": d.get("category_matrix") or {},
        "notes": d.get("notes") or [],
    }


def load_short_repro() -> dict[str, Any]:
    got = _load("short_repro_results.json", "python -m evals.repro_check")
    if not got["available"]:
        return got
    d = got["data"]
    return {
        "available": True,
        "path": got["path"],
        "generated_at": d.get("generated_at"),
        "identical": bool(d.get("identical")),
        "diffs": d.get("diffs") or [],
        "conclusions_count": d.get("conclusions_count") or [],
    }


def load_short_agent_eval() -> dict[str, Any]:
    got = _load("agent_eval_short_results.json", "python -m evals.agent_eval --input ...")
    if not got["available"]:
        return got
    d = got["data"]
    s = d.get("summary") or {}
    source = d.get("source") or s.get("source")
    return {
        "available": True,
        "judged": source == "real",
        "source": source,
        "path": got["path"],
        "generated_at": d.get("generated_at"),
        "input_path": d.get("input_path"),
        "model": d.get("model"),
        "summary": s,
    }


def load_short_conclusion_eval() -> dict[str, Any]:
    return load_conclusion_eval("conclusion_eval_short_agent_results.json")


def flatten(conclusion_eval: dict[str, Any], conclusion_diff: dict[str, Any],
            short_repro: dict[str, Any], short_agent: dict[str, Any],
            short_conclusion: dict[str, Any]) -> dict[str, Any]:
    """把关键结论级数字平铺进 eval_results.json；不设阈值、不判通过。"""
    out: dict[str, Any] = {}
    if conclusion_eval.get("available"):
        for mode in ("pipeline", "agent"):
            m = conclusion_eval.get(mode) or {}
            ap = m.get("anomaly_point") or {}
            ae = m.get("anomaly_event") or {}
            cp = m.get("changepoint") or {}
            out[f"结论级异常点P({mode})"] = ap.get("precision")
            out[f"结论级异常点R({mode})"] = ap.get("recall")
            out[f"结论级异常点F1({mode})"] = ap.get("f1")
            out[f"结论级异常事件F1({mode})"] = ae.get("f1")
            out[f"结论级变化点F1({mode})"] = cp.get("f1")
            out[f"结论级趋势正确率({mode})"] = (m.get("trend") or {}).get("accuracy")
            out[f"结论级季节正确率({mode})"] = (m.get("seasonal") or {}).get("accuracy")
            out[f"结论级结论需求覆盖率({mode})"] = (m.get("conclusion_need") or {}).get("coverage")
    if conclusion_diff.get("available"):
        s = conclusion_diff.get("summary") or {}
        out["结论差异总数"] = s.get("missing_total")
        out["结论差异无法归因率"] = s.get("unattributed_rate")
        out["结论差异归因完整"] = bool(s.get("attribution_complete"))
    if short_repro.get("available"):
        out["短数据复现一致"] = 1.0 if short_repro.get("identical") else 0.0
    if short_agent.get("available") and short_agent.get("judged"):
        s = short_agent.get("summary") or {}
        out["短数据Agent成功率"] = s.get("success_rate")
        out["短数据Agent需求覆盖率"] = s.get("mean_need_coverage")
        out["短数据Agent可重放率"] = s.get("replayable_rate")
    if short_conclusion.get("available"):
        m = short_conclusion.get("agent") or {}
        out["短数据结论需求覆盖率(agent)"] = (m.get("conclusion_need") or {}).get("coverage")
        out["短数据异常点F1(agent)"] = (m.get("anomaly_point") or {}).get("f1")
    return out


def load_all() -> dict[str, Any]:
    ce = load_conclusion_eval()
    cd = load_conclusion_diff()
    sr = load_short_repro()
    sa = load_short_agent_eval()
    sc = load_short_conclusion_eval()
    return {"conclusion_eval": ce, "conclusion_diff": cd,
            "short_repro": sr, "short_agent": sa, "short_conclusion": sc,
            "flattened": flatten(ce, cd, sr, sa, sc)}