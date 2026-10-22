# -*- coding: utf-8 -*-
"""LLM 模式评估：计划合法率、回退率、分析项覆盖率与计划一致率。

为什么单独成脚本而不并入 run_evals
----------------------------------
1. **依赖外部服务**：无密钥、网络不通、额度耗尽都会失败。让它阻塞离线评估，
   等于把「模型服务可用」变成了「整套评估可用」的前提。
2. **有成本**：每次真实调用消耗 token，不能每轮评估都跑。
3. **有非确定性**：结果随模型版本与采样变化，需要单独留档做时间对照。

因此本脚本独立运行，结果写 `evals/llm_eval_results.json`；`run_evals` 只读缓存，
缺失时不判失败，仅记录「未运行」。

判据定义
--------
- **合法**：计划通过三级校验，且 `plan_source` 为 `llm`。`cache` 也算失败——
  评估要检验的是模型本次的输出，命中缓存说明没有新的模型输出。
- **回退**：模型输出被拒（格式 / schema / 语义任一不通过），`plan_source` 为 `rule`。
- **覆盖率**：问题所期望的分析项中被模型选中的比例。
- **一致率**：同一问题两次调用，分析项集合是否相同。

覆盖率的局限必须说清
--------------------
期望项由人工按问题语义给出，衡量的是「模型有没有选中该问题明显需要的那几项」，
**不衡量「多选或少选是否恰当」**——后者无法在没有标准答案的情况下自动判定。
因此覆盖率是一个下界性质的指标：高分不代表计划好，低分一定说明漏了明显该做的事。

另一个必须注意的统计陷阱：**覆盖率只统计合法计划**。回退时选中的是规则计划的
12 项，几乎必然覆盖期望项，把它算进平均会把「模型全挂」粉饰成「覆盖率很高」。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from typing import Any, Optional

from mtsa import io_loaders, profiler, quality
from mtsa.llm_client import client_from_env
from mtsa.planner import PLANNER_VERSION, LLMPlanner
from mtsa.scenarios import stock as _stock          # noqa: F401  适配器靠 import 副作用注册
from mtsa.scenarios import weather as _weather      # noqa: F401
from mtsa.scenarios.base import get_adapter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "fixtures")
DEFAULT_OUT = os.path.join(ROOT, "evals", "llm_eval_results.json")

# 细粒度检验项的上位开关：模型只选 test 时，应视为覆盖了其下的四项
PARENT_TOOL = {
    "stationarity": "test",
    "autocorr": "test",
    "normality": "test",
    "randomness": "test",
}

# 期望项按「问题明显需要什么」给出，不做穷举
CASES: list[dict[str, Any]] = [
    {"id": "wx_trend", "scene": "weather", "csv": "weather_v1.csv",
     "question": "分析温度的趋势、季节性与极端事件",
     "expect": ["trend", "seasonal", "extreme"]},
    {"id": "wx_anomaly", "scene": "weather", "csv": "weather_v1.csv",
     "question": "检查数据中的异常点和突变更点",
     "expect": ["anomaly", "changepoint"]},
    {"id": "wx_corr", "scene": "weather", "csv": "weather_v1.csv",
     "question": "分析温度与湿度之间的相关性以及领先滞后关系",
     "expect": ["corr", "lag"]},
    {"id": "wx_stationary", "scene": "weather", "csv": "weather_v1.csv",
     "question": "评估这条时间序列是否平稳，并说明其自相关结构",
     "expect": ["stationarity"]},
    {"id": "st_trend", "scene": "stock", "csv": "stock_v1.csv",
     "question": "分析股价的趋势与波动特征",
     "expect": ["trend"]},
    {"id": "st_anomaly", "scene": "stock", "csv": "stock_v1.csv",
     "question": "检查价格数据中的异常跳变与均值突变",
     "expect": ["anomaly", "changepoint"]},
    {"id": "st_corr", "scene": "stock", "csv": "stock_v1.csv",
     "question": "分析股价与基准指数的相关性和领先滞后关系",
     "expect": ["corr", "lag"]},
    # 探针用例：不给期望项，只记录模型面对开放问题选了什么。
    # 存在的意义是暴露上面七个用例的偏直白——问题与期望项几乎一一对应，
    # 覆盖率 1.00 更多说明用例简单，而非模型理解力强。探针不参与覆盖率统计。
    {"id": "wx_open", "scene": "weather", "csv": "weather_v1.csv",
     "question": "帮我看看这两年的温度数据有什么值得注意的地方",
     "expect": [], "probe": True},
    {"id": "st_open", "scene": "stock", "csv": "stock_v1.csv",
     "question": "这份股价数据能看出什么问题",
     "expect": [], "probe": True},
]

# 这两个用例额外重复调用一次，用于观察计划一致率
REPEAT_IDS = {"wx_trend", "st_anomaly"}

MIN_VALID_RATE = 0.90
MAX_FALLBACK_RATE = 0.10
MIN_TOOL_RECALL = 0.60


def _hit(selected: set[str], tool: str) -> bool:
    """期望项是否被覆盖；细项允许被上位开关覆盖。"""
    if tool in selected:
        return True
    parent = PARENT_TOOL.get(tool)
    return parent is not None and parent in selected


def tool_recall(selected: list[str], expected: list[str]) -> tuple[Optional[float], list[str]]:
    """返回（覆盖率, 未覆盖项）。无期望项（探针用例）返回 None，不参与统计。"""
    if not expected:
        return None, []
    sel = set(selected or [])
    missing = [e for e in expected if not _hit(sel, e)]
    return round((len(expected) - len(missing)) / len(expected), 4), missing


# ---------------- 单次调用 ----------------

def _context(case: dict, cache: dict) -> tuple[Any, Any, Any, Any]:
    """按 CSV 缓存加载结果：同场景多个用例不必重复导入与摘要。"""
    key = case["csv"]
    if key in cache:
        return cache[key]
    raw = io_loaders.load_dataset(os.path.join(FIX, key))
    profile, parsed_time = profiler.profile_dataset(raw)
    qrep = quality.check_quality(raw, profile, parsed_time)
    adapter = get_adapter(case["scene"])
    cache[key] = (profile, qrep, raw.frame, adapter)
    return cache[key]


def _run_once(client, case: dict, cache: dict) -> dict[str, Any]:
    profile, qrep, frame, adapter = _context(case, cache)
    # cache_dir=None 彻底禁用缓存：评估要的就是本次真实输出
    planner = LLMPlanner(client, cache_dir=None)
    started = time.time()
    plan, source = planner.plan(case["question"], profile, qrep, adapter, frame)
    elapsed = round(time.time() - started, 2)

    selected = list(plan.analyses or [])
    recall, missing = tool_recall(selected, case["expect"])
    return {
        "selected": selected,
        "source": source,
        "valid": source == "llm",
        "error": planner.last_error,
        "tool_recall": recall,
        "missing": missing,
        "elapsed_s": elapsed,
    }


def _run_case(client, case: dict, cache: dict, repeat: bool, verbose: bool) -> dict[str, Any]:
    first = _run_once(client, case, cache)
    row: dict[str, Any] = {
        "id": case["id"], "scene": case["scene"], "csv": case["csv"],
        "question": case["question"], "expect": list(case["expect"]),
        **first,
    }
    if repeat:
        second = _run_once(client, case, cache)
        row["repeat_selected"] = second["selected"]
        row["repeat_source"] = second["source"]
        # 只比较「有序去重后的集合」：顺序无意义，重复项也无意义
        row["consistent"] = sorted(set(first["selected"])) == sorted(set(second["selected"]))
    if verbose:
        flag = "OK " if row["valid"] else "回退"
        cons = "" if "consistent" not in row else (" 一致" if row["consistent"] else " 不一致")
        cov = "探针" if row["tool_recall"] is None else "%.2f" % row["tool_recall"]
        print("  %-14s %s 覆盖率=%-5s 缺=%-12s 用时=%.1fs%s" % (
            row["id"], flag, cov, "、".join(row["missing"]) or "无", row["elapsed_s"], cons))
    return row


# ---------------- 汇总 ----------------

def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总判据。覆盖率只在合法计划上平均，理由见模块文档。"""
    total = len(rows)
    if total == 0:
        return {"passed": False, "reason": "无有效用例", "n_cases": 0}

    valid = [r for r in rows if r.get("valid")]
    valid_rate = round(len(valid) / total, 4)
    fallback_rate = round(1.0 - valid_rate, 4)

    # 关键：覆盖率只取合法计划。回退时选中的是规则计划（12 项），
    # 混进来会把「模型全挂」算成高覆盖率，属统计陷阱。
    recall_pool = [r["tool_recall"] for r in valid if r.get("tool_recall") is not None]
    mean_recall: Optional[float] = round(statistics.mean(recall_pool), 4) if recall_pool else None

    cons_pool = [r["consistent"] for r in rows if "consistent" in r]
    consistency: Optional[float] = round(sum(1 for c in cons_pool if c) / len(cons_pool), 4) \
        if cons_pool else None

    checks = {
        f"合法率>={MIN_VALID_RATE}": valid_rate >= MIN_VALID_RATE,
        f"回退率<={MAX_FALLBACK_RATE}": fallback_rate <= MAX_FALLBACK_RATE,
        f"覆盖率>={MIN_TOOL_RECALL}": (mean_recall is not None and mean_recall >= MIN_TOOL_RECALL),
    }
    return {
        "n_cases": total,
        "n_judged": len([r for r in rows if r.get("tool_recall") is not None]),
        "n_valid": len(valid),
        "valid_rate": valid_rate,
        "fallback_rate": fallback_rate,
        "mean_tool_recall": mean_recall,
        "plan_consistency": consistency,
        "consistency_samples": len(cons_pool),
        "checks": checks,
        "passed": all(checks.values()),
        "scope_note": "覆盖率仅统计合法且带期望项的用例；一致率仅对有重复调用的用例计算",
    }


# ---------------- 入口 ----------------

def run_llm_eval(cases: Optional[list[dict]] = None,
                 repeat_ids: Optional[set[str]] = None,
                 client=None,
                 out_path: Optional[str] = None,
                 verbose: bool = True) -> dict[str, Any]:
    cases = cases if cases is not None else CASES
    repeat_ids = REPEAT_IDS if repeat_ids is None else repeat_ids
    client = client if client is not None else client_from_env()
    if client is None:
        raise RuntimeError(
            "未配置 LLM：需要环境变量 MTSA_LLM_BASE_URL 与 MTSA_LLM_API_KEY")

    cache: dict = {}
    rows = [_run_case(client, c, cache, repeat=(c["id"] in repeat_ids), verbose=verbose)
            for c in cases]
    summary = summarize(rows)

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": getattr(client, "model", None),
        "base_url": getattr(client, "base_url", None),
        "planner_version": PLANNER_VERSION,
        "n_calls": len(rows) + len([r for r in rows if "consistent" in r]),
        "cases": rows,
        "summary": summary,
    }
    out_path = out_path or DEFAULT_OUT
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def format_table(result: dict[str, Any]) -> str:
    lines = ["| 用例 | 场景 | 合法 | 选中分析项 | 覆盖率 | 未覆盖 | 用时 |",
             "|---|---|---|---|---|---|---|"]
    for r in result["cases"]:
        cons = "" if "consistent" not in r else ("（重复一致）" if r["consistent"] else "（重复不一致）")
        cov = "—（探针）" if r.get("tool_recall") is None else "%.2f" % r["tool_recall"]
        lines.append("| %s | %s | %s | %s | %s | %s | %.1fs%s |" % (
            r["id"], r["scene"], "是" if r["valid"] else "否",
            "、".join(r["selected"]) or "—", cov,
            "、".join(r["missing"]) or "无", r["elapsed_s"], cons))
    s = result["summary"]
    lines.append("")
    lines.append("汇总：合法率 %.2f，回退率 %.2f，覆盖率 %s，一致率 %s" % (
        s["valid_rate"], s["fallback_rate"],
        "—" if s["mean_tool_recall"] is None else "%.2f" % s["mean_tool_recall"],
        "—" if s["plan_consistency"] is None else "%.2f" % s["plan_consistency"]))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="LLM 计划层评估（真实调用，禁用缓存）")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--no-repeat", action="store_true", help="不做重复调用，跳过一致率")
    ap.add_argument("--dry", action="store_true", help="只列出用例，不调用模型")
    args = ap.parse_args(argv)

    if args.dry:
        print("共 %d 个用例：" % len(CASES))
        for c in CASES:
            print("  %-14s %-8s %s  期望=%s" % (
                c["id"], c["scene"], c["question"], "、".join(c["expect"])))
        return 0

    print("=== LLM 计划层评估 ===")
    result = run_llm_eval(repeat_ids=set() if args.no_repeat else None,
                          out_path=args.out, verbose=True)
    print()
    print(format_table(result))
    print()
    for k, v in result["summary"]["checks"].items():
        print("  %-18s %s" % (k, "通过" if v else "未通过"))
    print()
    print("结果已写入：", args.out)
    return 0 if result["summary"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
