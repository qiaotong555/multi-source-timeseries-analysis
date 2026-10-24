# -*- coding: utf-8 -*-
"""报告生成：把质量、计划、指标、检验、异常、图表与结论汇总为可复现报告。

每条结论必须绑定 evidence；未绑定证据的结论计数返回给调用方用于校验。
"""
from __future__ import annotations

import math
import os
from typing import Optional

import pandas as pd

from .models import Artifact, Conclusion, Evidence
from .scenarios.base import ScenarioContext

MAX_TABLE_ROWS = 120
# 质量表逐段列出区间（最多与 runs 的上限一致）；结论里只取前 3 段。
MAX_RUNS_IN_TABLE = 20


def _fmt(v, nd: int = 6) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        # Agent 工具会跑完整检验集，个别列样本不足时统计量为 NaN/inf。
        # 这里必须显式落成「—」，否则 int(nan) 会让整张报告在收尾层崩掉。
        if not math.isfinite(v):
            return "—"
        if v == int(v) and abs(v) < 1e12:
            return f"{v:.0f}"
        return f"{v:.{nd}g}"
    return str(v)


_STATE_CN = {"ok": "已执行", "skipped": "跳过", "failed": "失败", "not_run": "未执行"}


def _budget_label(budget: dict) -> str:
    """报告里的 token 预算一行：几种状态必须能一眼区分。"""
    if not budget or not budget.get("enforced"):
        used = budget.get("used")
        return "未设限" + (f"（已用 {used}）" if used else "")
    limit = budget.get("limit")
    if budget.get("unavailable"):
        return f"上限 {limit}；服务端未返回 usage，未生效"
    text = f"{budget.get('used') or 0} / {limit}"
    if not budget.get("exhausted"):
        return text
    pre = budget.get("precheck") or {}
    if pre.get("blocked"):
        return text + f"（调用前拦截：下次约需 {pre.get('reserve')}）"
    return text + "（已用尽）"


def _cost_label(cost: dict) -> str:
    """报告里的成本一行。金额一律标「估算」，并写出 token 分项依据。"""
    total = cost.get("total")
    if total is None:
        return "—"
    tokens = cost.get("tokens") or {}
    return (f"{cost.get('currency') or ''} {total:.6f}（估算；输入 {tokens.get('prompt', 0)} / "
            f"缓存 {tokens.get('cached', 0)} / 输出 {tokens.get('completion', 0)} token）")


def _issue_range_cell(issue: object) -> str:
    """质量表「区间」列：有 runs 时给前几段，否则退回 start~end。"""
    runs = getattr(issue, "runs", None) or []
    if runs:
        parts = []
        for r in runs[:MAX_RUNS_IN_TABLE]:
            a = r.get("start") or "—"
            b = r.get("end") or "—"
            who = f" [分组 {r['group']}]" if r.get("group") else ""
            body = a if a == b else f"{a}~{b}"
            parts.append(f"{body}（{r.get('count')} 点）{who}")
        total = getattr(issue, "run_count", 0) or len(runs)
        if total > len(runs[:MAX_RUNS_IN_TABLE]):
            parts.append(f"共 {total} 段")
        return "；".join(parts)
    start = getattr(issue, "start", None)
    if start:
        end = getattr(issue, "end", None)
        return f"{start}~{end}" if end and end != start else str(start)
    return "—"


def _extreme_longest(item: dict) -> str:
    runs = item.get("runs") or []
    if not runs:
        return "—"
    r = max(runs, key=lambda x: x.get("count") or 0)
    a, b = r.get("start"), r.get("end")
    body = a if a == b else f"{a}~{b}"
    return f"{body}（{r.get('count')} 点）"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def render_report(
    ctx: ScenarioContext,
    artifacts: list[Artifact],
    conclusions: list[Conclusion],
    evidences: list[Evidence],
    quality: object,
    profile: object,
    dataset_info: dict,
    outdir: str,
    warnings: Optional[list[str]] = None,
    agent: Optional[dict] = None,
) -> Artifact:
    os.makedirs(outdir, exist_ok=True)
    lines: list[str] = []
    frame = ctx.clean.frame

    lines.append("# 多源时序数据量化分析报告")
    lines.append("")
    lines.append(f"- `task_id`: `{dataset_info.get('task_id')}`")
    lines.append(f"- `run_id`: `{dataset_info.get('run_id')}`")
    lines.append(f"- 场景：`{ctx.plan.scene_id}`（置信度 {ctx.plan.scene_confidence:.3f}）")
    lines.append(f"- 分析问题：{ctx.plan.question}")
    lines.append(f"- 生成时间：{dataset_info.get('created_at')}")
    lines.append("")

    # 1) 数据集
    lines.append("## 1. 数据集")
    lines.append("")
    lines.append(_md_table(
        ["项目", "取值"],
        [["输入路径", dataset_info.get("input_path", "—")],
         ["文件 SHA1", dataset_info.get("file_hash", "—")],
         ["原始行数", str(dataset_info.get("row_count", "—"))],
         ["清洗后行数", str(len(frame))],
         ["时间范围", f"{frame.index.min()} ~ {frame.index.max()}" if len(frame) else "—"],
         ["目标列", "、".join(ctx.plan.target_columns)],
         ["分组列", ctx.plan.group_column or "—"],
         ["运行模式", dataset_info.get("mode", "pipeline")]],
    ))
    lines.append("")

    # 2) 数据质量
    lines.append("## 2. 数据质量报告")
    lines.append("")
    issues = getattr(quality, "issues", [])
    if issues:
        rows = [[i.code, i.severity, i.column or "—", i.group or "—",
                 str(i.count), _issue_range_cell(i), i.detail] for i in issues]
        lines.append(_md_table(["问题代码", "级别", "列", "分组", "数量", "区间", "说明"],
                               rows[:MAX_TABLE_ROWS]))
        if len(rows) > MAX_TABLE_ROWS:
            lines.append("")
            lines.append(f"> 仅列出前 {MAX_TABLE_ROWS} 条，完整清单见 `result.json`。")
    else:
        lines.append("未检出数据质量问题。")
    lines.append("")

    # 3) 分析计划
    lines.append("## 3. 分析计划")
    lines.append("")
    lines.append(_md_table(
        ["项目", "取值"],
        [["计划分析项（基底）" if agent else "分析项", "、".join(ctx.plan.analyses)],
         ["重采样规则", ctx.plan.resample_rule or "不做重采样"],
         ["关键参数", "、".join(f"{k}={v}" for k, v in ctx.plan.parameters.items()
                                if k not in ("unit_conversions",))],
         ["不支持项", "、".join(ctx.plan.not_supported) or "—"]],
    ))
    lines.append("")

    # 3.1) Agent 执行轨迹（仅 agent 模式）
    if agent:
        executed = list(agent.get("tools_run") or [])
        allowed = list(agent.get("allowed_tools") or [])
        states = agent.get("tool_states") or {}
        not_run = [x for x in allowed if states.get(x, "not_run") == "not_run"]
        failed = [x for x in allowed if states.get(x) == "failed"]
        skipped = [x for x in allowed if states.get(x) == "skipped"]
        cov = agent.get("coverage") or {}
        scope = agent.get("scope") or {}
        lines.append("### 3.1 Agent 执行轨迹")
        lines.append("")
        lines.append(_md_table(
            ["项目", "取值"],
            [["运行模式", "agent"],
             ["模型", agent.get("model") or "—"],
             ["终止原因", agent.get("terminate_reason") or "—"],
             ["范围策略", agent.get("scope_mode") or "—"],
             ["范围下限", (("、".join(scope.get("tools") or []) or "—")
                          + ("（已覆盖）" if scope.get("met")
                             else "（未覆盖：" + "、".join(scope.get("missing") or []) + "）"))
                         if scope.get("enforced") else "—（free：范围由模型决定）"],
             ["下限外追加步数", (f"{scope.get('extra_used', 0)}/{scope.get('max_extra_steps', 0)}"
                               if scope.get("enforced") else "—")],
             ["token 预算", _budget_label(agent.get("budget") or {})],
             *([["成本（估算）", _cost_label(agent.get("cost") or {})]]
               if agent.get("cost") else []),
             ["步数", str(agent.get("step_count", 0))],
             ["已执行工具", "、".join(executed) or "—"],
             ["未执行工具", "、".join(not_run) or "—"],
             ["失败工具", "、".join(failed) or "—"],
             ["跳过工具", "、".join(skipped) or "—"],
             ["工具覆盖率", f"{cov.get('tool_coverage', 0):.2f}（{len(cov.get('executed_tools') or [])}/{len(allowed)}）" if cov else "—"],
             ["计划分析项覆盖率", f"{cov.get('analysis_coverage', 0):.2f}（{len(cov.get('covered_analyses') or [])}/{len(cov.get('planned_analyses') or [])}）" if cov else "—"],
             ["被拒动作", str(len(agent.get("rejects") or []))]],
        ))
        if agent.get("cost"):
            lines.append("")
            lines.append("> 金额按给定价目表与 token 用量估算，**不等于供应商账单**；"
                         "它只用于成本展示与预算提示，不参与任何统计计算或结论。")
        lines.append("")
        cov_tools = cov.get("tools") or []
        if cov_tools:
            lines.append("**覆盖矩阵**：每个工具的状态、它负责的分析项、以及未执行的原因。")
            lines.append("")
            lines.append(_md_table(
                ["工具", "状态", "覆盖分析项", "原因"],
                [[t.get("tool", ""), _STATE_CN.get(t.get("state", ""), t.get("state", "")),
                  "、".join(t.get("covers") or []) or "—",
                  t.get("reason") or "—"] for t in cov_tools],
            ))
            lines.append("")
        steps = list(agent.get("steps") or [])
        if steps:
            rows = []
            for s in steps:
                reason = str(s.get("reason") or "")
                if len(reason) > 80:
                    reason = reason[:80] + "…"
                rows.append([str(int(s.get("index", 0)) + 1), str(s.get("action")),
                             str(s.get("outcome")), f"{s.get('elapsed_ms', 0)} ms", reason])
            lines.append(_md_table(["步序", "工具", "结果", "耗时", "理由"], rows))
            lines.append("")
        lines.append("> Agent 只负责选择工具；数值由工具内的确定性代码计算。"
                     "未执行的分析在报告与 `result.json` 中保留为空，不能解读为「未发现问题」。")
        lines.append("")

    # 4) 清洗过程
    lines.append("## 4. 清洗过程")
    lines.append("")
    ops = ctx.clean.operations
    if ops:
        lines.append(_md_table(["步骤", "操作", "影响"],
                               [[str(i + 1), o.get("op"),
                                 ", ".join(f"{k}={v}" for k, v in o.items() if k != "op")]
                                for i, o in enumerate(ops)]))
    else:
        lines.append("未执行清洗操作。")
    lines.append("")
    lines.append(f"- 丢弃行数：{ctx.clean.dropped_rows}")
    lines.append(f"- 插补单元格：{ctx.clean.imputed_cells}")
    lines.append("")

    # 5) 量化指标
    lines.append("## 5. 量化指标表")
    lines.append("")
    all_rows = list(ctx.metrics.rows) + list(ctx.extra.rows)
    if all_rows:
        rows = [[r.metric_id, r.name, r.column, r.group or "—", _fmt(r.value), r.unit or "—",
                 r.window or "—", r.method] for r in all_rows]
        lines.append(_md_table(["指标 ID", "名称", "列", "分组", "取值", "单位", "窗口", "方法"],
                               rows[:MAX_TABLE_ROWS]))
        if len(rows) > MAX_TABLE_ROWS:
            lines.append("")
            lines.append(f"> 共 {len(rows)} 行，此处列出前 {MAX_TABLE_ROWS} 行，完整表见 `result.json`。")
    else:
        lines.append("未产出指标。")
    lines.append("")

    # 6) 频域
    if ctx.spectrum:
        lines.append("## 6. 频域分析")
        lines.append("")
        lines.append(_md_table(
            ["列", "采样率(Hz)", "主频(Hz)", "主周期", "低频占比", "中频占比", "高频占比", "方法"],
            [[s.column, _fmt(s.sampling_rate_hz), _fmt(s.dominant_freq), s.dominant_period or "—",
              f"{s.band_power.get('低频', 0):.2%}", f"{s.band_power.get('中频', 0):.2%}",
              f"{s.band_power.get('高频', 0):.2%}", s.method] for s in ctx.spectrum],
        ))
        lines.append("")

    # 7) 统计检验
    if ctx.tests:
        lines.append("## 7. 统计检验结果")
        lines.append("")
        lines.append(_md_table(
            ["检验", "列", "统计量", "p 值", "临界值", "拒绝原假设", "结论", "假设前提"],
            [[t.name, t.column, _fmt(t.statistic), _fmt(t.p_value), _fmt(t.critical_value),
              "是" if t.reject_null else ("否" if t.reject_null is False else "—"),
              t.conclusion, t.assumptions] for t in ctx.tests],
        ))
        lines.append("")

    # 8) 异常与变化点
    lines.append("## 8. 异常点与变化点")
    lines.append("")
    if ctx.anomalies:
        lines.append(f"### 异常点（共 {len(ctx.anomalies)} 个）")
        lines.append("")
        lines.append(_md_table(
            ["时间", "序号", "列", "取值", "稳健 Z", "方法"],
            [[a.timestamp, str(a.index), a.column, _fmt(a.value), _fmt(a.score), a.method]
             for a in ctx.anomalies[:MAX_TABLE_ROWS]],
        ))
    else:
        lines.append("异常点：未检出。")
    lines.append("")
    if ctx.change_points:
        lines.append(f"### 变化点（共 {len(ctx.change_points)} 个）")
        lines.append("")
        lines.append(_md_table(
            ["时间", "序号", "列", "类型", "统计量", "前中位数", "后中位数", "置信度"],
            [[c.timestamp, str(c.index), c.column, c.kind, _fmt(c.statistic),
              _fmt(c.before), _fmt(c.after), c.confidence] for c in ctx.change_points],
        ))
    else:
        lines.append("变化点：未检出。")
    lines.append("")

    # 8b) 绝对极端
    exts = getattr(ctx, "extremes", None) or []
    lines.append("## 8b. 极端值（绝对阈值）")
    lines.append("")
    if exts:
        rows = [[e.get("column"), "高" if e.get("tail") == "high" else "低",
                 f"p{e.get('quantile')}", _fmt(e.get("threshold")),
                 str(e.get("count")), str(e.get("run_count")), _extreme_longest(e)]
                for e in exts[:MAX_TABLE_ROWS]]
        lines.append(_md_table(["列", "方向", "分位", "阈值", "点数", "段数", "最长段"], rows))
        if len(exts) > MAX_TABLE_ROWS:
            lines.append("")
            lines.append(f"> 仅列出前 {MAX_TABLE_ROWS} 项，完整清单见 `result.json` 的 `data.extremes`。")
        lines.append("")
        lines.append("> 极端值按「本列经验分位」判定，回答「有没有超过自身分布的两端」；"
                     "与第 8 节的统计异常不是同一判据，不能互相替代。")
    else:
        lines.append("极端值：未检出（样本不足或列无变化）。")
    lines.append("")

    # 8c) 预测（仅 --forecast 时存在）
    fc = getattr(ctx, "forecast", None)
    if fc:
        fc_cols = list(fc.get("columns") or [])
        ok_cols = [c for c in fc_cols if c.get("status") == "ok"]
        used = {}
        for c in ok_cols:
            used[c.get("selected")] = used.get(c.get("selected"), 0) + 1
        beats = [c for c in ok_cols if c.get("beats_baseline")]
        covs = [c.get("interval_coverage") for c in ok_cols
                if c.get("interval_coverage") is not None]
        lines.append("## 8c. 预测（回测口径）")
        lines.append("")
        lines.append(_md_table(
            ["项目", "取值"],
            [["预测步数（horizon）", f"{fc.get('horizon')} 步"],
             ["季节周期", f"{fc.get('period')}"],
             ["区间水平", f"{float(fc.get('interval_level') or 0):.0%}"],
             ["回测设置", f"最后 30% 内等距 {fc.get('origins')} 个 origin；"
                         f"区间标定取最近 {fc.get('calib_window')} 个"],
             ["基线方法", "seasonal_naive（候选须把 MASE 压低 ≥5% 才替换）"],
             ["选中方法分布", "、".join(f"{k} {v} 列" for k, v in sorted(used.items())) or "—"],
             ["优于基线的列", f"{len(beats)}/{len(ok_cols)}" if ok_cols else "—"],
             ["区间覆盖率区间", (f"{min(covs):.1%} ~ {max(covs):.1%}" if covs else "—")],
             ["未预测的列", "、".join(c.get("column") for c in fc_cols
                                  if c.get("status") != "ok") or "—"]],
        ))
        lines.append("")
        rows = []
        for c in fc_cols:
            if c.get("status") != "ok":
                rows.append([c.get("column"), "未预测", "—", "—", "—", "—",
                             c.get("note") or "样本不足"])
                continue
            rows.append([c.get("column"), c.get("selected"),
                         _fmt(c.get("selected_mase"), 4), _fmt(c.get("baseline_mase"), 4),
                         "是" if c.get("beats_baseline") else "否",
                         _fmt(c.get("interval_coverage"), 4), c.get("selected_reason") or "—"])
        if rows:
            lines.append(_md_table(
                ["列", "选中方法", "MASE", "基线 MASE", "优于基线", "区间覆盖率", "选型说明"],
                rows[:MAX_TABLE_ROWS]))
            lines.append("")
        for c in ok_cols[:3]:
            pts = list(c.get("forecast") or [])
            lines.append(f"### {c['column']}：未来 {len(pts)} 步")
            lines.append("")
            shown = pts[:8] + ([pts[-1]] if len(pts) > 8 else [])
            lines.append(_md_table(
                ["时间", "预测值", "90% 下界", "90% 上界"],
                [[p.get("timestamp") or "—", _fmt(p.get("value"), 4),
                  _fmt(p.get("lo"), 4), _fmt(p.get("hi"), 4)] for p in shown]))
            lines.append("")
            if len(pts) > 8:
                lines.append(f"> 另有 {len(pts) - 9} 步未列出，完整未来点见 `result.json` 的 "
                             f"`data.forecast.columns[].forecast`。")
                lines.append("")
        lines.append("> 预测是回测口径下的条件推断，**不是事实**。假设是「未来 p 阶季节形态与"
                     "回测窗口内一致」；horizon 越长误差越大，90% 区间由历史回测误差标定，"
                     "覆盖率见上表。与第 8 节的异常、第 8b 节的绝对极端不是同一口径。")
        lines.append("")

    # 9) 结论
    lines.append("## 9. 结论与限制")
    lines.append("")
    ev_map = {e.evidence_id: e for e in evidences}
    for c in conclusions:
        evs = [ev_map[i] for i in c.evidence_ids if i in ev_map]
        metric_ids = sorted({m for e in evs for m in e.metric_ids})
        art_ids = sorted({a for e in evs for a in e.artifact_ids})
        lines.append(f"**[{c.category}]** {c.text}")
        lines.append("")
        lines.append(f"  - 证据指标：{', '.join(metric_ids) if metric_ids else '（无）'}")
        lines.append(f"  - 证据产物：{', '.join(art_ids) if art_ids else '（无）'}")
        lines.append(f"  - 计算位置：{', '.join(sorted({e.code_ref for e in evs})) or '（无）'}")
        if c.limitations:
            lines.append(f"  - 限制：{c.limitations}")
        lines.append("")

    # 10) 图表与产物
    lines.append("## 10. 图表与产物清单")
    lines.append("")
    lines.append(_md_table(
        ["产物 ID", "类型", "标题", "相对路径"],
        [[a.artifact_id, a.kind, a.title, os.path.basename(a.path)] for a in artifacts],
    ))
    lines.append("")

    if warnings:
        lines.append("## 11. 运行告警")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## 12. 复现方式")
    lines.append("")
    lines.append("```bash")
    lines.append(f"python -m mtsa.cli run --scene {ctx.plan.scene_id} \\")
    lines.append(f"    --input {dataset_info.get('input_path')} \\")
    lines.append(f"    --question \"{ctx.plan.question}\" --seed {dataset_info.get('seed')}")
    lines.append("```")
    lines.append("")

    path = os.path.join(outdir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return Artifact(artifact_id="art_report_md", kind="report_md", path=path,
                    title="分析报告（Markdown）", fields=["all"])


def write_clean_csv(ctx: ScenarioContext, outdir: str) -> Artifact:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "clean_data.csv")
    ctx.clean.frame.to_csv(path, encoding="utf-8-sig")
    return Artifact(artifact_id="art_clean_csv", kind="table_csv", path=path,
                    title="清洗后数据", fields=list(ctx.clean.frame.columns))


def write_repro_script(dataset_info: dict, outdir: str) -> Artifact:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "reproduce.py")
    # 开启预测时把参数写进脚本，否则「复现」复现出来的是一份没有预测的报告
    forecast_args = ""
    if dataset_info.get("forecast"):
        forecast_args = ('\n        "--forecast",\n'
                         '        "--forecast-horizon", "%s",' % dataset_info.get("forecast_horizon"))
    content = f'''# -*- coding: utf-8 -*-
"""复现脚本：与本次运行完全相同的参数。"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from mtsa.cli import main

if __name__ == "__main__":
    sys.argv = [
        "mtsa.cli", "run",
        "--scene", "{dataset_info.get("scene_id")}",
        "--input", r"{dataset_info.get("input_path")}",
        "--question", "{dataset_info.get("question")}",
        "--seed", "{dataset_info.get("seed")}",{forecast_args}
    ]
    raise SystemExit(main())
'''
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return Artifact(artifact_id="art_script", kind="script", path=path,
                    title="可复现脚本", fields=[])
