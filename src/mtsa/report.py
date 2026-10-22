# -*- coding: utf-8 -*-
"""报告生成：把质量、计划、指标、检验、异常、图表与结论汇总为可复现报告。

每条结论必须绑定 evidence；未绑定证据的结论计数返回给调用方用于校验。
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from .models import Artifact, Conclusion, Evidence
from .scenarios.base import ScenarioContext

MAX_TABLE_ROWS = 120


def _fmt(v, nd: int = 6) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e12:
            return f"{v:.0f}"
        return f"{v:.{nd}g}"
    return str(v)


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
         ["分组列", ctx.plan.group_column or "—"]],
    ))
    lines.append("")

    # 2) 数据质量
    lines.append("## 2. 数据质量报告")
    lines.append("")
    issues = getattr(quality, "issues", [])
    if issues:
        rows = [[i.code, i.severity, i.column or "—", str(i.count), i.detail] for i in issues]
        lines.append(_md_table(["问题代码", "级别", "列", "数量", "说明"], rows[:MAX_TABLE_ROWS]))
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
        [["分析项", "、".join(ctx.plan.analyses)],
         ["重采样规则", ctx.plan.resample_rule or "不做重采样"],
         ["关键参数", "、".join(f"{k}={v}" for k, v in ctx.plan.parameters.items()
                                if k not in ("unit_conversions",))],
         ["不支持项", "、".join(ctx.plan.not_supported) or "—"]],
    ))
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
        "--seed", "{dataset_info.get("seed")}",
    ]
    raise SystemExit(main())
'''
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return Artifact(artifact_id="art_script", kind="script", path=path,
                    title="可复现脚本", fields=[])
