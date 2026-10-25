# -*- coding: utf-8 -*-
"""结果校验与「依校验结果改参重跑」（审计 B2）。

三条可判定检查，分级明确：

| kind | 严重度 | 能不能自动修 |
| --- | --- | --- |
| `no_evidence` | error | 不能：删结论或补证据都属于改口径 |
| `figure_field_missing` | error | 能：从 spec 里摘掉数据中不存在的列 |
| `conclusion_number_unmatched` | advisory | 不能：只提示，不当作失败 |

为什么第三条只是 advisory：结论里的数字大多由适配器现算现格式化（差值、占比、周期倒数），
`result.json` 里并没有同名的记录值。实测 weather 51 条结论里有 35 个小数对不上任何
已落盘的数值——如果按「对不上就失败」处理，正常运行会被判死。要把它变成硬判定，
需要让每条结论在生成时登记「这个数字来自哪个指标」，那是适配器层面的改造，单独立项。

因此自动重跑只覆盖可修的一类：图表 spec 引用了不存在的列（渲染必然失败或画空图）。
重跑次数上限用 `MAX_RERUN`，并带「修不动就停」的空转保护。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

# 只认带小数点（或科学计数法）的字面量：整数多是计数、周期、序号，不是度量值。
NUMBER_RE = re.compile(r"-?\d+\.\d+(?:e-?\d+)?")


@dataclass(frozen=True)
class VerifyIssue:
    kind: str
    severity: str                 # error | advisory
    detail: str
    repair: Optional[str] = None  # figure_spec；None 表示不可自动修复
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "severity": self.severity,
                "detail": self.detail, "repair": self.repair,
                "payload": self.payload}


# ---------------- 图表 spec ----------------

def specs_referencing_missing_columns(specs: Iterable[dict], columns: Iterable[str]) -> list[dict]:
    present = {str(c) for c in columns}
    out = []
    for spec in specs:
        refs = [str(c) for c in (spec.get("columns") or [])]
        missing = [c for c in refs if c not in present]
        if missing:
            out.append({"spec_id": str(spec.get("id")), "missing": missing})
    return out


def repair_figure_specs(specs: list[dict], columns: Iterable[str]) -> tuple[list[dict], dict]:
    """摘掉数据里不存在的列；整张图都不剩列时连同这张图一起删掉。

    只动 `columns`，不动 `kind` / `groupby` / 标题——参数调整的范围就这么大。
    """
    present = {str(c) for c in columns}
    fixed: list[dict] = []
    dropped_columns: dict[str, list[str]] = {}
    dropped_specs: list[str] = []
    for spec in specs:
        refs = [str(c) for c in (spec.get("columns") or [])]
        if not refs:
            fixed.append(spec)
            continue
        keep = [c for c in refs if c in present]
        if not keep:
            dropped_specs.append(str(spec.get("id")))
            continue
        if keep != refs:
            dropped_columns[str(spec.get("id"))] = [c for c in refs if c not in present]
            spec = {**spec, "columns": keep}
        fixed.append(spec)
    changed = bool(dropped_columns or dropped_specs)
    return fixed, {"changed": changed, "dropped_columns": dropped_columns,
                   "dropped_specs": dropped_specs}


def figure_problems(specs: Iterable[dict], artifacts: Iterable[Any],
                    columns: Iterable[str]) -> list[VerifyIssue]:
    """两条：引用了不存在的列；或 spec 没产出任何图表产物（多为渲染失败）。"""
    missing = specs_referencing_missing_columns(specs, columns)
    built = " ".join(getattr(a, "path", "") or "" for a in artifacts)
    unbuilt = [str(s.get("id")) for s in specs
               if s.get("columns") and f"_{s.get('id')}" not in built]
    issues: list[VerifyIssue] = []
    if missing:
        issues.append(VerifyIssue(
            kind="figure_field_missing", severity="error", repair="figure_spec",
            detail="图表 spec 引用了数据中不存在的列：" +
                   "；".join(f"{m['spec_id']} 缺 {m['missing']}" for m in missing),
            payload={"missing": missing}))
    if unbuilt:
        issues.append(VerifyIssue(
            kind="figure_not_rendered", severity="error", repair="figure_spec",
            detail=f"以下图表没有产出任何文件：{unbuilt}",
            payload={"spec_ids": unbuilt}))
    return issues


# ---------------- 结论 ----------------

def check_no_evidence(conclusions: Iterable[Any]) -> list[VerifyIssue]:
    bad = [c.conclusion_id for c in conclusions if not getattr(c, "evidence_ids", None)]
    if not bad:
        return []
    return [VerifyIssue(kind="no_evidence", severity="error",
                        detail=f"存在未绑定证据的结论：{bad}", payload={"ids": bad})]


def numeric_pool(objects: Iterable[Any]) -> set[float]:
    """把已落盘的结构化数值收进池子，供结论文字比对。"""
    pool: set[float] = set()

    def walk(node: Any) -> None:
        if isinstance(node, bool) or node is None:
            return
        if isinstance(node, (int, float)):
            pool.add(float(node))
            return
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
            return
        if isinstance(node, (list, tuple)):
            for v in node:
                walk(v)
            return
        dump = getattr(node, "model_dump", None)
        if callable(dump):
            walk(dump(mode="json"))

    for obj in objects:
        walk(obj)
    return pool


def _tolerance(literal: str) -> float:
    mantissa, _, exp = literal.partition("e")
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    power = int(exp) if exp else 0
    return 0.5 * 10 ** (-decimals + power) + 1e-12


def check_conclusion_numbers(conclusions: Iterable[Any], evidences: Iterable[Any],
                             pool: set[float]) -> list[VerifyIssue]:
    """保守检查：结论里的小数必须能在已落盘的数值里找到。

    只报 advisory。命中的常见原因是派生值（差值、占比、周期倒数），不是错误，
    所以既不改文本也不当作失败——但会出现在 `data.verify` 里，供人工抽查。
    """
    if not pool:
        return []
    by_id = {e.evidence_id: e for e in evidences}
    flagged: list[dict] = []
    for c in conclusions:
        cited = []
        for eid in getattr(c, "evidence_ids", []) or []:
            ev = by_id.get(eid)
            if ev is not None:
                cited.extend(getattr(ev, "metric_ids", []) or [])
        if not cited:
            continue
        for literal in NUMBER_RE.findall(c.text or ""):
            value = float(literal)
            tol = _tolerance(literal)
            if not any(abs(value - x) <= max(tol, 1e-12) for x in pool):
                flagged.append({"conclusion_id": c.conclusion_id, "number": literal,
                                "cited_metrics": cited[:4]})
    if not flagged:
        return []
    return [VerifyIssue(
        kind="conclusion_number_unmatched", severity="advisory", repair=None,
        detail=(f"{len(flagged)} 个结论数值在已落盘的指标里找不到同值；"
                "多为派生值（差值/占比/周期倒数），仅作抽查线索，不影响结论成立性"),
        payload={"count": len(flagged), "samples": flagged[:10]})]
