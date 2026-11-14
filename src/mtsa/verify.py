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


# ---------------- 数据引用（A5 证据链） ----------------

_REF_PART = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)((?:\[\d+\])*)$")


def resolve_ref(data: Any, ref: str) -> tuple[bool, Any]:
    """把 `data.quality.issues[3].count` 解析成值，返回（是否解析成功, 值）。

    只支持两种语法：点号取字典键、方括号取列表下标。刻意不支持通配与表达式——
    引用必须能一眼看出指向哪个节点，否则它就不算「可定位」。
    """
    path = ref[5:] if ref.startswith("data.") else ref
    if not path:
        return False, None
    node = data
    for part in path.split("."):
        m = _REF_PART.match(part)
        if not m:
            return False, None
        key, idx_text = m.group(1), m.group(2)
        if not isinstance(node, dict) or key not in node:
            return False, None
        node = node[key]
        for idx in re.findall(r"\[(\d+)\]", idx_text):
            i = int(idx)
            if not isinstance(node, list) or i >= len(node):
                return False, None
            node = node[i]
    return True, node


def _numbers_under(node: Any) -> set[float]:
    out: set[float] = set()

    def walk(n: Any) -> None:
        if isinstance(n, bool) or n is None:
            return
        if isinstance(n, (int, float)):
            out.add(float(n))
        elif isinstance(n, str):
            # 有些数值只以文本形式存在（如 `method="...threshold=8.0"`、
            # `dominant_period="23.27 小时"`）。它们同样是落盘值，按同一个小数
            # 正则抽出来即可——只抽带小数点的，日期与计数不会被误当成度量。
            for tok in NUMBER_RE.findall(n):
                out.add(float(tok))
        elif isinstance(n, dict):
            for v in n.values():
                walk(v)
        elif isinstance(n, (list, tuple)):
            for v in n:
                walk(v)

    walk(node)
    return out


def refs_numbers(data: Any, refs: Iterable[str]) -> tuple[set[float], list[str]]:
    """收集引用节点下的全部数值；同时返回解析失败的引用（用于报错）。"""
    values: set[float] = set()
    broken: list[str] = []
    for ref in refs or []:
        ok, node = resolve_ref(data, ref)
        if not ok:
            broken.append(ref)
            continue
        values |= _numbers_under(node)
    return values, broken


def evidence_coverage(conclusions: Iterable[Any], evidences: Iterable[Any]) -> dict:
    """统计「结论的每个数字能不能回到一个已落盘节点」。

    分三种口径，避免用一个比率掩盖不同问题：

    - `with_metric_id`：证据挂了指标编号（可在 `data.metrics` 里查到值）；
    - `with_data_ref`：证据挂了 `data.*` 节点（周期/极值/异常点这类不在指标表里的量）；
    - `numeric_locatable_rate`：**只统计含小数的结论**。免责声明、方法说明这类
      没有数值的结论本来就不需要来源，把它们算进分母会把比率稀释成好看的数字。
    """
    by_id = {e.evidence_id: e for e in evidences}
    total = metric = ref = numeric = numeric_locatable = 0
    for c in conclusions:
        total += 1
        evs = [by_id[i] for i in c.evidence_ids if i in by_id]
        has_m = any(getattr(e, "metric_ids", None) for e in evs)
        has_r = any(getattr(e, "data_refs", None) for e in evs)
        metric += int(has_m)
        ref += int(has_r)
        if NUMBER_RE.search(c.text or ""):
            numeric += 1
            numeric_locatable += int(has_m or has_r)
    return {
        "conclusions": total,
        "with_metric_id": metric,
        "with_data_ref": ref,
        "numeric_conclusions": numeric,
        "numeric_locatable": numeric_locatable,
        "numeric_locatable_rate": (round(numeric_locatable / numeric, 4) if numeric else None),
    }


# 结论里的数字常做过量纲换算：比例写成百分比（×100）、秒写成毫秒（×1000）。
# 这类换算不改变可复算性，比对时按这两个倍率各试一次，避免把格式化当成失配。
_SCALES: tuple[float, ...] = (1.0, 100.0, 1000.0)


def _matches(value: float, tol: float, targets: set[float]) -> bool:
    for x in targets:
        for k in _SCALES:
            if abs(value - x * k) <= tol or abs(value - x / k) <= tol:
                return True
    return False


def _tolerance(literal: str) -> float:
    mantissa, _, exp = literal.partition("e")
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    power = int(exp) if exp else 0
    return 0.5 * 10 ** (-decimals + power) + 1e-12


def check_conclusion_numbers(conclusions: Iterable[Any], evidences: Iterable[Any],
                             pool: set[float], data: Any = None,
                             metric_values: Optional[dict[str, float]] = None) -> list[VerifyIssue]:
    """保守检查：结论里的小数必须能在**它自己引用的**数值里找到。

    只报 advisory。命中的常见原因是派生值（差值、占比、周期倒数），不是错误，
    所以既不改文本也不当作失败——但会出现在 `data.verify` 里，供人工抽查。

    `data` / `metric_values` 是 A5 加的：有 `data_refs` 的结论按引用节点取数，
    不再拿全库数值兜底。过去只比全局池子，等于「这个数在结果里任何地方出现过就算过」，
    周期、极值、异常点这些描述性结论根本进不了检查（它们没有 metric_ids，被整条跳过）。
    """
    if not pool and data is None:
        return []
    by_id = {e.evidence_id: e for e in evidences}
    flagged: list[dict] = []
    broken: list[dict] = []
    for c in conclusions:
        cited: list[str] = []
        refs: list[str] = []
        for eid in getattr(c, "evidence_ids", []) or []:
            ev = by_id.get(eid)
            if ev is not None:
                cited.extend(getattr(ev, "metric_ids", []) or [])
                refs.extend(getattr(ev, "data_refs", []) or [])
        if not cited and not refs:
            continue
        target: set[float] = set()
        if metric_values:
            target |= {float(metric_values[m]) for m in cited if m in metric_values}
        if data is not None and refs:
            values, missing = refs_numbers(data, refs)
            target |= values
            if missing:
                broken.append({"conclusion_id": c.conclusion_id, "refs": missing[:4]})
        if not target:
            target = pool
        for literal in NUMBER_RE.findall(c.text or ""):
            value = float(literal)
            tol = max(_tolerance(literal), 1e-12)
            if not _matches(value, tol, target):
                flagged.append({"conclusion_id": c.conclusion_id, "number": literal,
                                "cited_metrics": cited[:4], "cited_refs": refs[:3]})
    issues: list[VerifyIssue] = []
    if flagged:
        issues.append(VerifyIssue(
            kind="conclusion_number_unmatched", severity="advisory", repair=None,
            detail=(f"{len(flagged)} 个结论数值在它自己引用的指标/数据节点里找不到同值；"
                    "多为派生值（差值/占比/周期倒数），仅作抽查线索，不影响结论成立性"),
            payload={"count": len(flagged), "samples": flagged[:10]}))
    if broken:
        issues.append(VerifyIssue(
            kind="evidence_ref_broken", severity="advisory", repair=None,
            detail=(f"{len(broken)} 条结论的数据引用解析不到节点；"
                    "引用应指向 `result.data` 里真实存在的键或下标"),
            payload={"count": len(broken), "samples": broken[:10]}))
    return issues
