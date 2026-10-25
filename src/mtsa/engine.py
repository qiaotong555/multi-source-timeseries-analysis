# -*- coding: utf-8 -*-
"""流水线编排：确定性状态机，节点边界与 doc/架构设计.md 一致。

每个节点写入 events；任何节点失败即中止并返回 status=failed，不返回占位结果。
"""
from __future__ import annotations

import json
import math
import os
import time
import uuid
from datetime import datetime
from typing import Any, Callable

import pandas as pd

from . import io_loaders, profiler, quality, preprocess, metrics, freq, stats_tests, anomaly
from . import forecast as forecast_mod
from . import replay as replay_mod
from .agent_runtime import WEATHER_AGENT_TOOLS, build_agent_runner
from .observation import coverage_summary, tool_states
from .models import (
    DEFAULT_SEED,
    ERR_AGENT_NO_RESULT,
    ERR_AGENT_SCENE_UNSUPPORTED,
    ERR_AGENT_UNAVAILABLE,
    ERR_DATA_LOAD_FAILED,
    ERR_DATASET_TOO_LARGE,
    ERR_EMPTY_NUMERIC_COLUMN,
    ERR_INTERNAL_ERROR,
    ERR_LOOKAHEAD_VIOLATION,
    ERR_PRICE_TABLE,
    ERR_REPLAY_INVALID,
    ERR_REPLAY_MISMATCH,
    ERR_SCENE_NOT_DETECTED,
    ERR_TIME_AXIS_NOT_FOUND,
    MAX_RERUN,
    SCENE_CONFIDENCE_THRESHOLD,
    TASK_ID,
    AnalysisPlan,
    AnalysisRequest,
    MetricTable,
    AnalysisResult,
    Artifact,
    ErrorDetail,
    EventRecord,
)
from .scenarios import stock as stock_scene  # noqa: F401  注册适配器
from .scenarios import weather as weather_scene  # noqa: F401
from .scenarios.base import ScenarioContext, detect_scene, get_adapter
from .tools import ToolContext
from . import cost as cost_mod
from . import report as report_mod
from . import viz as viz_mod
from .planner import TOOL_SPECS, build_planner

# 规模上限。实测（10 列、含全部统计检验）：1e5 行 37 s、2e5 行 124 s、5e5 行 736 s，
# 超线性来自 ADF 的自动滞后阶随样本量增长。上限取 2e5，最坏约 2 分钟。
# 超过即明确拒绝——原先的行为是「给多少算多少」，表现为长时间无响应而非可预期的失败。
DEFAULT_MAX_ROWS = 200_000
WARN_ROWS = 100_000
# 可用 MTSA_MAX_ROWS 临时放宽，便于确认更大的机器或更快的实现是否真的可行
ENV_MAX_ROWS = "MTSA_MAX_ROWS"


def _max_rows() -> int:
    raw = os.environ.get(ENV_MAX_ROWS, "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_MAX_ROWS


def _new_run_id() -> str:
    return f"run_{datetime.now():%Y%m%d}_{uuid.uuid4().hex[:8]}"


def _empty_metrics(tag: str) -> MetricTable:
    """Agent 未执行指标工具时的占位表。

    用显式的 ``generated_by`` 标明「没算」，而不是伪造一张空但看似正常的表；
    报告和下游据此可以判断指标缺失，而不是把缺失读成「没有指标」。
    """
    return MetricTable(rows=[], generated_by=tag, code_ref="")


def _agent_outputs(traj) -> tuple[MetricTable, MetricTable, list, list, list, list]:
    """把 Agent 轨迹里的工具产出还原成收尾层需要的六类结果。

    只接受 ``outcome=ok`` 的结果；失败或跳过的工具按「没有产出」处理，
    由调用方把它写进 warning，不能静默当成空结果。
    """
    def ok(name: str):
        res = traj.results.get(name)
        return res if (res is not None and res.outcome == "ok") else None

    def rows(name: str) -> list:
        res = ok(name)
        return list(res.data) if res is not None and res.data is not None else []

    metric_res = ok("compute_metrics")
    if metric_res is not None and isinstance(metric_res.data, dict):
        generic = metric_res.data.get("generic") or _empty_metrics("agent:compute_metrics")
        extra = metric_res.data.get("scenario") or _empty_metrics("agent:compute_metrics")
    else:
        generic = _empty_metrics("agent:not_run")
        extra = _empty_metrics("agent:not_run")

    return (generic, extra,
            rows("analyze_spectrum"), rows("run_stat_tests"),
            rows("detect_anomalies"), rows("detect_change_points"))


def _sample_scale_warnings(clean, qrep, profile, plan) -> list[str]:
    """短样本/高缺失的显式提示。

    C 任务的第二份数据只有 30 天、15 分钟、10% 缺失。ADF/KPSS/STL 在这类样本上
    会退化，不能静默给结论；这里只加 warning，不改变任何判定或数值。
    """
    out: list[str] = []
    rows = len(clean.frame)
    if rows < 5000:
        out.append(
            f"样本量仅 {rows} 行（<5000），ADF/KPSS/STL 等检验在小样本下可能退化；"
            "相关结论只能作探索性参考，不能当作稳定结论。"
        )
    ta = profile.time_axis
    if ta and ta.start and ta.end:
        try:
            span_days = (datetime.fromisoformat(ta.end) - datetime.fromisoformat(ta.start)).total_seconds() / 86400.0
            if span_days < 60:
                out.append(
                    f"时间跨度仅 {span_days:.1f} 天（<60 天），无法可靠估计年周期/季节性；"
                    "季节性结论应视为未验证。"
                )
        except ValueError:
            pass
    ratios = list((qrep.missing_ratio or {}).values())
    max_missing = max(ratios) if ratios else 0.0
    if max_missing > 0.05:
        out.append(
            f"最大缺失率 {max_missing:.2%}（>5%），共插值 {clean.imputed_cells} 个单元；"
            "插值点不作为异常判定依据，缺失段附近的结论需谨慎。"
        )
    return out


class _Timer:
    """记录节点耗时与状态，写入 events。"""

    def __init__(self, events: list[EventRecord]):
        self.events = events

    def run(self, node: str, fn: Callable[[], dict[str, Any]]) -> Any:
        started = datetime.now().isoformat(timespec="seconds")
        t0 = time.perf_counter()
        try:
            res = fn()
            self.events.append(EventRecord(
                node=node, started_at=started,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                status="succeeded", summary=res if isinstance(res, dict) else {},
            ))
            return res
        except Exception as exc:
            self.events.append(EventRecord(
                node=node, started_at=started,
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
                status="failed", summary={"error": f"{type(exc).__name__}: {exc}"},
            ))
            raise


def _json_safe(value):
    """把非有限浮点替换成 null，保证失败结果也能写成合法 JSON。"""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _unexpected_failure(request: AnalysisRequest, run_id: str, outdir: str,
                        events: list[EventRecord], warnings: list[str],
                        exc: Exception, verbose: bool) -> AnalysisResult:
    """把未覆盖的节点异常收敛成 INTERNAL_ERROR，而不是让调用方看到崩溃。

    这是鲁棒性矩阵的底线：任何输入都必须有 result.json，且能看出失败在哪个节点。
    """
    last = next((e for e in reversed(events) if e.status == "failed"), None)
    node = last.node if last else "unknown"
    result = AnalysisResult(request_id=request.request_id, run_id=run_id, status="failed")
    result.error = ErrorDetail(
        code=ERR_INTERNAL_ERROR,
        message=f"分析节点 {node} 发生未分类异常，已中止并保留结果",
        details={
            "node": node,
            "exception": f"{type(exc).__name__}: {exc}"[:500],
            "hint": "该输入触发了未覆盖的边界条件；请保留原始输入与 result.json 以便复现",
        },
    )
    result.events = events
    result.warnings = warnings
    try:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(_json_safe(result.model_dump(mode="python")), f,
                      ensure_ascii=False, indent=2, default=str)
    except OSError:
        pass
    if verbose:
        print(f"[{ERR_INTERNAL_ERROR}] 节点 {node}: {type(exc).__name__}: {exc}")
    return result


def run_analysis(request: AnalysisRequest, verbose: bool = True,
                 dry_run: bool = False) -> AnalysisResult:
    """执行一次分析。

    `dry_run=True` 只走到「计划已产出」为止：返回字段摘要、场景、计划与将要执行的
    工具，不写任何文件、不运行检测器。供 `mtsa run --dry-run` 让用户先确认目标。
    """
    run_id = _new_run_id()
    outdir = request.output_dir or os.path.join("artifacts", run_id)
    events: list[EventRecord] = []
    warnings: list[str] = []
    try:
        return _run_analysis_impl(request, verbose, run_id, outdir, events, warnings,
                                  dry_run=dry_run)
    except Exception as exc:                      # noqa: BLE001
        if dry_run:
            res = AnalysisResult(request_id=request.request_id, run_id=run_id,
                                 status="failed")
            res.error = ErrorDetail(
                code=ERR_INTERNAL_ERROR,
                message="预演阶段发生未分类异常（dry-run 不落盘）",
                details={"exception": f"{type(exc).__name__}: {exc}"[:500]},
            )
            res.events = events
            res.warnings = warnings
            return res
        return _unexpected_failure(request, run_id, outdir, events, warnings, exc, verbose)


# Agent 白名单工具的说明，仅用于 --dry-run 的展示。
_AGENT_TOOL_DESC = {
    "compute_metrics": "指标表：趋势、波动、分位等基础统计",
    "analyze_spectrum": "频谱：主频与周期（Welch 功率谱）",
    "run_stat_tests": "统计检验：平稳性 / 自相关 / 正态性 / 趋势",
    "detect_anomalies": "异常点：离群与故障定位",
    "detect_change_points": "变化点：均值或方差突变",
}


def _fmt_interval(sec) -> str:
    if not sec:
        return "未知"
    if float(sec) % 3600 == 0:
        return f"{int(float(sec) // 3600)}h"
    if float(sec) % 60 == 0:
        return f"{int(float(sec) // 60)}min"
    return f"{float(sec):g}s"


def _resolve_effective_mode(request: AnalysisRequest, plan: AnalysisPlan,
                            replay_source, warnings: list[str]) -> tuple[str, str]:
    """把请求里的模式解析成实际执行模式：auto 要看场景与模型可用性。"""
    requested = request.mode
    if replay_source is not None:
        return requested, "agent"
    if requested != "auto":
        return requested, requested
    from .llm_client import client_from_env
    if plan.scene_id != "weather":
        warnings.append("auto 模式：当前场景不是天气，Agent 尚未支持该场景，已回退 pipeline")
        return requested, "pipeline"
    if client_from_env() is None:
        warnings.append("auto 模式：未配置可用的模型客户端，已回退 pipeline")
        return requested, "pipeline"
    return requested, "agent"


def _preview_payload(request: AnalysisRequest, raw, profile, qrep, plan: AnalysisPlan,
                     requested_mode: str, effective_mode: str, outdir: str,
                     warnings: list[str]) -> dict[str, Any]:
    """dry-run 的输出：只描述「接下来要做什么」，不含任何计算结果。"""
    axis = profile.time_axis
    group_col = plan.group_column
    groups = 1
    if group_col and group_col in raw.frame.columns:
        groups = int(raw.frame[group_col].nunique(dropna=True))
    if effective_mode == "agent":
        tools = [{"name": n, "description": _AGENT_TOOL_DESC.get(n, "")}
                 for n in WEATHER_AGENT_TOOLS]
    else:
        tools = [{"name": a, "description": TOOL_SPECS.get(a, "场景专属分析（不由模型裁剪）")}
                 for a in plan.analyses]
    missing = [v for v in (qrep.missing_ratio or {}).values() if v is not None]
    return {
        "dry_run": True,
        "input": {
            "path": request.input_path,
            "format": raw.spec.format,
            "rows": raw.row_count,
            "columns": list(raw.frame.columns),
            "time_column": axis.column if axis else None,
            "time_start": axis.start if axis else None,
            "time_end": axis.end if axis else None,
            "interval": _fmt_interval(axis.median_interval_seconds if axis else None),
            "interval_stable": bool(profile.sampling.interval_stable),
            "group_column": group_col,
            "groups": groups,
            "numeric_columns": list(profile.numeric_columns),
        },
        "quality": {
            "issues": len(qrep.issues),
            "max_missing_ratio": max(missing) if missing else 0.0,
            "duplicates": qrep.duplicate_timestamps,
            "out_of_order": qrep.out_of_order,
            "gaps": qrep.gap_count,
            "anomaly_candidates": qrep.anomaly_candidate_count,
        },
        "scene": {
            "scene_id": plan.scene_id,
            "confidence": plan.scene_confidence,
            "source": "显式指定 --scene" if request.scene_id else "自动识别",
        },
        "mode": effective_mode,
        "mode_requested": requested_mode,
        "plan": {
            "plan_source": plan.plan_source,
            "planner_version": plan.planner_version,
            "analyses": list(plan.analyses),
            "target_columns": list(plan.target_columns),
            "resample_rule": plan.resample_rule,
            "parameters": dict(plan.parameters or {}),
        },
        "tools": tools,
        "warnings": list(warnings),
        "output_dir": os.path.abspath(outdir),
    }


def _dry_run_result(request: AnalysisRequest, run_id: str, raw, profile, qrep,
                    plan: AnalysisPlan, requested_mode: str, effective_mode: str,
                    outdir: str, warnings: list[str],
                    events: list[EventRecord]) -> AnalysisResult:
    result = AnalysisResult(request_id=request.request_id, run_id=run_id,
                            status="succeeded")
    result.data = {"plan": plan.model_dump(),
                   "preview": _preview_payload(request, raw, profile, qrep, plan,
                                               requested_mode, effective_mode,
                                               outdir, warnings)}
    # 只保留只读节点的轨迹：清洗之后的节点在 dry-run 里根本没有执行
    result.events = list(events)
    result.warnings = warnings
    return result


def _run_analysis_impl(request: AnalysisRequest, verbose: bool, run_id: str,
                       outdir: str, events: list[EventRecord],
                       warnings: list[str], dry_run: bool = False) -> AnalysisResult:
    timer = _Timer(events)
    result = AnalysisResult(request_id=request.request_id, run_id=run_id, status="running")

    def fail(code: str, message: str, **details) -> AnalysisResult:
        result.status = "failed"
        result.error = ErrorDetail(code=code, message=message, details=details)
        result.events = events
        result.warnings = warnings
        # 失败也要落盘：既便于事后排查，也让进程隔离的调用方能区分
        # 「分析失败」与「子进程崩溃」——后者根本没有 result.json。
        # 例外是 --dry-run：它对用户的承诺就是「什么都不写」，失败时同样不落盘。
        if not dry_run:
            try:
                os.makedirs(outdir, exist_ok=True)
                with open(os.path.join(outdir, "result.json"), "w", encoding="utf-8") as f:
                    f.write(result.model_dump_json(indent=2))
            except OSError:
                pass                              # 目录不可写不应掩盖原始错误
        if verbose:
            print(f"[{code}] {message}")
        return result

    # ---- 0. 重放源 ----
    # 重放只读取 agent 运行的 result.json；计划、步骤与原始输入都来自这份记录。
    replay_source = None
    if request.replay_from:
        try:
            replay_source = replay_mod.load_replay(request.replay_from)
        except replay_mod.ReplayError as exc:
            return fail(ERR_REPLAY_INVALID, str(exc), path=request.replay_from)
        if request.mode != "agent":
            return fail(ERR_REPLAY_INVALID,
                        "重放只适用于 agent 结果；请使用 --mode agent",
                        mode=request.mode)
        if request.param_overrides:
            return fail(ERR_REPLAY_INVALID,
                        "重放不接受 --param：计划参数已记录在轨迹中",
                        overrides=sorted(request.param_overrides))

    # ---- 1. 导入 ----
    try:
        timer.run("load_dataset", lambda: {})
        raw = io_loaders.load_dataset(
            request.input_path,
            time_column=request.time_column,
            value_columns=request.value_columns,
            group_column=request.group_column,
            unit=request.unit,
            sampling_interval=request.sampling_interval,
        )
        events[-1].summary = {"rows": raw.row_count, "columns": len(raw.frame.columns),
                              "format": raw.spec.format}
    except ValueError as exc:
        return fail(ERR_DATA_LOAD_FAILED, str(exc), path=request.input_path)
    except Exception as exc:
        return fail(ERR_INTERNAL_ERROR, "导入节点发生未分类异常", detail=f"{type(exc).__name__}: {exc}")

    # ---- 1b. 规模检查 ----
    # 必须在摘要之前：否则摘要、清洗、指标、统计检验都要跑一遍才失败，
    # 用户已经等了几分钟才看到报错，与「卡死」的体感没有区别。
    _limit = _max_rows()
    if raw.row_count > _limit:
        return fail(ERR_DATASET_TOO_LARGE,
                    f"数据行数 {raw.row_count} 超过本实现的上限 {_limit}，已在导入后中止",
                    rows=raw.row_count, limit=_limit,
                    measured={"1e5": "37 s", "2e5": "124 s", "5e5": "736 s"},
                    hint=("统计检验的滞后阶随样本量增长，耗时超线性。请先降采样"
                          "（如按小时聚合成日频）或按时间分片分别分析；"
                          f"确需一次跑完可用环境变量 {ENV_MAX_ROWS} 放宽上限"))
    if raw.row_count > WARN_ROWS:
        warnings.append(
            f"数据行数 {raw.row_count} 较大，统计检验预计耗时数分钟（实测 2e5 行约 124 s）")

    # ---- 2. 摘要 ----
    profile, parsed_time = profiler.profile_dataset(raw)
    timer.run("profile_dataset", lambda: {
        "time_axis": profile.time_axis.column if profile.time_axis else None,
        "numeric_columns": len(profile.numeric_columns),
        "gaps": profile.sampling.gap_count,
    })
    if profile.time_axis is None:
        return fail(ERR_TIME_AXIS_NOT_FOUND,
                    "未识别到可解析的时间字段（格式不一致、混合时区或全为空都会导致此结果）",
                    columns=list(raw.frame.columns),
                    hint="请先统一时间格式与时区，或用 --time-column 显式指定时间列")
    if not profile.numeric_columns:
        return fail(ERR_EMPTY_NUMERIC_COLUMN, "未识别到任何有效数值列",
                    columns=list(raw.frame.columns))

    # ---- 3. 质量检查 ----
    qrep = quality.check_quality(raw, profile, parsed_time)
    timer.run("check_quality", lambda: {"issues": len(qrep.issues),
                                        "anomaly_candidates": qrep.anomaly_candidate_count})

    # ---- 4. 场景识别与计划 ----
    # 计划由「计划层」产出：有 LLM 配置且校验通过则用模型计划，否则回退规则计划。
    # 无论来源如何，节点 5 至 12 的执行逻辑完全一致，数值结论仍由代码产出。
    if replay_source is not None:
        # 重放不重新规划：使用记录里的计划快照，参数与目标列完全一致。
        plan_data = replay_source["plan"]
        adapter = get_adapter(plan_data["scene_id"])
        confidence = float(plan_data.get("scene_confidence") or 1.0)
        plan = AnalysisPlan.model_validate(plan_data)
        plan_source = plan_data.get("plan_source", "replay")
    else:
        adapter, confidence, scores = detect_scene(raw.frame, profile)
        if request.scene_id:
            adapter = get_adapter(request.scene_id)
            confidence = 1.0
        elif adapter is None or confidence < SCENE_CONFIDENCE_THRESHOLD:
            return fail(ERR_SCENE_NOT_DETECTED,
                        f"场景识别置信度不足（最高 {confidence:.3f} < {SCENE_CONFIDENCE_THRESHOLD}），"
                        f"请显式指定 --scene",
                        scores={k: round(v, 4) for k, v in scores.items()})
        planner = build_planner(adapter)
        plan, plan_source = planner.plan(request.question, profile, qrep, adapter, raw.frame)
        plan.scene_confidence = round(float(confidence), 4)
        plan.plan_source = plan_source
        if getattr(planner, "last_error", None):
            warnings.append(f"计划层回退到规则模式：{planner.last_error}")
        open_units = (plan.parameters or {}).get("unit_change_open_intervals") or []
        if open_units:
            where = "、".join("%s@%s" % (x.get("column"), x.get("start_time"))
                             for x in open_units[:3])
            warnings.append(
                f"检测到单侧疑似单位变化（{where}）：未自动换算。"
                "单侧转移无法区分量纲变化与真实水平跳变，请在报告中人工核对。")

    # ---- 4b. 参数覆盖（调参入口，不修改适配器默认值）----
    if request.param_overrides:
        unknown = sorted(set(request.param_overrides) - set(plan.parameters))
        if unknown:
            return fail(ERR_INTERNAL_ERROR,
                        f"未知的计划参数：{unknown}，不会生效，已中止",
                        reason="invalid_param_override",
                        unknown=unknown, allowed=sorted(plan.parameters))
        plan.parameters.update(request.param_overrides)
        events.append(EventRecord(
            node="param_override", started_at=datetime.now().isoformat(timespec="seconds"),
            elapsed_ms=0, status="succeeded",
            summary={"applied": request.param_overrides},
        ))

    timer.run("plan", lambda: {"scene": plan.scene_id, "confidence": plan.scene_confidence,
                                "analyses": len(plan.analyses),
                                "overrides": len(request.param_overrides),
                                "plan_source": plan.plan_source})

    # ---- 4c. 预演（--dry-run）----
    # 到此为止只做了只读操作：导入、摘要、质量检查、场景识别、出计划。
    # 清洗（会改数据）、检测器、图表、报告都在后面，所以这里直接返回就能满足
    # 「不写 artifacts、不调检测器」；计划与真跑同源，不存在两套逻辑。
    if dry_run:
        requested_mode, effective_mode = _resolve_effective_mode(
            request, plan, replay_source, warnings)
        return _dry_run_result(request, run_id, raw, profile, qrep, plan,
                               requested_mode, effective_mode, outdir,
                               warnings, events)

    # ---- 5. 清洗 ----
    clean = preprocess.clean_dataset(raw, profile, parsed_time, plan)
    timer.run("clean_dataset", lambda: {"rows": len(clean.frame),
                                        "imputed": clean.imputed_cells,
                                        "unit_conversions": len(clean.unit_conversions)})
    if clean.frame.empty:
        return fail(ERR_EMPTY_NUMERIC_COLUMN, "清洗后数据集为空，无法继续分析")
    for msg in _sample_scale_warnings(clean, qrep, profile, plan):
        if msg not in warnings:
            warnings.append(msg)
    if any(op.get("op") == "replace_nonfinite_with_nan" for op in clean.operations):
        warnings.append("检测到 inf/-inf 等非有限数值，已按缺失处理并插值；相关结论需谨慎。")
    const_cols = [c for c in plan.target_columns
                  if c in clean.frame.columns and clean.frame[c].nunique(dropna=True) <= 1]
    if const_cols:
        warnings.append(f"常数列（方差为 0）不参与统计检验：{const_cols}")

    # ---- 5b. 运行模式解析 ----
    # auto：有模型且场景支持 → agent；否则回退 pipeline，并把原因写进 warnings。
    # 重放不需要模型，因此 replay 始终按 agent 处理。
    requested_mode, effective_mode = _resolve_effective_mode(
        request, plan, replay_source, warnings)

    # ---- 6-10. 分析层 ----
    # pipeline：固定顺序，行为与 T1.8 保持一致。
    # agent：仅天气场景，由模型在白名单内选择工具；数值仍由工具内部的确定性代码产出。
    agent_info: Optional[dict[str, Any]] = None
    if effective_mode == "agent":
        if plan.scene_id != "weather":
            return fail(
                ERR_AGENT_SCENE_UNSUPPORTED,
                f"Agent 模式本轮只支持天气场景，当前场景为 {plan.scene_id}",
                scene=plan.scene_id, supported=["weather"],
                hint="请改用 --mode pipeline，或等待股票场景的 Agent 实现",
            )

        # E2 成本语义：价目表先校验再跑，配置错误不该等到花完 token 才暴露。
        # 未配置时 price_table 为 None，金额字段整体缺失（不写 0 冒充免费）。
        price_table = None
        price_path = request.price_table or cost_mod.price_table_path()
        if price_path:
            try:
                price_table = cost_mod.load_price_table(price_path)
            except cost_mod.PriceTableError as exc:
                return fail(ERR_PRICE_TABLE, str(exc), path=price_path,
                            hint="金额是可选能力；不需要金额时去掉 --price-table / "
                                 f"{cost_mod.ENV_PRICE_TABLE}")
        tool_ctx = ToolContext(clean=clean, plan=plan, adapter=adapter, quality=qrep)
        agent_started = datetime.now().isoformat(timespec="seconds")
        t_agent = time.perf_counter()
        usage_client = None            # 重放不调用模型，因此没有客户端也没有 token
        if replay_source is not None:
            traj = replay_mod.replay_steps(replay_source, tool_ctx)
        else:
            runner = build_agent_runner(
                tool_ctx,
                request.question,
                max_steps=request.agent_max_steps,
                allowed_tools=WEATHER_AGENT_TOOLS,
                scope_mode=request.agent_scope,
                max_extra_steps=request.agent_max_extra_steps,
                max_tokens=request.agent_max_tokens,
            )
            if runner is None:
                return fail(
                    ERR_AGENT_UNAVAILABLE,
                    "Agent 模式需要模型配置，但当前没有可用的客户端",
                    required_env=["MTSA_LLM_BASE_URL", "MTSA_LLM_API_KEY", "MTSA_LLM_MODEL"],
                    hint="配置环境变量后重试；未配置时请使用 --mode pipeline",
                )
            traj = runner.run()
            usage_client = runner.client
        agent_elapsed = int((time.perf_counter() - t_agent) * 1000)
        agent_info = traj.to_dict()
        states = tool_states(traj, WEATHER_AGENT_TOOLS)
        agent_info.update({"mode": "agent", "elapsed_ms": agent_elapsed,
                           "max_steps": request.agent_max_steps,
                           "max_extra_steps": request.agent_max_extra_steps,
                           "max_tokens": request.agent_max_tokens,
                           "tool_states": states,
                           "replay_version": replay_mod.REPLAY_VERSION,
                           "coverage": coverage_summary(plan, traj, WEATHER_AGENT_TOOLS)})
        # 模型 token 用量：服务端未返回 usage 时留空，不用估算值冒充成本。
        usage_fn = getattr(usage_client, "usage_summary", None)
        tokens = usage_fn() if callable(usage_fn) else None
        if tokens:
            agent_info["tokens"] = tokens
        if price_table is not None:
            if tokens:
                try:
                    agent_info["cost"] = cost_mod.estimate_cost(tokens, price_table)
                except cost_mod.CostError as exc:
                    warnings.append(f"金额未计算：{exc}")
            else:
                warnings.append(
                    "已配置价目表，但服务端未返回 usage：本次金额无法计算（不估算）")
        budget = agent_info.get("budget") or {}
        if budget.get("unavailable"):
            warnings.append("token 预算未生效：服务端未返回 usage；本次运行未按预算截断")
        if budget.get("exhausted"):
            precheck = budget.get("precheck") or {}
            if precheck.get("blocked"):
                warnings.append(
                    "token 预算前置拦截（已用 {}/{}，下次调用预计再需 {}，会超限），"
                    "运行被截断；未覆盖的分析见 scope.missing，不能当作完整分析引用".format(
                        budget.get("used"), budget.get("limit"), precheck.get("reserve")))
            else:
                warnings.append(
                    "token 预算已用尽（{}/{}），运行被截断；未覆盖的分析见 scope.missing，"
                    "不能当作完整分析引用".format(budget.get("used"), budget.get("limit")))
        # 哈希只覆盖决策路径；耗时、时间戳、run_id 不参与
        agent_info["trajectory_sha256"] = replay_mod.trajectory_hash(agent_info)
        # 先把轨迹放进结果：即使下面判定为失败，result.json 里也能看到模型做了什么。
        result.data = {"agent": agent_info}

        for step in traj.steps:
            events.append(EventRecord(
                node=step.action,
                started_at=step.started_at or agent_started,
                elapsed_ms=step.elapsed_ms,
                status=("succeeded" if step.outcome == "ok"
                        else "skipped" if step.outcome == "skipped" else "failed"),
                summary={"outcome": step.outcome, "reason": step.reason,
                         "summary": step.summary},
            ))
        events.append(EventRecord(
            node="agent_analysis", started_at=agent_started, elapsed_ms=agent_elapsed,
            status="succeeded",
            summary={"steps": traj.step_count, "terminate_reason": traj.terminate_reason,
                     "tools_run": traj.tools_run, "allowed_tools": list(WEATHER_AGENT_TOOLS),
                     "replayed": replay_source is not None},
        ))

        if not traj.tools_run:
            return fail(
                ERR_AGENT_NO_RESULT,
                "Agent 没有成功执行任何分析工具，不能生成空报告冒充成功",
                terminate_reason=traj.terminate_reason,
                rejects=traj.rejects[-3:],
            )

        mtab, extra, spectra, tests, anomalies, change_points = _agent_outputs(traj)
        missing = [t for t in WEATHER_AGENT_TOOLS if states[t] == "not_run"]
        failed = [t for t in WEATHER_AGENT_TOOLS if states[t] == "failed"]
        skipped = [t for t in WEATHER_AGENT_TOOLS if states[t] == "skipped"]
        if missing:
            warnings.append(
                "Agent 未执行以下分析工具：" + "、".join(missing)
                + "。对应分析为空，不能解读为「未发现问题」。"
            )
        if failed:
            warnings.append(
                "Agent 执行失败的分析工具：" + "、".join(failed)
                + "。失败不等于没有问题，报告中对应分析为空。"
            )
        if skipped:
            warnings.append(
                "Agent 跳过的分析工具：" + "、".join(skipped)
                + "。跳过表示前置条件不满足，不是执行成功。"
            )
        if "compute_metrics" in missing or "compute_metrics" in failed:
            warnings.append("指标工具未成功执行：报告中的指标表与依赖指标的结论会缺失。")
        if not spectra and "analyze_spectrum" not in missing:
            warnings.append("频域分析未产出结果：采样率无法识别或样本长度不足")
    else:
        # ---- 6. 指标 ----
        # 计算必须放进 timer.run：否则 compute_metrics 抛异常时没有失败节点记录，
        # 顶层护栏只能报 node=unknown。
        def _compute_metrics():
            return metrics.compute_metrics(clean, plan), adapter.extra_metrics(clean, plan)

        mtab, extra = timer.run("compute_metrics", _compute_metrics)
        events[-1].summary = {"generic": len(mtab.rows), "scenario": len(extra.rows)}

        # ---- 7. 频域 ----
        spectra = []
        if "spectrum" in plan.analyses:
            spectra = freq.analyze_spectrum(clean, plan)
            if not spectra:
                warnings.append("频域分析未产出结果：采样率无法识别或样本长度不足")
        timer.run("analyze_spectrum", lambda: {"columns": len(spectra)})

        # ---- 8. 统计检验 ----
        tests = stats_tests.run_stat_tests(clean, plan)
        timer.run("run_stat_tests", lambda: {"tests": len(tests)})

        # ---- 9. 异常与变化点 ----
        anomalies = anomaly.detect_anomalies(clean, plan) if "anomaly" in plan.analyses else []
        change_points = anomaly.detect_change_points(clean, plan) if "changepoint" in plan.analyses else []
        timer.run("detect_anomalies", lambda: {"anomalies": len(anomalies),
                                               "change_points": len(change_points)})

    # ---- 9b. 绝对极端（描述性；不并入 anomalies，不参与异常判定） ----
    # 与模式无关：分位数口径固定，cost 很低；Agent 是否执行检测不影响这个字段。
    extremes = anomaly.detect_extremes(clean, plan)
    # 段上限截断会让「事件是否被覆盖」失真（D2 的 20→100 教训）。截断就必须说。
    _trunc = sorted({e["column"] for e in extremes if e.get("runs_truncated")})
    if _trunc:
        warnings.append(
            f"极端值段超过上限 {anomaly.EXTREME_MAX_RUNS} 被截断：{_trunc}；"
            "按时间窗口统计的覆盖类指标可能失真")

    # ---- 9c. 预测（仅天气；默认关闭） ----
    # 关闭时不进入分支，data 与报告逐位不变（F5）。重放时若记录里带 forecast，
    # 就重算一次，让重放比对覆盖这个字段。
    want_forecast = request.forecast or bool(
        replay_source is not None and "forecast" in (replay_source.get("data") or {}))
    forecast_res = None
    if want_forecast:
        if plan.scene_id != "weather":
            warnings.append("预测只支持天气场景，已跳过（未写入 data.forecast）")
        else:
            forecast_res = timer.run(
                "forecast", lambda: forecast_mod.build_forecast(
                    clean, plan, request.forecast_horizon))
            events[-1].summary = {"columns": len(forecast_res["columns"]),
                                  "period": forecast_res["period"],
                                  "horizon": forecast_res["horizon"]}
            bad = [c["column"] for c in forecast_res["columns"]
                   if c.get("status") != "ok"]
            if bad:
                warnings.append(f"以下列因样本不足未做预测：{bad}")

    # ---- 10. 未来数据泄漏检查（股票场景） ----
    leak: list[dict] = []
    if plan.scene_id == "stock":
        close_col = plan.parameters.get("close_column")
        leak = stock_scene.check_lookahead(clean.frame, close_col or "close")
        timer.run("check_lookahead", lambda: {"issues": len(leak)})
        if leak:
            return fail(ERR_LOOKAHEAD_VIOLATION,
                        "检测到未来数据泄漏，回测结果不予输出",
                        issues=leak,
                        hint="信号列与次日收益相关系数绝对值超过 0.80，不能使用未来信息构造信号")

    # ---- 11. 图表 ----
    executed_tools = None
    if agent_info is not None:
        executed_tools = set(agent_info.get("tools_run") or [])
    ctx = ScenarioContext(clean, plan, mtab, spectra, tests, anomalies, change_points, extra, qrep,
                          executed_tools=executed_tools, extremes=extremes,
                          forecast=forecast_res)
    specs = adapter.figure_specs(plan)
    artifacts, fig_warnings = viz_mod.build_figures(ctx, specs, outdir)
    warnings.extend(fig_warnings)
    timer.run("build_figures", lambda: {"artifacts": len(artifacts), "warnings": len(fig_warnings)})

    # ---- 12. 报告与结论 ----
    conclusions, evidences = adapter.conclusions(ctx)
    clean_art = report_mod.write_clean_csv(ctx, outdir)
    artifacts.append(clean_art)
    dataset_info = {
        "task_id": TASK_ID,
        "run_id": run_id,
        "created_at": result.created_at,
        "input_path": request.input_path,
        "file_hash": raw.file_hash,
        "row_count": raw.row_count,
        "scene_id": plan.scene_id,
        "question": request.question,
        "seed": request.seed,
        "mode": effective_mode,
        "mode_requested": requested_mode,
    }
    # 只在开启预测时才写这两个键：关闭时 datasets[0] 的字段集合保持不变（F5）
    if forecast_res is not None:
        dataset_info["forecast"] = True
        dataset_info["forecast_horizon"] = forecast_res["horizon"]
    rep_art = report_mod.render_report(ctx, artifacts, conclusions, evidences, qrep, profile,
                                       dataset_info, outdir, warnings, agent=agent_info)
    script_art = report_mod.write_repro_script(dataset_info, outdir)
    artifacts.extend([rep_art, script_art])
    timer.run("render_report", lambda: {"conclusions": len(conclusions), "evidence": len(evidences)})

    # ---- 13. 校验：结论必须有证据 ----
    no_evidence = [c.conclusion_id for c in conclusions if not c.evidence_ids]
    if no_evidence:
        warnings.append(f"存在未绑定证据的结论：{no_evidence}")
    timer.run("verify", lambda: {"no_evidence_conclusions": len(no_evidence)})

    result.status = "succeeded"
    data = {
        "profile": profile.model_dump(mode="json"),
        "quality": qrep.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "clean": {
            "rows": len(clean.frame),
            "operations": clean.operations,
            "imputed_cells": clean.imputed_cells,
            "dropped_rows": clean.dropped_rows,
            "unit_conversions": clean.unit_conversions,
        },
        "metrics": [r.model_dump(mode="json") for r in mtab.rows],
        "scenario_metrics": [r.model_dump(mode="json") for r in extra.rows],
        "spectrum": [s.model_dump(mode="json") for s in spectra],
        "tests": [t.model_dump(mode="json") for t in tests],
        "anomalies": [a.model_dump(mode="json") for a in anomalies],
        "change_points": [c.model_dump(mode="json") for c in change_points],
        "extremes": extremes,
        **({"forecast": forecast_res} if forecast_res is not None else {}),
        "lookahead": leak,
    }
    if agent_info is not None and replay_source is not None:
        recorded = {k: v for k, v in replay_source["data"].items() if k != "agent"}
        diffs = replay_mod.diff_paths(recorded, data)
        source_hash = replay_source.get("trajectory_sha256")
        replay_hash = agent_info["trajectory_sha256"]
        trajectory_match = (source_hash == replay_hash) if source_hash else None
        data_match = not diffs
        match = data_match and trajectory_match is not False
        agent_info["replay"] = {
            "source": replay_source["path"],
            "source_hash": source_hash,
            "replay_hash": replay_hash,
            "trajectory_match": trajectory_match,
            "data_match": data_match,
            "match": match,
            "mismatches": diffs[:10],
        }
        if not match:
            data["agent"] = agent_info
            result.data = data
            return fail(
                ERR_REPLAY_MISMATCH,
                "重放结果与原运行不一致，可复现性未通过",
                source=replay_source["path"],
                trajectory_match=trajectory_match,
                mismatches=diffs[:10],
                hint="先检查工具是否引入了随机源或未固定时间戳，再重新采集轨迹",
            )
    if agent_info is not None:
        data["agent"] = agent_info
    result.data = data
    result.evidence = evidences
    result.datasets = [dataset_info]
    result.artifacts = artifacts
    result.warnings = warnings
    result.events = events
    result.conclusions = conclusions

    # 落盘
    os.makedirs(outdir, exist_ok=True)
    out_json = os.path.join(outdir, "result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))
    result.artifacts.append(
        Artifact(artifact_id="art_result_json", kind="table_csv", path=out_json,
                 title="结构化结果", fields=["all"])
    )
    if verbose:
        print(f"完成：run_id={run_id}  场景={plan.scene_id}  置信度={plan.scene_confidence}")
        print(f"产物目录：{os.path.abspath(outdir)}")
    return result
