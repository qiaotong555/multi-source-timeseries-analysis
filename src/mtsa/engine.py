# -*- coding: utf-8 -*-
"""流水线编排：确定性状态机，节点边界与 doc/架构设计.md 一致。

每个节点写入 events；任何节点失败即中止并返回 status=failed，不返回占位结果。
"""
from __future__ import annotations

import os
import time
import uuid
from datetime import datetime
from typing import Any, Callable

import pandas as pd

from . import io_loaders, profiler, quality, preprocess, metrics, freq, stats_tests, anomaly
from .agent_runtime import WEATHER_AGENT_TOOLS, build_agent_runner
from .observation import coverage_summary
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
    ERR_SCENE_NOT_DETECTED,
    ERR_TIME_AXIS_NOT_FOUND,
    MAX_RERUN,
    SCENE_CONFIDENCE_THRESHOLD,
    TASK_ID,
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
from . import report as report_mod
from . import viz as viz_mod
from .planner import build_planner

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


def run_analysis(request: AnalysisRequest, verbose: bool = True) -> AnalysisResult:
    run_id = _new_run_id()
    outdir = request.output_dir or os.path.join("artifacts", run_id)
    events: list[EventRecord] = []
    timer = _Timer(events)
    warnings: list[str] = []
    result = AnalysisResult(request_id=request.request_id, run_id=run_id, status="running")

    def fail(code: str, message: str, **details) -> AnalysisResult:
        result.status = "failed"
        result.error = ErrorDetail(code=code, message=message, details=details)
        result.events = events
        result.warnings = warnings
        # 失败也要落盘：既便于事后排查，也让进程隔离的调用方能区分
        # 「分析失败」与「子进程崩溃」——后者根本没有 result.json。
        try:
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, "result.json"), "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
        except OSError:
            pass                                  # 目录不可写不应掩盖原始错误
        if verbose:
            print(f"[{code}] {message}")
        return result

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
                    "未识别到时间字段，无法执行时序分析",
                    columns=list(raw.frame.columns),
                    hint="请通过 --time-column 显式指定时间列")
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

    # ---- 5. 清洗 ----
    clean = preprocess.clean_dataset(raw, profile, parsed_time, plan)
    timer.run("clean_dataset", lambda: {"rows": len(clean.frame),
                                        "imputed": clean.imputed_cells,
                                        "unit_conversions": len(clean.unit_conversions)})
    if clean.frame.empty:
        return fail(ERR_EMPTY_NUMERIC_COLUMN, "清洗后数据集为空，无法继续分析")

    # ---- 6-10. 分析层 ----
    # pipeline：固定顺序，行为与 T1.8 保持一致。
    # agent：仅天气场景，由模型在白名单内选择工具；数值仍由工具内部的确定性代码产出。
    agent_info: Optional[dict[str, Any]] = None
    if request.mode == "agent":
        if plan.scene_id != "weather":
            return fail(
                ERR_AGENT_SCENE_UNSUPPORTED,
                f"Agent 模式本轮只支持天气场景，当前场景为 {plan.scene_id}",
                scene=plan.scene_id, supported=["weather"],
                hint="请改用 --mode pipeline，或等待股票场景的 Agent 实现",
            )

        runner = build_agent_runner(
            ToolContext(clean=clean, plan=plan, adapter=adapter, quality=qrep),
            request.question,
            max_steps=request.agent_max_steps,
            allowed_tools=WEATHER_AGENT_TOOLS,
        )
        if runner is None:
            return fail(
                ERR_AGENT_UNAVAILABLE,
                "Agent 模式需要模型配置，但当前没有可用的客户端",
                required_env=["MTSA_LLM_BASE_URL", "MTSA_LLM_API_KEY", "MTSA_LLM_MODEL"],
                hint="配置环境变量后重试；未配置时请使用 --mode pipeline",
            )

        agent_started = datetime.now().isoformat(timespec="seconds")
        t_agent = time.perf_counter()
        traj = runner.run()
        agent_elapsed = int((time.perf_counter() - t_agent) * 1000)
        agent_info = traj.to_dict()
        agent_info.update({"mode": "agent", "elapsed_ms": agent_elapsed,
                           "max_steps": request.agent_max_steps,
                           "coverage": coverage_summary(plan, traj, WEATHER_AGENT_TOOLS)})
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
                     "tools_run": traj.tools_run, "allowed_tools": list(WEATHER_AGENT_TOOLS)},
        ))

        if not traj.tools_run:
            return fail(
                ERR_AGENT_NO_RESULT,
                "Agent 没有成功执行任何分析工具，不能生成空报告冒充成功",
                terminate_reason=traj.terminate_reason,
                rejects=traj.rejects[-3:],
            )

        mtab, extra, spectra, tests, anomalies, change_points = _agent_outputs(traj)
        missing = [t for t in WEATHER_AGENT_TOOLS if t not in traj.results]
        failed = [s.action for s in traj.steps if s.outcome == "failed"]
        if missing:
            warnings.append(
                "Agent 未执行以下分析工具：" + "、".join(missing)
                + "。对应分析为空，不能解读为「未发现问题」。"
            )
        if failed:
            warnings.append("Agent 执行失败的分析工具：" + "、".join(failed))
        if "compute_metrics" in missing or "compute_metrics" in failed:
            warnings.append("指标工具未成功执行：报告中的指标表与依赖指标的结论会缺失。")
        if not spectra and "analyze_spectrum" not in missing:
            warnings.append("频域分析未产出结果：采样率无法识别或样本长度不足")
    else:
        # ---- 6. 指标 ----
        mtab = metrics.compute_metrics(clean, plan)
        extra = adapter.extra_metrics(clean, plan)
        timer.run("compute_metrics", lambda: {"generic": len(mtab.rows), "scenario": len(extra.rows)})

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
    ctx = ScenarioContext(clean, plan, mtab, spectra, tests, anomalies, change_points, extra, qrep)
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
        "mode": request.mode,
    }
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
        "lookahead": leak,
    }
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
