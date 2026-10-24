# -*- coding: utf-8 -*-
"""数据契约：所有输入、输出、错误和证据均通过 Pydantic 校验。

字段命名与取值见 doc/接口与字段.md；已发布字段不得直接改名。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

# ---------------- 常量 ----------------
TASK_ID = "multi_source_timeseries_analysis"
CAPABILITY_ID = "science.multi_source_timeseries_analysis"
MAX_RERUN = 2
MAX_TOOL_CALLS = 64
NODE_TIMEOUT_SEC = 120
DEFAULT_SEED = 20260927
SCENE_CONFIDENCE_THRESHOLD = 0.34
SUPPORTED_FORMATS = (".csv", ".xlsx", ".xls", ".parquet", ".npy", ".mat", ".h5", ".hdf5")

Status = Literal[
    "queued", "running", "waiting_for_input", "cancel_requested",
    "succeeded", "failed", "cancelled",
]
SceneId = Literal["weather", "stock"]

ERR_DATA_LOAD_FAILED = "DATA_LOAD_FAILED"
ERR_TIME_AXIS_NOT_FOUND = "TIME_AXIS_NOT_FOUND"
ERR_EMPTY_NUMERIC_COLUMN = "EMPTY_NUMERIC_COLUMN"
ERR_SCENE_NOT_DETECTED = "SCENE_NOT_DETECTED"
ERR_SAMPLING_TOO_COARSE = "SAMPLING_TOO_COARSE"
ERR_LOOKAHEAD_VIOLATION = "LOOKAHEAD_VIOLATION"
ERR_NODE_TIMEOUT = "NODE_TIMEOUT"
ERR_DATASET_TOO_LARGE = "DATASET_TOO_LARGE"
ERR_INTERNAL_ERROR = "INTERNAL_ERROR"
# Agent 模式专用错误。分开定义而不是都塞进 INTERNAL_ERROR：
# 调用方需要区分「环境没配好」「场景不支持」「模型什么都没做」这三种可操作信息。
ERR_AGENT_UNAVAILABLE = "AGENT_UNAVAILABLE"
ERR_AGENT_SCENE_UNSUPPORTED = "AGENT_SCENE_UNSUPPORTED"
ERR_AGENT_NO_RESULT = "AGENT_NO_RESULT"
# 重放专用：源文件不可用，或重放结果与原轨迹不一致
ERR_REPLAY_INVALID = "REPLAY_INVALID"
ERR_REPLAY_MISMATCH = "REPLAY_MISMATCH"
# E2：价目表缺失或非法。不静默按 0 计价，必须让调用方看到。
ERR_PRICE_TABLE = "PRICE_TABLE_INVALID"


class MtsaModel(BaseModel):
    """允许承载 DataFrame 等任意类型的基类配置。"""
    model_config = ConfigDict(arbitrary_types_allowed=True)


# ---------------- 错误 ----------------
class ErrorDetail(MtsaModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


# ---------------- 请求 ----------------
class AnalysisRequest(MtsaModel):
    request_id: str
    input_path: str
    question: str
    scene_id: Optional[SceneId] = None
    time_column: Optional[str] = None
    value_columns: Optional[list[str]] = None
    group_column: Optional[str] = None
    unit: Optional[str] = None
    sampling_interval: Optional[str] = None
    # 覆盖场景适配器给出的计划参数（如 anomaly_threshold）。键必须是计划中已存在的参数，
    # 未知键直接报错——静默忽略会让「调参」变成假象。
    param_overrides: dict[str, Any] = Field(default_factory=dict)
    seed: int = DEFAULT_SEED
    output_dir: Optional[str] = None
    # 运行模式。本轮 Agent 只支持天气场景；pipeline 是默认值，保持既有行为不变。
    mode: Literal["pipeline", "agent", "auto"] = "pipeline"
    # Agent 最大决策步数。默认 8，足以覆盖天气场景的 5 个工具并留出纠错余量。
    agent_max_steps: int = 8
    # 范围下限模式（T2.11）。guided：关键词下限 + 达标前不得结束 + 最多 1 步扩展；
    # free：范围完全由模型判断（T2.2 原始行为，步数会波动）。
    agent_scope: Literal["guided", "free"] = "guided"
    # 下限之外允许追加的步数，仅 guided 生效。
    agent_max_extra_steps: int = 1
    # Agent 累计 token 上限（0 = 不限制）。只约束模型调用，不约束工具执行。
    # E2 成本语义：价目表路径（也可用环境变量 MTSA_PRICE_TABLE）。
    # 未配置时不做金额换算；配置了但读不到会直接失败，不静默跳过。
    price_table: Optional[str] = None
    agent_max_tokens: int = 0
    # 预测（F）：默认关闭，避免改变既有报告；开启后附带 walk-forward 回测。
    forecast: bool = False
    forecast_horizon: Optional[int] = None
    # 从一次 agent 运行的 result.json 重放；设置后不调用模型，按记录顺序重跑工具。
    replay_from: Optional[str] = None


# ---------------- 导入与摘要 ----------------
class DatasetSpec(MtsaModel):
    path: str
    format: str
    time_column: Optional[str] = None
    value_columns: Optional[list[str]] = None
    group_column: Optional[str] = None
    unit: Optional[str] = None
    sampling_interval: Optional[str] = None


class RawDataset(MtsaModel):
    frame: pd.DataFrame
    spec: DatasetSpec
    file_hash: str
    row_count: int
    column_dtypes: dict[str, str]


class TimeAxisInfo(MtsaModel):
    column: str
    parse_ratio: float
    start: Optional[str] = None
    end: Optional[str] = None
    is_monotonic: bool
    median_interval_seconds: Optional[float] = None
    duplicated_count: int = 0
    out_of_order_count: int = 0


class SamplingInfo(MtsaModel):
    inferred_freq: Optional[str] = None
    interval_stable: bool
    gap_count: int = 0
    gaps: list[dict[str, Any]] = Field(default_factory=list)


class DatasetProfile(MtsaModel):
    time_axis: Optional[TimeAxisInfo] = None
    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    group_columns: list[str] = Field(default_factory=list)
    sampling: SamplingInfo = Field(default_factory=SamplingInfo)
    row_count: int
    column_count: int


# ---------------- 质量 ----------------
class QualityIssue(MtsaModel):
    code: str
    severity: Literal["error", "warn", "info"]
    column: Optional[str] = None
    group: Optional[str] = None       # 问题所属的分组取值（如站点），多实体数据必填
    row_label: Optional[int] = None   # 触发位置对应的原始行标签，供清洗阶段定位
    start: Optional[str] = None
    end: Optional[str] = None
    count: int = 0
    detail: str = ""
    # 可定位区间：连续段 [{start, end, count}]，按时间排序；多实体时每段额外带 group。
    # 最多保留 MAX_ISSUE_RUNS 段，截断前总段数见 run_count。
    runs: list[dict[str, Any]] = Field(default_factory=list)
    run_count: int = 0
    runs_truncated: bool = False


class QualityReport(MtsaModel):
    issues: list[QualityIssue] = Field(default_factory=list)
    missing_ratio: dict[str, float] = Field(default_factory=dict)
    duplicate_timestamps: int = 0
    # 多实体数据按 (实体, 时间) 判重后的分组计数；空表示无重复或无分组
    duplicate_by_group: dict[str, int] = Field(default_factory=dict)
    out_of_order: int = 0
    gap_count: int = 0
    anomaly_candidate_count: int = 0
    as_of: str = ""


# ---------------- 清洗 ----------------
class CleanDataset(MtsaModel):
    frame: pd.DataFrame
    operations: list[dict[str, Any]] = Field(default_factory=list)
    dropped_rows: int = 0
    imputed_cells: int = 0
    # 被插值填补的采样点：这些点的值来自填补而非观测，不能作为异常点判定依据。
    # 注意这是**行级并集**：任一行只要有任一列被插补就会计入。
    imputed_index: list[str] = Field(default_factory=list)
    # 逐列的插补位置（列名 → 时间戳列表）。行级并集会把「别列的缺失」算到这一列头上：
    # R17 上 61098 行里有 4930+ 个时间戳被别的列连坐，导致 75% 的阵风 p99 观测被误剔除。
    imputed_columns: dict[str, list[str]] = Field(default_factory=dict)
    unit_conversions: list[dict[str, Any]] = Field(default_factory=list)


# ---------------- 分析计划 ----------------
class AnalysisPlan(MtsaModel):
    scene_id: SceneId
    scene_confidence: float
    question: str
    target_columns: list[str]
    group_column: Optional[str] = None
    resample_rule: Optional[str] = None
    analyses: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    not_supported: list[str] = Field(default_factory=list)
    # 计划从哪来：rule（适配器规则）/ llm（模型生成）/ cache（命中缓存）/ mock（测试桩）
    plan_source: str = "rule"
    planner_version: str = "v1"


# ---------------- 结果 ----------------
class MetricRow(MtsaModel):
    metric_id: str
    name: str
    column: str
    group: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    window: Optional[str] = None
    method: str
    formula: str


class MetricTable(MtsaModel):
    rows: list[MetricRow] = Field(default_factory=list)
    generated_by: str
    code_ref: str


class SpectrumResult(MtsaModel):
    column: str
    sampling_rate_hz: Optional[float] = None
    freqs: list[float] = Field(default_factory=list)
    power: list[float] = Field(default_factory=list)
    dominant_freq: Optional[float] = None
    dominant_period: Optional[str] = None
    band_power: dict[str, float] = Field(default_factory=dict)
    method: str = "welch"


class TestResultRow(MtsaModel):
    test_id: str
    name: str
    column: str
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    critical_value: Optional[float] = None
    reject_null: Optional[bool] = None
    conclusion: str
    assumptions: str = ""


class AnomalyPoint(MtsaModel):
    timestamp: str
    index: int
    column: str
    value: float
    score: float
    method: str


class ChangePoint(MtsaModel):
    timestamp: str
    index: int
    column: str
    kind: Literal["mean", "variance", "unit"]
    statistic: float
    before: Optional[float] = None
    after: Optional[float] = None
    confidence: str = "medium"


class Artifact(MtsaModel):
    artifact_id: str
    kind: Literal["figure_static", "figure_interactive", "table_csv", "report_md", "script"]
    path: str
    title: str
    fields: list[str] = Field(default_factory=list)
    sha1: str = ""


class Evidence(MtsaModel):
    evidence_id: str
    claim: str
    metric_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    code_ref: str
    caveat: str = ""


class Conclusion(MtsaModel):
    conclusion_id: str
    category: str
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    limitations: str = ""


class EventRecord(MtsaModel):
    node: str
    started_at: str
    elapsed_ms: int
    status: Literal["succeeded", "failed", "skipped"]
    summary: dict[str, Any] = Field(default_factory=dict)


class AnalysisResult(MtsaModel):
    request_id: str
    task_id: str = TASK_ID
    run_id: str
    status: Status
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    data: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    datasets: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    events: list[EventRecord] = Field(default_factory=list)
    error: Optional[ErrorDetail] = None
    conclusions: list[Conclusion] = Field(default_factory=list)
