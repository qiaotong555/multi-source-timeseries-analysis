# 方案 2：独立结论门禁报告

生成时间：2026-10-24T12:45:15
数据：`fixtures/weather_eval_v2.csv`（SHA1 `2ee8077e5ddc8f446a793e04729ad50dd85c1eac`）
标签：`fixtures/labels_weather_eval_v2.csv`（SHA1 `3afbe2a3ee7a1cc9e3d11ee7465c090fc92da460`）
真值：`fixtures/truth_weather_eval_v2.json`（SHA1 `5ffb2e5eb29df3889fa2bf0c4a6887db19439d92`）
阈值：运行前冻结；agent 非劣于 pipeline − 0.10。

## 数据规格

| 项 | 值 |
| --- | --- |
| 行数 | 17520 |
| 站点 | STA_A, STA_B |
| 列 | temperature, humidity, pressure, wind_speed, rainfall |
| 时间范围 | 2024-01-01 00:00:00 → 2024-12-30 23:00:00 |
| 异常点 / 异常事件 | 100 / 30 |
| 变化点 | 2 |
| 缺失缺口标签 | 1 |

> v2 是独立于旧 fixture 的新数据，但生成器在构造阶段经过修订（修复缺失遮蔽、去掉变化点叠加、把注入幅度调整到可检测区间）。因此本报告验证的是**固定版 v2** 上的表现，不能宣称是完全未看过的隐藏盲测。检测算法与阈值均未因本数据修改。

## 结论

**通过**

读数注意：

- 压线通过（无余量）：pipeline.changepoint_f1 = 0.5000（阈值 0.50）、agent.changepoint_f1 = 0.5000（阈值 0.50）。
- agent 终止原因：budget_exhausted 3 次，model_finished 1 次。
- `budget_exhausted` 指 guided 模式下限之外的**追加步数**用尽，不是 token 预算耗尽（见 `rejects[].kind=scope_exceeded` 与 `tokens`）。
- agent 未执行工具：run_stat_tests 3 次。结论级指标相同不等于工具覆盖相同，应结合 `data.agent.coverage` 阅读。

## 分模式指标

| 指标 | pipeline | agent |
| --- | --- | --- |
| 异常事件 Recall（未执行计入分母） | 0.9667 [0.917, 0.987] | 0.9667 [0.917, 0.987] |
| 异常事件 Precision | 1.0000 [0.968, 1.000] | 1.0000 [0.968, 1.000] |
| 异常事件 F1（未执行计入分母） | 0.9831 | 0.9831 |
| 异常点级 F1（参考，无门槛） | 0.9130 | 0.9130 |
| 变化点 F1（未执行计入分母） | 0.5000 | 0.5000 |
| 趋势正确率 | 1.0000 [0.676, 1.000] | 1.0000 [0.676, 1.000] |
| 季节正确率 | 1.0000 [0.676, 1.000] | 1.0000 [0.676, 1.000] |
| 趋势已评分率 | 1.0000 | 1.0000 |
| 季节已评分率 | 1.0000 | 1.0000 |
| 结论需求覆盖率 | 1.0000 [0.796, 1.000] | 1.0000 [0.796, 1.000] |
| 证据覆盖率 | 1.0000 [0.510, 1.000] | 1.0000 [0.510, 1.000] |
| 成功率 | 1.0000 [0.510, 1.000] | 1.0000 [0.510, 1.000] |

置信区间为 Wilson 95%；F1 由 Precision/Recall 区间端点不自洽，故只给点估计。

## 逐用例

| 用例 | 模式 | 状态 | 步数 | 终止 | 结论数 | 证据覆盖 | 需求覆盖 | 异常点 TP/FP/FN | 异常事件 TP/FP/FN | 变化点 TP/FP/FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_full_1 | pipeline | succeeded | — | — | 47 | 1.000 | 1.00 | 84/0/16 | 29/0/1 | 2/4/0 |
| v2_full_1 | agent | succeeded | 4 | budget_exhausted | 47 | 1.000 | 1.00 | 84/0/16 | 29/0/1 | 2/4/0 |
| v2_full_2 | pipeline | succeeded | — | — | 47 | 1.000 | 1.00 | 84/0/16 | 29/0/1 | 2/4/0 |
| v2_full_2 | agent | succeeded | 5 | model_finished | 47 | 1.000 | 1.00 | 84/0/16 | 29/0/1 | 2/4/0 |
| v2_full_3 | pipeline | succeeded | — | — | 47 | 1.000 | 1.00 | 84/0/16 | 29/0/1 | 2/4/0 |
| v2_full_3 | agent | succeeded | 4 | budget_exhausted | 47 | 1.000 | 1.00 | 84/0/16 | 29/0/1 | 2/4/0 |
| v2_full_4 | pipeline | succeeded | — | — | 47 | 1.000 | 1.00 | 84/0/16 | 29/0/1 | 2/4/0 |
| v2_full_4 | agent | succeeded | 4 | budget_exhausted | 47 | 1.000 | 1.00 | 84/0/16 | 29/0/1 | 2/4/0 |

## 阈值检查

| 检查 | 实际 | 阈值 | 结果 |
| --- | --- | --- | --- |
| pipeline.anomaly_event_recall | 0.9667 | >= 0.8 | PASS |
| pipeline.anomaly_event_f1 | 0.9831 | >= 0.4 | PASS |
| pipeline.changepoint_f1 | 0.5000 | >= 0.5 | PASS |
| pipeline.trend_accuracy | 1.0000 | >= 0.8 | PASS |
| pipeline.seasonal_accuracy | 1.0000 | >= 0.8 | PASS |
| pipeline.trend_scored_rate | 1.0000 | >= 0.9 | PASS |
| pipeline.seasonal_scored_rate | 1.0000 | >= 0.9 | PASS |
| pipeline.need_coverage | 1.0000 | >= 0.9 | PASS |
| pipeline.evidence_coverage | 1.0000 | >= 1.0 | PASS |
| pipeline.success_rate | 1.0000 | >= 0.9 | PASS |
| agent.anomaly_event_recall | 0.9667 | >= 0.8 | PASS |
| agent.anomaly_event_f1 | 0.9831 | >= 0.4 | PASS |
| agent.changepoint_f1 | 0.5000 | >= 0.5 | PASS |
| agent.trend_accuracy | 1.0000 | >= 0.8 | PASS |
| agent.seasonal_accuracy | 1.0000 | >= 0.8 | PASS |
| agent.trend_scored_rate | 1.0000 | >= 0.9 | PASS |
| agent.seasonal_scored_rate | 1.0000 | >= 0.9 | PASS |
| agent.need_coverage | 1.0000 | >= 0.9 | PASS |
| agent.evidence_coverage | 1.0000 | >= 1.0 | PASS |
| agent.success_rate | 1.0000 | >= 0.9 | PASS |
| agent_non_inferior.anomaly_event_f1 | 0.9831 | >= 0.8831 | PASS |
| agent_non_inferior.changepoint_f1 | 0.5000 | >= 0.4 | PASS |
| agent_non_inferior.trend_accuracy | 1.0000 | >= 0.9 | PASS |
| agent_non_inferior.seasonal_accuracy | 1.0000 | >= 0.9 | PASS |

## 复现

```bash
PYTHONPATH=src python -m evals.conclusion_gate_eval
PYTHONPATH=src python -m evals.conclusion_gate_eval --pipeline-only
```
