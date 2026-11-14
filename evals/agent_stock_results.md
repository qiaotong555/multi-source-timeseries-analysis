# 股票 Agent 评估（E4）

> 生成：2026-11-14T20:11:08　场景：`stock`　模式：`agent`　计划器：`rule`

## 汇总

| 指标 | 实测 | 阈值 | 结论 |
| --- | --- | --- | --- |
| success_rate | 1.0 | >= 1.0 | PASS |
| gate_consistency | 1.0 | >= 1.0 | PASS |
| scenario_layer_rate | 1.0 | >= 1.0 | PASS |
| need_coverage | 1.0 | >= 1.0 | PASS |

平均步数 2.75；终止原因 {'budget_exhausted': 4}

## 逐例

| 用例 | 状态 | 步数 | 已执行工具 | 门控 | 问题 |
| --- | --- | --- | --- | --- | --- |
| s_return | succeeded | 2 | compute_metrics、run_stat_tests | 一致 | — |
| s_anomaly | succeeded | 3 | detect_anomalies、compute_metrics、detect_change_points | 一致 | — |
| s_changepoint | succeeded | 2 | compute_metrics、detect_change_points | 一致 | — |
| s_mixed | succeeded | 4 | compute_metrics、run_stat_tests、detect_anomalies、detect_change_points | 一致 | — |

口径：「门控一致」要求 —— 工具执行过，结论不得写「未执行」；工具没执行，结论必须写「未执行」。两条都查，防止把没查说成没发现。
