# 鲁棒性矩阵报告

生成时间：2026-10-24T11:35:34
口径：固定 `MTSA_PLANNER=rule`；Agent 用例只测边界，不调用真实模型。

## 汇总

| 用例 | 通过 | 失败 | 崩溃 | 缺 result.json | 非有限 JSON |
| --- | --- | --- | --- | --- | --- |
| 37 | 37 | 0 | 0 | 1 | 0 |

> 缺 result.json 的用例：q_empty（CLI 参数拒绝按设计不落盘，不计为失败）。

| 组 | 总数 | 通过 | 失败 |
| --- | --- | --- | --- |
| 导入 | 7 | 7 | 0 |
| 时间轴 | 7 | 7 | 0 |
| 数值 | 9 | 9 | 0 |
| 编码 | 5 | 5 | 0 |
| 规模 | 2 | 2 | 0 |
| 问题 | 4 | 4 | 0 |
| Agent | 3 | 3 | 0 |

## 用例矩阵

| 编号 | 组 | 输入 | 预期 | 实际 | 错误码 | 警告 | 结论 | 无证据 | 耗时(ms) | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| load_missing | 导入 | 文件不存在 | failed | failed | DATA_LOAD_FAILED | 0 | 0 | 0 | 0.8 | PASS |
| load_empty | 导入 | 0 字节文件 | failed | failed | DATA_LOAD_FAILED | 0 | 0 | 0 | 0.8 | PASS |
| load_header_only | 导入 | 只有表头 | failed | failed | DATA_LOAD_FAILED | 0 | 0 | 0 | 13.0 | PASS |
| load_malformed | 导入 | 畸形 CSV（列数不一致） | failed | failed | DATA_LOAD_FAILED | 0 | 0 | 0 | 6.2 | PASS |
| load_unsupported | 导入 | 不支持的 .txt | failed | failed | DATA_LOAD_FAILED | 0 | 0 | 0 | 0.8 | PASS |
| load_directory | 导入 | 把目录当输入文件 | failed | failed | DATA_LOAD_FAILED | 0 | 0 | 0 | 0.7 | PASS |
| load_json_as_csv | 导入 | JSON 内容但扩展名 .csv | failed | failed | DATA_LOAD_FAILED | 0 | 0 | 0 | 5.7 | PASS |
| time_missing | 时间轴 | 没有时间列 | failed | failed | TIME_AXIS_NOT_FOUND | 0 | 0 | 0 | 18.1 | PASS |
| time_unparseable | 时间轴 | 时间列全部不可解析 | failed | failed | TIME_AXIS_NOT_FOUND | 0 | 0 | 0 | 38.3 | PASS |
| time_mixed_tz | 时间轴 | 混合时区偏移 | any | failed | TIME_AXIS_NOT_FOUND | 0 | 0 | 0 | 23.0 | PASS |
| time_all_same | 时间轴 | 所有时间戳相同 | any | succeeded | — | 4 | 1 | 0 | 3314.2 | PASS |
| time_out_of_order | 时间轴 | 时间戳乱序 | succeeded | succeeded | — | 2 | 14 | 0 | 3039.9 | PASS |
| time_duplicates | 时间轴 | 重复时间戳 | succeeded | succeeded | — | 2 | 14 | 0 | 3635.8 | PASS |
| time_gap | 时间轴 | 48 小时缺口 | succeeded | succeeded | — | 2 | 13 | 0 | 3376.2 | PASS |
| num_none | 数值 | 只有时间与文本列 | failed | failed | EMPTY_NUMERIC_COLUMN | 0 | 0 | 0 | 28.3 | PASS |
| num_all_nan | 数值 | 目标列全 NaN | any | succeeded | — | 2 | 20 | 0 | 3356.0 | PASS |
| num_constant | 数值 | 目标列恒定 | succeeded | succeeded | — | 3 | 18 | 0 | 3511.9 | PASS |
| num_inf | 数值 | 含 +inf / -inf | succeeded | succeeded | — | 3 | 26 | 0 | 3588.7 | PASS |
| num_huge | 数值 | 1e12 量级 | any | succeeded | — | 2 | 26 | 0 | 3661.1 | PASS |
| num_negative_rain | 数值 | 降雨为负 | succeeded | succeeded | — | 2 | 26 | 0 | 3603.1 | PASS |
| num_missing_50 | 数值 | 温度 50% 缺失 | succeeded | succeeded | — | 3 | 25 | 0 | 3584.1 | PASS |
| num_one_row | 数值 | 1 行样本 | any | succeeded | — | 4 | 1 | 0 | 2820.2 | PASS |
| num_ten_rows | 数值 | 10 行样本 | any | succeeded | — | 4 | 7 | 0 | 2833.7 | PASS |
| enc_gbk | 编码 | GBK 编码中文站点 | succeeded | succeeded | — | 2 | 13 | 0 | 2912.4 | PASS |
| enc_bom | 编码 | UTF-8 BOM | succeeded | succeeded | — | 2 | 26 | 0 | 3896.9 | PASS |
| schema_duplicate_columns | 编码 | 重复列名 | any | succeeded | — | 2 | 26 | 0 | 3589.4 | PASS |
| schema_weird_names | 编码 | 列名含换行与指令文本 | any | succeeded | — | 2 | 26 | 0 | 4237.6 | PASS |
| schema_no_target | 编码 | 没有场景可识别的数值列 | any | failed | SCENE_NOT_DETECTED | 0 | 0 | 0 | 34.4 | PASS |
| scale_over_limit | 规模 | 超过行数上限 | failed | failed | DATASET_TOO_LARGE | 0 | 0 | 0 | 12.5 | PASS |
| scale_at_limit | 规模 | 恰好等于行数上限 | succeeded | succeeded | — | 3 | 13 | 0 | 3222.6 | PASS |
| q_empty | 问题 | 空问题（CLI） | cli | — | — | 0 | 0 | 0 | 1.4 | PASS |
| q_long | 问题 | 3000 字长问题 | succeeded | succeeded | — | 2 | 13 | 0 | 3022.0 | PASS |
| q_forecast | 问题 | 越界：预测未来 | succeeded | succeeded | — | 2 | 13 | 0 | 3464.8 | PASS |
| q_injection | 问题 | 提示注入 | succeeded | succeeded | — | 2 | 13 | 0 | 2997.5 | PASS |
| agent_stock | Agent | agent + 股票场景 | failed | failed | AGENT_SCENE_UNSUPPORTED | 1 | 0 | 0 | 100.6 | PASS |
| agent_no_model | Agent | agent + 无模型配置 | failed | failed | AGENT_UNAVAILABLE | 2 | 0 | 0 | 304.9 | PASS |
| agent_bad_input | Agent | agent + 无有效数值列 | failed | failed | EMPTY_NUMERIC_COLUMN | 0 | 0 | 0 | 29.2 | PASS |

## 失败与发现

本轮无失败用例。
## 复现

```bash
PYTHONPATH=src python -m evals.robustness_eval
# 只跑某一组
PYTHONPATH=src python -m evals.robustness_eval --only load,time
```
