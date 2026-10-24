# B：十份真实天气数据验证报告

生成时间：2026-10-24T11:44:46
运行矩阵：10 数据集 × 4 问题 × 2 模式 = **80 次运行**
模型：`deepseek-chat`；agent 固定 `guided` + `MTSA_PLANNER=rule`，token 上限 8000。
数据：`fixtures/real_weather/`（registry.json 记录来源与 SHA1）。

## 0. 结论

80 次运行 80 成功、0 失败；pipeline 证据覆盖率 1.000，agent 1.000；L1 质量事件 Recall 1.000（可定位 0.500），需求覆盖 0.800。主要问题不是「跑不动」，而是口径与推断缺陷：多站点重复时间戳误报；质量问题不进结论；质量事件不可定位。

## 1. 十份数据

| ID | 数据 | 来源 | 频率 | 行数 | 站点 | 分析列 | 源不提供 |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| R01 | berlin_hourly_2024 | Open-Meteo Archive API | hourly | 8784 | BER | temperature,humidity,pressure,wind_speed,rainfall | — |
| R02 | singapore_hourly_2024 | Open-Meteo Archive API | hourly | 8784 | SIN | temperature,humidity,pressure,wind_speed,rainfall | — |
| R03 | phoenix_daily_2015_2024 | Open-Meteo Archive API | daily | 3653 | PHX | temperature,humidity,pressure,wind_speed,rainfall | — |
| R04 | reykjavik_hourly_2024 | Open-Meteo Archive API | hourly | 8784 | REK | temperature,humidity,pressure,wind_speed,rainfall | — |
| R05 | three_cities_hourly_2024 | Open-Meteo Archive API | hourly | 26352 | SIN,SYD,TYO | temperature,humidity,pressure,wind_speed,rainfall | — |
| R06 | nyc_central_park_daily_2018_2024 | NOAA GHCN-Daily | daily | 2557 | NYC | temperature,rainfall,wind_speed | — |
| R07 | melbourne_daily_2018_2024 | NOAA GHCN-Daily | daily | 2557 | MEL | temperature,rainfall | wind_speed |
| R08 | berlin_tegel_hourly_2024 | DWD Open Data | hourly | 8780 | BER_TEGEL | temperature,humidity | — |
| R09 | munich_hourly_2024 | DWD Open Data | hourly | 8784 | MUC | temperature,humidity | — |
| R10 | toronto_hourly_2024 | ECCC Bulk Data | hourly | 8783 | YYZ | temperature,humidity,pressure,wind_speed | rainfall |

> 「源不提供」是按整列为空剔除的字段，属于数据源事实，不计入质量事件。

## 2. 运行矩阵

| 数据集 | q1P | q1A | q2P | q2A | q3P | q3A | q4P | q4A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R01 | OK | OK | OK | OK | OK | OK | OK | OK |
| R02 | OK | OK | OK | OK | OK | OK | OK | OK |
| R03 | OK | OK | OK | OK | OK | OK | OK | OK |
| R04 | OK | OK | OK | OK | OK | OK | OK | OK |
| R05 | OK | OK | OK | OK | OK | OK | OK | OK |
| R06 | OK | OK | OK | OK | OK | OK | OK | OK |
| R07 | OK | OK | OK | OK | OK | OK | OK | OK |
| R08 | OK | OK | OK | OK | OK | OK | OK | OK |
| R09 | OK | OK | OK | OK | OK | OK | OK | OK |
| R10 | OK | OK | OK | OK | OK | OK | OK | OK |

P=pipeline，A=agent；每格是该数据集在该问题下的运行状态。

## 3. 逐数据集指标

| 数据集 | 模式 | 成功 | 证据覆盖 | 需求覆盖 | L1 Recall | 可定位 | 异常均值 | 变化点均值 | 工具覆盖 | token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R01 | pipeline | 4/4 | 1.000 | 1.000 | — | — | 124.000 | 1.000 | — | — |
| R01 | agent | 4/4 | 1.000 | 1.000 | — | — | 124.000 | 0.750 | 0.650 | 4518 |
| R02 | pipeline | 4/4 | 1.000 | 0.833 | — | — | 32.000 | 0.000 | — | — |
| R02 | agent | 4/4 | 1.000 | 0.833 | — | — | 32.000 | 0.000 | 0.650 | 4424 |
| R03 | pipeline | 4/4 | 1.000 | 1.000 | — | — | 14.000 | 1.000 | — | — |
| R03 | agent | 4/4 | 1.000 | 1.000 | — | — | 14.000 | 0.750 | 0.650 | 4458 |
| R04 | pipeline | 4/4 | 1.000 | 0.833 | — | — | 226.000 | 0.000 | — | — |
| R04 | agent | 4/4 | 1.000 | 0.833 | — | — | 226.000 | 0.000 | 0.650 | 4440 |
| R05 | pipeline | 4/4 | 1.000 | 1.000 | — | — | 156.000 | 2.000 | — | — |
| R05 | agent | 4/4 | 1.000 | 1.000 | — | — | 156.000 | 1.500 | 0.650 | 4776 |
| R06 | pipeline | 4/4 | 1.000 | 0.833 | 1.000 | 0.000 | 5.000 | 0.000 | — | — |
| R06 | agent | 4/4 | 1.000 | 0.833 | 1.000 | 0.000 | 5.000 | 0.000 | 0.600 | 4170 |
| R07 | pipeline | 4/4 | 1.000 | 0.833 | — | — | 5.000 | 0.000 | — | — |
| R07 | agent | 4/4 | 1.000 | 0.833 | — | — | 5.000 | 0.000 | 0.600 | 4134 |
| R08 | pipeline | 4/4 | 1.000 | 0.333 | 1.000 | 1.000 | 0.000 | 0.000 | — | — |
| R08 | agent | 4/4 | 1.000 | 0.333 | 1.000 | 1.000 | 0.000 | 0.000 | 0.650 | 4324 |
| R09 | pipeline | 4/4 | 1.000 | 0.333 | — | — | 0.000 | 0.000 | — | — |
| R09 | agent | 4/4 | 1.000 | 0.333 | — | — | 0.000 | 0.000 | 0.650 | 4246 |
| R10 | pipeline | 4/4 | 1.000 | 1.000 | — | — | 51.000 | 2.000 | — | — |
| R10 | agent | 4/4 | 1.000 | 1.000 | — | — | 51.000 | 1.500 | 0.650 | 4456 |

### 3.1 宏平均与最差数据集

| 指标 | 宏平均 | 最差数据集 |
| --- | ---: | --- |
| 成功率 pipeline | 1.000 | 1.000（R01,R02,R03,R04,R05,R06,R07,R08,R09,R10） |
| 成功率 agent | 1.000 | 1.000（R01,R02,R03,R04,R05,R06,R07,R08,R09,R10） |
| 证据覆盖率 pipeline | 1.000 | 1.000（R01,R02,R03,R04,R05,R06,R07,R08,R09,R10） |
| 证据覆盖率 agent | 1.000 | 1.000（R01,R02,R03,R04,R05,R06,R07,R08,R09,R10） |
| 需求覆盖率 pipeline | 0.800 | 0.333（R08,R09） |
| 需求覆盖率 agent | 0.800 | 0.333（R08,R09） |
| L1 质量事件 Recall | 1.000 | 1.000（R06,R08） |
| L1 可定位 Recall | 0.500 | 0.000（R06） |
| 两模式异常 Jaccard | 1.000 | 1.000（R01,R02,R03,R04,R05,R06,R07,R08,R09,R10） |
| 两模式变化点 Jaccard | 0.900 | 0.750（R01,R03,R05,R10） |

宏平均 = 10 份数据各自指标的等权平均；没有把所有行拼成一个总分。

### 3.2 两模式一致性（不是正确性）

| 数据集 | 异常 Jaccard | 变化点 Jaccard | 说明 |
| --- | ---: | ---: | --- |
| R01 | 1.000 | 0.750 | q1: 124 vs 124; q2: 124 vs 124; q3: 124 vs 124; q4: 124 vs 124 |
| R02 | 1.000 | 1.000 | q1: 32 vs 32; q2: 32 vs 32; q3: 32 vs 32; q4: 32 vs 32 |
| R03 | 1.000 | 0.750 | q1: 14 vs 14; q2: 14 vs 14; q3: 14 vs 14; q4: 14 vs 14 |
| R04 | 1.000 | 1.000 | q1: 226 vs 226; q2: 226 vs 226; q3: 226 vs 226; q4: 226 vs 226 |
| R05 | 1.000 | 0.750 | q1: 156 vs 156; q2: 156 vs 156; q3: 156 vs 156; q4: 156 vs 156 |
| R06 | 1.000 | 1.000 | q1: 5 vs 5; q2: 5 vs 5; q3: 5 vs 5; q4: 5 vs 5 |
| R07 | 1.000 | 1.000 | q1: 5 vs 5; q2: 5 vs 5; q3: 5 vs 5; q4: 5 vs 5 |
| R08 | 1.000 | 1.000 | q1: 0 vs 0; q2: 0 vs 0; q3: 0 vs 0; q4: 0 vs 0 |
| R09 | 1.000 | 1.000 | q1: 0 vs 0; q2: 0 vs 0; q3: 0 vs 0; q4: 0 vs 0 |
| R10 | 1.000 | 0.750 | q1: 51 vs 51; q2: 51 vs 51; q3: 51 vs 51; q4: 51 vs 51 |

Jaccard = 两模式检出集合的交集/并集；1.0 表示逐点相同。它只说明两种模式是否一致，不说明哪一边正确。

## 4. L1 标注与匹配明细

| 数据集 | 种类 | 列 | 站点 | 起点 | 点数 | 系统状态 | 说明 |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| R06 | missing_gap | wind_speed | NYC | 2018-09-13 00:00:00 | 4 | reported_unlocalized | 命中 MISSING_VALUES |
| R06 | missing_gap | wind_speed | NYC | 2018-10-13 00:00:00 | 149 | reported_unlocalized | 命中 MISSING_VALUES |
| R06 | missing_gap | wind_speed | NYC | 2020-01-22 00:00:00 | 5 | reported_unlocalized | 命中 MISSING_VALUES |
| R06 | missing_gap | wind_speed | NYC | 2020-05-18 00:00:00 | 32 | reported_unlocalized | 命中 MISSING_VALUES |
| R06 | missing_gap | wind_speed | NYC | 2021-09-15 00:00:00 | 8 | reported_unlocalized | 命中 MISSING_VALUES |
| R06 | missing_gap | wind_speed | NYC | 2021-09-27 00:00:00 | 8 | reported_unlocalized | 命中 MISSING_VALUES |
| R08 | sampling_gap | timestamp | BER_TEGEL | 2024-04-23 06:00:00 | 4 | detected_localized | 命中 SAMPLING_GAP |

> `reported_unlocalized` = 只报「这一列有缺失」，给不出缺口起止；`detected_localized` = 能对到具体区间。两者分开统计。

## 5. 关键发现

> **C1 修复后复跑说明**：首轮暴露的 F1（R08 季节性升温被误判为华氏度段，5,977 行被改写）已在 C1 中修复，本次复跑不再复现（R08 单位换算 0 次，峰值月恢复为 8 月）。修复前的原始数据与证据见 `spec/项目承接已完成.md` 33.6 与提交 `3404b43`。

### F2 R05：多站点长表被误报重复时间戳

该文件是 3 个站点共享同一时间轴的长表。质量层按整表 `timestamp` 判重，把「同一时刻的不同站点」算成重复；按 `(station, timestamp)` 判重实际为 0 条。清洗阶段本身正确：`group_columns=['station']`，透视后为 8,784 行 × 15 列，三个站点分别出结论。缺陷只在质量报告的计数口径。

证据：

- `fixtures/real_weather/three_cities_hourly_2024/data.csv`

### F3 质量问题只进报告第 2 节，不进结论

问题 2 明确问「缺失、重复、乱序、采样缺口、单位变化」，但 20/20 次成功运行里没有一条 `数据口径` 结论；质量信息只出现在 `report.md` 的「2. 数据质量报告」表格和 `warnings` 中。因此结论级指标无法评价数据质量问答。

证据：

- `evals/real_data_report.md 第 2 节`

### F4 L1 质量事件 7 条，能定位到区间的只有 1 条

`MISSING_VALUES` 只给列的缺失率与计数，没有缺口起止；`DUPLICATE_TIMESTAMPS`、`OUT_OF_ORDER` 也只给计数。只有 `SAMPLING_GAP` 带起止时间。所以 Recall 高不代表可追溯，可定位 Recall 才是更严格的读法。

证据：

- `fixtures/real_weather/*/labels.csv`
- `src/mtsa/quality.py`

### F5 L2 为空：异常 P/R/F1 本轮不计分

没有取得可离线复核的官方极端事件清单，`l2_verified_events.csv` 只有表头。异常与变化点只报数量（L3），不把无标签候选当成误报，也不当成命中。

证据：

- `fixtures/real_weather/l2_verified_events.csv`

### F6 两个源整列不报送，已在转换阶段剔除

R07 缺 wind_speed；R10 缺 rainfall。这是数据源事实，不是系统漏检。

证据：

- `fixtures/real_weather/registry.json`

### F7 零检出时没有「未发现异常」结论（15 次运行）

这些运行确实执行了 `detect_anomalies` 且结果为 0 条，但报告结论里没有任何「异常检测」条目，用户只看到趋势/季节/相关。问「找出异常」时这等于没有回答；在 rule 计划下问题意图也不会裁剪分析项，Q1/Q3 的结论集合基本相同。涉及：R08/q1、R08/q2、R08/q3、R08/q4、R09/q1、R09/q2、R09/q3、R09/q4。

证据：

- `evals/real_data_eval/*/q3_*/result.json`

## 6. 验收检查（先写后测）

| 编号 | 检查 | 实际 | 要求 | 结果 |
| --- | --- | --- | --- | --- |
| R1 | R1 数据来源/时间/字段/行数/SHA1 完整 | 10/10 | 全部且 SHA1 与文件一致 | PASS |
| R2 | R2 覆盖 >=3 来源 / >=2 频率 / >=2 气候带 | 来源 4，频率 ['daily', 'hourly']，气候带 ['南半球', '寒带', '干旱', '温带', '热带'] | >=3 / >=2 / >=2 | PASS |
| R3 | R3 运行矩阵完整且每次都有明确状态 | 80/80 | 全部有状态 | PASS |
| R4 | R4 成功运行都有 report.md 且证据覆盖率 1.00 | 成功 80，缺报告 0，证据不足 0 | 缺 0 / 不足 0 | PASS |
| R5 | R5 L1 质量事件 Recall >= 0.80 | 1.0 | >= 0.80 | PASS |
| R6 | R6 未标注事件不计入 FP/FN | L3 只报数量；无 L2 标签的异常/变化点指标标 N/A | 口径检查 | PASS |
| R7 | R7 >=5 份数据有可解释的质量事实 | 5/10 | >=5 | PASS |
| R8 | R8 失败用例有错误码与节点 | 0 失败，0 缺错误码/节点 | 缺 0 | PASS |
| R9 | R9 报告含逐数据集/宏平均/最差数据集 | 由报告结构保证 | 三项齐全 | PASS |
| R10 | R10 分析代码改动仅限 C1 允许清单 | 变更 3 项，清单外 0 项（src/mtsa/engine.py,src/mtsa/quality.py,src/mtsa/scenarios/weather.py） | 清单外 0 项 | PASS |

## 7. 复现

```bash
PYTHONPATH=src python scripts/build_real_labels.py
PYTHONPATH=src python -m evals.real_data_eval --pipeline-only
PYTHONPATH=src python -m evals.real_data_eval --agent-only
PYTHONPATH=src python -m evals.real_data_eval
```

## 8. 局限

- L2 为空：没有可离线复核的官方事件清单，异常 P/R/F1 标 N/A，不用 L3 冒充真值；
- L1 只有 7 条标签且集中在少数数据集，Wilson 区间宽，只能说明「已标注事件被报到」，不能证明质量层普遍可靠；
- Open-Meteo 的 5 份是再分析数据，不是气象站原始观测；
- 日级数据按 365 天周期，样本 <5000 行时引擎给出小样本警告，属预期行为。
