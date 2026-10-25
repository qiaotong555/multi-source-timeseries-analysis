# B：十份真实天气数据验证报告

生成时间：2026-10-24T14:23:33
运行矩阵：10 数据集 × 4 问题 × 2 模式 = **80 次运行**
模型：`deepseek-chat`；agent 固定 `guided` + `MTSA_PLANNER=rule`，token 上限 8000。
数据：`fixtures/real_weather/`（registry.json 记录来源与 SHA1）。

## 0. 结论

80 次运行 80 成功、0 失败；pipeline 证据覆盖率 1.000，agent 1.000；L1 质量事件 Recall 1.000（可定位 1.000），需求覆盖 1.000。剩余问题：L2 事件漏检；F6 两个源整列不报送，已在转换阶段剔除；L2 事件 Recall 0.071 / Precision 0.100（pipeline q1）。

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
| R01 | agent | 4/4 | 1.000 | 1.000 | — | — | 124.000 | 0.750 | 0.650 | 4517 |
| R02 | pipeline | 4/4 | 1.000 | 1.000 | — | — | 32.000 | 0.000 | — | — |
| R02 | agent | 4/4 | 1.000 | 1.000 | — | — | 32.000 | 0.000 | 0.650 | 4427 |
| R03 | pipeline | 4/4 | 1.000 | 1.000 | — | — | 14.000 | 1.000 | — | — |
| R03 | agent | 4/4 | 1.000 | 1.000 | — | — | 14.000 | 0.750 | 0.650 | 4453 |
| R04 | pipeline | 4/4 | 1.000 | 1.000 | — | — | 226.000 | 0.000 | — | — |
| R04 | agent | 4/4 | 1.000 | 1.000 | — | — | 226.000 | 0.000 | 0.650 | 4442 |
| R05 | pipeline | 4/4 | 1.000 | 1.000 | — | — | 156.000 | 2.000 | — | — |
| R05 | agent | 4/4 | 1.000 | 1.000 | — | — | 156.000 | 1.500 | 0.650 | 4734 |
| R06 | pipeline | 4/4 | 1.000 | 1.000 | 1.000 | 1.000 | 5.000 | 0.000 | — | — |
| R06 | agent | 4/4 | 1.000 | 1.000 | 1.000 | 1.000 | 5.000 | 0.000 | 0.650 | 4424 |
| R07 | pipeline | 4/4 | 1.000 | 1.000 | — | — | 5.000 | 0.000 | — | — |
| R07 | agent | 4/4 | 1.000 | 1.000 | — | — | 5.000 | 0.000 | 0.650 | 4385 |
| R08 | pipeline | 4/4 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | — | — |
| R08 | agent | 4/4 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.650 | 4319 |
| R09 | pipeline | 4/4 | 1.000 | 1.000 | — | — | 0.000 | 0.000 | — | — |
| R09 | agent | 4/4 | 1.000 | 1.000 | — | — | 0.000 | 0.000 | 0.650 | 4240 |
| R10 | pipeline | 4/4 | 1.000 | 1.000 | — | — | 51.000 | 2.000 | — | — |
| R10 | agent | 4/4 | 1.000 | 1.000 | — | — | 51.000 | 1.500 | 0.650 | 4465 |

### 3.1 宏平均与最差数据集

| 指标 | 宏平均 | 最差数据集 |
| --- | ---: | --- |
| 成功率 pipeline | 1.000 | 1.000（R01,R02,R03,R04,R05,R06,R07,R08,R09,R10） |
| 成功率 agent | 1.000 | 1.000（R01,R02,R03,R04,R05,R06,R07,R08,R09,R10） |
| 证据覆盖率 pipeline | 1.000 | 1.000（R01,R02,R03,R04,R05,R06,R07,R08,R09,R10） |
| 证据覆盖率 agent | 1.000 | 1.000（R01,R02,R03,R04,R05,R06,R07,R08,R09,R10） |
| 需求覆盖率 pipeline | 1.000 | 1.000（R01,R02,R03,R04,R05,R06,R07,R08,R09,R10） |
| 需求覆盖率 agent | 1.000 | 1.000（R01,R02,R03,R04,R05,R06,R07,R08,R09,R10） |
| L1 质量事件 Recall | 1.000 | 1.000（R06,R08） |
| L1 可定位 Recall | 1.000 | 1.000（R06,R08） |
| 事件 Recall（pipeline q1） | 0.071 | 0.000（R03） |
| 事件 Precision（pipeline q1） | 0.100 | 0.000（R03） |
| 事件 F1（pipeline q1） | 0.083 | 0.000（R03） |
| 事件 Recall（agent q1） | 0.071 | 0.000（R03） |
| 极端值事件覆盖率 | 0.500 | 0.429（R03） |
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
| R06 | missing_gap | wind_speed | NYC | 2018-09-13 00:00:00 | 4 | detected_localized | 命中 MISSING_VALUES |
| R06 | missing_gap | wind_speed | NYC | 2018-10-13 00:00:00 | 149 | detected_localized | 命中 MISSING_VALUES |
| R06 | missing_gap | wind_speed | NYC | 2020-01-22 00:00:00 | 5 | detected_localized | 命中 MISSING_VALUES |
| R06 | missing_gap | wind_speed | NYC | 2020-05-18 00:00:00 | 32 | detected_localized | 命中 MISSING_VALUES |
| R06 | missing_gap | wind_speed | NYC | 2021-09-15 00:00:00 | 8 | detected_localized | 命中 MISSING_VALUES |
| R06 | missing_gap | wind_speed | NYC | 2021-09-27 00:00:00 | 8 | detected_localized | 命中 MISSING_VALUES |
| R08 | sampling_gap | timestamp | BER_TEGEL | 2024-04-23 06:00:00 | 4 | detected_localized | 命中 SAMPLING_GAP |

> `reported_unlocalized` = 只报「这一列有缺失」，给不出缺口起止；`detected_localized` = 能对到具体区间。两者分开统计。

## 4b. L2 事件级异常评分（NOAA StormEvents）

> 事件窗口内（同列）有任一异常点即命中；episode = 同列相邻天异常聚类；
> Precision = 落在事件窗口内的 episode / 全部 episode。
> Agent 未执行 `detect_anomalies` 时标 N/A，不当成 0。

| 数据集 | 模式 | 事件数 | 命中 | Recall | Recall 95% CI | episode | 命中 episode | Precision | F1 | 状态 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| R03 | pipeline | 7 | 0 | 0.000 | [0.000, 0.354] | 9 | 0 | 0.000 | 0.000 | scored |
| R03 | agent | 7 | 0 | 0.000 | [0.000, 0.354] | 9 | 0 | 0.000 | 0.000 | scored |
| R06 | pipeline | 7 | 1 | 0.143 | [0.026, 0.513] | 5 | 1 | 0.200 | 0.167 | scored |
| R06 | agent | 7 | 1 | 0.143 | [0.026, 0.513] | 5 | 1 | 0.200 | 0.167 | scored |

### 4b.1 事件明细（pipeline q1）

| 数据集 | 列 | 事件窗口 | 官方类型 | 命中 |
| --- | --- | --- | --- | --- |
| R03 | temperature | 2018-08-19 ~ 2018-08-31 | Excessive Heat/Heat | 否 |
| R03 | temperature | 2019-07-22 ~ 2019-07-31 | Excessive Heat/Heat | 否 |
| R03 | temperature | 2020-07-03 ~ 2020-07-22 | Excessive Heat/Heat | 否 |
| R03 | temperature | 2021-06-29 ~ 2021-07-11 | Excessive Heat/Heat | 否 |
| R03 | temperature | 2022-07-08 ~ 2022-07-23 | Excessive Heat/Heat | 否 |
| R03 | temperature | 2023-07-01 ~ 2023-07-29 | Excessive Heat | 否 |
| R03 | temperature | 2024-06-09 ~ 2024-06-16 | Excessive Heat/Heat | 否 |
| R06 | wind_speed | 2018-03-02 ~ 2018-03-02 | High Wind | 否 |
| R06 | temperature | 2019-07-19 ~ 2019-07-21 | Excessive Heat/Heat | 否 |
| R06 | wind_speed | 2020-04-13 ~ 2020-04-13 | High Wind | 否 |
| R06 | rainfall | 2021-09-01 ~ 2021-09-02 | Flash Flood | 是 |
| R06 | rainfall | 2022-07-18 ~ 2022-07-18 | Flash Flood | 否 |
| R06 | wind_speed | 2023-08-13 ~ 2023-08-13 | Thunderstorm Wind | 否 |
| R06 | wind_speed | 2024-03-11 ~ 2024-03-11 | High Wind | 否 |

> L2 只覆盖 R03（Phoenix）与 R06（NYC）；其余 8 份数据没有可离线复核的官方事件，异常指标对这些数据保持 N/A。来源与选样规则见 `fixtures/real_weather/l2_sources/sources.json`。

## 4c. 极端值通道对 L2 事件的覆盖

> 绝对极端 = 超过本列经验 p99（或低于 p1）的观测点，与统计异常分开。
> 事件被覆盖 = 窗口内同列、同方向存在任一极端段。

| 数据集 | 事件 | 覆盖 | 覆盖率 |
| --- | ---: | ---: | ---: |
| R03 | 7 | 3 | 0.429 |
| R06 | 7 | 4 | 0.571 |

### 4c.1 事件明细

| 数据集 | 列 | 事件窗口 | 官方类型 | 方向 | 覆盖 |
| --- | --- | --- | --- | --- | --- |
| R03 | temperature | 2018-08-19 ~ 2018-08-31 | Excessive Heat/Heat | 高 | 否 |
| R03 | temperature | 2019-07-22 ~ 2019-07-31 | Excessive Heat/Heat | 高 | 否 |
| R03 | temperature | 2020-07-03 ~ 2020-07-22 | Excessive Heat/Heat | 高 | 是 |
| R03 | temperature | 2021-06-29 ~ 2021-07-11 | Excessive Heat/Heat | 高 | 否 |
| R03 | temperature | 2022-07-08 ~ 2022-07-23 | Excessive Heat/Heat | 高 | 是 |
| R03 | temperature | 2023-07-01 ~ 2023-07-29 | Excessive Heat | 高 | 是 |
| R03 | temperature | 2024-06-09 ~ 2024-06-16 | Excessive Heat/Heat | 高 | 否 |
| R06 | wind_speed | 2018-03-02 ~ 2018-03-02 | High Wind | 高 | 否 |
| R06 | temperature | 2019-07-19 ~ 2019-07-21 | Excessive Heat/Heat | 高 | 是 |
| R06 | wind_speed | 2020-04-13 ~ 2020-04-13 | High Wind | 高 | 否 |
| R06 | rainfall | 2021-09-01 ~ 2021-09-02 | Flash Flood | 高 | 是 |
| R06 | rainfall | 2022-07-18 ~ 2022-07-18 | Flash Flood | 高 | 是 |
| R06 | wind_speed | 2023-08-13 ~ 2023-08-13 | Thunderstorm Wind | 高 | 否 |
| R06 | wind_speed | 2024-03-11 ~ 2024-03-11 | High Wind | 高 | 是 |

> 极端值回答「有没有超过自身分布的两端」，不是「偏离趋势/季节」。两者不能互相替代；统计异常的事件 Recall 见 4b。

## 5. 关键发现

> **C1 修复后复跑说明**：首轮暴露的 F1（R08 季节性升温被误判为华氏度段，5,977 行被改写）已在 C1 中修复，本次复跑不再复现（R08 单位换算 0 次，峰值月恢复为 8 月）。修复前的原始数据与证据见 `spec/archive/项目承接已完成.md` 33.6 与提交 `3404b43`。

> **C2 修复后复跑说明**：首轮暴露的 F2（R05 三站点共享时间轴被报 17,568 条重复）已在 C2 中修复，本次复跑不再复现（duplicate_timestamps=0，清洗仍为 8,784 行 × 15 列）。见 `spec/archive/项目承接已完成.md` 第 35 章。

> **C3 修复后复跑说明**：首轮暴露的 F3（质量问题不进结论）与 F7（零检出无结论）已在 C3 中修复：Q2 的 20 次运行全部有 `数据口径` 结论（修复前 2/20），12 个零异常运行全部给出「未发现异常/变化点」（修复前 12 个静默），需求覆盖 0.800 → 1.000。见 `spec/archive/项目承接已完成.md` 第 36 章。

> **C4 修复后复跑说明**：首轮暴露的 F4（7 条 L1 只有 1 条能定位到区间）已在 C4 中修复：本次复跑 7/7 可定位（可定位 Recall 1.000，最差数据集 1.000）。`MISSING_VALUES` 现在带连续缺失段的起止与点数，完整清单见 `result.json` 的 `data.quality.issues[].runs`。见 `spec/archive/项目承接已完成.md` 第 37 章。

### F5 事件级 Recall 0.07：官方事件仍有漏检

按 NOAA StormEvents 的 14 条官方事件计分，pipeline q1 的事件级 Recall 为 0.071（阈值 0.50）。漏检事件见报告 4b.1 的「命中=否」行。

证据：

- `fixtures/real_weather/l2_verified_events.csv`

### F6 两个源整列不报送，已在转换阶段剔除

R07 缺 wind_speed；R10 缺 rainfall。这是数据源事实，不是系统漏检。

证据：

- `fixtures/real_weather/registry.json`

## 6. 验收检查（先写后测）

| 编号 | 检查 | 实际 | 要求 | 结果 |
| --- | --- | --- | --- | --- |
| R1 | R1 数据来源/时间/字段/行数/SHA1 完整 | 10/10 | 全部且 SHA1 与文件一致 | PASS |
| R2 | R2 覆盖 >=3 来源 / >=2 频率 / >=2 气候带 | 来源 4，频率 ['daily', 'hourly']，气候带 ['南半球', '寒带', '干旱', '温带', '热带'] | >=3 / >=2 / >=2 | PASS |
| R3 | R3 运行矩阵完整且每次都有明确状态 | 80/80 | 全部有状态 | PASS |
| R4 | R4 成功运行都有 report.md 且证据覆盖率 1.00 | 成功 80，缺报告 0，证据不足 0 | 缺 0 / 不足 0 | PASS |
| R5 | R5 L1 质量事件 Recall >= 0.80 | 1.0 | >= 0.80 | PASS |
| R6 | R6 未标注事件不计入 FN | L2 事件只用于 Recall/Precision；未标注异常不作为 FN，只影响 Precision | 口径检查 | PASS |
| R7 | R7 >=5 份数据有可解释的质量事实 | 5/10 | >=5 | PASS |
| R8 | R8 失败用例有错误码与节点 | 0 失败，0 缺错误码/节点 | 缺 0 | PASS |
| R9 | R9 报告含逐数据集/宏平均/最差数据集 | 由报告结构保证 | 三项齐全 | PASS |
| R10 | R10 分析代码改动仅限 C1/C2/C3 允许清单 | 变更 8 项，清单外 0 项（src/mtsa/anomaly.py,src/mtsa/engine.py,src/mtsa/models.py,src/mtsa/profiler.py,src/mtsa/quality.py,src/mtsa/report.py,src/mtsa/scenarios/base.py,src/mtsa/scenarios/weather.py） | 清单外 0 项 | PASS |
| R11 | R11 L2 事件清单可离线复核 | 14 条 / 2 数据集 / 3 列 / 5 类型 | >=12 / >=2 / >=3 / >=3 | PASS |
| R12 | R12 每条 L2 事件都能在落盘原始行中找到 | 77/77 EVENT_ID 命中 | 全部命中 | PASS |
| R13 | R13 事件级 Recall >= 0.50（pipeline q1） | 0.0714 | >= 0.50 | FAIL |
| R14 | R14 极端值通道覆盖 L2 事件 >= 0.50 | 7/14 | >= 0.50 | PASS |

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
