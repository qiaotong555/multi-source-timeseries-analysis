# D3 数据扩容验收

生成时间：2026-10-24T15:31:59
新增：7 份公开数据（南美 3 / 非洲 3 / 北美 1），不修改分析阈值。

## 门禁

| 编号 | 检查 | 结果 | 证据 |
| --- | --- | --- | --- |
| D3-1 | 新增 7 份、≥2 大洲、海拔≥1500m ≥4、来源可复核、旧 SHA1 不变 | PASS | 7 份 / 北美、南美、非洲 / 高海拔 4 份 / 来源 齐全 / 旧 SHA1 全同 True |
| D3-2 | R17 的 wind_gusts 是独立目标列 | PASS | R17=['temperature', 'humidity', 'pressure', 'wind_speed', 'wind_gusts', 'rainfall']；其余 6 份无阵风列 |
| D3-3 | 7×4=28 次全部成功，证据覆盖率 1.000 | PASS | 28/28 成功，最低证据覆盖 1.0 |
| D3-4 | 4 条 L2 大风事件窗口内阵风有 p99、小时平均风速没有 | FAIL | 阵风命中 3/4、小时平均风速命中 3/4（R17 阵风 p99=12.86 m/s） |
| D3-4b | （事后）分辨率对比：R06 日均 1/4 vs R17 小时 3/4 | INFO（不计门禁） | R06 日均风速覆盖 1/4（D2 已发布），R17 小时阵风/平均风速各 3/4；唯一未覆盖的 2023-08-13 为 Thunderstorm Wind，Central Park 当日阵风峰值仅 8.2 m/s |
| D3-2b | 极端段未被上限截断（时间覆盖完整） | PASS | 截断的数据集/列：无 |

**存在未通过项**

## 新增数据

| 编号 | 站点 | 大洲 | 行数 | 海拔 (m) | 列 |
| --- | --- | --- | ---: | ---: | --- |
| R11 | la_paz_hourly_2024 | 南美 | 8784 | 3767.0 | temperature、humidity、pressure、wind_speed、rainfall |
| R12 | quito_hourly_2024 | 南美 | 8784 | 2903.0 | temperature、humidity、pressure、wind_speed、rainfall |
| R13 | addis_ababa_hourly_2024 | 非洲 | 8784 | 2444.0 | temperature、humidity、pressure、wind_speed、rainfall |
| R14 | nairobi_hourly_2024 | 非洲 | 8784 | 1668.0 | temperature、humidity、pressure、wind_speed、rainfall |
| R15 | cairo_hourly_2024 | 非洲 | 8784 | 29.0 | temperature、humidity、pressure、wind_speed、rainfall |
| R16 | ushuaia_hourly_2024 | 南美 | 8784 | 52.0 | temperature、humidity、pressure、wind_speed、rainfall |
| R17 | nyc_asos_hourly_2018_2024 | 北美 | 61098 | — | temperature、humidity、pressure、wind_speed、wind_gusts、rainfall |

## 目标列识别

| 数据集 | 目标列 |
| --- | --- |
| R11 | temperature、humidity、pressure、wind_speed、rainfall |
| R12 | temperature、humidity、pressure、wind_speed、rainfall |
| R13 | temperature、humidity、pressure、wind_speed、rainfall |
| R14 | temperature、humidity、pressure、wind_speed、rainfall |
| R15 | temperature、humidity、pressure、wind_speed、rainfall |
| R16 | temperature、humidity、pressure、wind_speed、rainfall |
| R17 | temperature、humidity、pressure、wind_speed、wind_gusts、rainfall |

## 阵风可观测性（R17，观测，非再分析）

R17 阵风 p99 = 12.8611 m/s；4 条 L2 大风事件窗口内：

| 事件日 | 类型 | 阵风有 p99 观测 | 平均风速有 p99 观测 |
| --- | --- | --- | --- |
| 2018-03-02 | High Wind | 是 | 是 |
| 2020-04-13 | High Wind | 是 | 是 |
| 2023-08-13 | Thunderstorm Wind | 否 | 否 |
| 2024-03-11 | High Wind | 是 | 是 |

命中：阵风 3/4，平均风速 3/4。

## 运行矩阵（7 × 4）

| 数据集 | 问题 | 状态 | 结论数 | 证据覆盖 | 极端项 | 质量码 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| R11 | q1 | succeeded | 28 | 1.0 | 8 | OUTLIER_CANDIDATE、POSSIBLE_UNIT_CHANGE |
| R11 | q2 | succeeded | 28 | 1.0 | 8 | OUTLIER_CANDIDATE、POSSIBLE_UNIT_CHANGE |
| R11 | q3 | succeeded | 28 | 1.0 | 8 | OUTLIER_CANDIDATE、POSSIBLE_UNIT_CHANGE |
| R11 | q4 | succeeded | 28 | 1.0 | 8 | OUTLIER_CANDIDATE、POSSIBLE_UNIT_CHANGE |
| R12 | q1 | succeeded | 29 | 1.0 | 8 | LEVEL_SHIFT、OUTLIER_CANDIDATE |
| R12 | q2 | succeeded | 29 | 1.0 | 8 | LEVEL_SHIFT、OUTLIER_CANDIDATE |
| R12 | q3 | succeeded | 29 | 1.0 | 8 | LEVEL_SHIFT、OUTLIER_CANDIDATE |
| R12 | q4 | succeeded | 29 | 1.0 | 8 | LEVEL_SHIFT、OUTLIER_CANDIDATE |
| R13 | q1 | succeeded | 27 | 1.0 | 8 | LEVEL_SHIFT、OUTLIER_CANDIDATE |
| R13 | q2 | succeeded | 27 | 1.0 | 8 | LEVEL_SHIFT、OUTLIER_CANDIDATE |
| R13 | q3 | succeeded | 27 | 1.0 | 8 | LEVEL_SHIFT、OUTLIER_CANDIDATE |
| R13 | q4 | succeeded | 27 | 1.0 | 8 | LEVEL_SHIFT、OUTLIER_CANDIDATE |
| R14 | q1 | succeeded | 30 | 1.0 | 8 | LEVEL_SHIFT、OUTLIER_CANDIDATE |
| R14 | q2 | succeeded | 30 | 1.0 | 8 | LEVEL_SHIFT、OUTLIER_CANDIDATE |
| R14 | q3 | succeeded | 30 | 1.0 | 8 | LEVEL_SHIFT、OUTLIER_CANDIDATE |
| R14 | q4 | succeeded | 30 | 1.0 | 8 | LEVEL_SHIFT、OUTLIER_CANDIDATE |
| R15 | q1 | succeeded | 29 | 1.0 | 7 | LEVEL_SHIFT、OUTLIER_CANDIDATE、SAMPLING_GAP |
| R15 | q2 | succeeded | 29 | 1.0 | 7 | LEVEL_SHIFT、OUTLIER_CANDIDATE、SAMPLING_GAP |
| R15 | q3 | succeeded | 29 | 1.0 | 7 | LEVEL_SHIFT、OUTLIER_CANDIDATE、SAMPLING_GAP |
| R15 | q4 | succeeded | 29 | 1.0 | 7 | LEVEL_SHIFT、OUTLIER_CANDIDATE、SAMPLING_GAP |
| R16 | q1 | succeeded | 29 | 1.0 | 8 | OUTLIER_CANDIDATE |
| R16 | q2 | succeeded | 29 | 1.0 | 8 | OUTLIER_CANDIDATE |
| R16 | q3 | succeeded | 29 | 1.0 | 8 | OUTLIER_CANDIDATE |
| R16 | q4 | succeeded | 29 | 1.0 | 8 | OUTLIER_CANDIDATE |
| R17 | q1 | succeeded | 38 | 1.0 | 9 | MISSING_VALUES、OUTLIER_CANDIDATE、SAMPLING_GAP |
| R17 | q2 | succeeded | 38 | 1.0 | 9 | MISSING_VALUES、OUTLIER_CANDIDATE、SAMPLING_GAP |
| R17 | q3 | succeeded | 38 | 1.0 | 9 | MISSING_VALUES、OUTLIER_CANDIDATE、SAMPLING_GAP |
| R17 | q4 | succeeded | 38 | 1.0 | 9 | MISSING_VALUES、OUTLIER_CANDIDATE、SAMPLING_GAP |

## 复现

```bash
PYTHONPATH=src python -m evals.d3_expansion_eval
```
