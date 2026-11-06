# F：预测回测评估

生成时间：2026-11-06T11:21:09
数据：18 份（v2 fixture + 17 份公开真实天气数据）
方法：seasonal_naive（基线）/ moving_average（窗=周期）/ drift（末值+平均增量）；按 walk-forward MASE 选型，候选须相对基线下降 ≥5% 才允许替换
区间：最近 12 个 origin 的误差按 horizon 归一、分 4 箱、取 conformal 有限样本修正分位

## 门禁

| 项 | 实际 | 阈值 | 结果 |
| --- | --- | --- | --- |
| F3 汇总 90% 区间覆盖率 | 0.8799 | [0.80, 0.98] | PASS |
| F4 MASE≤1.0 的列占比 | 0.7234（68/94） | ≥ 0.50 | PASS |
| F4 未优于基线的列都有选型说明 | 是 | 0 缺失 | PASS |
| F6 零膨胀列条件区间覆盖率 ∈ [0.80, 0.98]（非零点 ≥ 100） | 13 列参与，全部在带内 | 每列在带内 | PASS |
| F5 零膨胀列事件概率预报 BSS > 0（事件点 ≥ 100） | 13 列参与，存在非正 BSS | 每列 > 0 | FAIL |

**存在未通过项**

## 汇总

| 项 | 值 |
| --- | --- |
| 参与评估的列 | 94 |
| 样本不足未预测的列 | 0 |
| 零膨胀列（零值 ≥ 50%） | 18 |
| 覆盖率（按点加权） | 0.8799 |
| 覆盖率（逐列最小 / 最大） | 0.8069 / 0.9944 |
| 覆盖率参与点数 | 125160 |
| 判定分布 | 带内 76、带外 0、零膨胀不适用 18 |
| 覆盖率带外（Wilson 95% 区间与带不相交） | 无 |
| 零膨胀不判区间覆盖率的列 | 18 列（全部为降水类）；整体覆盖率 0.845 ~ 0.994，非零点 3 ~ 682；逐列明细见 `coverage_not_applicable` |
| 选中方法 MASE≤1.0 | 68 / 94 = 0.7234 |
| 选中方法优于基线 | 40 / 94 = 0.4255 |
| 选中的方法 | drift 7 列、moving_average 33 列、seasonal_naive 54 列 |

## 条件区间（零膨胀列：真下雨时的覆盖率）

整体覆盖率在这类列上不算判据，看「观测非零」时刻的覆盖率。`overall` 是用整体区间统计的（必然偏窄），`conditional` 是用只按非零误差标定的区间。

| 数据集 | 列 | 非零点 | 整体区间覆盖率 | 条件区间覆盖率 |
| --- | --- | --- | --- | --- |
| v2_fixture | rainfall@STA_A | 117 | 0.5043 | **0.9231** |
| R01_berlin_hourly_2024 | rainfall | 262 | 0.7061 | **0.9122** |
| R02_singapore_hourly_2024 | rainfall | 592 | 0.8125 | **0.9358** |
| R04_reykjavik_hourly_2024 | rainfall | 435 | 0.7494 | **0.8943** |
| R05_three_cities_hourly_2024 | rainfall@SIN | 592 | 0.8125 | **0.9358** |
| R05_three_cities_hourly_2024 | rainfall@SYD | 230 | 0.5348 | **0.8913** |
| R06_nyc_central_park_daily_2018_2024 | rainfall | 138 | 0.7536 | **0.913** |
| R07_melbourne_daily_2018_2024 | rainfall | 146 | 0.8288 | **0.9178** |
| R11_la_paz_hourly_2024 | rainfall | 314 | 0.7006 | **0.9172** |
| R12_quito_hourly_2024 | rainfall | 682 | 0.7815 | **0.8783** |
| R14_nairobi_hourly_2024 | rainfall | 494 | 0.8381 | **0.8401** |
| R16_ushuaia_hourly_2024 | rainfall | 527 | 0.8046 | **0.8748** |
| R17_nyc_asos_hourly_2018_2024 | rainfall | 207 | 0.6135 | **0.8454** |

## 事件概率预报（零膨胀列）

阈值 0.1；主模型 = climatology 与 persistence_short 的等权平均；BSS 相对 climatology。事件点少于 100 的列不判，因为「基本不报」也能把 BSS 抬高。

| 数据集 | 列 | 事件点 | 实测发生率 | BSS | 等级 | ECE | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2_fixture | rainfall@STA_A | 104 | 0.0722 | 0.002 | very_weak | 0.0077 | 参与门禁 |
| v2_fixture | rainfall@STA_B | 94 | 0.0653 | 0.0074 | insufficient_events | 0.0027 | 未判（事件点不足） |
| R01_berlin_hourly_2024 | rainfall | 165 | 0.1146 | 0.0221 | very_weak | 0.0227 | 参与门禁 |
| R02_singapore_hourly_2024 | rainfall | 443 | 0.3076 | -0.0328 | none | 0.0676 | 参与门禁 |
| R03_phoenix_daily_2015_2024 | rainfall | 49 | 0.1167 | 0.023 | insufficient_events | 0.044 | 未判（事件点不足） |
| R04_reykjavik_hourly_2024 | rainfall | 313 | 0.2174 | 0.0666 | weak | 0.0464 | 参与门禁 |
| R05_three_cities_hourly_2024 | rainfall@SIN | 443 | 0.3076 | -0.0328 | none | 0.0676 | 参与门禁 |
| R05_three_cities_hourly_2024 | rainfall@SYD | 145 | 0.1007 | 0.0753 | weak | 0.038 | 参与门禁 |
| R05_three_cities_hourly_2024 | rainfall@TYO | 50 | 0.0347 | 0.1105 | insufficient_events | 0.0888 | 未判（事件点不足） |
| R06_nyc_central_park_daily_2018_2024 | rainfall | 138 | 0.3286 | 0.0023 | very_weak | 0.0474 | 参与门禁 |
| R07_melbourne_daily_2018_2024 | rainfall | 146 | 0.3476 | 0.0141 | very_weak | 0.0262 | 参与门禁 |
| R11_la_paz_hourly_2024 | rainfall | 191 | 0.1326 | -0.0025 | none | 0.0592 | 参与门禁 |
| R12_quito_hourly_2024 | rainfall | 527 | 0.366 | 0.0005 | very_weak | 0.1333 | 参与门禁 |
| R13_addis_ababa_hourly_2024 | rainfall | 1 | 0.0007 | 0.3734 | insufficient_events | 0.1061 | 未判（事件点不足） |
| R14_nairobi_hourly_2024 | rainfall | 213 | 0.1479 | -0.002 | none | 0.0458 | 参与门禁 |
| R15_cairo_hourly_2024 | rainfall | 8 | 0.0056 | -0.0057 | insufficient_events | 0.0034 | 未判（事件点不足） |
| R16_ushuaia_hourly_2024 | rainfall | 334 | 0.2319 | 0.0128 | very_weak | 0.0363 | 参与门禁 |
| R17_nyc_asos_hourly_2018_2024 | rainfall | 126 | 0.0875 | 0.0235 | very_weak | 0.0081 | 参与门禁 |

## 逐数据集

| 数据集 | 行数 | 周期 | horizon | 列 | 选中方法 | MASE | 基线 MASE | 覆盖率 | 零膨胀 | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_fixture | 8760 | 24 | 24 | temperature@STA_A | seasonal_naive | 0.9626 | 0.9626 | 0.8826 | 否 | 带内 |
| v2_fixture | 8760 | 24 | 24 | temperature@STA_B | seasonal_naive | 0.9935 | 0.9935 | 0.8896 | 否 | 带内 |
| v2_fixture | 8760 | 24 | 24 | humidity@STA_A | moving_average | 0.9234 | 0.9839 | 0.8583 | 否 | 带内 |
| v2_fixture | 8760 | 24 | 24 | humidity@STA_B | moving_average | 1.0064 | 1.073 | 0.8736 | 否 | 带内 |
| v2_fixture | 8760 | 24 | 24 | pressure@STA_A | moving_average | 0.7887 | 0.9614 | 0.8764 | 否 | 带内 |
| v2_fixture | 8760 | 24 | 24 | pressure@STA_B | moving_average | 0.708 | 0.8638 | 0.884 | 否 | 带内 |
| v2_fixture | 8760 | 24 | 24 | wind_speed@STA_A | moving_average | 0.8101 | 0.9437 | 0.8819 | 否 | 带内 |
| v2_fixture | 8760 | 24 | 24 | wind_speed@STA_B | moving_average | 0.8072 | 0.9441 | 0.8701 | 否 | 带内 |
| v2_fixture | 8760 | 24 | 24 | rainfall@STA_A | seasonal_naive | 0.6496 | 0.6496 | 0.9243 | 是 | 零膨胀不适用 |
| v2_fixture | 8760 | 24 | 24 | rainfall@STA_B | seasonal_naive | 1.9976 | 1.9976 | 0.9118 | 是 | 零膨胀不适用 |
| R01_berlin_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 0.8704 | 0.8704 | 0.8562 | 否 | 带内 |
| R01_berlin_hourly_2024 | 8784 | 24 | 24 | humidity | seasonal_naive | 0.7645 | 0.7645 | 0.8667 | 否 | 带内 |
| R01_berlin_hourly_2024 | 8784 | 24 | 24 | pressure | drift | 0.8428 | 1.1994 | 0.8778 | 否 | 带内 |
| R01_berlin_hourly_2024 | 8784 | 24 | 24 | wind_speed | moving_average | 0.8165 | 0.9416 | 0.8639 | 否 | 带内 |
| R01_berlin_hourly_2024 | 8784 | 24 | 24 | rainfall | moving_average | 0.8218 | 0.8944 | 0.8931 | 是 | 零膨胀不适用 |
| R02_singapore_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 1.0539 | 1.0539 | 0.9021 | 否 | 带内 |
| R02_singapore_hourly_2024 | 8784 | 24 | 24 | humidity | seasonal_naive | 1.1209 | 1.1209 | 0.8931 | 否 | 带内 |
| R02_singapore_hourly_2024 | 8784 | 24 | 24 | pressure | seasonal_naive | 0.8882 | 0.8882 | 0.8535 | 否 | 带内 |
| R02_singapore_hourly_2024 | 8784 | 24 | 24 | wind_speed | moving_average | 0.8762 | 0.9536 | 0.8861 | 否 | 带内 |
| R02_singapore_hourly_2024 | 8784 | 24 | 24 | rainfall | seasonal_naive | 1.1335 | 1.1335 | 0.9042 | 是 | 零膨胀不适用 |
| R03_phoenix_daily_2015_2024 | 3653 | 7 | 7 | temperature | moving_average | 0.8186 | 0.9685 | 0.8929 | 否 | 带内 |
| R03_phoenix_daily_2015_2024 | 3653 | 7 | 7 | humidity | moving_average | 0.7747 | 0.9218 | 0.8762 | 否 | 带内 |
| R03_phoenix_daily_2015_2024 | 3653 | 7 | 7 | pressure | moving_average | 0.7945 | 1.0075 | 0.881 | 否 | 带内 |
| R03_phoenix_daily_2015_2024 | 3653 | 7 | 7 | wind_speed | moving_average | 0.8539 | 1.0653 | 0.8929 | 否 | 带内 |
| R03_phoenix_daily_2015_2024 | 3653 | 7 | 7 | rainfall | seasonal_naive | 0.9585 | 0.9585 | 0.8857 | 是 | 零膨胀不适用 |
| R04_reykjavik_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 1.4028 | 1.4028 | 0.8757 | 否 | 带内 |
| R04_reykjavik_hourly_2024 | 8784 | 24 | 24 | humidity | moving_average | 0.9361 | 1.0737 | 0.859 | 否 | 带内 |
| R04_reykjavik_hourly_2024 | 8784 | 24 | 24 | pressure | drift | 1.1296 | 1.348 | 0.8069 | 否 | 带内 |
| R04_reykjavik_hourly_2024 | 8784 | 24 | 24 | wind_speed | moving_average | 0.979 | 1.1985 | 0.8597 | 否 | 带内 |
| R04_reykjavik_hourly_2024 | 8784 | 24 | 24 | rainfall | moving_average | 1.0406 | 1.1994 | 0.8701 | 是 | 零膨胀不适用 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | temperature@SIN | seasonal_naive | 1.0539 | 1.0539 | 0.9021 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | temperature@SYD | seasonal_naive | 1.1396 | 1.1396 | 0.8569 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | temperature@TYO | seasonal_naive | 0.9426 | 0.9426 | 0.8576 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | humidity@SIN | seasonal_naive | 1.1209 | 1.1209 | 0.8931 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | humidity@SYD | seasonal_naive | 1.1922 | 1.1922 | 0.8375 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | humidity@TYO | seasonal_naive | 1.0014 | 1.0014 | 0.85 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | pressure@SIN | seasonal_naive | 0.8882 | 0.8882 | 0.8535 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | pressure@SYD | drift | 0.8807 | 1.1751 | 0.8549 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | pressure@TYO | drift | 0.8553 | 1.0603 | 0.8417 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | wind_speed@SIN | moving_average | 0.8762 | 0.9536 | 0.8861 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | wind_speed@SYD | moving_average | 0.9086 | 0.9929 | 0.9028 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | wind_speed@TYO | moving_average | 0.577 | 0.6931 | 0.8562 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | rainfall@SIN | seasonal_naive | 1.1335 | 1.1335 | 0.9042 | 是 | 零膨胀不适用 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | rainfall@SYD | moving_average | 0.8191 | 0.8624 | 0.8799 | 是 | 零膨胀不适用 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | rainfall@TYO | moving_average | 0.6156 | 0.6562 | 0.9146 | 是 | 零膨胀不适用 |
| R06_nyc_central_park_daily_2018_2024 | 2557 | 7 | 7 | temperature | moving_average | 0.8236 | 0.9947 | 0.9024 | 否 | 带内 |
| R06_nyc_central_park_daily_2018_2024 | 2557 | 7 | 7 | wind_speed | moving_average | 0.8443 | 1.0163 | 0.8762 | 否 | 带内 |
| R06_nyc_central_park_daily_2018_2024 | 2557 | 7 | 7 | rainfall | moving_average | 0.9001 | 0.9662 | 0.8452 | 是 | 零膨胀不适用 |
| R07_melbourne_daily_2018_2024 | 2557 | 7 | 7 | temperature | moving_average | 0.7776 | 0.9932 | 0.8738 | 否 | 带内 |
| R07_melbourne_daily_2018_2024 | 2557 | 7 | 7 | rainfall | moving_average | 0.7683 | 0.8423 | 0.8881 | 是 | 零膨胀不适用 |
| R08_berlin_tegel_hourly_2024 | 8780 | 24 | 24 | temperature | seasonal_naive | 0.8343 | 0.8343 | 0.8681 | 否 | 带内 |
| R08_berlin_tegel_hourly_2024 | 8780 | 24 | 24 | humidity | seasonal_naive | 0.6914 | 0.6914 | 0.8903 | 否 | 带内 |
| R09_munich_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 0.9304 | 0.9304 | 0.8674 | 否 | 带内 |
| R09_munich_hourly_2024 | 8784 | 24 | 24 | humidity | seasonal_naive | 0.6932 | 0.6932 | 0.8757 | 否 | 带内 |
| R10_toronto_hourly_2024 | 8783 | 24 | 24 | temperature | seasonal_naive | 1.0774 | 1.0774 | 0.8743 | 否 | 带内 |
| R10_toronto_hourly_2024 | 8783 | 24 | 24 | humidity | seasonal_naive | 0.9028 | 0.9028 | 0.8938 | 否 | 带内 |
| R10_toronto_hourly_2024 | 8783 | 24 | 24 | pressure | drift | 0.8871 | 1.1314 | 0.8583 | 否 | 带内 |
| R10_toronto_hourly_2024 | 8783 | 24 | 24 | wind_speed | moving_average | 0.9054 | 1.0645 | 0.8715 | 否 | 带内 |
| R11_la_paz_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 0.8853 | 0.8853 | 0.8771 | 否 | 带内 |
| R11_la_paz_hourly_2024 | 8784 | 24 | 24 | humidity | seasonal_naive | 0.8915 | 0.8915 | 0.8826 | 否 | 带内 |
| R11_la_paz_hourly_2024 | 8784 | 24 | 24 | pressure | seasonal_naive | 0.8204 | 0.8204 | 0.8674 | 否 | 带内 |
| R11_la_paz_hourly_2024 | 8784 | 24 | 24 | wind_speed | seasonal_naive | 0.9413 | 0.9413 | 0.8792 | 否 | 带内 |
| R11_la_paz_hourly_2024 | 8784 | 24 | 24 | rainfall | seasonal_naive | 1.2532 | 1.2532 | 0.8986 | 是 | 零膨胀不适用 |
| R12_quito_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 0.9752 | 0.9752 | 0.8924 | 否 | 带内 |
| R12_quito_hourly_2024 | 8784 | 24 | 24 | humidity | seasonal_naive | 1.0042 | 1.0042 | 0.8868 | 否 | 带内 |
| R12_quito_hourly_2024 | 8784 | 24 | 24 | pressure | seasonal_naive | 0.927 | 0.927 | 0.875 | 否 | 带内 |
| R12_quito_hourly_2024 | 8784 | 24 | 24 | wind_speed | seasonal_naive | 1.0102 | 1.0102 | 0.8958 | 否 | 带内 |
| R12_quito_hourly_2024 | 8784 | 24 | 24 | rainfall | seasonal_naive | 1.095 | 1.095 | 0.8924 | 是 | 零膨胀不适用 |
| R13_addis_ababa_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 0.8204 | 0.8204 | 0.8854 | 否 | 带内 |
| R13_addis_ababa_hourly_2024 | 8784 | 24 | 24 | humidity | seasonal_naive | 1.0214 | 1.0214 | 0.8757 | 否 | 带内 |
| R13_addis_ababa_hourly_2024 | 8784 | 24 | 24 | pressure | seasonal_naive | 0.8846 | 0.8846 | 0.8924 | 否 | 带内 |
| R13_addis_ababa_hourly_2024 | 8784 | 24 | 24 | wind_speed | seasonal_naive | 0.8254 | 0.8254 | 0.8986 | 否 | 带内 |
| R13_addis_ababa_hourly_2024 | 8784 | 24 | 24 | rainfall | seasonal_naive | 0.3813 | 0.3813 | 0.9944 | 是 | 零膨胀不适用 |
| R14_nairobi_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 1.0049 | 1.0049 | 0.9132 | 否 | 带内 |
| R14_nairobi_hourly_2024 | 8784 | 24 | 24 | humidity | seasonal_naive | 1.0428 | 1.0428 | 0.8896 | 否 | 带内 |
| R14_nairobi_hourly_2024 | 8784 | 24 | 24 | pressure | seasonal_naive | 0.8491 | 0.8491 | 0.9042 | 否 | 带内 |
| R14_nairobi_hourly_2024 | 8784 | 24 | 24 | wind_speed | seasonal_naive | 0.9987 | 0.9987 | 0.9083 | 否 | 带内 |
| R14_nairobi_hourly_2024 | 8784 | 24 | 24 | rainfall | moving_average | 1.0811 | 1.1595 | 0.9014 | 是 | 零膨胀不适用 |
| R15_cairo_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 0.6864 | 0.6864 | 0.8576 | 否 | 带内 |
| R15_cairo_hourly_2024 | 8784 | 24 | 24 | humidity | seasonal_naive | 0.8379 | 0.8379 | 0.8646 | 否 | 带内 |
| R15_cairo_hourly_2024 | 8784 | 24 | 24 | pressure | seasonal_naive | 0.8931 | 0.8931 | 0.8306 | 否 | 带内 |
| R15_cairo_hourly_2024 | 8784 | 24 | 24 | wind_speed | seasonal_naive | 0.8765 | 0.8765 | 0.8882 | 否 | 带内 |
| R15_cairo_hourly_2024 | 8784 | 24 | 24 | rainfall | seasonal_naive | 0.6307 | 0.6307 | 0.9826 | 是 | 零膨胀不适用 |
| R16_ushuaia_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 0.9698 | 0.9698 | 0.8688 | 否 | 带内 |
| R16_ushuaia_hourly_2024 | 8784 | 24 | 24 | humidity | seasonal_naive | 1.137 | 1.137 | 0.8688 | 否 | 带内 |
| R16_ushuaia_hourly_2024 | 8784 | 24 | 24 | pressure | drift | 0.8209 | 0.9475 | 0.8715 | 否 | 带内 |
| R16_ushuaia_hourly_2024 | 8784 | 24 | 24 | wind_speed | moving_average | 1.0384 | 1.2006 | 0.8736 | 否 | 带内 |
| R16_ushuaia_hourly_2024 | 8784 | 24 | 24 | rainfall | moving_average | 0.8923 | 1.0174 | 0.8722 | 是 | 零膨胀不适用 |
| R17_nyc_asos_hourly_2018_2024 | 61098 | 24 | 24 | temperature | seasonal_naive | 0.8533 | 0.8533 | 0.8708 | 否 | 带内 |
| R17_nyc_asos_hourly_2018_2024 | 61098 | 24 | 24 | humidity | seasonal_naive | 0.8923 | 0.8923 | 0.8611 | 否 | 带内 |
| R17_nyc_asos_hourly_2018_2024 | 61098 | 24 | 24 | pressure | drift | 0.6935 | 0.9445 | 0.866 | 否 | 带内 |
| R17_nyc_asos_hourly_2018_2024 | 61098 | 24 | 24 | wind_speed | moving_average | 0.8743 | 1.0965 | 0.8708 | 否 | 带内 |
| R17_nyc_asos_hourly_2018_2024 | 61098 | 24 | 24 | wind_gusts | moving_average | 0.9004 | 1.087 | 0.8681 | 否 | 带内 |
| R17_nyc_asos_hourly_2018_2024 | 61098 | 24 | 24 | rainfall | moving_average | 1.1374 | 1.2078 | 0.8715 | 是 | 零膨胀不适用 |

## 读数注意

- MASE<1 的含义是「误差小于 seasonal naive 的一步朴素参照」，不是「预测准确」；天气预报本身有不可约的随机分量，MASE≈1 在气温上很常见。
- 覆盖率按点加权汇总，逐列偏离的少数列单独列在上表，不掩盖。
- **零膨胀列不判区间覆盖率**：降水类列 79%–99% 的时刻观测为 0，「预测 0、实际也是 0」就能拿 0.99 的覆盖率；它衡量的是猜没下雨，不是报得准。这类列改看事件概率预报的 Brier / BSS / 可靠性。
- 逐列出界用 Wilson 95% 置信区间判定，不用点估计卡边界：384 个点上的标准误约 1.5–2 个百分点，0.7995 与 0.8000 的差别没有统计意义。
- 事件点数（`event_points`）少于 100 时 BSS 不可解释——测试期几乎没发生事件时，「基本不报」的模型会赢过气候基线，那个 BSS 不能当能力。
- 事件技能分档：BSS≤0 无技能；0–0.05 极弱（几乎无实用价值）；0.05–0.15 弱；>0.15 中等以上。F5 只要求「BSS>0」，通过不等于能用于业务预报。
- **数据来源有重复**：`R05_three_cities` 的 `rainfall@SIN` 与 `R02_singapore` 是同一份新加坡数据，两列数字必然相同；算独立证据时必须按来源去重，不能当两列。

## 复现

```bash
PYTHONPATH=src python -m evals.forecast_eval
```
