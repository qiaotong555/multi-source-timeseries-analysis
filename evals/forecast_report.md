# F：预测回测评估

生成时间：2026-11-01T09:41:32
数据：18 份（v2 fixture + 10 份公开真实天气数据）
方法：seasonal_naive（基线）/ moving_average（窗=周期）/ drift（末值+平均增量）；按 walk-forward MASE 选型，候选须相对基线下降 ≥5% 才允许替换
区间：最近 12 个 origin 的误差按 horizon 归一、分 4 箱、取 conformal 有限样本修正分位

## 门禁

| 项 | 实际 | 阈值 | 结果 |
| --- | --- | --- | --- |
| F3 汇总 90% 区间覆盖率 | 0.8857 | [0.80, 0.98] | PASS |
| F4 MASE≤1.0 的列占比 | 0.7234（68/94） | ≥ 0.50 | PASS |
| F4 未优于基线的列都有选型说明 | 是 | 0 缺失 | PASS |
| F5 零膨胀列事件概率预报 BSS > 0（事件点 ≥ 100） | 3 列参与，全部为正 | 每列 > 0 | PASS |

**全部通过**

## 汇总

| 项 | 值 |
| --- | --- |
| 参与评估的列 | 94 |
| 样本不足未预测的列 | 0 |
| 零膨胀列（零值 ≥ 50%） | 18 |
| 覆盖率（按点加权） | 0.8857 |
| 覆盖率（逐列最小 / 最大） | 0.7995 / 0.9948 |
| 覆盖率参与点数 | 33376 |
| 判定分布 | 带内 76、带外 0、零膨胀不适用 18 |
| 覆盖率带外（Wilson 95% 区间与带不相交） | 无 |
| 零膨胀不判区间覆盖率的列 | v2_fixture/rainfall@STA_A（覆盖率 0.9219，零值 93%，非零点 34）；v2_fixture/rainfall@STA_B（覆盖率 0.9167，零值 92%，非零点 30）；R01_berlin_hourly_2024/rainfall（覆盖率 0.8646，零值 84%，非零点 76）；R02_singapore_hourly_2024/rainfall（覆盖率 0.8906，零值 65%，非零点 165）；R03_phoenix_daily_2015_2024/rainfall（覆盖率 0.9643，零值 83%，非零点 13）；R04_reykjavik_hourly_2024/rainfall（覆盖率 0.8724，零值 68%，非零点 129）；R05_three_cities_hourly_2024/rainfall@SIN（覆盖率 0.8906，零值 65%，非零点 165）；R05_three_cities_hourly_2024/rainfall@SYD（覆盖率 0.9062，零值 80%，非零点 57）；R05_three_cities_hourly_2024/rainfall@TYO（覆盖率 0.9818，零值 83%，非零点 4）；R06_nyc_central_park_daily_2018_2024/rainfall（覆盖率 0.9018，零值 63%，非零点 35）；R07_melbourne_daily_2018_2024/rainfall（覆盖率 0.9464，零值 63%，非零点 24）；R11_la_paz_hourly_2024/rainfall（覆盖率 0.8854，零值 87%，非零点 101）；R12_quito_hourly_2024/rainfall（覆盖率 0.8620，零值 70%，非零点 192）；R13_addis_ababa_hourly_2024/rainfall（覆盖率 0.9948，零值 79%，非零点 2）；R14_nairobi_hourly_2024/rainfall（覆盖率 0.9010，零值 79%，非零点 131）；R15_cairo_hourly_2024/rainfall（覆盖率 0.9870，零值 99%，非零点 5）；R16_ushuaia_hourly_2024/rainfall（覆盖率 0.8750，零值 67%，非零点 122）；R17_nyc_asos_hourly_2018_2024/rainfall（覆盖率 0.8542，零值 87%，非零点 40） |
| 选中方法 MASE≤1.0 | 68 / 94 = 0.7234 |
| 选中方法优于基线 | 40 / 94 = 0.4255 |
| 选中的方法 | drift 10 列、moving_average 30 列、seasonal_naive 54 列 |

## 事件概率预报（零膨胀列）

阈值 0.1；主模型 = climatology 与 persistence_short 的等权平均；BSS 相对 climatology。事件点少于 100 的列不判，因为「基本不报」也能把 BSS 抬高。

| 数据集 | 列 | 事件点 | 实测发生率 | BSS | 等级 | ECE | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2_fixture | rainfall@STA_A | 30 | 0.0781 | -0.0003 | insufficient_events | 0.0144 | 未判（事件点不足） |
| v2_fixture | rainfall@STA_B | 29 | 0.0755 | -0.0058 | insufficient_events | 0.0105 | 未判（事件点不足） |
| R01_berlin_hourly_2024 | rainfall | 55 | 0.1432 | -0.0087 | insufficient_events | 0.0685 | 未判（事件点不足） |
| R02_singapore_hourly_2024 | rainfall | 130 | 0.3385 | 0.0214 | very_weak | 0.0999 | 参与门禁 |
| R03_phoenix_daily_2015_2024 | rainfall | 8 | 0.0714 | 0.0645 | insufficient_events | 0.0808 | 未判（事件点不足） |
| R04_reykjavik_hourly_2024 | rainfall | 99 | 0.2578 | 0.1144 | insufficient_events | 0.0778 | 未判（事件点不足） |
| R05_three_cities_hourly_2024 | rainfall@SIN | 130 | 0.3385 | 0.0214 | very_weak | 0.0999 | 参与门禁 |
| R05_three_cities_hourly_2024 | rainfall@SYD | 39 | 0.1016 | 0.0834 | insufficient_events | 0.0439 | 未判（事件点不足） |
| R05_three_cities_hourly_2024 | rainfall@TYO | 1 | 0.0026 | 0.2303 | insufficient_events | 0.1118 | 未判（事件点不足） |
| R06_nyc_central_park_daily_2018_2024 | rainfall | 35 | 0.3125 | 0.0173 | insufficient_events | 0.057 | 未判（事件点不足） |
| R07_melbourne_daily_2018_2024 | rainfall | 24 | 0.2143 | 0.0254 | insufficient_events | 0.1622 | 未判（事件点不足） |
| R11_la_paz_hourly_2024 | rainfall | 63 | 0.1641 | 0.0794 | insufficient_events | 0.0676 | 未判（事件点不足） |
| R12_quito_hourly_2024 | rainfall | 159 | 0.4141 | 0.039 | very_weak | 0.1637 | 参与门禁 |
| R13_addis_ababa_hourly_2024 | rainfall | 1 | 0.0026 | 0.3579 | insufficient_events | 0.1027 | 未判（事件点不足） |
| R14_nairobi_hourly_2024 | rainfall | 62 | 0.1615 | 0.0402 | insufficient_events | 0.0501 | 未判（事件点不足） |
| R15_cairo_hourly_2024 | rainfall | 2 | 0.0052 | 0.0034 | insufficient_events | 0.0029 | 未判（事件点不足） |
| R16_ushuaia_hourly_2024 | rainfall | 76 | 0.1979 | 0.034 | insufficient_events | 0.0483 | 未判（事件点不足） |
| R17_nyc_asos_hourly_2018_2024 | rainfall | 25 | 0.0651 | 0.011 | insufficient_events | 0.0069 | 未判（事件点不足） |

## 逐数据集

| 数据集 | 行数 | 周期 | horizon | 列 | 选中方法 | MASE | 基线 MASE | 覆盖率 | 零膨胀 | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_fixture | 8760 | 24 | 24 | temperature@STA_A | seasonal_naive | 0.9449 | 0.9449 | 0.9219 | 否 | 带内 |
| v2_fixture | 8760 | 24 | 24 | temperature@STA_B | seasonal_naive | 1.0268 | 1.0268 | 0.9219 | 否 | 带内 |
| v2_fixture | 8760 | 24 | 24 | humidity@STA_A | moving_average | 0.9431 | 1.0296 | 0.8776 | 否 | 带内 |
| v2_fixture | 8760 | 24 | 24 | humidity@STA_B | moving_average | 1.04 | 1.0994 | 0.8594 | 否 | 带内 |
| v2_fixture | 8760 | 24 | 24 | pressure@STA_A | moving_average | 0.7487 | 0.9057 | 0.8568 | 否 | 带内 |
| v2_fixture | 8760 | 24 | 24 | pressure@STA_B | moving_average | 0.6641 | 0.8605 | 0.901 | 否 | 带内 |
| v2_fixture | 8760 | 24 | 24 | wind_speed@STA_A | moving_average | 0.8434 | 0.978 | 0.8568 | 否 | 带内 |
| v2_fixture | 8760 | 24 | 24 | wind_speed@STA_B | moving_average | 0.8267 | 0.9881 | 0.8568 | 否 | 带内 |
| v2_fixture | 8760 | 24 | 24 | rainfall@STA_A | seasonal_naive | 0.5919 | 0.5919 | 0.9219 | 是 | 零膨胀不适用 |
| v2_fixture | 8760 | 24 | 24 | rainfall@STA_B | seasonal_naive | 1.1342 | 1.1342 | 0.9167 | 是 | 零膨胀不适用 |
| R01_berlin_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 0.981 | 0.981 | 0.8333 | 否 | 带内 |
| R01_berlin_hourly_2024 | 8784 | 24 | 24 | humidity | seasonal_naive | 0.7905 | 0.7905 | 0.8438 | 否 | 带内 |
| R01_berlin_hourly_2024 | 8784 | 24 | 24 | pressure | drift | 0.8259 | 1.2437 | 0.8438 | 否 | 带内 |
| R01_berlin_hourly_2024 | 8784 | 24 | 24 | wind_speed | moving_average | 0.8298 | 0.926 | 0.8958 | 否 | 带内 |
| R01_berlin_hourly_2024 | 8784 | 24 | 24 | rainfall | drift | 0.9241 | 1.1001 | 0.8646 | 是 | 零膨胀不适用 |
| R02_singapore_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 1.1082 | 1.1082 | 0.9323 | 否 | 带内 |
| R02_singapore_hourly_2024 | 8784 | 24 | 24 | humidity | seasonal_naive | 1.133 | 1.133 | 0.9089 | 否 | 带内 |
| R02_singapore_hourly_2024 | 8784 | 24 | 24 | pressure | seasonal_naive | 0.9223 | 0.9223 | 0.8542 | 否 | 带内 |
| R02_singapore_hourly_2024 | 8784 | 24 | 24 | wind_speed | moving_average | 0.8385 | 0.9395 | 0.8594 | 否 | 带内 |
| R02_singapore_hourly_2024 | 8784 | 24 | 24 | rainfall | seasonal_naive | 1.1618 | 1.1618 | 0.8906 | 是 | 零膨胀不适用 |
| R03_phoenix_daily_2015_2024 | 3653 | 7 | 7 | temperature | moving_average | 0.7713 | 0.8804 | 0.8571 | 否 | 带内 |
| R03_phoenix_daily_2015_2024 | 3653 | 7 | 7 | humidity | moving_average | 0.8209 | 1.014 | 0.9286 | 否 | 带内 |
| R03_phoenix_daily_2015_2024 | 3653 | 7 | 7 | pressure | moving_average | 0.7828 | 1.0515 | 0.8839 | 否 | 带内 |
| R03_phoenix_daily_2015_2024 | 3653 | 7 | 7 | wind_speed | moving_average | 0.7236 | 0.9783 | 0.8929 | 否 | 带内 |
| R03_phoenix_daily_2015_2024 | 3653 | 7 | 7 | rainfall | moving_average | 0.7194 | 0.8 | 0.9643 | 是 | 零膨胀不适用 |
| R04_reykjavik_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 1.369 | 1.369 | 0.7995 | 否 | 带内 |
| R04_reykjavik_hourly_2024 | 8784 | 24 | 24 | humidity | drift | 0.9455 | 1.0861 | 0.8646 | 否 | 带内 |
| R04_reykjavik_hourly_2024 | 8784 | 24 | 24 | pressure | drift | 1.0993 | 1.4648 | 0.8906 | 否 | 带内 |
| R04_reykjavik_hourly_2024 | 8784 | 24 | 24 | wind_speed | moving_average | 0.9731 | 1.229 | 0.901 | 否 | 带内 |
| R04_reykjavik_hourly_2024 | 8784 | 24 | 24 | rainfall | moving_average | 1.1615 | 1.2614 | 0.8724 | 是 | 零膨胀不适用 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | temperature@SIN | seasonal_naive | 1.1082 | 1.1082 | 0.9323 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | temperature@SYD | seasonal_naive | 0.992 | 0.992 | 0.8672 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | temperature@TYO | seasonal_naive | 0.9189 | 0.9189 | 0.888 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | humidity@SIN | seasonal_naive | 1.133 | 1.133 | 0.9089 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | humidity@SYD | seasonal_naive | 1.1606 | 1.1606 | 0.862 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | humidity@TYO | seasonal_naive | 1.0689 | 1.0689 | 0.8854 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | pressure@SIN | seasonal_naive | 0.9223 | 0.9223 | 0.8542 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | pressure@SYD | drift | 0.8534 | 1.121 | 0.9115 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | pressure@TYO | drift | 0.9205 | 1.0394 | 0.875 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | wind_speed@SIN | moving_average | 0.8385 | 0.9395 | 0.8594 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | wind_speed@SYD | moving_average | 0.9467 | 1.0223 | 0.9089 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | wind_speed@TYO | moving_average | 0.6025 | 0.6994 | 0.8698 | 否 | 带内 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | rainfall@SIN | seasonal_naive | 1.1618 | 1.1618 | 0.8906 | 是 | 零膨胀不适用 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | rainfall@SYD | seasonal_naive | 0.7759 | 0.7759 | 0.9062 | 是 | 零膨胀不适用 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | rainfall@TYO | seasonal_naive | 0.703 | 0.703 | 0.9818 | 是 | 零膨胀不适用 |
| R06_nyc_central_park_daily_2018_2024 | 2557 | 7 | 7 | temperature | moving_average | 0.7366 | 0.9916 | 0.9196 | 否 | 带内 |
| R06_nyc_central_park_daily_2018_2024 | 2557 | 7 | 7 | wind_speed | moving_average | 0.8205 | 1.089 | 0.875 | 否 | 带内 |
| R06_nyc_central_park_daily_2018_2024 | 2557 | 7 | 7 | rainfall | seasonal_naive | 0.8852 | 0.8852 | 0.9018 | 是 | 零膨胀不适用 |
| R07_melbourne_daily_2018_2024 | 2557 | 7 | 7 | temperature | moving_average | 0.7848 | 1.0234 | 0.8661 | 否 | 带内 |
| R07_melbourne_daily_2018_2024 | 2557 | 7 | 7 | rainfall | moving_average | 0.67 | 0.7424 | 0.9464 | 是 | 零膨胀不适用 |
| R08_berlin_tegel_hourly_2024 | 8780 | 24 | 24 | temperature | seasonal_naive | 0.8952 | 0.8952 | 0.8594 | 否 | 带内 |
| R08_berlin_tegel_hourly_2024 | 8780 | 24 | 24 | humidity | seasonal_naive | 0.7346 | 0.7346 | 0.8698 | 否 | 带内 |
| R09_munich_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 1.005 | 1.005 | 0.8099 | 否 | 带内 |
| R09_munich_hourly_2024 | 8784 | 24 | 24 | humidity | seasonal_naive | 0.667 | 0.667 | 0.9062 | 否 | 带内 |
| R10_toronto_hourly_2024 | 8783 | 24 | 24 | temperature | seasonal_naive | 0.9659 | 0.9659 | 0.8047 | 否 | 带内 |
| R10_toronto_hourly_2024 | 8783 | 24 | 24 | humidity | seasonal_naive | 0.8715 | 0.8715 | 0.8724 | 否 | 带内 |
| R10_toronto_hourly_2024 | 8783 | 24 | 24 | pressure | moving_average | 0.8192 | 0.9394 | 0.8698 | 否 | 带内 |
| R10_toronto_hourly_2024 | 8783 | 24 | 24 | wind_speed | moving_average | 0.9863 | 1.1187 | 0.888 | 否 | 带内 |
| R11_la_paz_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 0.7775 | 0.7775 | 0.9036 | 否 | 带内 |
| R11_la_paz_hourly_2024 | 8784 | 24 | 24 | humidity | seasonal_naive | 0.7904 | 0.7904 | 0.9141 | 否 | 带内 |
| R11_la_paz_hourly_2024 | 8784 | 24 | 24 | pressure | seasonal_naive | 0.7418 | 0.7418 | 0.888 | 否 | 带内 |
| R11_la_paz_hourly_2024 | 8784 | 24 | 24 | wind_speed | seasonal_naive | 0.9075 | 0.9075 | 0.8672 | 否 | 带内 |
| R11_la_paz_hourly_2024 | 8784 | 24 | 24 | rainfall | seasonal_naive | 1.2539 | 1.2539 | 0.8854 | 是 | 零膨胀不适用 |
| R12_quito_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 0.9656 | 0.9656 | 0.9167 | 否 | 带内 |
| R12_quito_hourly_2024 | 8784 | 24 | 24 | humidity | seasonal_naive | 0.9993 | 0.9993 | 0.8906 | 否 | 带内 |
| R12_quito_hourly_2024 | 8784 | 24 | 24 | pressure | seasonal_naive | 0.9603 | 0.9603 | 0.9219 | 否 | 带内 |
| R12_quito_hourly_2024 | 8784 | 24 | 24 | wind_speed | seasonal_naive | 1.0165 | 1.0165 | 0.9062 | 否 | 带内 |
| R12_quito_hourly_2024 | 8784 | 24 | 24 | rainfall | seasonal_naive | 1.0484 | 1.0484 | 0.862 | 是 | 零膨胀不适用 |
| R13_addis_ababa_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 0.7768 | 0.7768 | 0.9271 | 否 | 带内 |
| R13_addis_ababa_hourly_2024 | 8784 | 24 | 24 | humidity | seasonal_naive | 1.0543 | 1.0543 | 0.8906 | 否 | 带内 |
| R13_addis_ababa_hourly_2024 | 8784 | 24 | 24 | pressure | seasonal_naive | 0.8456 | 0.8456 | 0.9193 | 否 | 带内 |
| R13_addis_ababa_hourly_2024 | 8784 | 24 | 24 | wind_speed | seasonal_naive | 0.7784 | 0.7784 | 0.9245 | 否 | 带内 |
| R13_addis_ababa_hourly_2024 | 8784 | 24 | 24 | rainfall | seasonal_naive | 0.4373 | 0.4373 | 0.9948 | 是 | 零膨胀不适用 |
| R14_nairobi_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 0.9143 | 0.9143 | 0.9349 | 否 | 带内 |
| R14_nairobi_hourly_2024 | 8784 | 24 | 24 | humidity | seasonal_naive | 1.0127 | 1.0127 | 0.9141 | 否 | 带内 |
| R14_nairobi_hourly_2024 | 8784 | 24 | 24 | pressure | seasonal_naive | 0.8327 | 0.8327 | 0.9245 | 否 | 带内 |
| R14_nairobi_hourly_2024 | 8784 | 24 | 24 | wind_speed | seasonal_naive | 1.0084 | 1.0084 | 0.8776 | 否 | 带内 |
| R14_nairobi_hourly_2024 | 8784 | 24 | 24 | rainfall | moving_average | 1.2513 | 1.3601 | 0.901 | 是 | 零膨胀不适用 |
| R15_cairo_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 0.6901 | 0.6901 | 0.8646 | 否 | 带内 |
| R15_cairo_hourly_2024 | 8784 | 24 | 24 | humidity | seasonal_naive | 0.905 | 0.905 | 0.8672 | 否 | 带内 |
| R15_cairo_hourly_2024 | 8784 | 24 | 24 | pressure | drift | 0.8209 | 0.9231 | 0.9089 | 否 | 带内 |
| R15_cairo_hourly_2024 | 8784 | 24 | 24 | wind_speed | seasonal_naive | 0.8997 | 0.8997 | 0.8828 | 否 | 带内 |
| R15_cairo_hourly_2024 | 8784 | 24 | 24 | rainfall | drift | 0.2633 | 0.7153 | 0.987 | 是 | 零膨胀不适用 |
| R16_ushuaia_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 1.012 | 1.012 | 0.8698 | 否 | 带内 |
| R16_ushuaia_hourly_2024 | 8784 | 24 | 24 | humidity | seasonal_naive | 1.1119 | 1.1119 | 0.8802 | 否 | 带内 |
| R16_ushuaia_hourly_2024 | 8784 | 24 | 24 | pressure | drift | 0.7393 | 0.9702 | 0.9141 | 否 | 带内 |
| R16_ushuaia_hourly_2024 | 8784 | 24 | 24 | wind_speed | moving_average | 1.0453 | 1.1595 | 0.8203 | 否 | 带内 |
| R16_ushuaia_hourly_2024 | 8784 | 24 | 24 | rainfall | moving_average | 0.7905 | 0.9644 | 0.875 | 是 | 零膨胀不适用 |
| R17_nyc_asos_hourly_2018_2024 | 61098 | 24 | 24 | temperature | seasonal_naive | 0.7932 | 0.7932 | 0.8307 | 否 | 带内 |
| R17_nyc_asos_hourly_2018_2024 | 61098 | 24 | 24 | humidity | seasonal_naive | 0.9299 | 0.9299 | 0.8177 | 否 | 带内 |
| R17_nyc_asos_hourly_2018_2024 | 61098 | 24 | 24 | pressure | drift | 0.7071 | 0.8937 | 0.875 | 否 | 带内 |
| R17_nyc_asos_hourly_2018_2024 | 61098 | 24 | 24 | wind_speed | moving_average | 0.9418 | 1.1275 | 0.8828 | 否 | 带内 |
| R17_nyc_asos_hourly_2018_2024 | 61098 | 24 | 24 | wind_gusts | moving_average | 1.0379 | 1.2114 | 0.8281 | 否 | 带内 |
| R17_nyc_asos_hourly_2018_2024 | 61098 | 24 | 24 | rainfall | moving_average | 0.9239 | 0.9895 | 0.8542 | 是 | 零膨胀不适用 |

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
