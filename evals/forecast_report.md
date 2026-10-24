# F：预测回测评估

生成时间：2026-10-24T14:53:23
数据：11 份（v2 fixture + 10 份公开真实天气数据）
方法：seasonal_naive（基线）/ moving_average（窗=周期）/ drift（末值+平均增量）；按 walk-forward MASE 选型，候选须相对基线下降 ≥5% 才允许替换

## 门禁

| 项 | 实际 | 阈值 | 结果 |
| --- | --- | --- | --- |
| F3 汇总 90% 区间覆盖率 | 0.8673 | [0.80, 0.98] | PASS |
| F4 MASE≤1.0 的列占比 | 0.7414（43/58） | ≥ 0.50 | PASS |
| F4 未优于基线的列都有选型说明 | 是 | 0 缺失 | PASS |

**全部通过**

## 汇总

| 项 | 值 |
| --- | --- |
| 参与评估的列 | 58 |
| 样本不足未预测的列 | 0 |
| 覆盖率（按点加权） | 0.8673 |
| 覆盖率（逐列最小 / 最大） | 0.7995 / 0.987 |
| 覆盖率参与点数 | 19552 |
| 覆盖率落在 [0.80, 0.98] 之外的列 | rainfall@TYO=0.987、temperature=0.7995 |
| 选中方法 MASE≤1.0 | 43 / 58 = 0.7414 |
| 选中方法优于基线 | 30 / 58 = 0.5172 |
| 选中的方法 | drift 6 列、moving_average 24 列、seasonal_naive 28 列 |

## 逐数据集

| 数据集 | 行数 | 周期 | horizon | 列 | 选中方法 | MASE | 基线 MASE | 覆盖率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_fixture | 8760 | 24 | 24 | temperature@STA_A | seasonal_naive | 0.9449 | 0.9449 | 0.9089 |
| v2_fixture | 8760 | 24 | 24 | temperature@STA_B | seasonal_naive | 1.0268 | 1.0268 | 0.9036 |
| v2_fixture | 8760 | 24 | 24 | humidity@STA_A | moving_average | 0.9431 | 1.0296 | 0.8802 |
| v2_fixture | 8760 | 24 | 24 | humidity@STA_B | moving_average | 1.04 | 1.0994 | 0.8464 |
| v2_fixture | 8760 | 24 | 24 | pressure@STA_A | moving_average | 0.7487 | 0.9057 | 0.8542 |
| v2_fixture | 8760 | 24 | 24 | pressure@STA_B | moving_average | 0.6641 | 0.8605 | 0.8984 |
| v2_fixture | 8760 | 24 | 24 | wind_speed@STA_A | moving_average | 0.8434 | 0.978 | 0.8333 |
| v2_fixture | 8760 | 24 | 24 | wind_speed@STA_B | moving_average | 0.8267 | 0.9881 | 0.8255 |
| v2_fixture | 8760 | 24 | 24 | rainfall@STA_A | seasonal_naive | 0.5919 | 0.5919 | 0.901 |
| v2_fixture | 8760 | 24 | 24 | rainfall@STA_B | seasonal_naive | 1.1342 | 1.1342 | 0.8958 |
| R01_berlin_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 0.981 | 0.981 | 0.8255 |
| R01_berlin_hourly_2024 | 8784 | 24 | 24 | humidity | seasonal_naive | 0.7905 | 0.7905 | 0.8542 |
| R01_berlin_hourly_2024 | 8784 | 24 | 24 | pressure | drift | 0.8259 | 1.2437 | 0.8151 |
| R01_berlin_hourly_2024 | 8784 | 24 | 24 | wind_speed | moving_average | 0.8298 | 0.926 | 0.9141 |
| R01_berlin_hourly_2024 | 8784 | 24 | 24 | rainfall | drift | 0.9241 | 1.1001 | 0.862 |
| R02_singapore_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 1.1082 | 1.1082 | 0.9141 |
| R02_singapore_hourly_2024 | 8784 | 24 | 24 | humidity | seasonal_naive | 1.133 | 1.133 | 0.8906 |
| R02_singapore_hourly_2024 | 8784 | 24 | 24 | pressure | seasonal_naive | 0.9223 | 0.9223 | 0.8229 |
| R02_singapore_hourly_2024 | 8784 | 24 | 24 | wind_speed | moving_average | 0.8385 | 0.9395 | 0.8281 |
| R02_singapore_hourly_2024 | 8784 | 24 | 24 | rainfall | seasonal_naive | 1.1618 | 1.1618 | 0.8906 |
| R03_phoenix_daily_2015_2024 | 3653 | 7 | 7 | temperature | moving_average | 0.7713 | 0.8804 | 0.8304 |
| R03_phoenix_daily_2015_2024 | 3653 | 7 | 7 | humidity | moving_average | 0.8209 | 1.014 | 0.8929 |
| R03_phoenix_daily_2015_2024 | 3653 | 7 | 7 | pressure | moving_average | 0.7828 | 1.0515 | 0.8393 |
| R03_phoenix_daily_2015_2024 | 3653 | 7 | 7 | wind_speed | moving_average | 0.7236 | 0.9783 | 0.8661 |
| R03_phoenix_daily_2015_2024 | 3653 | 7 | 7 | rainfall | moving_average | 0.7194 | 0.8 | 0.9196 |
| R04_reykjavik_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 1.369 | 1.369 | 0.8229 |
| R04_reykjavik_hourly_2024 | 8784 | 24 | 24 | humidity | drift | 0.9455 | 1.0861 | 0.8698 |
| R04_reykjavik_hourly_2024 | 8784 | 24 | 24 | pressure | drift | 1.0993 | 1.4648 | 0.9089 |
| R04_reykjavik_hourly_2024 | 8784 | 24 | 24 | wind_speed | moving_average | 0.9731 | 1.229 | 0.8828 |
| R04_reykjavik_hourly_2024 | 8784 | 24 | 24 | rainfall | moving_average | 1.1615 | 1.2614 | 0.8698 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | temperature@SIN | seasonal_naive | 1.1082 | 1.1082 | 0.9141 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | temperature@SYD | seasonal_naive | 0.992 | 0.992 | 0.8828 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | temperature@TYO | seasonal_naive | 0.9189 | 0.9189 | 0.862 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | humidity@SIN | seasonal_naive | 1.133 | 1.133 | 0.8906 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | humidity@SYD | seasonal_naive | 1.1606 | 1.1606 | 0.8229 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | humidity@TYO | seasonal_naive | 1.0689 | 1.0689 | 0.8385 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | pressure@SIN | seasonal_naive | 0.9223 | 0.9223 | 0.8229 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | pressure@SYD | drift | 0.8534 | 1.121 | 0.8594 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | pressure@TYO | drift | 0.9205 | 1.0394 | 0.8438 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | wind_speed@SIN | moving_average | 0.8385 | 0.9395 | 0.8281 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | wind_speed@SYD | moving_average | 0.9467 | 1.0223 | 0.8828 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | wind_speed@TYO | moving_average | 0.6025 | 0.6994 | 0.8516 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | rainfall@SIN | seasonal_naive | 1.1618 | 1.1618 | 0.8906 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | rainfall@SYD | seasonal_naive | 0.7759 | 0.7759 | 0.8958 |
| R05_three_cities_hourly_2024 | 8784 | 24 | 24 | rainfall@TYO | seasonal_naive | 0.703 | 0.703 | 0.987 |
| R06_nyc_central_park_daily_2018_2024 | 2557 | 7 | 7 | temperature | moving_average | 0.7366 | 0.9916 | 0.8393 |
| R06_nyc_central_park_daily_2018_2024 | 2557 | 7 | 7 | wind_speed | moving_average | 0.8205 | 1.089 | 0.9018 |
| R06_nyc_central_park_daily_2018_2024 | 2557 | 7 | 7 | rainfall | seasonal_naive | 0.8852 | 0.8852 | 0.8571 |
| R07_melbourne_daily_2018_2024 | 2557 | 7 | 7 | temperature | moving_average | 0.7848 | 1.0234 | 0.8393 |
| R07_melbourne_daily_2018_2024 | 2557 | 7 | 7 | rainfall | moving_average | 0.67 | 0.7424 | 0.8839 |
| R08_berlin_tegel_hourly_2024 | 8780 | 24 | 24 | temperature | seasonal_naive | 0.8952 | 0.8952 | 0.8802 |
| R08_berlin_tegel_hourly_2024 | 8780 | 24 | 24 | humidity | seasonal_naive | 0.7346 | 0.7346 | 0.8984 |
| R09_munich_hourly_2024 | 8784 | 24 | 24 | temperature | seasonal_naive | 1.005 | 1.005 | 0.8229 |
| R09_munich_hourly_2024 | 8784 | 24 | 24 | humidity | seasonal_naive | 0.667 | 0.667 | 0.8724 |
| R10_toronto_hourly_2024 | 8783 | 24 | 24 | temperature | seasonal_naive | 0.9659 | 0.9659 | 0.7995 |
| R10_toronto_hourly_2024 | 8783 | 24 | 24 | humidity | seasonal_naive | 0.8715 | 0.8715 | 0.8516 |
| R10_toronto_hourly_2024 | 8783 | 24 | 24 | pressure | moving_average | 0.8192 | 0.9394 | 0.849 |
| R10_toronto_hourly_2024 | 8783 | 24 | 24 | wind_speed | moving_average | 0.9863 | 1.1187 | 0.8646 |

## 读数注意

- MASE<1 的含义是「误差小于 seasonal naive 的一步朴素参照」，不是「预测准确」；天气预报本身有不可约的随机分量，MASE≈1 在气温上很常见。
- 覆盖率按点加权汇总，逐列偏离的少数列单独列在上表，不掩盖。
- 未做多模型对比，也没有在评估集上调过任何阈值；三个方法在运行前就已固定。

## 复现

```bash
PYTHONPATH=src python -m evals.forecast_eval
```
