# Bug 调试文档

| 编号 | 触发场景 | 根因 | 修复 | 回归测试 | 遗留风险 |
| --- | --- | --- | --- | --- | --- |
| B01 | 合成数据基准夏普异常（4.43）远高于理论值 | 有限样本下 AR(1) 序列样本均值偏离中枢（PMI 样本均值 50.97 vs 设定 50.1），残余漂移被计入年化收益 | 合成序列生成后强制去均值到目标中枢；收益序列去均值到理论漂移 | `tests/test_fixture_drift.py` | 无：去均值不改变相关结构，仅校正一阶矩 |
| B02 | 流水线运行报 `NameError: name 'volume' is not defined` | 重构价格生成逻辑时删除了 `volume` 初始化语句 | 恢复 `volume = []`；运行前增加冒烟检查 | `tests/test_pipeline_smoke.py` | 曾导致 `analysis.json` 停留在上一版旧结果，已通过 CLI 退出码校验避免 |
| B03 | 因子权重被低频因子主导（PMI 权重 0.42） | 月频因子在 520 日样本仅约 25 个独立观测，IC 估计方差大且虚高 | IC 加权向等权收缩（λ=0.4），并在结论中标注小样本风险 | `tests/test_signal_weight.py` | 收缩系数 λ 为经验值，未做敏感性扫描 |
| B04 | 情绪变量相关系数为负，与数据生成设定矛盾 | 混淆「水平」与「一阶差分」：水平领先收益 2 日，差分同期呈弱负相关 | 相关矩阵与领先滞后检验统一改用水平值，利率因近似单位根保留差分；字段标注口径 | `tests/test_leadlag.py` | 无 |
| B05 | 多实体数据采样间隔判为不稳定、缺口数为 0 | 多站点共享时间戳使差分出现 0，中位间隔失去意义；按时间戳并集检测缺口会掩盖单个实体的缺失段 | 间隔在去重时间点上计算；缺口改为按分组分别检测 | `tests/test_profiler.py::test_profile_detects_gap`、`tests/test_pipeline_smoke.py` | 分组数很多时缺口数会成倍增加，已截断到 50 条 |
| B06 | 温度序列单列检出 3562 个异常点 | 季节相位按「行位置 % 周期」计算，样例存在整段缺失后位置与小时错位，日周期未被剔除，残差残留振幅 6°C 的日波动 | 季节相位改为按真实时间与中位步长计算 | `tests/test_anomaly.py::test_anomaly_precision_recall`、评估集 Precision 0.0026→0.75 | 相位依赖中位步长，步长本身异常时仍会错位 |
| B07 | 干净序列误报率偏高、端点成片误报 | STL 的移动平均会把单个极端值摊到相邻约一个周期；序列两端窗口不完整 | 改为滚动中位数趋势 + 相位中位数季节项的稳健分解，并排除首尾各一个周期 | `tests/test_anomaly.py::test_no_false_alarm_on_clean_series` | 无 |
| B08 | 72 小时插值缺口处连续误报 | 线性插值无法还原季节形态，填补误差被判定为异常 | 清洗阶段记录被插值填补的采样点，异常检测排除这些点 | `tests/test_anomaly.py` + 评估集 | 落在缺口内的真实异常不会被检出 |
| B09 | 华氏段换算作用到 14439 行（应为 689 行），且换算后中位数变成 -7.87 | 三个叠加原因：换算按行标签范围定位（输入乱序时无效）；一次量纲错误有进入/退出两个转移点却被当成两次独立换算；退出转移点比值为倒数（0.357）被区间过滤条件排除 | 换算改按时间戳 + 分组定位；转移点按方向相反配对成区间；比值判定同时接受比值与倒数 | `tests/test_preprocess.py::test_unit_conversion_respects_group` | 只检测到一个转移点时退化为开放区间，结论中已声明 |
| B10 | 复权最大相对偏差 1.23e-01，远超注入的 1% | 样例注入顺序错误：复权价在收盘价异常跳点之前计算，跳点后复权价未同步 | 调整注入顺序，复权价在收盘价跳点之后计算 | `tests/test_pipeline_smoke.py::test_stock_adjust_inconsistency_flagged` | 无 |
| B11 | 统计检验测试全部 StopIteration，`rows` 只有「样本不足」 | 测试辅助里 `pd.DataFrame({"v": series}, index=新索引)` 触发标签对齐，整列变 NaN | 改用 `series.to_numpy()` 构造 | `tests/test_stats_tests.py` | 无 |
| B12 | 同输入两次运行 `result.data` 不一致 | `QualityReport.as_of` 取墙钟时间 | `as_of` 改为取数据自身截止时间 | `tests/test_pipeline_smoke.py::test_determinism_same_input_twice` | 无 |

## 记录规则

修复 Bug 时先在此表追加一行（触发场景、根因），再修改测试与实现；修复后补齐回归测试与遗留风险。不得只改代码不记录。
