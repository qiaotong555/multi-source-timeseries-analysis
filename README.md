# 多源时序数据量化分析与可视化 Agent

输入带时间顺序的数据，自动识别数据结构，完成质量检查、预处理、量化指标计算、时域/频域分析、
异常检测、统计检验和可视化，输出可复现的分析报告。**引擎与场景解耦**，当前内置天气（主场景）
与股票（扩展场景）两个适配器，用同一工作流验证可迁移性。

```bash
# 环境：受管虚拟环境 ~/.workbuddy/binaries/python/envs/mtsa（Python 3.13.14）
# 依赖安装见 doc/架构设计.md；src 需在导入路径上（PYTHONPATH=src 或 pip install -e .）

# 测试（basetemp 每次独立，避免临时目录堆积被清理策略拦截）
python -m pytest tests -q --basetemp=artifacts/_pt_<时间戳>

# 运行分析
python -m mtsa.cli run --scene weather \
    --input fixtures/weather_v1.csv --question "分析温度的趋势、季节性与极端事件"
python -m mtsa.cli run --scene stock \
    --input fixtures/stock_v1.csv  --question "计算收益风险指标并做 walk-forward 回测"

# 批量评估（含 dev/holdout 划分）
python -m evals.run_evals
# 只复核划分与阈值选择
python -m evals.split_eval
```

评估口径：异常检测阈值只在各折 dev 段上选定，指标取各折 holdout 段平均；有效折不足时判为「不可判定」而非通过，详见 `doc/评估设计.md` 第 3 节。

数据集分工：`weather_v1` 用于交付演示与其余指标，`weather_eval_v1` 专用于异常检测的 dev/holdout 评估（异常密度更高，每折 holdout 约 12 个真值）。两者分离，避免在同一套数据上调参又报告。

产物写入 `artifacts/<run_id>/`：`result.json`（结构化结果与证据链）、`report.md`、
静态图 PNG、交互图 HTML、清洗后数据 CSV、可复现脚本。

## 目录

```text
src/mtsa/      引擎实现（导入 / 摘要 / 质量 / 预处理 / 指标 / 频域 / 检验 / 异常 / 图表 / 报告 / 编排）
  scenarios/   场景适配器（weather 主场景、stock 扩展场景）
tests/         单元测试与回归测试
evals/         评估脚本与评估记录
fixtures/      固定脱敏样例与真值标签
artifacts/     运行产物（按 run_id 隔离，不修改输入）
spec/          当前任务、未来规划、已完成任务、Bug 记录
doc/           架构设计、接口与字段、评估设计
```

## 边界

- 输入文件只读，从不修改、移动或删除。
- 股票场景不提供真实投资建议，不做自动交易；回测不含交易成本与滑点，结论中显式声明。
- 不做彩票中奖号码预测；彩票场景不在当前任务范围。
- 数值结论全部由计算代码产出，当前任务不启用 LLM 节点。
- 对外 REST 服务不在当前任务范围，见 `spec/未来规划.md`。

## 文档约定

开发以 `spec/当前开发任务.md` 为准；架构、接口与字段变更验证后同步到 `doc/`。
评估记录必须包含数据集版本、指标实测值、失败样例与修改前后对比；未实际运行不得声明通过。
