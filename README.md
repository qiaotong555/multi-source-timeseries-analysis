# 多源时序数据量化分析与可视化 Agent

输入带时间顺序的数据，自动识别数据结构，完成质量检查、预处理、量化指标计算、时域/频域分析、
异常检测、统计检验和可视化，输出可复现的分析报告。**引擎与场景解耦**，当前内置天气（主场景）
与股票（扩展场景）两个适配器，用同一工作流验证可迁移性。

```bash
# 环境：受管虚拟环境 ~/.workbuddy/binaries/python/envs/mtsa（Python 3.13.14）
# 依赖安装见 doc/架构设计.md；src 需在导入路径上（PYTHONPATH=src 或 pip install -e .）

# 测试（basetemp 每次独立，避免临时目录堆积被清理策略拦截）
python -m pytest tests -q --basetemp=artifacts/_pt_<时间戳>

# 低内存 / 可超时的逐文件入口（单进程卡死或内存紧张时用）
# 每个文件独立进程，固定 MTSA_PLANNER=rule、OPENBLAS_NUM_THREADS=1
python run_tests.py --timeout 300

# 运行分析（默认在子进程中执行，带 600 秒超时）
python -m mtsa.cli run --scene weather \
    --input fixtures/weather_v1.csv --question "分析温度的趋势、季节性与极端事件"
python -m mtsa.cli run --scene stock \
    --input fixtures/stock_v1.csv  --question "计算收益风险指标并做 walk-forward 回测"

# Agent 模式（本轮仅天气；需配置 MTSA_LLM_BASE_URL / MTSA_LLM_API_KEY / MTSA_LLM_MODEL）
python -m mtsa.cli run --mode agent --scene weather \
    --input fixtures/weather_v1.csv --question "分析温度的趋势、季节性和异常"

# 成本上限：累计 token 超限即以 token_budget 结束（默认 0 = 不限制）
python -m mtsa.cli run --mode agent --scene weather \
    --input fixtures/weather_v1.csv --question "分析温度的趋势、季节性与异常" \
    --agent-max-tokens 2000

# 离线重放：按已保存的轨迹重跑工具，不调用模型
python -m mtsa.cli run --replay artifacts/<run_id>/result.json

# 自动模式：有模型且天气走 Agent，否则自动回退固定流水线
python -m mtsa.cli run --mode auto --scene weather \
    --input fixtures/weather_v1.csv --question "分析温度的趋势、季节性和异常"

# 批量评估（含 dev/holdout 划分）
python -m evals.run_evals
# 只复核划分与阈值选择
python -m evals.split_eval
# 性能基准（约 8 分钟，结果供 run_evals 读取）
python -m evals.benchmark
# LLM 计划层评估（需配置 MTSA_LLM_BASE_URL / API_KEY / MODEL，约 15 秒）
python -m evals.llm_eval

# Agent 模式评估（13 个天气用例；步数、覆盖率、重放率、越界、失败率、token）
python -m evals.agent_eval
# 只验证指标管线，不调用模型（结果标 source=mock，不能当模型能力结论）
python -m evals.agent_eval --mock

# pipeline / agent 对照（同批 13 个天气用例，固定规则计划）
python -m evals.mode_compare

# 输出稳定性：同题重复 3 次，guided（关键词下限）vs free（模型自定范围）
python -m evals.stability_eval --repeats 3

# A 结论级评估：读 artifacts/t212_compare，不改检测阈值
python -m evals.conclusion_eval
# B 差异溯源：把 agent 少掉的结论按类别归因
python -m evals.conclusion_diff

# C 第二份数据：30 天 / 15 分钟 / 单站点 / DST / 10% 缺失
python -m evals.agent_eval --input fixtures/weather_short_v1.csv     --out evals/agent_eval_short_results.json --out-root artifacts/c_short_agent_eval
python -m evals.repro_check     artifacts/c_short_pipeline_rule_1/result.json     artifacts/c_short_pipeline_rule_2/result.json
python -m evals.conclusion_eval --result-root artifacts/c_short_agent_eval     --labels fixtures/labels_weather_short_v1.csv     --truth fixtures/truth_weather_short_v1.json     --out evals/conclusion_eval_short_agent_results.json
```

## 运行保护

| 限制 | 默认 | 调整方式 |
| --- | --- | --- |
| 数据规模 | 20 万行，超过即拒绝（`DATASET_TOO_LARGE`） | `MTSA_MAX_ROWS=500000` |
| 整体超时 | 600 秒，超时终止子进程（`NODE_TIMEOUT`） | `--timeout 900`；`0` 表示不限制且不进子进程 |
| 运行模式 | 默认 `pipeline` | `--mode agent` / `--mode auto`，或设置 `MTSA_AGENT` |
| Agent 范围策略 | `guided`：按问题关键词定下限，达标前不得结束，下限外最多追加 1 步 | `--agent-scope free` 恢复“范围完全由模型决定”（步数会波动） |
| Agent token 预算 | 0（不限制），只记录用量 | `--agent-max-tokens 2000`；超限以 `token_budget` 结束 |

CLI 默认在**子进程**中执行分析：Python 无法中断正在执行的 C 扩展调用（numpy、statsmodels），
主线程计时器拦不住卡死，只有进程级终止有效。代价是约 3 秒启动开销。
库调用 `engine.run_analysis()` 不走子进程，行为与以前一致。

**判断成败请读 `result.json` 的 `status`，不要以「产物目录非空」为准**——失败时也会落盘留档。

评估口径：异常检测阈值只在各折 dev 段上选定，指标取各折 holdout 段平均；有效折不足时判为「不可判定」而非通过，详见 `doc/评估设计.md` 第 3 节。

数据集分工：`weather_v1` 用于交付演示与其余指标，`weather_eval_v1` 专用于异常检测的 dev/holdout 评估（异常密度更高，每折 holdout 约 12 个真值）。两者分离，避免在同一套数据上调参又报告。

产物写入 `artifacts/<run_id>/`：`result.json`（结构化结果与证据链）、`report.md`、
静态图 PNG、交互图 HTML、清洗后数据 CSV、可复现脚本。

## 当前状态与能力边界

**已实现（天气 Agent）**：观察—决策—行动循环、五工具白名单、预算受控的观察摘要、
决策边界（锁定参数/未知参数/重复调用/步数上限）、失败重试与三态记录、
轨迹哈希与离线重放、`pipeline / agent / auto` 三模式、关键词范围下限（`guided`）、
13 用例评估与两模式对照。

**明确未实现**：股票场景的 Agent（请求 `mode=agent` 返回 `AGENT_SCENE_UNSUPPORTED`）；
多模型对比；按金额计费换算（只限 token，不换算货币）；`free` 模式下的范围稳定性。

| 模式 | 行为 | 无模型时 |
| --- | --- | --- |
| `pipeline`（默认） | 固定跑完 12 项计划 | 正常 |
| `agent` | 天气场景由模型选择工具 | `AGENT_UNAVAILABLE` |
| `auto` | 天气 + 有模型才升级 | 回退 pipeline 并写 warning |

**真实模型与 Mock 必须分清**：`evals/agent_eval.py --mock` 用脚本客户端，只验证指标管线，
结果标 `source=mock`，不参与 `run_evals` 判定；真实结论只来自 `source=real`。

**分析项 12 / 13 / 14 的口径**（常见混淆点）：

- **12 项**：天气规则计划实际选中的分析项（`trend`、`corr`、`lag`、`extreme`、`spectrum`、
  `changepoint`、`anomaly`、`test`、`stationarity`、`autocorr`、`normality`、`seasonal`）；
- **13 项**：计划层 `TOOL_SPECS` 在 12 项基础上多一个 `randomness`；
- **14 项**：引擎内部开关 `ALL_ANALYSES` 在 13 项基础上多一个 `volatility`。

评估里说的「规则计划覆盖率」固定以 **12 项**为分母；「工具覆盖率」以 **5 个白名单工具**为分母；
两者不能互相替代。

**两模式实测（13 个天气用例，`guided`，2026-10-23）**：

| 维度 | pipeline | agent |
| --- | --- | --- |
| 成功 | 13/13 | 13/13 |
| 平均步数 | 5.00 | 2.54 |
| 结论（均值） | 39.00 | 29.92（−23.3%） |
| 需求覆盖率 | 1.00 | 1.00 |
| 规则计划覆盖率 | 1.00 | 0.66 |
| 墙钟（均值） | 12.09 s | 12.77 s（+5.6%） |
| token | 0 | 53,924 |

9 用例时期的数字（结论 −31.3%、墙钟 +4.7%）保留在 `doc/评估设计.md` 6.5/6.6，
**不同用例集的降幅不可直接比较**。现行口径见 6.8，逐用例数据见
`evals/mode_compare_results.json`。

**结论级评估（A/B，2026-10-23）**：在这份演示数据上，异常点级 P/R/F1 为
0.130 / 1.000 / 0.229（真值 18、FP 121）；agent 只在 7/13 个用例执行异常检测，
未执行的 6 个用例少看 108 个真值实例，但去重后真值命中仍为 18/18。两模式结论
总数 507 vs 389，差 118 条（9.08/用例），**100% 归因于 `tool_not_run`**，无法归因 0 条。
这说明 Agent 的差异主要是范围而不是精度；也说明「检出 139 个」不等于「139 个都对」。

结论：Agent 适合窄问题与需要审计路径的场景；需要完整报告、或没有模型配额时用 pipeline。

## 目录

```text
src/mtsa/      引擎实现（导入 / 摘要 / 质量 / 预处理 / 指标 / 频域 / 检验 / 异常 / 图表 / 报告 / 编排）
  scenarios/   场景适配器（weather 主场景、stock 扩展场景）
tests/         单元测试与回归测试
evals/         评估脚本与评估记录
fixtures/      固定脱敏样例与真值标签
artifacts/     运行产物（按 run_id 隔离，不修改输入）
spec/          当前任务、未来规划、已完成任务、Bug 记录
doc/           架构设计、接口与字段、评估设计、改进记录、Agent 化改造方案与任务清单、报告源稿（reports/）
```

## 边界

- 输入文件只读，从不修改、移动或删除。
- 股票场景不提供真实投资建议，不做自动交易；回测不含交易成本与滑点，结论中显式声明。
- 不做彩票中奖号码预测；彩票场景不在当前任务范围。
- 数值结论全部由计算代码产出，模型不参与任何数值计算。LLM 只参与两处选择：计划层选择分析项，
  以及 Agent 模式选择下一步调用哪个工具。未配置模型时，pipeline 仍可用规则计划运行。
- **Agent 模式本轮只支持天气场景**；股票场景请求 `--mode agent` 会返回 `AGENT_SCENE_UNSUPPORTED`。
  Agent 只决定「调用哪些工具」，工具的完整产出与未执行项都会写入 `result.json` 与报告。
- 对外 REST 服务不在当前任务范围，见 `spec/未来规划.md`。

## 文档约定

开发以 `spec/当前开发任务.md` 为准；架构、接口与字段变更验证后同步到 `doc/`。
评估记录必须包含数据集版本、指标实测值、失败样例与修改前后对比；未实际运行不得声明通过。
