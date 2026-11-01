# 变更记录

版本号与 PyPI 上的发行版一一对应。每个版本的能力边界与实测数字见 `README.md`
第 3、4 节，完整评估结论在 `evals/`。

## 未发布

**预测：区间标定与零膨胀列口径**

- 区间标定改为「按 horizon 分 4 箱 + conformal 有限样本分位」。名义 90% 区间与实际的
  逐列平均绝对偏差从 0.0390 降到 0.0311，点估计出界列 6 → 0（R04 温度 0.7995 落在
  Wilson 95% 区间 [0.758, 0.838] 内，未为它单独调参）。
- 新增列级字段 `zero_share` / `zero_inflated` / `interval_coverage_nonzero` /
  `interval_coverage_nonzero_points` / `coverage_note`。零膨胀列（零值 ≥50%）不再用
  整体覆盖率作为质量判据——降水列 79%–99% 的时刻为 0，那个数字衡量的是「猜没下雨」。
- 新增 `mtsa/prob_forecast.py`：零膨胀列的事件概率预报（climatology / persistence_long /
  persistence_short / 等权 blend），输出 Brier、BSS、技能等级、可靠性分箱与 ECE。
  `data.forecast.events` 随 `--forecast` 输出，报告新增 8d 节。
- 评估新增 F5 门禁：事件点 ≥100 的零膨胀列 BSS > 0。当前 3 列参与、全部为正，
  但技能只有 0.02–0.04（极弱）；事件点不足 100 的列标「未判」，不评级。
  逐列覆盖率判定改为 Wilson 95% 区间与 [0.80, 0.98] 是否相交。
- 局限：事件点每列仅 2–159，降水概率预报暂无实用价值；这件事没有调参掩盖，
  详见 `doc/预测场景化.md`。

## 0.4.0 — 2026-10-26

第一个带 HTTP 接口的版本。

**新增**

- `mtsa-api`：HTTP 薄壳 7 个路由——提交分析返回 202、轮询状态、取 `result.json`、
  下载单个产物、上传 CSV、同步预演、`/healthz`。单 token 鉴权（`X-API-Token`），
  不配置 token 时拒绝启动。
- `[api]` extra（fastapi / uvicorn / python-multipart）与 `mtsa-api` 入口脚本；
  Docker 镜像同一份支持 CLI 与 api 两种入口。
- 计划器可按请求指定：CLI 的 `--planner rule|llm|auto`、API 的 `planner` 字段，
  请求优先于服务端 `MTSA_PLANNER`；响应里回显「请求的」与「实际用的」两种来源。
- 日历热力图（weather 场景默认多出一张 `fig_05_wx_calendar.png`）。
- 保留集（C1）：3 份此前未用过的新气候带数据，12/12 成功、证据与需求覆盖 1.000。
- 开箱自检 `mtsa/verify.py`：三条校验 + 图表 spec 自动修正重跑。

**修复**

- `dev` extra 补齐 fastapi / python-multipart。此前只装 `[dev]` 跑测试，
  `tests/test_api.py` 会在收集阶段 `ModuleNotFoundError`（本机因 venv 里手工装过
  fastapi 而一直掩盖）。

**工程**

- CI 的 docker job 增加 api 冒烟：起容器 → 鉴权 401 → 202 提交 → 轮询 → 取
  `result.json` 与 `report.md`。
- 文档：AI 协作记录、技术栈对照、改进记录；旧规划归档到 `spec/archive/`。

**升级提示**

- 想要 HTTP 接口：`pip install "mtsa-agent[api]"`，然后 `mtsa-api --port 8000`
  （需先设 `MTSA_API_TOKEN`）。
- 想逐位复现文档里的 12 项规则计划：`--planner rule`。

## 0.3.0 — 2026-10-24

PyPI 首个公开版本：CLI（`mtsa run` / `mtsa demo`）、Docker 镜像、天气与股票两个场景、
CI 全量测试与 wheel 构建。该版本不含 HTTP 接口，也没有 `[api]` extra。
