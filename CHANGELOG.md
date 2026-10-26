# 变更记录

版本号与 PyPI 上的发行版一一对应。每个版本的能力边界与实测数字见 `README.md`
第 3、4 节，完整评估结论在 `evals/`。

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
