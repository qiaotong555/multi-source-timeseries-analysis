# 第三方组件与许可证

本文件列出运行与构建所需的第三方组件及其许可证。它**不替代**各项目的许可证原文；
对外分发（含打 wheel、做镜像）时应随附对应许可证副本。

## 运行时依赖

| 组件 | 本项目实测版本 | 许可证 | 用途 |
| --- | --- | --- | --- |
| pandas | 3.0.6 | BSD-3-Clause | 数据框与重采样 |
| numpy | 2.5.3 | BSD-3-Clause | 数值计算 |
| scipy | 1.18.1 | BSD-3-Clause | 信号处理、统计分布 |
| statsmodels | 0.15.0 | BSD-3-Clause | ADF / KPSS / Ljung-Box / STL |
| plotly | 7.1.0 | MIT | 交互图与静态图 |
| kaleido | 1.4.0 | MIT | 静态图 PNG 导出（需浏览器） |
| pydantic | 2.13.5 | MIT | 输入输出与证据 schema |
| openpyxl | 3.1.5 | MIT | 读 `.xlsx`（可选 `mtsa[io]`） |
| pyarrow | 25.0.1 | Apache-2.0 | 读 `.parquet`（可选 `mtsa[io]`） |
| h5py | 3.16.0 | BSD-3-Clause | 读 `.h5`（可选 `mtsa[io]`） |

## 开发与文档工具

| 组件 | 版本 | 许可证 | 用途 |
| --- | --- | --- | --- |
| pytest | 9.1.1 | MIT | 测试 |
| python-docx | 1.2.0 | MIT | 生成 Word 演示文档（可选 `mtsa[docx]`） |

## 需要知道的实践细节

1. **BSD-3-Clause 与 MIT 都允许闭源分发**，但要求保留版权声明与许可证文本。
   这也是本项目代码采用 Apache-2.0 的依据：组件的既有授权不因本项目的许可证而改变，
   本项目只对自己的代码授权。
2. **kaleido 1.x 需要真实浏览器**（Chrome / Edge / Chromium）导出 PNG；
   浏览器本身不是本项目的依赖，也不随本项目分发。
3. **字体**：容器镜像里安装的 Noto CJK 字体按其自身许可（SIL OFL）使用，
   仅用于渲染中文图表，不随源码分发。
4. 版本号来自本项目实测通过的组合（见 `pyproject.toml`）；升级依赖前请复跑全量测试。

## 复核方法

```bash
pip show pandas numpy scipy statsmodels plotly kaleido pydantic
```

或直接查看各项目仓库的 `LICENSE` 文件。若本表与实际情况不一致，以组件自身的许可证原文为准。
