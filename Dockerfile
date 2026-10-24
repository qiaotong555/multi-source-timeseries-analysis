# 多源时序分析 Agent —— 最小可运行镜像
#
# 三个容易踩的点，写在这里而不是只写进文档：
# 1. PNG 静态图由 kaleido 1.x 导出，它需要**真实浏览器**（不是纯 Python 渲染）；
#    这里装 Debian 的 chromium。没有浏览器时流水线仍然成功：HTML 图与 report.md 正常，
#    只有 PNG 失败并在 warnings 里写明原因——这是设计，不是静默降级。
# 2. 图表与报告是中文，必须装 CJK 字体，否则图里全是方块。
# 3. 非 root 运行；/data 放输入，/out 放产物。
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        chromium \
        fonts-noto-cjk \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 1000 mtsa \
    && mkdir -p /data /out && chown -R mtsa /data /out
USER mtsa
WORKDIR /data

ENTRYPOINT ["mtsa"]
CMD ["list-scenes"]
