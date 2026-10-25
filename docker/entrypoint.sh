#!/bin/sh
# 同一个镜像跑两种入口：
#   docker run ... <img>               → mtsa CLI（默认 CMD 是 list-scenes）
#   docker run ... <img> api           → HTTP 薄壳（uvicorn，监听 0.0.0.0）
# 不把 ENTRYPOINT 直接写成 uvicorn，是为了让原来的 CLI 用法一行都不用改。
set -e
if [ "$1" = "api" ]; then
    shift
    exec uvicorn mtsa.api:app --host 0.0.0.0 --port "${MTSA_API_PORT:-8000}" "$@"
fi
exec mtsa "$@"