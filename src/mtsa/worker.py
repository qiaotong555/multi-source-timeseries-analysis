# -*- coding: utf-8 -*-
"""子进程入口：从 JSON 文件读请求并执行一次分析。

只供 `timeout_guard.run_with_timeout` 使用。独立成模块而不是塞进 cli，
是为了让子进程的命令行参数只有一个文件路径，避免把 --param 等结构再拼一遍。
"""
from __future__ import annotations

import sys

from .engine import run_analysis
from .models import AnalysisRequest
from .scenarios import stock as _stock      # noqa: F401  注册适配器
from .scenarios import weather as _weather  # noqa: F401


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("用法：python -m mtsa.worker <request.json>", file=sys.stderr)
        return 2
    with open(argv[0], encoding="utf-8") as f:
        request = AnalysisRequest.model_validate_json(f.read())
    result = run_analysis(request, verbose=False)
    # 结果已由 run_analysis 写入 output_dir/result.json，这里只回退出码
    return 0 if result.status == "succeeded" else 1


if __name__ == "__main__":
    sys.exit(main())
