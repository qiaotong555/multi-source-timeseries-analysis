# -*- coding: utf-8 -*-
"""`mtsa-api` 的入口垫片：没装 HTTP 依赖时给一句人话，而不是 ModuleNotFoundError。

真正的实现在 `mtsa.api`，但那个模块在导入时就要 fastapi；把导入延后到这里，
才能只在用户真的执行 `mtsa-api` 时才提示"装哪个 extra"。
"""
from __future__ import annotations

import sys
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    try:
        from .api import main as api_main
    except ModuleNotFoundError as exc:              # 只拦 HTTP 依赖，别的照常抛
        missing = str(exc).split("'")[1] if "'" in str(exc) else str(exc)
        if missing in ("fastapi", "uvicorn", "multipart", "starlette"):
            print('缺少 HTTP 依赖（%s）。安装：pip install "mtsa-agent[api]"' % missing,
                  file=sys.stderr)
            return 2
        raise
    return api_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())