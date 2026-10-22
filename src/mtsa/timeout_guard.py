# -*- coding: utf-8 -*-
"""进程级超时保护：分析超时即终止，返回可预期的失败而不是无限等待。

为什么必须用子进程
------------------
Python 无法从外部中断正在执行的 C 扩展调用（numpy、statsmodels、LAPACK）。
主线程里「计时器 + 检查点」的方案只能发现「上一段耗时过长」，无法中止一个
已经开始且不会返回的调用——而那恰恰就是团队最怕的形态：无输出、无报错、
无法判断是正在算还是已经挂死。

真正的防护只有一种：把工作放进子进程，超时直接终止进程。

设计取舍
--------
1. **不改 `run_analysis` 本身。** 库调用者仍可直接调用，既有行为与测试完全不变；
   超时保护是可选外壳，由 CLI 默认启用。
2. **用临时 JSON 文件传参**，不走 pickle，避免 Windows spawn 与跨版本序列化的坑。
3. **不改产物格式。** 子进程照常把 `result.json` 写到输出目录，父进程读回即可；
   超时被终止时该文件可能不存在或残留半成品，因此在 `details` 里如实标注。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from typing import Optional

from .models import (
    ERR_INTERNAL_ERROR,
    ERR_NODE_TIMEOUT,
    AnalysisRequest,
    AnalysisResult,
    ErrorDetail,
)

# 默认 10 分钟。按实测：20 万行（规模上限）完整流程约 2 分钟，
# 留 5 倍余量以容纳机器差异与图表渲染波动；超过这个量级基本可判定为异常。
DEFAULT_TIMEOUT_S = 600.0
# 超时终止后等待进程退出的时间
_KILL_GRACE_S = 5.0


def _failed(req: AnalysisRequest, code: str, message: str, **details) -> AnalysisResult:
    return AnalysisResult(
        request_id=req.request_id,
        run_id=details.pop("run_id", None) or f"run_{uuid.uuid4().hex[:8]}",
        status="failed",
        error=ErrorDetail(code=code, message=message, details=details),
        warnings=[],
        events=[],
    )


def _read_result(path: str) -> Optional[AnalysisResult]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return AnalysisResult.model_validate_json(f.read())
    except Exception:                             # noqa: BLE001  半成品文件按无结果处理
        return None


def run_with_timeout(request: AnalysisRequest,
                     timeout_s: Optional[float] = None,
                     verbose: bool = True) -> AnalysisResult:
    """在子进程中执行分析；超时终止并返回 `NODE_TIMEOUT`。

    `timeout_s <= 0` 表示不设超时，此时仍走子进程（保留隔离），只是不主动终止。
    """
    limit = DEFAULT_TIMEOUT_S if timeout_s is None else float(timeout_s)

    # 必须固定输出目录：子进程的 run_id 由它自己生成，父进程只能靠目录读回结果
    outdir = request.output_dir or os.path.join(
        "artifacts", f"run_{uuid.uuid4().hex[:8]}")
    req = request.model_copy(update={"output_dir": outdir})
    os.makedirs(outdir, exist_ok=True)

    payload_path = os.path.join(outdir, "_request.json")
    with open(payload_path, "w", encoding="utf-8") as f:
        f.write(req.model_dump_json())

    cmd = [sys.executable, "-m", "mtsa.worker", payload_path]
    # 子进程必须能找到 mtsa：不依赖调用方是否设了 PYTHONPATH
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = pkg_parent + (os.pathsep + existing if existing else "")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              timeout=limit if limit > 0 else None)
    except subprocess.TimeoutExpired:
        # subprocess 已终止子进程；这里只需回收并如实报告
        return _failed(
            req, ERR_NODE_TIMEOUT,
            f"分析超过 {limit:.0f} 秒未完成，已终止子进程",
            timeout_s=limit, output_dir=outdir,
            hint=("常见原因：数据规模超出承受范围、单个检验在异常输入上退化。"
                  "请先确认行数是否超过上限；如需完整排查可加 --timeout 调大后重跑"),
            partial_artifacts=os.listdir(outdir) if os.path.isdir(outdir) else [],
        )

    result_path = os.path.join(outdir, "result.json")
    parsed = _read_result(result_path)
    if parsed is not None:
        if verbose and parsed.status != "succeeded":
            print(f"[{parsed.error.code if parsed.error else 'FAILED'}] "
                  f"{parsed.error.message if parsed.error else '分析失败'}")
        return parsed

    # 子进程结束但没有产出 result.json：进程崩溃、导入失败或被杀
    tail = (proc.stderr or "").strip().splitlines()[-8:]
    return _failed(
        req, ERR_INTERNAL_ERROR,
        "子进程已退出但未产出 result.json",
        returncode=proc.returncode, output_dir=outdir,
        stderr_tail=tail,
        hint="多为环境问题（依赖缺失、导入失败）；stderr_tail 为最后几行日志",
    )


def main(argv: Optional[list[str]] = None) -> int:
    """自检入口：对一个请求文件执行带超时的分析，便于手工验证超时行为。"""
    import argparse

    ap = argparse.ArgumentParser(description="带超时保护地执行一次分析")
    ap.add_argument("request_json", help="AnalysisRequest 的 JSON 文件")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                    help=f"超时秒数，默认 {DEFAULT_TIMEOUT_S:.0f}；0 表示不限制")
    args = ap.parse_args(argv)

    with open(args.request_json, encoding="utf-8") as f:
        req = AnalysisRequest.model_validate_json(f.read())
    res = run_with_timeout(req, timeout_s=args.timeout)
    print(json.dumps({"status": res.status,
                      "code": res.error.code if res.error else None,
                      "run_id": res.run_id}, ensure_ascii=False))
    return 0 if res.status == "succeeded" else 1


if __name__ == "__main__":
    sys.exit(main())
