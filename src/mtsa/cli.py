# -*- coding: utf-8 -*-
"""命令行入口：run / list-scenes。

用法：
    python -m mtsa.cli run --scene weather --input fixtures/weather_v1.csv \
        --question "分析温度的趋势、季节性与极端事件"

`run` 默认在子进程中执行并带超时（默认 600 秒）：Python 无法中止正在执行的
C 扩展调用，只有进程级终止才能真正防住「无输出、无报错」的卡死。
`--timeout 0` 退回进程内直接执行，便于调试与被外部调度器托管。
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

from .engine import run_analysis
from .models import AnalysisRequest, DEFAULT_SEED
from .timeout_guard import DEFAULT_TIMEOUT_S, run_with_timeout
from .scenarios import stock as _stock  # noqa: F401  注册
from .scenarios import weather as _weather  # noqa: F401
from .scenarios.base import _REGISTRY


def _parse_overrides(items: list[str]) -> dict:
    """把 `key=value` 列表解析为字典；整数优先转 int，其次 float，其余保留字符串。"""
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"参数格式应为 KEY=VALUE，收到：{item}")
        key, raw = item.split("=", 1)
        key, raw = key.strip(), raw.strip()
        if not key:
            raise SystemExit(f"参数名为空：{item}")
        try:
            out[key] = int(raw)
        except ValueError:
            try:
                out[key] = float(raw)
            except ValueError:
                out[key] = raw
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mtsa.cli", description="多源时序数据量化分析与可视化 Agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="执行一次分析")
    r.add_argument("--scene", choices=sorted(_REGISTRY), default=None,
                   help="场景标识；缺省时自动识别")
    r.add_argument("--input", default=None,
                   help="输入文件路径；使用 --replay 时可省略（从原结果读取）")
    r.add_argument("--question", default=None,
                   help="自然语言分析问题；使用 --replay 时可省略（沿用原问题）")
    r.add_argument("--time-column", default=None)
    r.add_argument("--value-columns", default=None, help="逗号分隔")
    r.add_argument("--group-column", default=None)
    r.add_argument("--seed", type=int, default=DEFAULT_SEED)
    r.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                   help="覆盖计划参数，可重复，如 --param anomaly_threshold=7.5")
    r.add_argument("--output", default=None, help="产物目录，默认 artifacts/<run_id>")
    r.add_argument("--request-id", default=None)
    r.add_argument("--mode", choices=("pipeline", "agent"), default="pipeline",
                   help="运行模式：pipeline 为固定流水线（默认）；agent 由模型选择分析工具，本轮仅天气场景")
    r.add_argument("--agent-max-steps", type=int, default=8, metavar="N",
                   help="Agent 模式最大决策步数，默认 8")
    r.add_argument("--replay", default=None, metavar="RESULT_JSON",
                   help="从一次 agent 运行的 result.json 离线重放；不调用模型")
    r.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                   metavar="SEC",
                   help=f"分析超时秒数，默认 {DEFAULT_TIMEOUT_S:.0f}；0 表示不设超时且不进子进程")

    sub.add_parser("list-scenes", help="列出已注册场景")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "list-scenes":
        for k, ad in _REGISTRY.items():
            print(f"{k}\t{ad.name}")
        return 0

    if args.replay:
        from .replay import ReplayError, load_replay
        try:
            source = load_replay(args.replay)
        except ReplayError as exc:
            print(f"[REPLAY_INVALID] {exc}", file=sys.stderr)
            return 1
        args.mode = "agent"
        args.input = args.input or source["input_path"]
        args.question = args.question or source["question"]
    if not args.input:
        print("缺少 --input（--replay 未能从原结果读取输入路径）", file=sys.stderr)
        return 1
    if not args.question:
        print("缺少 --question", file=sys.stderr)
        return 1

    req = AnalysisRequest(
        request_id=args.request_id or f"req_{datetime.now():%Y%m%d%H%M%S}",
        input_path=args.input,
        question=args.question,
        scene_id=args.scene,
        time_column=args.time_column,
        value_columns=args.value_columns.split(",") if args.value_columns else None,
        group_column=args.group_column,
        param_overrides=_parse_overrides(args.param),
        seed=args.seed,
        output_dir=args.output,
        mode=args.mode,
        agent_max_steps=args.agent_max_steps,
        replay_from=args.replay,
    )
    if args.timeout and args.timeout > 0:
        result = run_with_timeout(req, timeout_s=args.timeout, verbose=False)
    else:
        result = run_analysis(req)

    if result.status == "succeeded":
        plan = (result.data or {}).get("plan") or {}
        outdir = os.path.dirname(result.artifacts[0].path) if result.artifacts else req.output_dir
        print(f"完成：run_id={result.run_id}  场景={plan.get('scene_id')}  "
              f"置信度={plan.get('scene_confidence')}  模式={req.mode}")
        print(f"产物目录：{os.path.abspath(outdir) if outdir else '—'}")
        return 0

    code = result.error.code if result.error else "FAILED"
    message = result.error.message if result.error else "分析失败"
    print(f"[{code}] {message}", file=sys.stderr)
    details = result.error.details if result.error else {}
    if details.get("hint"):
        print(f"提示：{details['hint']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
