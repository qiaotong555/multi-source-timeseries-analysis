# -*- coding: utf-8 -*-
"""命令行入口：run / demo / list-scenes。

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

from . import cost as cost_mod
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


def _resolve_mode(cli_mode: str | None, env: dict | None = None) -> str:
    """确定运行模式：显式 CLI > MTSA_AGENT > pipeline。

    默认仍是 pipeline，避免补上模型配置后，原本的固定流水线运行被悄悄改成 Agent。
    """
    if cli_mode:
        return cli_mode
    env = os.environ if env is None else env
    raw = (env.get("MTSA_AGENT", "") or "").strip().lower()
    if raw in ("agent", "on", "1", "true", "yes"):
        return "agent"
    if raw == "auto":
        return "auto"
    return "pipeline"


def _version() -> str:
    """装了包就报包版本；直接用源码跑（未安装）时如实写 uninstalled。"""
    try:
        from importlib.metadata import version
        return version("mtsa-agent")
    except Exception:                            # noqa: BLE001  未安装时 metadata 不存在
        return "uninstalled"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mtsa", description="多源时序数据量化分析与可视化 Agent")
    p.add_argument("--version", action="version", version=f"mtsa {_version()}")
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
    r.add_argument("--mode", choices=("pipeline", "agent", "auto"), default=None,
                   help="运行模式：pipeline 固定流水线；agent 由模型选择工具（本轮仅天气）；"
                        "auto 有模型且场景支持时走 agent，否则回退 pipeline。"
                        "缺省读 MTSA_AGENT，再缺省 pipeline。"
                         "注意：这与计划器不是同一个开关——分析哪些项由 --planner 决定")
    r.add_argument("--planner", choices=("auto", "rule", "llm"), default=None,
                   help="计划器（决定分析哪些项/哪些列）：rule 固定规则计划（12 项，"
                        "与评估口径一致）；llm 强制模型（配置不全时回退规则并告警）；"
                        "auto 按模型可用性选。缺省读 MTSA_PLANNER，再缺省 auto")
    r.add_argument("--dry-run", action="store_true",
                   help="只预演：打印字段摘要、场景、计划与将要执行的工具，然后退出；"
                        "不写产物、不运行检测器。用于确认分析目标是否正确")
    r.add_argument("--agent-max-steps", type=int, default=8, metavar="N",
                   help="Agent 模式最大决策步数，默认 8")
    r.add_argument("--forecast", action="store_true",
                   help="附加预测：点预测 + 90%% 区间 + walk-forward 回测（仅天气）")
    r.add_argument("--forecast-horizon", type=int, default=None, metavar="N",
                   help="预测步数，默认一个季节周期，上限 168")
    r.add_argument("--agent-scope", choices=("guided", "free"), default="guided",
                   help="Agent 范围策略：guided 按问题关键词设下限（步数稳定）；"
                        "free 完全由模型决定范围（步数会波动）")
    r.add_argument("--agent-max-extra-steps", type=int, default=1, metavar="N",
                   help="guided 模式下限之外允许追加的步数，默认 1")
    r.add_argument("--agent-max-tokens", type=int, default=0, metavar="N",
                   help="Agent 累计 token 上限，默认 0（不限制）；到限前一次调用会被拦下，"
                        "以 token_budget 结束")
    r.add_argument("--strict-verify", action="store_true",
                   help="校验时额外抽查「结论里的数值能否在指标里找到」；"
                        "结果写进 data.verify.advisory，不改变成败")
    r.add_argument("--max-cost", type=float, default=None, metavar="AMOUNT",
                   help="金额预算上限，币种与价目表 currency 一致；超过只标记、不中止。"
                        "缺省读 MTSA_MAX_COST。设了上限却没配价目表会直接失败")
    r.add_argument("--price-table", default=None, metavar="PATH",
                   help="价目表 JSON 路径，用于把 token 估算成金额；缺省读环境变量 "
                        "MTSA_PRICE_TABLE。都未配置时不做金额换算，也不按 0 计价；"
                        "路径无效会直接失败")
    r.add_argument("--replay", default=None, metavar="RESULT_JSON",
                   help="从一次 agent 运行的 result.json 离线重放；不调用模型")
    r.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                   metavar="SEC",
                   help=f"分析超时秒数，默认 {DEFAULT_TIMEOUT_S:.0f}；0 表示不设超时且不进子进程")

    d = sub.add_parser("demo", help="一键演示：合成数据 → 跑完整链路 → 出报告；不需要模型、数据或网络")
    d.add_argument("--out", default="mtsa_demo", help="输出目录，默认 ./mtsa_demo")
    d.add_argument("--seed", type=int, default=DEFAULT_SEED, help="合成数据种子，默认与全局一致")
    d.add_argument("--question", default=None, help="覆盖默认的分析问题")
    d.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, metavar="SEC",
                   help=f"分析超时秒数，默认 {DEFAULT_TIMEOUT_S:.0f}；0 表示进程内直接执行")

    sub.add_parser("list-scenes", help="列出已注册场景")
    return p


def _resolve_max_cost(cli_value: float | None) -> float | None:
    """金额上限：显式参数 > 环境变量；格式错误直接退出而不是悄悄忽略。"""
    value = cli_value
    if value is None:
        try:
            value = cost_mod.max_cost_from_env()
        except cost_mod.CostError as exc:
            raise SystemExit(f"[COST_CONFIG_INVALID] {exc}") from None
    if value is not None and value <= 0:
        raise SystemExit(f"[COST_CONFIG_INVALID] --max-cost 必须大于 0，实际为 {value}")
    return value


def _print_preview(p: dict) -> None:
    """把 dry-run 的预览打印成人读的样子。"""
    inp, q, sc, pl = p["input"], p["quality"], p["scene"], p["plan"]
    print("[dry-run] 只做预演：没有写入任何文件，也没有运行检测器。\n")
    print(f"数据   {inp['path']}（{inp['format']}）")
    print(f"       {inp['rows']:,} 行 × {len(inp['columns'])} 列"
          f" | 时间列 {inp['time_column']} {inp['time_start']} → {inp['time_end']}"
          f" | 采样 {inp['interval']}"
          f" | 分组 {inp['group_column'] or '无'}（{inp['groups']} 组）")
    nums = inp["numeric_columns"]
    shown = ", ".join(nums[:8]) + ("  …" if len(nums) > 8 else "")
    print(f"       数值列 {len(nums)} 个：{shown}")
    print(f"       质量问题 {q['issues']} 条：最大缺失率 {q['max_missing_ratio']:.2%}"
          f" / 重复时间戳 {q['duplicates']}"
          f" / 乱序 {q['out_of_order']}{'（多站长表按各组合计）' if inp['groups'] > 1 else ''}"
          f" / 采样缺口 {q['gaps']} / 异常候选 {q['anomaly_candidates']}")
    print(f"\n场景   {sc['scene_id']}（置信度 {sc['confidence']:.3f}，{sc['source']}）")
    print(f"模式   {p['mode']}（请求 {p['mode_requested']}）")
    period = pl["parameters"].get("seasonal_period")
    print(f"计划   {pl['plan_source']}（{pl['planner_version']}）"
          f" / {len(pl['analyses'])} 项分析 / 目标列 {len(pl['target_columns'])} 个"
          f" / 周期 {period if period is not None else '—'}")
    print("\n将要执行：")
    for i, t in enumerate(p["tools"], 1):
        print(f"  {i}. {t['name']:<20} {t['description']}")
    if p["warnings"]:
        print("\n先留意：")
        for w in p["warnings"]:
            print(f"  - {w}")
    print(f"\n产物目录（本次不会创建）：{p['output_dir']}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "demo":
        from .demo import DEMO_CSV_NAME, DEMO_QUESTION, run_demo

        out_dir = os.path.abspath(args.out)
        result = run_demo(out_dir, seed=args.seed,
                          question=args.question or DEMO_QUESTION, timeout=args.timeout)
        if result.status != "succeeded":
            code = result.error.code if result.error else "FAILED"
            message = result.error.message if result.error else "演示失败"
            print(f"[{code}] {message}", file=sys.stderr)
            return 1
        csv_path = os.path.join(out_dir, DEMO_CSV_NAME)
        rows = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
        print(f"完成：run_id={result.run_id}  模式=pipeline（未调用模型）")
        print(f"数据：{csv_path}（{rows} 行，合成数据，seed={args.seed}）")
        print(f"报告：{os.path.join(out_dir, 'report.md')}")
        print(f"结果：{os.path.join(out_dir, 'result.json')}")
        print('换成自己的数据：mtsa run --scene weather --input <你的.csv> --question "分析温度的趋势和异常"')
        return 0

    if args.cmd == "list-scenes":
        for k, ad in _REGISTRY.items():
            print(f"{k}\t{ad.name}")
        return 0

    # 冲突先于 replay 解析判断：否则不存在的文件会先报 REPLAY_INVALID，掩盖真正原因
    if getattr(args, "dry_run", False) and args.replay:
        print("--dry-run 不能与 --replay 同时使用：重放的是已有轨迹，没有可确认的计划",
              file=sys.stderr)
        return 1

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

    mode = _resolve_mode(args.mode)
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
        mode=mode,
        planner=args.planner,
        agent_max_steps=args.agent_max_steps,
        agent_scope=args.agent_scope,
        agent_max_extra_steps=args.agent_max_extra_steps,
        agent_max_tokens=args.agent_max_tokens,
        price_table=args.price_table,
        max_cost=_resolve_max_cost(args.max_cost),
        strict_verify=args.strict_verify,
        replay_from=args.replay,
        forecast=args.forecast,
        forecast_horizon=args.forecast_horizon,
    )
    if args.dry_run:
        # 不进子进程：预演不写任何文件，也不需要超时保护
        result = run_analysis(req, verbose=False, dry_run=True)
        if result.status != "succeeded":
            code = result.error.code if result.error else "FAILED"
            message = result.error.message if result.error else "预演失败"
            print(f"[{code}] {message}", file=sys.stderr)
            return 1
        _print_preview(result.data["preview"])
        return 0
    if args.timeout and args.timeout > 0:
        result = run_with_timeout(req, timeout_s=args.timeout, verbose=False)
    else:
        result = run_analysis(req)

    if result.status == "succeeded":
        plan = (result.data or {}).get("plan") or {}
        outdir = os.path.dirname(result.artifacts[0].path) if result.artifacts else req.output_dir
        effective = result.datasets[0].get("mode") if result.datasets else req.mode
        print(f"完成：run_id={result.run_id}  场景={plan.get('scene_id')}  "
              f"置信度={plan.get('scene_confidence')}  模式={effective}")
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
