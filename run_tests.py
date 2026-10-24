# -*- coding: utf-8 -*-
"""低内存、可超时的测试入口：逐文件跑 pytest。

为什么需要它
------------
本机单进程跑全量时会出现「CPU 归零、无报错、停在同一位置」的卡死：
numpy/OpenBLAS 在内存紧张时重试分配，pytest 既不报错也不退出。
把测试按文件拆成独立进程可以绕开这一点，但没有汇总和超时就会变成「卡了也不知道卡在哪」。

本入口做四件事：
1. 每个测试文件一个子进程，互不污染；
2. 每个文件可设超时，超时即杀掉并标记 STALLED，而不是无限等待；
3. 固定 `MTSA_PLANNER=rule`、`OPENBLAS_NUM_THREADS=1`，避免联网与线程抢占内存；
4. 汇总通过数并写 `evals/test_logs/summary.json`。

用法：
    python run_tests.py                 # 全部文件，默认 900 s/文件
    python run_tests.py --timeout 300
    python run_tests.py --only "test_viz_export.py"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent
TESTS = ROOT / "tests"
LOG_DIR = ROOT / "evals" / "test_logs"
BASETEMP = ROOT / "artifacts" / "_pt_lowmem"

_PASSED_RE = re.compile(r"(\d+) passed")
_FAILED_RE = re.compile(r"(\d+) failed")
_ERROR_RE = re.compile(r"(\d+) error")


def parse_counts(output: str) -> dict[str, int]:
    """从 pytest 输出里取通过/失败/错误数；取最后一次出现，避免摘要里的引用干扰。"""
    def _last(pattern: re.Pattern[str]) -> int:
        hits = pattern.findall(output)
        return int(hits[-1]) if hits else 0

    return {"passed": _last(_PASSED_RE), "failed": _last(_FAILED_RE),
            "errors": _last(_ERROR_RE)}


def discover(pattern: str, only: Optional[str]) -> list[Path]:
    if only:
        names = [n.strip() for n in only.split(",") if n.strip()]
        return [TESTS / n for n in names if (TESTS / n).exists()]
    return sorted(TESTS.glob(pattern))


def build_command(path: Path) -> list[str]:
    """构造单文件 pytest 命令。

    不要在这里再加 `-q`：pyproject 的 addopts 已经是 `-q`，再加一次变成 `-qq`，
    pytest 会连「N passed」的汇总行一起省掉，runner 就无法统计通过数。
    """
    return [sys.executable, "-m", "pytest", str(path),
            "--basetemp", str(BASETEMP / path.stem)]


def sanitize_paths(text: str) -> str:
    """把本机绝对路径换成占位符，日志要进公开仓库。

    不能只做 `str(ROOT)` 字符串替换：pytest 的警告段落是子进程按本地代码页写出的，
    解码后用户名可能已经变成替换字符，前缀匹配不上。这里按「目录结构」用正则兜底。
    """
    text = text.replace(str(ROOT), "<repo>")
    text = re.sub(
        r"C:[\\/]Users[\\/][^\\/\r\n]{1,64}[\\/]WorkBuddy[\\/][^\\/\r\n]*[\\/]"
        r"multi_source_timeseries_analysis",
        "<repo>", text,
    )
    text = re.sub(r"C:[\\/]Users[\\/][^\\/\r\n]{1,64}(?=[\\/])", "<home>", text)
    return text


def run_file(path: Path, timeout: float, logs: Path) -> dict[str, Any]:
    name = path.stem
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{name}.log"
    env = os.environ.copy()
    env.setdefault("MTSA_PLANNER", "rule")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)

    cmd = build_command(path)
    started = time.perf_counter()
    timed_out = False
    try:
        cp = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True,
                            timeout=timeout)
        code: Optional[int] = cp.returncode
        raw = (cp.stdout or b"") + b"\n" + (cp.stderr or b"")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        code = None
        raw = (exc.stdout or b"") + b"\n" + (exc.stderr or b"")
    duration = round(time.perf_counter() - started, 2)
    output = sanitize_paths(raw.decode("utf-8", errors="replace"))

    counts = parse_counts(output)
    status = "STALLED" if timed_out else ("PASS" if code == 0 else "FAIL")
    header = (f"# {name}\n# status={status} exit={code} duration={duration}s "
              f"passed={counts['passed']} failed={counts['failed']} errors={counts['errors']}\n")
    log_path.write_text(header + output, encoding="utf-8")

    return {"file": path.name, "status": status, "exit": code,
            "duration_s": duration, **counts, "log": str(log_path.relative_to(ROOT))}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    passed_tests = sum(r["passed"] for r in rows)
    return {
        "files": total,
        "passed_files": sum(1 for r in rows if r["status"] == "PASS"),
        "failed_files": sum(1 for r in rows if r["status"] == "FAIL"),
        "stalled_files": sum(1 for r in rows if r["status"] == "STALLED"),
        "tests_passed": passed_tests,
        "tests_failed": sum(r["failed"] for r in rows),
        "errors": sum(r["errors"] for r in rows),
        "total_duration_s": round(sum(r["duration_s"] for r in rows), 2),
        "ok": bool(rows) and all(r["status"] == "PASS" for r in rows),
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="逐文件跑测试（低内存、带超时）")
    ap.add_argument("--timeout", type=float, default=900.0, help="单个文件超时秒数，默认 900")
    ap.add_argument("--only", default=None, help="只跑指定文件，逗号分隔")
    ap.add_argument("--pattern", default="test_*.py", help="文件匹配模式")
    ap.add_argument("--summary", default=str(LOG_DIR / "summary.json"))
    ap.add_argument("--dry", action="store_true", help="只列文件")
    args = ap.parse_args(argv)

    files = discover(args.pattern, args.only)
    if not files:
        print("没有匹配的测试文件", file=sys.stderr)
        return 2
    if args.dry:
        print("共 %d 个文件：" % len(files))
        for f in files:
            print("  " + f.name)
        return 0

    BASETEMP.mkdir(parents=True, exist_ok=True)   # pytest 只建叶子目录，父目录必须已存在
    rows = []
    for f in files:
        row = run_file(f, args.timeout, LOG_DIR)
        rows.append(row)
        print("  %-28s %-7s %6.1fs  passed=%d failed=%d" % (
            row["file"], row["status"], row["duration_s"],
            row["passed"], row["failed"]))

    summary = summarize(rows)
    summary["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary["timeout_s"] = args.timeout
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps({"summary": summary, "files": rows},
                                             ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print("汇总：文件 %d（通过 %d / 失败 %d / 卡死 %d），测试通过 %d，失败 %d，错误 %d，耗时 %.1fs" % (
        summary["files"], summary["passed_files"], summary["failed_files"],
        summary["stalled_files"], summary["tests_passed"], summary["tests_failed"],
        summary["errors"], summary["total_duration_s"]))
    print("明细：", args.summary)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())