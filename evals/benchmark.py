# -*- coding: utf-8 -*-
"""性能与规模基准。

解决的问题
----------
此前从未测量过分析耗时与内存占用，因此无法回答「1 万行要多久」「10 万行会不会
拖垮机器」「100 万行还能不能跑」。没有这组数字，任何「性能可接受」的说法都是
没有依据的。

测量方式
--------
1. **子进程隔离**：每个规模在独立子进程中运行，主进程设置超时并可强制终止。
   这样做是因为 100 万行有 OOM 风险，若在主进程内测量，一次崩溃会带走整个基准。
2. **计时与内存分开测**：`tracemalloc` 会让执行变慢数倍，若与计时同轮次运行，
   报出的耗时不真实。因此每个规模跑两次——一次纯计时，一次只读内存峰值。
3. **只测分析链，不测图表**：图表渲染依赖 kaleido 外部进程，耗时与数据行数弱相关、
   与图片数量强相关，混入分析链会掩盖真正的规模效应。图表成本单独说明。

用法
----
    python -m evals.benchmark                 # 跑全部规模
    python -m evals.benchmark --scales 1e4 1e5
    python -m evals.benchmark --worker 10000 --mode timed   # 内部使用
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for _p in (SRC, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT_PATH = os.path.join(ROOT, "evals", "benchmark_results.json")

# 规模档位：总行数（两站合计）
SCALES = [("1e4", 10_000), ("1e5", 100_000), ("1e6", 1_000_000)]
WORKER_TIMEOUT = 300          # 单进程超时（秒）


# ---------------- 数据构造 ----------------

def build_frame(n_rows: int, stations: int = 2, seed: int = 20260927) -> pd.DataFrame:
    """构造与天气场景同结构的合成数据（向量化生成，避免逐点循环拖慢准备阶段）。"""
    per = max(1, n_rows // stations)
    idx = pd.date_range("2024-01-01", periods=per, freq="h")
    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    rng = np.random.default_rng(seed)

    frames = []
    for s in range(stations):
        phase = 0.0 if s == 0 else 0.6
        temp = (15.0
                + 12.0 * np.sin(2 * np.pi * (doy - 100) / 365.25 + phase)
                + 6.0 * np.sin(2 * np.pi * (hour - 14) / 24.0 + phase)
                + rng.normal(0.0, 1.5, per))
        humidity = np.clip(65.0 - 0.8 * (temp - 15.0) + rng.normal(0.0, 8.0, per), 5.0, 100.0)
        pressure = 1013.0 + 8.0 * np.sin(2 * np.pi * doy / 365.25) + rng.normal(0.0, 2.3, per)
        wind = np.abs(3.0 + rng.normal(0.0, 2.5, per) + 1.2 * np.sin(2 * np.pi * hour / 24.0))
        rain_u = rng.random(per)
        rainfall = np.where(rain_u < 0.94, 0.0, rng.exponential(2.5, per) * (rain_u - 0.93) / 0.07)

        frames.append(pd.DataFrame({
            "timestamp": idx,
            "station": f"STA_{'AB'[s % 2]}",
            "temperature": temp, "humidity": humidity, "pressure": pressure,
            "wind_speed": wind, "rainfall": rainfall,
        }))
    return pd.concat(frames, ignore_index=True)


# ---------------- 分析链 ----------------

def run_chain(csv_path: str) -> tuple[dict, dict]:
    """按引擎的节点顺序逐段计时，返回 (各节点耗时, 结果摘要)。"""
    from mtsa import anomaly, freq, io_loaders, metrics, preprocess, profiler, quality, stats_tests
    from mtsa.scenarios import stock as _stock  # noqa: F401  确保适配器注册
    from mtsa.scenarios import weather as _weather  # noqa: F401
    from mtsa.scenarios.base import get_adapter

    t: dict[str, float] = {}
    summary: dict[str, int] = {}

    def step(name, fn):
        t0 = time.perf_counter()
        r = fn()
        t[name] = round(time.perf_counter() - t0, 4)
        return r

    raw = step("load", lambda: io_loaders.load_dataset(csv_path))
    profile, parsed = step("profile", lambda: profiler.profile_dataset(raw))
    qrep = step("quality", lambda: quality.check_quality(raw, profile, parsed))
    adapter = get_adapter("weather")
    plan = step("plan", lambda: adapter.plan("分析温度的趋势、季节性与极端事件",
                                             profile, qrep, raw.frame))
    clean = step("clean", lambda: preprocess.clean_dataset(raw, profile, parsed, plan))
    summary["rows"] = int(len(clean.frame))
    summary["columns"] = int(len(clean.frame.columns))

    mtab = step("metrics", lambda: metrics.compute_metrics(clean, plan))
    step("scenario_metrics", lambda: adapter.extra_metrics(clean, plan))
    summary["metrics"] = int(len(mtab.rows))
    step("spectrum", lambda: freq.analyze_spectrum(clean, plan))
    step("stat_tests", lambda: stats_tests.run_stat_tests(clean, plan))
    anoms = step("anomalies", lambda: anomaly.detect_anomalies(clean, plan))
    cps = step("change_points", lambda: anomaly.detect_change_points(clean, plan))
    summary["anomalies"] = int(len(anoms))
    summary["change_points"] = int(len(cps))
    return t, summary


def _worker(n_rows: int, mode: str) -> int:
    """子进程入口：生成数据、跑分析链、输出 JSON。"""
    tmp = tempfile.mkdtemp(prefix="mtsa_bench_")
    try:
        csv_path = os.path.join(tmp, "bench.csv")
        df = build_frame(n_rows)
        prep_t0 = time.perf_counter()
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        prep_s = round(time.perf_counter() - prep_t0, 4)
        del df

        if mode == "memory":
            tracemalloc.start()

        timings, summary = run_chain(csv_path)

        peak_mb = None
        if mode == "memory":
            _cur, peak_b = tracemalloc.get_traced_memory()
            peak_mb = round(peak_b / 1024 / 1024, 1)
            tracemalloc.stop()

        print(json.dumps({
            "rows": n_rows, "mode": mode,
            "prep_s": prep_s,
            "timings": timings,
            "total_s": round(sum(timings.values()), 4),
            "peak_mb": peak_mb,
            "summary": summary,
        }, ensure_ascii=False))
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------- 主控 ----------------

def _run_one(n_rows: int, mode: str) -> dict:
    cmd = [sys.executable, "-m", "evals.benchmark", "--worker", str(n_rows), "--mode", mode]
    try:
        cp = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=WORKER_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"rows": n_rows, "mode": mode, "status": "timeout",
                "timeout_s": WORKER_TIMEOUT}
    if cp.returncode != 0:
        tail = (cp.stderr or "")[-400:]
        kind = "oom" if ("MemoryError" in tail or "Unable to allocate" in tail) else "error"
        return {"rows": n_rows, "mode": mode, "status": kind, "stderr_tail": tail}
    try:
        data = json.loads(cp.stdout.strip().splitlines()[-1])
        data["status"] = "ok"
        return data
    except Exception as exc:                       # noqa: BLE001
        return {"rows": n_rows, "mode": mode, "status": "parse_error",
                "detail": str(exc), "stdout_tail": cp.stdout[-300:]}


def benchmark(scales: list[tuple[str, int]] | None = None) -> dict:
    scales = scales or SCALES
    out = {"env": {"python": sys.version.split()[0], "platform": sys.platform,
                   "cpu": os.cpu_count()},
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "timeout_s": WORKER_TIMEOUT, "scales": []}

    for label, rows in scales:
        timed = _run_one(rows, "timed")
        mem = _run_one(rows, "memory") if timed.get("status") == "ok" else {"status": "skipped"}
        out["scales"].append({
            "label": label, "rows": rows,
            "status": timed.get("status"),
            "total_s": timed.get("total_s"),
            "peak_mb": mem.get("peak_mb"),
            "timings": timed.get("timings"),
            "summary": timed.get("summary"),
            "prep_s": timed.get("prep_s"),
            "memory_status": mem.get("status"),
        })
        print(f"[{label}] status={timed.get('status')} total={timed.get('total_s')}s "
              f"peak={mem.get('peak_mb')}MB", flush=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="时序分析引擎性能基准")
    ap.add_argument("--scales", nargs="*", help="要跑的档位，如 1e4 1e5")
    ap.add_argument("--worker", type=int, help="内部使用：子进程行数")
    ap.add_argument("--mode", default="timed", choices=["timed", "memory"])
    args = ap.parse_args()

    if args.worker:
        return _worker(args.worker, args.mode)

    sel = None
    if args.scales:
        table = dict(SCALES)
        sel = [(k, table[k]) for k in args.scales if k in table]
        if not sel:
            print(f"未知档位：{args.scales}，可选 {list(table)}")
            return 2

    res = benchmark(sel)
    print()
    print("| 规模 | 行数 | 状态 | 分析链耗时(s) | 峰值内存(MB) |")
    print("| --- | --- | --- | --- | --- |")
    for s in res["scales"]:
        print(f"| {s['label']} | {s['rows']} | {s['status']} | "
              f"{s['total_s']} | {s['peak_mb']} |")
    print(f"\n已写入 {OUT_PATH}")
    return 0 if all(s["status"] == "ok" for s in res["scales"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
