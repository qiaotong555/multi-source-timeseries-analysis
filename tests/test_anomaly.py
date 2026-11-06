# -*- coding: utf-8 -*-
"""异常检测与变化点测试：对照注入真值计算 Precision / Recall。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mtsa import anomaly, profiler, preprocess
from mtsa.io_loaders import build_spec
from mtsa.models import RawDataset
from .helpers import make_plan, make_weather_frame

MIN_PRECISION = 0.70
MIN_RECALL = 0.70


def _pipeline(df: pd.DataFrame, plan):
    raw = RawDataset(frame=df, spec=build_spec("mem.csv"), file_hash="0" * 40,
                     row_count=len(df), column_dtypes={c: str(t) for c, t in df.dtypes.items()})
    profile, parsed = profiler.profile_dataset(raw)
    clean = preprocess.clean_dataset(raw, profile, parsed, plan)
    return clean


def test_anomaly_precision_recall(make_plan):
    n = 3000
    df = make_weather_frame(n=n, seed=21, stations=1)
    rng = np.random.default_rng(99)
    sd = float(df["temperature"].std())
    truth = set()
    for _ in range(12):
        i = int(rng.integers(200, n - 200))
        df.loc[i, "temperature"] += (1 if rng.random() > 0.5 else -1) * (7.0 + 2.0 * rng.random()) * sd
        truth.add(str(df.loc[i, "timestamp"]))

    plan = make_plan("weather", ["temperature"], seasonal_period=24,
                     rolling_window=24, anomaly_threshold=4.0)
    clean = _pipeline(df, plan)
    detected = anomaly.detect_anomalies(clean, plan)
    got = {a.timestamp for a in detected if a.column == "temperature"}
    tp = len(truth & got)
    precision = tp / len(got) if got else 0.0
    recall = tp / len(truth)
    assert recall >= MIN_RECALL, f"Recall={recall:.3f} 低于 {MIN_RECALL}"
    assert precision >= MIN_PRECISION, f"Precision={precision:.3f} 低于 {MIN_PRECISION}"


def test_no_false_alarm_on_clean_series(make_plan):
    df = make_weather_frame(n=3000, seed=33, stations=1)
    plan = make_plan("weather", ["temperature"], seasonal_period=24, anomaly_threshold=4.5)
    clean = _pipeline(df, plan)
    detected = anomaly.detect_anomalies(clean, plan)
    rate = len(detected) / len(clean.frame)
    assert rate < 0.005, f"干净序列误报率过高：{rate:.4%}"


def test_change_point_locates_level_shift(make_plan):
    n = 2400
    df = make_weather_frame(n=n, seed=44, stations=1)
    shift_at = 1200
    df.loc[shift_at:, "temperature"] += 6.0      # 持续水平抬升 6°C
    plan = make_plan("weather", ["temperature"], seasonal_period=24, change_point_depth=2)
    clean = _pipeline(df, plan)
    cps = [c for c in anomaly.detect_change_points(clean, plan)
           if c.column == "temperature" and c.kind == "mean"]
    assert cps, "未检出变化点"
    best = min(cps, key=lambda c: abs(c.index - shift_at))
    assert abs(best.index - shift_at) <= 24, f"变化点定位偏差过大：{best.index} vs {shift_at}"


def test_constant_block_does_not_produce_change_point(make_plan):
    """局部散布为 0 时不得判显著。

    降水列有整段为 0 的区段。曾经的实现写成「jump > 0 就给 inf」，
    于是两侧中位数差 0.1 mm 被当成最高置信度的变化点报出来
    （实测 4 个降水列各报 2–3 个）。没有噪声模型就没有证据。
    """
    n = 3000
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    vals = np.zeros(n)
    vals[1500:] = 0.1                       # 常数 → 常数，差 0.1
    df = pd.DataFrame({"timestamp": idx, "temperature": 20.0,
                       "humidity": 50.0, "pressure": 1013.0,
                       "wind_speed": 2.0, "rainfall": vals})
    plan = make_plan("weather", ["rainfall"], seasonal_period=24, change_point_depth=2)
    clean = _pipeline(df, plan)
    cps = [c for c in anomaly.detect_change_points(clean, plan) if c.column == "rainfall"]
    assert not cps, [(c.index, c.statistic) for c in cps]


def test_variance_channel_is_disabled(make_plan):
    """方差突变通道已停用：纯方差变化（σ×2）不再报出。

    停用不是因为「数字不好看」，而是实测没有分辨力（`doc/A3方差通道.md`）：
    - 量纲错：σ→2σ 时跳变 ≈ σ、基线也 ≈ σ，比值恒 ≈1.0，够不到 2.0 的阈值；
    - 未注入时同一判据在四个列上各报一个 σ 比 2.4–3.4 的假点，σ×2 的真变化只有 2.1。

    这条测试锁住「不输出 `kind="variance"`」，避免以后无意中把它加回来。
    """
    df = make_weather_frame(n=3000, seed=71, stations=1)
    vals = df["temperature"].to_numpy(dtype=float).copy()
    half = len(vals) // 2
    mid = float(vals[half:].mean())
    vals[half:] = mid + (vals[half:] - mid) * 2.0      # 只放大残差，均值不变
    df["temperature"] = vals
    plan = make_plan("weather", ["temperature"], seasonal_period=24, change_point_depth=2)
    clean = _pipeline(df, plan)
    cps = anomaly.detect_change_points(clean, plan)
    assert not [c for c in cps if c.kind == "variance"], \
        [(c.index, c.statistic) for c in cps if c.kind == "variance"]


def test_anomaly_output_schema(make_plan):
    df = make_weather_frame(n=1500, seed=55, stations=1)
    sd = float(df["temperature"].std())
    df.loc[500, "temperature"] += 9 * sd
    plan = make_plan("weather", ["temperature"], seasonal_period=24)
    clean = _pipeline(df, plan)
    pts = anomaly.detect_anomalies(clean, plan)
    assert pts
    p = pts[0]
    assert isinstance(p.value, float) and isinstance(p.score, float)
    assert p.method and p.column == "temperature"


# ---------------- 段级通道（持续偏移） ----------------
# 点判据减一个周期的滚动中位数，会把 ≥半个周期的台阶吸收进趋势项；
# 段通道换 7 周期基线 + 窗口均值。默认关闭，只有显式开参数才生效。

def _inject_step(col: str, k: float, hours: int, seed: int = 7):
    """在序列中段注入 k·σ 的台阶，返回 (df, 开启段通道的计划, 注入起止时间戳)。"""
    df = make_weather_frame(n=24 * 200, seed=seed, stations=1)
    vals = df[col].to_numpy(dtype=float).copy()
    sd = float(np.std(vals - pd.Series(vals).rolling(24, center=True).mean().bfill().ffill()))
    start = len(vals) // 2
    vals[start:start + hours] += k * sd
    df[col] = vals
    plan_on = make_plan("weather", [col], seasonal_period=24, anomaly_threshold=7.0,
                        segment_anomaly=True)
    t0 = pd.Timestamp(df["timestamp"].iloc[start])
    t1 = pd.Timestamp(df["timestamp"].iloc[start + hours - 1])
    return df, plan_on, t0, t1


def test_segment_channel_off_by_default(make_plan):
    """默认不跑段通道：同一份注入了台阶的数据，不应出现 sustained_offset。"""
    df, _plan_on, _, _ = _inject_step("temperature", k=5.0, hours=48)
    plan = make_plan("weather", ["temperature"], seasonal_period=24, anomaly_threshold=7.0)
    clean = _pipeline(df, plan)
    pts = anomaly.detect_anomalies(clean, plan)
    assert not [p for p in pts if p.method.startswith("sustained_offset")]


def test_segment_channel_detects_sustained_step():
    """开段通道后，48 小时 5σ 的台阶应被报成一段，且代表点落在台阶内。"""
    df, plan, start, end = _inject_step("temperature", k=5.0, hours=48)
    clean = _pipeline(df, plan)
    pts = anomaly.detect_anomalies(clean, plan)
    seg = [p for p in pts if p.method.startswith("sustained_offset")]
    assert seg, "48h/5σ 的台阶没有被段通道检出"
    assert len(seg) <= 2, [(p.timestamp, p.score) for p in seg]
    for p in seg:
        assert start <= pd.Timestamp(p.timestamp) <= end
        assert "window=24" in p.method and "span=" in p.method


def test_segment_channel_ignores_short_blip():
    """只有 12 小时的偏移不该算「持续」：持续性要求是区间 ≥2 个周期。"""
    df, plan, _, _ = _inject_step("temperature", k=6.0, hours=12)
    clean = _pipeline(df, plan)
    pts = anomaly.detect_anomalies(clean, plan)
    assert not [p for p in pts if p.method.startswith("sustained_offset")]


def test_segment_channel_skips_zero_inflated_column():
    """零值占比过半的列（降水）不参与段级判定——窗口均值衡量的是「猜没下雨」。"""
    df, plan, _, _ = _inject_step("rainfall", k=0.0, hours=48)
    clean = _pipeline(df, plan)
    pts = anomaly.detect_anomalies(clean, plan)
    assert not [p for p in pts if p.method.startswith("sustained_offset")]


def test_segment_channel_tolerates_short_series(make_plan):
    """序列短于 3 个基线时直接跳过，不报错、不产出。"""
    df = make_weather_frame(n=24 * 10, seed=13, stations=1)
    plan = make_plan("weather", ["temperature"], seasonal_period=24, segment_anomaly=True)
    clean = _pipeline(df, plan)
    pts = anomaly.detect_anomalies(clean, plan)
    assert not [p for p in pts if p.method.startswith("sustained_offset")]
