# -*- coding: utf-8 -*-
"""F：预测（点预测 + 区间 + walk-forward 回测）。

接缝：
- `forecast.infer_period` / `_forecast_from` / `_fit_column` / `build_forecast`；
- `engine`：`--forecast` 才写 `data.forecast`，关闭时既有字段逐位不变；
- `report` 8c 节、`weather.conclusions` 的「预测」分类。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mtsa import forecast
from mtsa.engine import run_analysis
from mtsa.models import AnalysisRequest

from .helpers import make_weather_frame


def _series(n: int = 2000, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    hour = idx.hour.to_numpy()
    vals = 15 + 6 * np.sin(2 * np.pi * (hour - 14) / 24) + rng.normal(0, 0.6, n)
    return pd.Series(vals, index=idx, name="temperature")


def test_infer_period_from_spacing():
    hourly = pd.date_range("2024-01-01", periods=100, freq="h")
    daily = pd.date_range("2024-01-01", periods=100, freq="D")
    quarter = pd.date_range("2024-01-01", periods=100, freq="15min")
    assert forecast.infer_period(hourly) == 24
    assert forecast.infer_period(daily) == 7
    assert forecast.infer_period(quarter) == 96
    # 非时间索引不猜周期
    assert forecast.infer_period(pd.RangeIndex(100)) == 1


def test_prediction_never_reads_the_future():
    """F2：同一条序列，改动 origin 之后的观测，该 origin 的预测必须完全不变。"""
    s = _series(1200, seed=5)
    y = s.to_numpy(dtype=float)
    n = len(y)
    origins = forecast._origins(n, 24, forecast.DEFAULT_ORIGINS)
    assert len(origins) >= 10
    rng = np.random.default_rng(9)
    for t in (origins[0], origins[len(origins) // 2], origins[-1]):
        for m in forecast.METHODS:
            base = forecast._forecast_from(y[:t], 24, 24, m)
            future = y.copy()
            future[t:] = rng.normal(0, 50, n - t)      # 把「未来」换成完全不同的值
            assert np.array_equal(base, forecast._forecast_from(future[:t], 24, 24, m)), m


def test_constant_series_falls_back_to_baseline():
    idx = pd.date_range("2024-01-01", periods=800, freq="h")
    out = forecast._fit_column(pd.Series(np.full(800, 5.0), index=idx, name="c"), 24, 24, 40)
    assert out["status"] == "ok"
    assert out["selected"] == "seasonal_naive"
    assert ("回退" in out["selected_reason"]) or ("未优于" in out["selected_reason"])
    assert out["beats_baseline"] is False
    assert out["selected_mase"] is None          # 常数序列 MASE 无定义，不伪造 0


def test_perfectly_periodic_series_keeps_baseline():
    idx = pd.date_range("2024-01-01", periods=1200, freq="h")
    vals = 10 + 5 * np.sin(2 * np.pi * np.arange(1200) / 24)
    out = forecast._fit_column(pd.Series(vals, index=idx, name="p"), 24, 24, 40)
    assert out["selected"] == "seasonal_naive"
    assert out["methods"][0]["mae"] == 0.0       # 基线在完美周期上误差为 0
    # 完美周期下 1 阶差分为 0，MASE 无定义：返回 None，不用 2e-14 的分母硬算
    assert out["methods"][0]["mase"] is None
    assert out["beats_baseline"] is False


def test_interval_coverage_is_near_nominal():
    """F3：90% 区间的回测覆盖率落在 [0.80, 0.98]。"""
    out = forecast._fit_column(_series(2000, seed=11), 24, 24, forecast.DEFAULT_ORIGINS)
    cov = out["interval_coverage"]
    assert cov is not None and 0.80 <= cov <= 0.98, cov
    assert out["interval_coverage_points"] > 100
    assert out["interval_method"] == "empirical_pooled"
    pts = out["forecast"]
    assert len(pts) == 24
    for p in pts:
        assert p["lo"] <= p["value"] <= p["hi"]


def test_forecast_off_by_default(tmp_path):
    df = make_weather_frame(n=600, stations=1)
    p = tmp_path / "w.csv"
    df.to_csv(p, index=False)
    out = tmp_path / "off"
    res = run_analysis(AnalysisRequest(
        request_id="f_off", input_path=str(p), scene_id="weather",
        question="分析温度的趋势", mode="pipeline", seed=20260927,
        output_dir=str(out)), verbose=False)
    assert res.status == "succeeded", res.error
    doc = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert "forecast" not in doc["data"]
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "## 8c." not in text
    assert "预测" not in {c["category"] for c in doc["conclusions"]}


def test_forecast_on_adds_section_and_keeps_other_fields(tmp_path):
    """F1/F5/F7：开启后有完整字段、报告分节、结论单列；其余字段与关闭时逐位相同。"""
    df = make_weather_frame(n=600, stations=1)
    p = tmp_path / "w.csv"
    df.to_csv(p, index=False)
    base = dict(input_path=str(p), scene_id="weather", question="分析温度的趋势",
                mode="pipeline", seed=20260927)
    off_dir, on_dir = tmp_path / "off", tmp_path / "on"
    r_off = run_analysis(AnalysisRequest(request_id="f_off2", output_dir=str(off_dir),
                                         **base), verbose=False)
    r_on = run_analysis(AnalysisRequest(request_id="f_on", output_dir=str(on_dir),
                                        forecast=True, **base), verbose=False)
    assert r_off.status == "succeeded", r_off.error
    assert r_on.status == "succeeded", r_on.error
    d_off = json.loads((off_dir / "result.json").read_text(encoding="utf-8"))
    d_on = json.loads((on_dir / "result.json").read_text(encoding="utf-8"))

    fc = d_on["data"]["forecast"]
    assert {"enabled", "period", "horizon", "interval_level", "methods",
            "baseline", "columns", "disclaimer"} <= set(fc)
    assert fc["horizon"] == fc["period"] == 24
    col = fc["columns"][0]
    assert {"column", "status", "points", "period", "horizon", "origins",
            "methods", "selected", "selected_reason", "beats_baseline",
            "baseline_mase", "interval_level", "interval_coverage",
            "forecast"} <= set(col)
    assert len(col["forecast"]) == 24
    assert {m["method"] for m in col["methods"]} == set(forecast.METHODS)

    # F5：除 forecast 外，data 其余字段逐位相同
    rest_on = {k: v for k, v in d_on["data"].items() if k != "forecast"}
    assert rest_on == d_off["data"]
    # 既有结论是新增结论的前缀，没有被改写
    assert d_on["conclusions"][:len(d_off["conclusions"])] == d_off["conclusions"]

    text = (on_dir / "report.md").read_text(encoding="utf-8")
    assert "## 8c. 预测（回测口径）" in text
    assert "不是事实" in text
    # 复现脚本必须带上同样参数，否则「复现」出来的是一份没有预测的报告
    repro = (on_dir / "reproduce.py").read_text(encoding="utf-8")
    assert "--forecast" in repro and "--forecast-horizon" in repro
    assert "--forecast" not in (off_dir / "reproduce.py").read_text(encoding="utf-8")

    cats = {c["category"] for c in d_on["conclusions"]}
    assert "预测" in cats
    forecast_con = [c for c in d_on["conclusions"] if c["category"] == "预测"]
    assert forecast_con and all(c["evidence_ids"] for c in forecast_con)
    assert any("horizon" in c["text"] for c in forecast_con)


def test_forecast_horizon_override(tmp_path):
    df = make_weather_frame(n=600, stations=1)
    p = tmp_path / "w.csv"
    df.to_csv(p, index=False)
    out = tmp_path / "h"
    res = run_analysis(AnalysisRequest(
        request_id="f_h", input_path=str(p), scene_id="weather",
        question="分析温度", mode="pipeline", seed=20260927,
        output_dir=str(out), forecast=True, forecast_horizon=12), verbose=False)
    assert res.status == "succeeded", res.error
    doc = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert doc["data"]["forecast"]["horizon"] == 12
    assert all(len(c["forecast"]) == 12 for c in doc["data"]["forecast"]["columns"])
