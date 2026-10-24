# -*- coding: utf-8 -*-
"""`mtsa demo`（CLI 子命令）回归：装完就能跑，且注入的已知情况确实被检出。

注意与 `tests/test_demo.py` 区分：那个测的是仓库内的 `scripts/demo.py` 对比脚本，
这里测的是打包进 wheel 的 `mtsa demo` 子命令。
"""
from __future__ import annotations

import json

from mtsa.cli import main
from mtsa.demo import (
    DEMO_CSV_NAME,
    DEMO_DAYS,
    DEMO_ROWS_PER_DAY,
    DUPLICATE_ROWS,
    build_demo_frame,
)


def test_demo_frame_shape_and_injected_events():
    frame = build_demo_frame()
    assert len(frame) == DEMO_DAYS * DEMO_ROWS_PER_DAY + DUPLICATE_ROWS
    assert list(frame.columns) == [
        "timestamp", "station", "temperature", "humidity",
        "pressure", "wind_speed", "rainfall",
    ]
    # 注入的连续缺失：6 个点 × 4 列
    for col in ("temperature", "humidity", "pressure", "wind_speed"):
        assert frame[col].isna().sum() == 6

    # 注入的气压台阶：前 3 天均值应比后 3 天高约 8 hPa
    step = 60 * DEMO_ROWS_PER_DAY
    before = frame["pressure"].iloc[step - 72:step].mean()
    after = frame["pressure"].iloc[step:step + 72].mean()
    assert 4.0 < before - after < 12.0


def test_cli_demo_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("MTSA_PLANNER", "rule")
    monkeypatch.delenv("MTSA_AGENT", raising=False)
    out = tmp_path / "demo"
    assert main(["demo", "--out", str(out)]) == 0

    assert (out / DEMO_CSV_NAME).is_file()
    assert (out / "report.md").is_file()
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "succeeded"

    codes = {i["code"] for i in result["data"]["quality"]["issues"]}
    assert {"MISSING_VALUES", "DUPLICATE_TIMESTAMPS"} <= codes
    assert result["data"]["change_points"], "注入的气压台阶没有被检出"
    assert result["data"]["anomalies"], "注入的持续高温没有被检出"
