# -*- coding: utf-8 -*-
"""`--dry-run` 的契约：只读、不写盘、不跑检测器，且计划与真跑同源。

这一层是「用户确认分析目标」的落点：先把要做什么摆出来，人点头了再花钱算。
所以测试盯的不是输出格式，而是三条硬约束：
1. 不创建产物目录；
2. 不调用任何检测器或渲染函数（用 monkeypatch 让它们一被调用就失败）；
3. 预览里的计划与真跑用的是同一份（同一 planner、同一参数）。
"""
from __future__ import annotations

import pytest

from mtsa import anomaly, freq, metrics, report, stats_tests, viz
from mtsa.cli import main
from mtsa.engine import run_analysis
from mtsa.models import AnalysisRequest
from .helpers import make_weather_frame


def _write_csv(tmp_path, n=96, stations=1):
    df = make_weather_frame(n=n, stations=stations)
    path = tmp_path / "w.csv"
    df.to_csv(path, index=False)
    return path, len(df)


def _req(tmp_path, rid="dry"):
    path, rows = _write_csv(tmp_path)
    req = AnalysisRequest(request_id=rid, input_path=str(path), scene_id="weather",
                          question="分析温度的趋势与异常",
                          output_dir=str(tmp_path / "out"))
    return req, rows


@pytest.fixture()
def no_detectors(monkeypatch):
    """检测器 / 渲染一旦被调用就失败：dry-run 只能走只读的那几步。"""
    def boom(*_args, **_kwargs):
        raise AssertionError("dry-run 不应调用检测器或渲染函数")

    for mod, name in ((metrics, "compute_metrics"),
                      (freq, "analyze_spectrum"),
                      (stats_tests, "run_stat_tests"),
                      (anomaly, "detect_anomalies"),
                      (anomaly, "detect_change_points"),
                      (anomaly, "detect_extremes"),
                      (viz, "build_figures"),
                      (report, "write_clean_csv"),
                      (report, "render_report"),
                      (report, "write_repro_script")):
        monkeypatch.setattr(mod, name, boom)


def test_dry_run_writes_nothing_and_skips_detectors(tmp_path, monkeypatch, no_detectors):
    monkeypatch.setenv("MTSA_PLANNER", "rule")
    req, rows = _req(tmp_path)
    res = run_analysis(req, verbose=False, dry_run=True)

    assert res.status == "succeeded"
    assert res.error is None
    assert res.artifacts == []
    assert not (tmp_path / "out").exists()        # 目录都没建

    p = res.data["preview"]
    assert p["dry_run"] is True
    assert p["input"]["rows"] == rows
    assert p["scene"]["scene_id"] == "weather"
    # 预览用的就是 plan 本身，不是另算一份
    assert p["plan"]["analyses"] == res.data["plan"]["analyses"]
    assert [t["name"] for t in p["tools"]] == p["plan"]["analyses"]


def test_dry_run_plan_matches_real_run(tmp_path, monkeypatch):
    """预演说要做 12 项，真跑就必须是同样 12 项、同样目标列。"""
    monkeypatch.setenv("MTSA_PLANNER", "rule")
    req, _ = _req(tmp_path, rid="parity")
    dry = run_analysis(req, verbose=False, dry_run=True)
    real = run_analysis(req, verbose=False)

    assert real.status == "succeeded"
    assert dry.data["plan"]["analyses"] == real.data["plan"]["analyses"]
    assert dry.data["plan"]["target_columns"] == real.data["plan"]["target_columns"]
    assert dry.data["preview"]["plan"]["plan_source"] == real.data["plan"]["plan_source"]


def test_cli_dry_run_prints_and_creates_no_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MTSA_PLANNER", "rule")
    path, _ = _write_csv(tmp_path)
    out = tmp_path / "cli_out"
    rc = main(["run", "--dry-run", "--scene", "weather", "--input", str(path),
               "--question", "分析温度的趋势", "--output", str(out)])

    assert rc == 0
    text = capsys.readouterr().out
    for keyword in ("dry-run", "场景", "将要执行", "不会创建"):
        assert keyword in text
    assert not out.exists()


def test_dry_run_rejects_replay(tmp_path, capsys):
    rc = main(["run", "--dry-run", "--replay", str(tmp_path / "missing.json")])
    assert rc == 1
    assert "不能与 --replay" in capsys.readouterr().err


def test_dry_run_failure_does_not_create_dir(tmp_path):
    """失败也不能建目录：--dry-run 对用户的承诺是「什么都不写」。"""
    out = tmp_path / "nope_out"
    req = AnalysisRequest(request_id="dry_bad", input_path=str(tmp_path / "missing.csv"),
                          scene_id="weather", question="分析温度",
                          output_dir=str(out))
    res = run_analysis(req, verbose=False, dry_run=True)

    assert res.status == "failed"
    assert res.error.code == "DATA_LOAD_FAILED"
    assert not out.exists()
