# -*- coding: utf-8 -*-
"""规模保护测试：超大文件必须在导入后立即明确拒绝，而不是跑到卡死。

真实上限是 2e5 行（实测约 2 分钟），测试里用 monkeypatch 调小，
避免生成几十万行的 fixture。
"""
from __future__ import annotations

import pytest

from mtsa import engine as engine_mod
from mtsa.engine import run_analysis
from mtsa.models import AnalysisRequest
from .helpers import make_weather_frame


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(engine_mod.ENV_MAX_ROWS, raising=False)


def _write(tmp_path, rows=100, name="w.csv"):
    df = make_weather_frame(n=rows, stations=1)
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def _req(path, out, rid="scale"):
    return AnalysisRequest(request_id=rid, input_path=str(path), scene_id="weather",
                           question="分析温度的趋势与季节性", output_dir=str(out))


def test_max_rows_defaults_and_env_parsing(monkeypatch):
    assert engine_mod._max_rows() == engine_mod.DEFAULT_MAX_ROWS

    monkeypatch.setenv(engine_mod.ENV_MAX_ROWS, "5000")
    assert engine_mod._max_rows() == 5000

    # 非法值一律回落到默认值，不能因为环境变量写错就让上限变成 0 或负数
    for bad in ("abc", "0", "-1", "", "  "):
        monkeypatch.setenv(engine_mod.ENV_MAX_ROWS, bad)
        assert engine_mod._max_rows() == engine_mod.DEFAULT_MAX_ROWS


def test_rejects_oversized_dataset(tmp_path, monkeypatch):
    """超过上限必须在导入后中止，并给出可操作的信息。"""
    monkeypatch.setattr(engine_mod, "DEFAULT_MAX_ROWS", 50)
    path = _write(tmp_path, rows=100)
    res = run_analysis(_req(path, tmp_path / "out"), verbose=False)

    assert res.status == "failed"
    assert res.error.code == "DATASET_TOO_LARGE"
    assert res.error.details["rows"] == 100
    assert res.error.details["limit"] == 50
    assert "降采样" in res.error.details["hint"]
    # 必须在导入后就失败：不应产生指标、图表等后续产物
    assert not res.artifacts


def test_oversized_dataset_does_not_run_later_nodes(tmp_path, monkeypatch):
    """拒绝要发生在摘要之前——否则用户已经等了很久才看到报错。"""
    monkeypatch.setattr(engine_mod, "DEFAULT_MAX_ROWS", 50)
    path = _write(tmp_path, rows=100)
    res = run_analysis(_req(path, tmp_path / "out", rid="scale2"), verbose=False)
    nodes = [e.node for e in res.events]
    assert "load_dataset" in nodes
    assert "profile_dataset" not in nodes


def test_warns_but_proceeds_between_thresholds(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_mod, "DEFAULT_MAX_ROWS", 10_000)
    monkeypatch.setattr(engine_mod, "WARN_ROWS", 50)
    path = _write(tmp_path, rows=100)
    res = run_analysis(_req(path, tmp_path / "out3", rid="scale3"), verbose=False)

    assert res.status == "succeeded"
    assert any("较大" in w for w in res.warnings)


def test_small_dataset_has_no_scale_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_mod, "WARN_ROWS", 10_000)
    path = _write(tmp_path, rows=100)
    res = run_analysis(_req(path, tmp_path / "out4", rid="scale4"), verbose=False)
    assert res.status == "succeeded"
    assert not any("较大" in w for w in res.warnings)


def test_env_can_raise_limit(tmp_path, monkeypatch):
    """上限可用环境变量放宽：便于确认更快的实现或更强的机器是否真的可行。"""
    monkeypatch.setattr(engine_mod, "DEFAULT_MAX_ROWS", 50)
    monkeypatch.setenv(engine_mod.ENV_MAX_ROWS, "500")
    path = _write(tmp_path, rows=100)
    res = run_analysis(_req(path, tmp_path / "out5", rid="scale5"), verbose=False)
    assert res.status == "succeeded"
