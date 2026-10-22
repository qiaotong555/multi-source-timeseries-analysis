# -*- coding: utf-8 -*-
"""场景识别测试：置信度、边界与无法识别时的失败行为。"""
from __future__ import annotations

import pandas as pd
import pytest

from mtsa import profiler
from mtsa.engine import run_analysis
from mtsa.io_loaders import build_spec
from mtsa.models import (ERR_SCENE_NOT_DETECTED, AnalysisRequest, RawDataset,
                         SCENE_CONFIDENCE_THRESHOLD)
from mtsa.scenarios import stock as _s  # noqa: F401
from mtsa.scenarios import weather as _w  # noqa: F401
from mtsa.scenarios.base import detect_scene
from .helpers import make_stock_frame, make_weather_frame


def _profile(df):
    raw = RawDataset(frame=df, spec=build_spec("mem.csv"), file_hash="0" * 40,
                     row_count=len(df), column_dtypes={c: str(t) for c, t in df.dtypes.items()})
    return raw, *profiler.profile_dataset(raw)


def test_detects_weather():
    raw, profile, _ = _profile(make_weather_frame(n=500))
    ad, conf, scores = detect_scene(raw.frame, profile)
    assert ad.scene_id == "weather"
    assert conf >= SCENE_CONFIDENCE_THRESHOLD


def test_detects_stock():
    raw, profile, _ = _profile(make_stock_frame(n=300))
    ad, conf, _ = detect_scene(raw.frame, profile)
    assert ad.scene_id == "stock"
    assert conf >= SCENE_CONFIDENCE_THRESHOLD


def test_unknown_scene_fails_explicitly(tmp_path):
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=60, freq="D"),
                       "x": range(60), "y": range(60)})
    p = tmp_path / "unknown.csv"
    df.to_csv(p, index=False)
    req = AnalysisRequest(request_id="req_scene", input_path=str(p),
                          question="分析一下", seed=1, output_dir=str(tmp_path / "out"))
    res = run_analysis(req, verbose=False)
    assert res.status == "failed"
    assert res.error.code == ERR_SCENE_NOT_DETECTED
    assert "scores" in res.error.details


def test_explicit_scene_overrides_detection(tmp_path):
    df = make_weather_frame(n=400)
    p = tmp_path / "w.csv"
    df.to_csv(p, index=False)
    req = AnalysisRequest(request_id="req_force", input_path=str(p), scene_id="weather",
                          question="分析温度趋势", seed=1, output_dir=str(tmp_path / "out2"))
    res = run_analysis(req, verbose=False)
    assert res.status == "succeeded"
    assert res.data["plan"]["scene_id"] == "weather"
    assert res.data["plan"]["scene_confidence"] == 1.0


def test_registry_contains_two_scenarios():
    from mtsa.scenarios.base import _REGISTRY
    assert set(_REGISTRY) >= {"weather", "stock"}


def test_get_unregistered_adapter_raises():
    from mtsa.scenarios.base import get_adapter
    with pytest.raises(KeyError):
        get_adapter("lottery")
