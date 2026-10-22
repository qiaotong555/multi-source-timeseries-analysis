# -*- coding: utf-8 -*-
"""未来数据泄漏测试：含泄漏必须报错，正常信号不得误报。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mtsa.engine import run_analysis
from mtsa.models import ERR_LOOKAHEAD_VIOLATION, AnalysisRequest
from mtsa.scenarios.stock import LOOKAHEAD_CORR_THRESHOLD, check_lookahead
from .helpers import make_stock_frame


def _leak_frame(n: int = 400) -> pd.DataFrame:
    """构造一个用次日收益直接构造的信号列。"""
    df = make_stock_frame(n=n, seed=77)
    fwd = df["close"].shift(-1) / df["close"] - 1.0
    df["signal_leak"] = (fwd + np.random.default_rng(3).normal(0, 1e-4, n)).fillna(0.0)
    return df


def test_check_lookahead_flags_leaky_signal():
    issues = check_lookahead(_leak_frame(), "close")
    assert issues, "未检出泄漏信号"
    assert abs(issues[0]["corr_with_next_return"]) > LOOKAHEAD_CORR_THRESHOLD


def test_check_lookahead_passes_legit_signal():
    df = make_stock_frame(n=400, seed=78)
    df["momentum_20"] = df["close"] / df["close"].shift(20) - 1.0
    assert check_lookahead(df, "close") == []


def test_engine_fails_on_leak_dataset(tmp_path):
    p = tmp_path / "leak.csv"
    _leak_frame().to_csv(p, index=False)
    req = AnalysisRequest(request_id="req_leak", input_path=str(p), scene_id="stock",
                          question="回测", seed=1, output_dir=str(tmp_path / "out"))
    res = run_analysis(req, verbose=False)
    assert res.status == "failed"
    assert res.error.code == ERR_LOOKAHEAD_VIOLATION
    assert res.error.details["issues"]
    # 失败时不得输出回测结果
    assert "scenario_metrics" not in res.data


def test_engine_succeeds_on_clean_stock(tmp_path):
    p = tmp_path / "ok.csv"
    make_stock_frame(n=400, seed=79).to_csv(p, index=False)
    req = AnalysisRequest(request_id="req_ok", input_path=str(p), scene_id="stock",
                          question="回测", seed=1, output_dir=str(tmp_path / "out2"))
    res = run_analysis(req, verbose=False)
    assert res.status == "succeeded"
    assert res.data["lookahead"] == []


def test_walk_forward_folds_do_not_overlap(tmp_path):
    p = tmp_path / "folds.csv"
    make_stock_frame(n=400, seed=80).to_csv(p, index=False)
    req = AnalysisRequest(request_id="req_f", input_path=str(p), scene_id="stock",
                          question="回测", seed=1, output_dir=str(tmp_path / "out3"))
    res = run_analysis(req, verbose=False)
    rows = res.data["scenario_metrics"]

    def val(mid):
        return next(r["value"] for r in rows if r["metric_id"] == mid)

    folds = [r for r in rows
             if r["metric_id"].startswith("stk.bt.fold") and r["metric_id"].endswith("_strategy")]
    assert len(folds) >= 3
    order = [int(r["window"].replace("fold", "")) for r in folds]
    assert order == sorted(order) and len(set(order)) == len(order)

    # 分段收益的复合必须等于全区间收益：证明分段无重叠且完整覆盖
    compounded = 1.0
    for r in folds:
        compounded *= (1.0 + r["value"])
    assert abs(compounded - 1.0 - val("stk.bt.total_strategy")) < 1e-9
