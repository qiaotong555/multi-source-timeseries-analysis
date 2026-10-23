# -*- coding: utf-8 -*-
"""鲁棒性矩阵的纯函数测试：契约判定与用例矩阵完整性。"""
from __future__ import annotations

from evals import robustness_eval as RE


def _case(**kw):
    base = {"id": "x", "group": "g", "desc": "d", "build": lambda d: d / "x.csv"}
    base.update(kw)
    return RE.Case(**base)


def _ok(**kw):
    m = {
        "result_exists": True, "report_exists": True, "report_chars": 1000,
        "status": "succeeded", "error_code": None, "warnings": [],
        "quality_codes": [], "conclusions": 2, "evidence": 2,
        "no_evidence": [], "forecast_claims": [], "leaks": [],
        "nonfinite": False, "elapsed_ms": 100, "crash": None,
    }
    m.update(kw)
    return m


def test_case_matrix_ids_unique_and_groups_present():
    cases = RE.build_cases()
    ids = [c.id for c in cases]
    assert len(ids) >= 30
    assert len(set(ids)) == len(ids), "用例 id 必须唯一"
    groups = {c.group for c in cases}
    assert {"导入", "时间轴", "数值", "编码", "规模", "问题", "Agent"} <= groups
    # 明确覆盖「检测器未执行」以外的三类边界
    assert any(c.kind == "cli" for c in cases)
    assert any(c.mode == "agent" for c in cases)


def test_judge_requires_result_json():
    probs = RE.judge(_case(), _ok(result_exists=False))
    assert any("result.json" in p for p in probs)


def test_judge_failed_code_must_be_allowed():
    case = _case(expect="failed", codes=("DATA_LOAD_FAILED",))
    m = _ok(status="failed", error_code="INTERNAL_ERROR", report_exists=False,
            report_chars=0, conclusions=0)
    assert any("不在预期集合" in p for p in RE.judge(case, m))
    m["error_code"] = "DATA_LOAD_FAILED"
    assert RE.judge(case, m) == []


def test_judge_success_requires_report_conclusions_and_evidence():
    case = _case()
    assert RE.judge(case, _ok()) == []
    assert any("report.md" in p for p in RE.judge(case, _ok(report_exists=False)))
    assert any("结论数为 0" in p for p in RE.judge(case, _ok(conclusions=0)))
    assert any("未绑定证据" in p for p in RE.judge(case, _ok(no_evidence=["c1"])))


def test_judge_warn_on_success():
    case = _case(warn_on_success=True)
    assert any("没有 warning" in p for p in RE.judge(case, _ok(warnings=[])))
    assert RE.judge(case, _ok(warnings=["降级"])) == []


def test_judge_quality_code_required():
    case = _case(require_quality=("DUPLICATE_TIMESTAMPS",))
    assert any("质量问题码" in p for p in RE.judge(case, _ok(quality_codes=[])))
    assert RE.judge(case, _ok(quality_codes=["DUPLICATE_TIMESTAMPS"])) == []


def test_judge_forbids_forecast_and_leak():
    assert RE.judge(_case(forbid_forecast=True), _ok(forecast_claims=["预测"])) != []
    assert RE.judge(_case(forbid_leak=True), _ok(leaks=["系统提示"])) != []


def test_judge_nonfinite_json_and_crash():
    assert any("非有限数值" in p for p in RE.judge(_case(), _ok(nonfinite=True)))
    assert any("crash" in p for p in RE.judge(_case(), _ok(crash="ValueError: x")))


def test_judge_cli_expectations():
    case = _case(kind="cli", question="", cli_expect={
        "returncode": 1, "stderr_contains": "缺少 --question", "result_json": False,
    })
    m = {"crash": None, "returncode": 1, "stderr": "缺少 --question", "result_exists": False,
         "elapsed_ms": 10}
    assert RE.judge(case, m) == []
    assert RE.judge(case, {**m, "returncode": 0}) != []
    assert RE.judge(case, {**m, "result_exists": True}) != []


def test_json_safe_replaces_nonfinite():
    from mtsa import engine as E
    assert E._json_safe({"x": float("inf"), "y": [float("nan"), 1.0]}) == {"x": None, "y": [None, 1.0]}


def test_unexpected_node_exception_is_contained(tmp_path, monkeypatch):
    """任何未覆盖的节点异常都必须变成 INTERNAL_ERROR + result.json，而不是抛给调用方。"""
    from mtsa import engine as E
    from mtsa.models import AnalysisRequest

    df = RE.weather_frame(n=60, stations=1)
    path = tmp_path / "w.csv"
    df.to_csv(path, index=False)
    out = tmp_path / "out"

    def boom(*args, **kwargs):
        raise RuntimeError("injected boom")

    monkeypatch.setattr(E.metrics, "compute_metrics", boom)
    res = E.run_analysis(AnalysisRequest(
        request_id="guard", input_path=str(path), scene_id="weather",
        question="分析温度的趋势", output_dir=str(out),
    ), verbose=False)
    assert res.status == "failed"
    assert res.error.code == "INTERNAL_ERROR"
    assert res.error.details["node"] == "compute_metrics"
    assert (out / "result.json").exists()
