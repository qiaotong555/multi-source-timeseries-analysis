# -*- coding: utf-8 -*-
"""LLM 模式评估的判据测试（全离线）。

重点锁住两条容易出错的统计规则：
1. 覆盖率只统计合法计划——回退时选中的是规则计划的 12 项，几乎必然覆盖期望项，
   混进平均会把「模型全挂」算成「覆盖率很高」；
2. 探针用例（无期望项）不得拉高覆盖率平均值。
"""
from __future__ import annotations

import pytest

from evals import llm_eval as LE


# ---------------- 覆盖率与父项覆盖 ----------------

def test_tool_recall_full():
    r, missing = LE.tool_recall(["trend", "seasonal"], ["trend", "seasonal"])
    assert r == 1.0
    assert missing == []


def test_tool_recall_partial_reports_missing():
    r, missing = LE.tool_recall(["trend"], ["trend", "seasonal", "extreme"])
    assert r == pytest.approx(1 / 3, abs=1e-4)
    assert missing == ["seasonal", "extreme"]


def test_parent_tool_covers_detail_items():
    """模型只选 test 时，应视为覆盖了 stationarity 等四项。"""
    r, missing = LE.tool_recall(["test"], ["stationarity"])
    assert r == 1.0
    assert missing == []
    assert LE.tool_recall(["test"], ["autocorr"])[0] == 1.0
    assert LE.tool_recall(["test"], ["normality"])[0] == 1.0


def test_parent_tool_does_not_cover_unrelated():
    """test 不能覆盖 trend 这类无关项，否则覆盖率全是 1。"""
    r, missing = LE.tool_recall(["test"], ["trend"])
    assert r == 0.0
    assert missing == ["trend"]


def test_probe_case_has_no_recall():
    """探针用例无期望项，覆盖率必须为 None 而不是 1.0。"""
    r, missing = LE.tool_recall(["trend", "anomaly"], [])
    assert r is None
    assert missing == []


# ---------------- 汇总 ----------------

def test_rollback_excluded_from_recall_average():
    """核心规则：回退计划的高覆盖率不得进入平均。"""
    rows = [
        {"id": "judged_ok", "valid": True, "tool_recall": 0.5, "source": "llm"},
        # 回退：选中规则计划全 12 项，覆盖率高，但不代表模型选对了
        {"id": "judged_fallback", "valid": False, "tool_recall": 1.0, "source": "rule"},
    ]
    s = LE.summarize(rows)
    assert s["mean_tool_recall"] == 0.5          # 只算合法那条
    assert s["valid_rate"] == 0.5
    assert s["fallback_rate"] == 0.5
    assert s["passed"] is False                  # 回退率超标


def test_probe_excluded_from_recall_average():
    rows = [
        {"id": "judged", "valid": True, "tool_recall": 0.5, "source": "llm"},
        {"id": "probe", "valid": True, "tool_recall": None, "source": "llm"},
    ]
    s = LE.summarize(rows)
    assert s["mean_tool_recall"] == 0.5
    assert s["n_judged"] == 1
    assert s["n_cases"] == 2


def test_all_fallback_yields_no_recall():
    """全回退时覆盖率不可计算，应为 None，且整体不通过。"""
    rows = [{"id": f"c{i}", "valid": False, "tool_recall": 1.0, "source": "rule"}
            for i in range(3)]
    s = LE.summarize(rows)
    assert s["mean_tool_recall"] is None
    assert s["valid_rate"] == 0.0
    assert s["passed"] is False


def test_consistency_only_from_repeated_cases():
    rows = [
        {"id": "a", "valid": True, "tool_recall": 1.0, "consistent": True},
        {"id": "b", "valid": True, "tool_recall": 1.0, "consistent": False},
        {"id": "c", "valid": True, "tool_recall": 1.0},           # 未重复，不参与一致率
    ]
    s = LE.summarize(rows)
    assert s["plan_consistency"] == 0.5
    assert s["consistency_samples"] == 2


def test_no_consistency_samples():
    rows = [{"id": "a", "valid": True, "tool_recall": 1.0}]
    s = LE.summarize(rows)
    assert s["plan_consistency"] is None


def test_empty_rows_not_passed():
    s = LE.summarize([])
    assert s["passed"] is False


# ---------------- 端到端（FakeClient，离线） ----------------

class _FakeClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0
        self.model = "fake-model"
        self.base_url = "https://fake.local/v1"

    def chat_json(self, system: str, user: str) -> dict:
        self.calls += 1
        return self.payload


def test_illegal_plan_counts_as_fallback(tmp_path):
    """返回未知工具名 → 三级校验拦截 → 记为回退，valid 为 False。"""
    client = _FakeClient({"analyses": ["不存在的工具"], "reason": "test"})
    case = {"id": "bad", "scene": "weather", "csv": "weather_v1.csv",
            "question": "分析温度趋势", "expect": ["trend"]}
    out = tmp_path / "r.json"
    res = LE.run_llm_eval(cases=[case], repeat_ids=set(), client=client,
                          out_path=str(out), verbose=False)
    row = res["cases"][0]
    assert row["valid"] is False
    assert row["source"] == "rule"
    assert row["error"] and "未知分析项" in row["error"]
    assert res["summary"]["fallback_rate"] == 1.0
    assert out.exists()


def test_valid_plan_marks_llm_source(tmp_path):
    client = _FakeClient({"analyses": ["trend", "seasonal"], "reason": "问题涉及趋势与季节性"})
    case = {"id": "good", "scene": "weather", "csv": "weather_v1.csv",
            "question": "分析温度的趋势与季节性", "expect": ["trend", "seasonal"]}
    res = LE.run_llm_eval(cases=[case], repeat_ids=set(), client=client,
                          out_path=str(tmp_path / "r.json"), verbose=False)
    row = res["cases"][0]
    assert row["valid"] is True
    assert row["source"] == "llm"
    assert row["tool_recall"] == 1.0
    assert res["summary"]["passed"] is True


def test_repeat_detects_inconsistency(tmp_path):
    """两次返回不同分析项时，一致率应为 0。"""
    class _Alt(_FakeClient):
        def chat_json(self, system, user):
            self.calls += 1
            return {"analyses": ["trend"] if self.calls == 1 else ["trend", "corr"],
                    "reason": "x"}

    case = {"id": "rep", "scene": "weather", "csv": "weather_v1.csv",
            "question": "随便看看", "expect": []}
    res = LE.run_llm_eval(cases=[case], repeat_ids={"rep"}, client=_Alt({}),
                          out_path=str(tmp_path / "r.json"), verbose=False)
    assert res["cases"][0]["consistent"] is False
    assert res["summary"]["plan_consistency"] == 0.0


def test_missing_client_raises_with_actionable_message(monkeypatch):
    """必须显式清空环境变量：开发者本机配置了 MTSA_LLM_* 时，测试不能因此改走联网路径。"""
    for key in ("MTSA_LLM_BASE_URL", "MTSA_LLM_API_KEY", "MTSA_LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="MTSA_LLM_BASE_URL"):
        LE.run_llm_eval(cases=[], client=None, out_path="unused.json")
