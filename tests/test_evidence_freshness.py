# -*- coding: utf-8 -*-
"""证据时效检查器的判据本身要被测住。

判据一句话：**产物 mtime 必须晚于它依赖源码的最后修改时间**。
这条判据是保守的（改到无关代码也会标过期），但方向必须是「宁多报、不漏报」——
反过来就是 2026-11-03 那次：`eval_results.json` 停在 10-25，
`结论级变化点F1` 还是 0.5（修完是 1.0），而 README 一直在引用它。
"""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evidence_freshness.py"
_spec = importlib.util.spec_from_file_location("evidence_freshness", _PATH)
EF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EF)


def _touch(path: Path, ts: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    os.utime(path, (ts, ts))


def _row(tmp_path: Path, source_ts: float, evidence_ts: float | None) -> dict:
    _touch(tmp_path / "src" / "a.py", source_ts)
    if evidence_ts is not None:
        _touch(tmp_path / "evals" / "e.json", evidence_ts)
    rows = EF.check([("evals/e.json", ("src/*.py",), "测试用")], root=str(tmp_path))
    return rows[0]


def test_evidence_newer_than_source_is_fresh(tmp_path):
    assert _row(tmp_path, source_ts=1000.0, evidence_ts=2000.0)["status"] == "fresh"


def test_evidence_older_than_source_is_stale_with_lag(tmp_path):
    row = _row(tmp_path, source_ts=3600.0 * 5, evidence_ts=0.0)
    assert row["status"] == "stale"
    assert row["lag_hours"] == 5.0


def test_equal_timestamps_count_as_fresh(tmp_path):
    """同一秒生成不算过期——否则跑完评估立刻检查就会误报。"""
    assert _row(tmp_path, source_ts=5000.0, evidence_ts=5000.0)["status"] == "fresh"


def test_missing_evidence_is_reported_not_silently_skipped(tmp_path):
    row = _row(tmp_path, source_ts=1000.0, evidence_ts=None)
    assert row["status"] == "missing"
    assert row["mtime"] == 0.0


def test_no_source_files_means_fresh(tmp_path):
    """源码 glob 什么都没匹配到时，不能把产物判成过期（否则空仓库里全是假警报）。"""
    _touch(tmp_path / "evals" / "e.json", 1000.0)
    rows = EF.check([("evals/e.json", ("nonexistent/*.py",), "x")], root=str(tmp_path))
    assert rows[0]["status"] == "fresh"


def test_render_flags_stale_and_missing(tmp_path):
    rows = [
        {"file": "a.json", "label": "甲", "status": "stale", "mtime": 0.0,
         "source_mtime": 3600.0, "source": "s.py", "lag_hours": 1.0},
        {"file": "b.json", "label": "乙", "status": "missing", "mtime": 0.0,
         "source_mtime": 3600.0, "source": "s.py", "lag_hours": None},
        {"file": "c.json", "label": "丙", "status": "fresh", "mtime": 7200.0,
         "source_mtime": 3600.0, "source": "s.py", "lag_hours": 0.0},
    ]
    text = EF.render(rows)
    assert "**过期**" in text and "**缺失**" in text and "新鲜" in text
    assert "a.json" in text and "c.json" in text


def test_repo_evidence_list_covers_readme_claims():
    """README 第 3 节引用的证据必须都在检查器清单里，否则等于没被检查。"""
    listed = {rel for rel, _, _ in EF.EVIDENCE}
    for must in ("evals/test_logs/summary.json",
                 "evals/robustness_results.json",
                 "evals/conclusion_gate_v2_results.json",
                 "evals/eval_results.json",
                 "evals/real_data_results.json",
                 "evals/forecast_results.json",
                 "evals/mode_compare_results.json"):
        assert must in listed, must
