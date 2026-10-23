# -*- coding: utf-8 -*-
"""低内存测试入口的纯函数测试（不真正起 pytest 子进程）。"""
from __future__ import annotations

import run_tests


def test_parse_counts_uses_last_match():
    out = "1 failed, 20 passed\n===== 21 passed, 1 warning in 3s ====="
    counts = run_tests.parse_counts(out)
    assert counts["passed"] == 21
    assert counts["failed"] == 1


def test_parse_counts_empty_output():
    assert run_tests.parse_counts("") == {"passed": 0, "failed": 0, "errors": 0}


def test_summarize_ok_only_when_all_files_pass():
    rows = [
        {"status": "PASS", "passed": 10, "failed": 0, "errors": 0, "duration_s": 1.0},
        {"status": "PASS", "passed": 5, "failed": 0, "errors": 0, "duration_s": 2.0},
    ]
    s = run_tests.summarize(rows)
    assert s["ok"] is True
    assert s["tests_passed"] == 15
    assert s["total_duration_s"] == 3.0


def test_summarize_flags_stall_and_failure():
    rows = [
        {"status": "STALLED", "passed": 3, "failed": 0, "errors": 0, "duration_s": 900.0},
        {"status": "FAIL", "passed": 2, "failed": 1, "errors": 0, "duration_s": 4.0},
    ]
    s = run_tests.summarize(rows)
    assert s["ok"] is False
    assert s["stalled_files"] == 1
    assert s["failed_files"] == 1
    assert s["tests_failed"] == 1


def test_summarize_empty_is_not_ok():
    assert run_tests.summarize([])["ok"] is False

def test_build_command_does_not_add_extra_quiet_flag():
    """pyproject 已有 -q；再传一次会变成 -qq，汇总行消失，统计全为 0。"""
    from pathlib import Path
    cmd = run_tests.build_command(Path("tests/test_tools.py"))
    assert "-q" not in cmd
    assert "--basetemp" in cmd
