# -*- coding: utf-8 -*-
"""进程级超时保护测试。

超时路径用一个必然不够的时限来触发：子进程光启动解释器与导入 numpy/statsmodels
就要数秒，给 0.5 秒必定超时，因此不必构造慢数据。
"""
from __future__ import annotations

import os
import subprocess

import pytest

from mtsa.timeout_guard import DEFAULT_TIMEOUT_S, run_with_timeout
from mtsa.models import AnalysisRequest
from .helpers import make_weather_frame


def _req(tmp_path, rows=80, rid="to"):
    df = make_weather_frame(n=rows, stations=1)
    path = tmp_path / "w.csv"
    df.to_csv(path, index=False)
    return AnalysisRequest(request_id=rid, input_path=str(path), scene_id="weather",
                           question="分析温度的趋势与季节性",
                           output_dir=str(tmp_path / "out"))


def test_completes_normally_within_timeout(tmp_path):
    req = _req(tmp_path, rid="to_ok")
    res = run_with_timeout(req, timeout_s=300, verbose=False)
    assert res.status == "succeeded"
    assert res.artifacts
    assert os.path.exists(os.path.join(tmp_path / "out", "result.json"))
    # 子进程的 run_id 由它自己生成，父进程读回后应保持一致
    assert res.run_id.startswith("run_")


def test_timeout_kills_child_and_reports_node_timeout(tmp_path):
    """核心用例：超时必须被真正终止，并返回可预期的错误码。"""
    req = _req(tmp_path, rid="to_kill")
    res = run_with_timeout(req, timeout_s=0.5, verbose=False)

    assert res.status == "failed"
    assert res.error.code == "NODE_TIMEOUT"
    assert res.error.details["timeout_s"] == 0.5
    assert "600" not in str(res.error.details["timeout_s"])   # 用的是传入值，不是默认值
    assert res.error.details["hint"]
    assert "output_dir" in res.error.details


def test_default_timeout_is_generous_enough():
    """默认值必须远大于规模上限下的正常耗时（约 2 分钟），否则会误杀。"""
    assert DEFAULT_TIMEOUT_S >= 300


def test_result_json_is_reused_not_recomputed(tmp_path):
    """父进程读回的是子进程写下的 result.json，不应有第二份产物目录。"""
    req = _req(tmp_path, rows=60, rid="to_reuse")
    res = run_with_timeout(req, timeout_s=300, verbose=False)
    assert res.status == "succeeded"
    outdir = tmp_path / "out"
    # 输入请求副本与产物同目录，但不应该出现第二个 output 目录
    assert (outdir / "_request.json").exists()
    dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert dirs == [outdir]


def test_child_can_import_package_without_external_pythonpath(tmp_path, monkeypatch):
    """子进程不应依赖调用方设置 PYTHONPATH。"""
    monkeypatch.delenv("PYTHONPATH", raising=False)
    req = _req(tmp_path, rows=60, rid="to_env")
    res = run_with_timeout(req, timeout_s=300, verbose=False)
    assert res.status == "succeeded"


def test_analysis_failure_is_parsed_not_masked(tmp_path):
    """子进程正常结束但分析失败（如文件不存在）时，应原样返回该失败，而不是报内部错误。"""
    req = AnalysisRequest(request_id="to_bad", input_path=str(tmp_path / "missing.csv"),
                          scene_id="weather", question="分析温度",
                          output_dir=str(tmp_path / "out2"))
    res = run_with_timeout(req, timeout_s=300, verbose=False)
    assert res.status == "failed"
    assert res.error.code == "DATA_LOAD_FAILED"


def test_child_stdio_is_forced_to_utf8(tmp_path, monkeypatch):
    """回归：子进程输出必须按 UTF-8 解码，不能跟着 locale 走。

    原实现 `subprocess.run(..., text=True)` 未指定 encoding，父进程用
    `locale.getpreferredencoding()` 解码。中文 Windows 上是 GBK，而子进程
    同样按 GBK 写 stderr——只要父进程处于 UTF-8 模式（`-X utf8` 或
    `PYTHONUTF8=1`，CLI 常规用法就是这样），reader 线程就会抛
    `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc7 ...`，
    主流程虽然成功，stderr 里却留下半截 traceback。

    触发物是 statsmodels 的 `InterpolationWarning`：告警文本带源码路径，
    路径里有中文用户名（`C:\\Users\\乔佟\\...`）。
    """
    import mtsa.timeout_guard as guard

    captured = {}

    class FakeProc:
        """替代 Popen：只实现 run_with_timeout 用到的那几个成员。"""

        def __init__(self, cmd, **kwargs):
            captured.update(kwargs)
            self.returncode = 1

        def communicate(self, timeout=None):
            return "", "boom"

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            pass

    monkeypatch.setattr(guard.subprocess, "Popen", FakeProc)
    res = run_with_timeout(_req(tmp_path, rid="to_enc"), timeout_s=30, verbose=False)

    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"          # 无法解码的字节降级，不抛异常
    assert captured["env"]["PYTHONIOENCODING"] == "utf-8"
    # 失败路径不受影响：stderr 仍以 str 进入 details
    assert res.status == "failed"
    assert res.error.details["stderr_tail"] == ["boom"]


def test_cancel_event_kills_child_and_reports_cancelled(tmp_path):
    """取消与超时必须分成两种结果：一个是「我不想要了」，一个是「它太慢了」。"""
    import threading
    import time as _time

    ev = threading.Event()
    timer = threading.Timer(0.4, ev.set)
    timer.daemon = True
    timer.start()
    t0 = _time.monotonic()
    try:
        res = run_with_timeout(_req(tmp_path, rid="to_cancel"), timeout_s=300,
                               verbose=False, cancel_event=ev)
    finally:
        timer.cancel()
    elapsed = _time.monotonic() - t0

    assert res.status == "cancelled"
    assert res.error is not None and res.error.code == "CANCELLED"
    # 子进程必须在取消后很快退出：留 15 秒给解释器启动与进程回收
    assert elapsed < 15, f"取消后仍未及时退出：{elapsed:.1f}s"
    assert not os.path.exists(os.path.join(tmp_path, "out", "result.json"))
