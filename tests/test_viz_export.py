# -*- coding: utf-8 -*-
"""静态图导出的健壮性测试。

这三个行为都直接对应真实故障，不能靠人工观察：
1. 浏览器启动失败多为瞬时，第二次尝试应能成功；
2. 彻底失败时必须删除残留旧图，否则目录里「有图」会掩盖本次失败；
3. 修复建议只在首次失败时给一次，避免重复七遍。
"""
from __future__ import annotations

import os

import pytest

from mtsa import viz


class _FakeFig:
    """只实现 write_image 的假图表；按 fail_times 决定前几次失败。"""

    def __init__(self, fail_times: int = 0, exc: Exception | None = None) -> None:
        self.calls = 0
        self.fail_times = fail_times
        self.exc = exc or RuntimeError("BrowserFailedError: browser closed immediately")

    def write_image(self, path: str, width: int | None = None,
                    height: int | None = None, **kwargs) -> None:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")


@pytest.fixture(autouse=True)
def _no_delay(monkeypatch):
    """测试里不要真的等 1.5 秒。"""
    monkeypatch.setattr(viz, "PNG_RETRY_DELAY_S", 0.0)


def test_export_png_writes_file_on_first_try(tmp_path):
    out = tmp_path / "fig.png"
    ok, reason = viz._export_png(_FakeFig(), str(out))
    assert ok is True
    assert reason == ""
    assert out.exists()


def test_export_png_retries_transient_failure(tmp_path):
    """首次失败、第二次成功——这是浏览器刚更新后最常见的情形。"""
    out = tmp_path / "fig.png"
    fig = _FakeFig(fail_times=1)
    ok, reason = viz._export_png(fig, str(out))
    assert ok is True
    assert fig.calls == 2
    assert out.exists()


def test_export_png_removes_stale_file_after_total_failure(tmp_path):
    """彻底失败时必须删掉上一次运行残留的同名图，否则会掩盖本次失败。"""
    out = tmp_path / "fig.png"
    out.write_bytes(b"stale-from-previous-run")
    fig = _FakeFig(fail_times=99)
    ok, reason = viz._export_png(fig, str(out))
    assert ok is False
    assert not out.exists()
    assert "BrowserFailedError" in reason
    assert fig.calls == viz.PNG_ATTEMPTS


def test_export_png_attempts_at_least_twice():
    """只尝试一次就等于没有重试，这个常量被改小要有测试挡住。"""
    assert viz.PNG_ATTEMPTS >= 2


def test_browser_hint_mentions_actionable_options():
    """提示必须能指导用户动手，不能只有「失败」两个字。"""
    hint = viz._BROWSER_HINT
    assert "chrome" in hint.lower() or "Chrome" in hint
    assert "重跑" in hint or "关闭" in hint
    assert "HTML" in hint          # 说明交互图不受影响


class _FakeHtmlFig:
    """只需 write_html 的假图表；PNG 由批量/单张导出函数负责。"""

    def write_html(self, path, include_plotlyjs=None, **kwargs):
        with open(path, "w", encoding="utf-8") as f:
            f.write("<html></html>")


def test_export_png_batch_uses_single_browser_session(tmp_path, monkeypatch):
    calls = {}

    def fake_write_images(figs, paths, width=None, height=None):
        calls["n"] = len(figs)
        for p in paths:
            with open(p, "wb") as f:
                f.write(b"PNG")

    monkeypatch.setattr(viz.pio, "write_images", fake_write_images)
    paths = [str(tmp_path / "a.png"), str(tmp_path / "b.png")]
    ok, reason = viz._export_png_batch([object(), object()], paths)
    assert ok is True and reason == ""
    assert calls["n"] == 2                      # 一次调用导出两张
    assert all(os.path.exists(p) for p in paths)


def test_export_png_batch_reports_failure_for_fallback(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("BrowserFailedError: closed immediately")

    monkeypatch.setattr(viz.pio, "write_images", boom)
    ok, reason = viz._export_png_batch([object(), object()], ["a.png", "b.png"])
    assert ok is False
    assert "BrowserFailedError" in reason


def test_build_figures_batches_png_and_keeps_artifact_order(tmp_path, monkeypatch):
    monkeypatch.setitem(viz._BUILDERS, "series", lambda ctx, spec: _FakeHtmlFig())
    seen = {}

    def fake_batch(figs, paths):
        seen["n"] = len(figs)
        for p in paths:
            with open(p, "wb") as f:
                f.write(b"PNG")
        return True, ""

    monkeypatch.setattr(viz, "_export_png_batch", fake_batch)
    specs = [{"id": "a", "kind": "series"}, {"id": "b", "kind": "series"}]
    artifacts, warnings = viz.build_figures(None, specs, str(tmp_path))
    assert warnings == []
    assert seen["n"] == 2
    assert [a.kind for a in artifacts] == ["figure_interactive", "figure_static"] * 2


def test_build_figures_falls_back_when_batch_fails(tmp_path, monkeypatch):
    monkeypatch.setitem(viz._BUILDERS, "series", lambda ctx, spec: _FakeHtmlFig())
    monkeypatch.setattr(viz, "_export_png_batch", lambda figs, paths: (False, "batch boom"))
    calls = []

    def fake_export(fig, path):
        calls.append(path)
        with open(path, "wb") as f:
            f.write(b"PNG")
        return True, ""

    monkeypatch.setattr(viz, "_export_png", fake_export)
    artifacts, warnings = viz.build_figures(None, [{"id": "a", "kind": "series"}], str(tmp_path))
    assert calls and warnings == []
    assert [a.kind for a in artifacts] == ["figure_interactive", "figure_static"]


def test_build_figures_warns_once_after_total_failure(tmp_path, monkeypatch):
    monkeypatch.setitem(viz._BUILDERS, "series", lambda ctx, spec: _FakeHtmlFig())
    monkeypatch.setattr(viz, "_export_png_batch", lambda figs, paths: (False, "batch boom"))
    monkeypatch.setattr(viz, "_export_png", lambda fig, path: (False, "still broken"))
    specs = [{"id": "a", "kind": "series"}, {"id": "b", "kind": "series"}]
    artifacts, warnings = viz.build_figures(None, specs, str(tmp_path))
    assert [a.kind for a in artifacts] == ["figure_interactive"] * 2
    assert len(warnings) == 2
    assert sum(1 for w in warnings if "若持续失败" in w) == 1     # 建议只出现一次
