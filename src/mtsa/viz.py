# -*- coding: utf-8 -*-
"""可视化：Plotly 生成静态图（PNG）与交互图（HTML）。

图表字段映射由场景适配器的 figure_specs 决定；渲染不修改任何数据。
静态图导出失败时明确记录 warning，不静默跳过。

静态图导出的稳定性
------------------
plotly 7 + kaleido 1.x 的 PNG 导出依赖外部浏览器（Chrome 优先，Edge 回退），
浏览器启动失败是这类环境最常见的故障，典型报错是
``BrowserFailedError: The browser seemed to close immediately after starting``。
常见诱因：浏览器刚自动更新、已有同名实例占用、机器繁忙。

因此导出做三件事：
1. **重试**：失败后短暂等待再试，绝大多数据偶发失败可在第二次成功。
2. **清理残留**：最终失败时删除同名旧 PNG。否则目录里留着上一次的图，
   看起来「有图」，实际上本次没生成——这种掩盖比缺少文件更危险。
3. **给可操作提示**：把「怎么修」写进 warning，而不是只把系统异常原文抛给人看。
"""
from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .models import Artifact
from .scenarios.base import ScenarioContext

FIG_WIDTH = 1280
FIG_HEIGHT = 640

# 静态图导出的总尝试次数与间隔。浏览器启动失败多为瞬时，重试一次即可覆盖多数情形；
# 不设更高次数，因为持续失败意味着环境问题，反复重试只会拖长运行时间。
PNG_ATTEMPTS = 2
PNG_RETRY_DELAY_S = 1.5

_BROWSER_HINT = (
    "若持续失败，可依次尝试："
    "① 重跑一次（浏览器刚自动更新时，重启后即可恢复）；"
    "② 关闭所有已打开的浏览器进程后重试（同名实例会抢占启动）；"
    "③ 安装本机 Chrome，并把环境变量 BROWSER_PATH 指向 chrome.exe——"
    "kaleido 优先用 Chrome，不受 Edge 自动更新影响。"
    "注意 kaleido 自带的 get_chrome_sync 需访问 storage.googleapis.com，国内网络通常不可达。"
    "交互图（HTML）不受影响，已正常生成。"
)


def _export_png(fig: go.Figure, path: str) -> tuple[bool, str]:
    """导出静态图；返回 (是否成功, 失败原因)。成功即写入文件。"""
    last_exc: Exception | None = None
    for attempt in range(PNG_ATTEMPTS):
        try:
            fig.write_image(path, width=FIG_WIDTH, height=FIG_HEIGHT)
            return True, ""
        except Exception as exc:                      # noqa: BLE001  导出失败原因不可枚举
            last_exc = exc
            if attempt + 1 < PNG_ATTEMPTS:
                time.sleep(PNG_RETRY_DELAY_S)
    # 全部尝试失败：删除可能残留的旧文件，避免「目录里有图」掩盖本次失败
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
    return False, f"{type(last_exc).__name__}: {last_exc}"



def _export_png_batch(figs: list[go.Figure], paths: list[str]) -> tuple[bool, str]:
    """用一次浏览器会话导出多张 PNG。

    kaleido 1.x 下每次 `write_image` 都会启动并回收一次浏览器：7 张图约 7 次启动，
    实测占整条流水线的三分之二。plotly 的 `write_images` 复用同一会话，
    官方文档明确「比逐张快得多」。

    批量是 all-or-nothing：任一张失败即返回 False，由调用方回退到 `_export_png`
    逐张重试——删除残留旧图与提示语义都保留在单张路径里。
    """
    if not figs:
        return True, ""
    if len(figs) == 1:
        return _export_png(figs[0], paths[0])
    try:
        pio.write_images(figs, paths, width=FIG_WIDTH, height=FIG_HEIGHT)
        return True, ""
    except Exception as exc:                      # noqa: BLE001  导出失败原因不可枚举
        return False, f"{type(exc).__name__}: {exc}"


def _downsample(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame
    step = max(1, len(frame) // n)
    return frame.iloc[::step]


def _fig_series(frame: pd.DataFrame, spec: dict) -> go.Figure:
    cols = [c for c in spec.get("columns", []) if c in frame.columns]
    d = _downsample(frame, int(spec.get("downsample", 4000)))
    fig = go.Figure()
    for c in cols:
        fig.add_trace(go.Scatter(x=d.index, y=pd.to_numeric(d[c], errors="coerce"),
                                 mode="lines", name=c, line=dict(width=1.2)))
    fig.update_layout(title=spec.get("title", "时间序列"), height=FIG_HEIGHT, width=FIG_WIDTH,
                      template="plotly_white", legend=dict(orientation="h"),
                      xaxis_title="时间", margin=dict(l=60, r=30, t=50, b=50))
    return fig


def _fig_heatmap(frame: pd.DataFrame, spec: dict) -> go.Figure:
    cols = [c for c in spec.get("columns", []) if c in frame.columns]
    num = frame[cols].apply(pd.to_numeric, errors="coerce")
    corr = num.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}", colorbar=dict(title="corr"),
    ))
    fig.update_layout(title=spec.get("title", "相关矩阵"), height=FIG_HEIGHT, width=FIG_WIDTH,
                      template="plotly_white", margin=dict(l=80, r=40, t=50, b=80))
    return fig


def _fig_spectrum(ctx: ScenarioContext, spec: dict) -> go.Figure:
    fig = go.Figure()
    for s in ctx.spectrum:
        if s.column not in spec.get("columns", []):
            continue
        f = np.array(s.freqs)
        p = np.array(s.power)
        m = f > 0
        if m.sum() < 2:
            continue
        fig.add_trace(go.Scatter(x=f[m], y=p[m], mode="lines", name=f"{s.column}（主频 {s.dominant_freq:.4g} Hz）"))
    fig.update_xaxes(type="log", title="频率 (Hz)")
    fig.update_yaxes(type="log", title="功率谱密度")
    fig.update_layout(title=spec.get("title", "功率谱"), height=FIG_HEIGHT, width=FIG_WIDTH,
                      template="plotly_white", legend=dict(orientation="h"),
                      margin=dict(l=70, r=30, t=50, b=50))
    return fig


def _fig_bar(ctx: ScenarioContext, spec: dict) -> go.Figure:
    frame = ctx.clean.frame
    groupby = spec.get("groupby")
    cols = [c for c in spec.get("columns", []) if c in frame.columns]
    fig = go.Figure()
    if groupby == "month" and isinstance(frame.index, pd.DatetimeIndex):
        key = frame.index.month
        labels = [f"{m}月" for m in range(1, 13)]
        for c in cols:
            means = pd.to_numeric(frame[c], errors="coerce").groupby(key).mean()
            fig.add_trace(go.Bar(x=[f"{i}月" for i in means.index], y=means.values, name=c))
        fig.update_xaxes(categoryorder="array", categoryarray=labels)
    elif groupby and groupby in frame.columns:
        for c in cols:
            means = pd.to_numeric(frame[c], errors="coerce").groupby(frame[groupby]).mean()
            fig.add_trace(go.Bar(x=[str(i) for i in means.index], y=means.values, name=c))
    else:
        for c in cols:
            s = pd.to_numeric(frame[c], errors="coerce").dropna()
            fig.add_trace(go.Bar(x=[c], y=[float(s.mean())], name=c))
    fig.update_layout(title=spec.get("title", "分组均值"), height=FIG_HEIGHT, width=FIG_WIDTH,
                      template="plotly_white", barmode="group", legend=dict(orientation="h"),
                      margin=dict(l=60, r=30, t=50, b=60))
    return fig


def _fig_box(ctx: ScenarioContext, spec: dict) -> go.Figure:
    frame = ctx.clean.frame
    cols = [c for c in spec.get("columns", []) if c in frame.columns]
    fig = go.Figure()
    if isinstance(frame.index, pd.DatetimeIndex):
        key = frame.index.month
        for c in cols:
            s = pd.to_numeric(frame[c], errors="coerce")
            for mth in sorted(set(key)):
                fig.add_trace(go.Box(y=s[key == mth], name=f"{c}-{mth}月", boxmean=True))
    else:
        for c in cols:
            fig.add_trace(go.Box(y=pd.to_numeric(frame[c], errors="coerce"), name=c, boxmean=True))
    fig.update_layout(title=spec.get("title", "分布"), height=FIG_HEIGHT, width=FIG_WIDTH,
                      template="plotly_white", showlegend=False, margin=dict(l=60, r=30, t=50, b=60))
    return fig


def _fig_anomaly(ctx: ScenarioContext, spec: dict) -> go.Figure:
    frame = _downsample(ctx.clean.frame, int(spec.get("downsample", 4000)))
    cols = [c for c in spec.get("columns", []) if c in frame.columns]
    fig = go.Figure()
    for c in cols:
        fig.add_trace(go.Scatter(x=frame.index, y=pd.to_numeric(frame[c], errors="coerce"),
                                 mode="lines", name=c, line=dict(width=1.1)))
    for c in cols:
        pts = [a for a in ctx.anomalies if a.column == c]
        if pts:
            fig.add_trace(go.Scatter(
                x=[pd.Timestamp(a.timestamp) if isinstance(frame.index, pd.DatetimeIndex) else a.index
                   for a in pts],
                y=[a.value for a in pts], mode="markers", name=f"{c} 异常点",
                marker=dict(color="#c0392b", size=8, symbol="x")))
    for cp in ctx.change_points:
        if cp.column not in cols:
            continue
        x = pd.Timestamp(cp.timestamp) if isinstance(frame.index, pd.DatetimeIndex) else cp.index
        fig.add_vline(x=x, line_width=1.5, line_dash="dash", line_color="#16a34a",
                      annotation_text=f"{cp.kind}@{cp.column}")
    fig.update_layout(title=spec.get("title", "异常点与变化点"), height=FIG_HEIGHT, width=FIG_WIDTH,
                      template="plotly_white", legend=dict(orientation="h"),
                      margin=dict(l=60, r=30, t=50, b=50))
    return fig


def _fig_nav(ctx: ScenarioContext, spec: dict) -> go.Figure:
    close = (spec.get("columns") or [None])[0]
    frame = ctx.clean.frame
    if not close or close not in frame.columns:
        close = ctx.plan.parameters.get("close_column")
    px = pd.to_numeric(frame[close], errors="coerce")
    rets = px.pct_change().fillna(0.0)
    nav = (1 + rets).cumprod()
    dd = nav / nav.cummax() - 1
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38],
                        subplot_titles=("累计净值（买入持有）", "水下回撤"))
    fig.add_trace(go.Scatter(x=frame.index, y=nav, mode="lines", name="净值",
                             line=dict(color="#1f4e79", width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=frame.index, y=dd, mode="lines", name="回撤", fill="tozeroy",
                             line=dict(color="#16a34a", width=1)), row=2, col=1)
    fig.update_layout(title=spec.get("title", "净值与回撤"), height=FIG_HEIGHT, width=FIG_WIDTH,
                      template="plotly_white", margin=dict(l=60, r=30, t=60, b=50))
    return fig


def _fig_rolling(ctx: ScenarioContext, spec: dict) -> go.Figure:
    frame = ctx.clean.frame
    p = ctx.plan.parameters
    close = p.get("close_column") or (spec.get("columns") or [None])[0]
    bench = p.get("benchmark_column")
    w = int(p.get("rolling_window", 20))
    rets = pd.to_numeric(frame[close], errors="coerce").pct_change() if close in frame.columns else None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=(f"滚动 {w} 日年化波动率", "与基准滚动相关"))
    if rets is not None:
        rv = rets.rolling(w, min_periods=w).std() * np.sqrt(252)
        fig.add_trace(go.Scatter(x=frame.index, y=rv, mode="lines", name="年化波动率",
                                 line=dict(color="#c0392b", width=1.3)), row=1, col=1)
        if bench and bench in frame.columns:
            b = pd.to_numeric(frame[bench], errors="coerce").pct_change()
            rc = rets.rolling(w, min_periods=w).corr(b)
            fig.add_trace(go.Scatter(x=frame.index, y=rc, mode="lines", name="滚动相关",
                                     line=dict(color="#1f4e79", width=1.3)), row=2, col=1)
    fig.update_layout(title=spec.get("title", "滚动指标"), height=FIG_HEIGHT, width=FIG_WIDTH,
                      template="plotly_white", margin=dict(l=60, r=30, t=60, b=50))
    return fig


def _fig_fold(ctx: ScenarioContext, spec: dict) -> go.Figure:
    rows = [r for r in ctx.extra.rows if r.metric_id.startswith("stk.bt.fold")]
    strat = {r.window: r.value for r in rows if r.metric_id.endswith("_strategy")}
    bench = {r.window: r.value for r in rows if r.metric_id.endswith("_benchmark")}
    order = sorted(strat, key=lambda x: x or "")
    labels = [f"第 {i + 1} 段" for i in range(len(order))]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[strat[k] for k in order], name="策略", marker_color="#1f4e79"))
    fig.add_trace(go.Bar(x=labels, y=[bench[k] for k in order], name="基准", marker_color="#c0392b"))
    fig.update_layout(title=spec.get("title", "Walk-forward 分段收益"), height=FIG_HEIGHT,
                      width=FIG_WIDTH, template="plotly_white", barmode="group",
                      yaxis_tickformat=".2%", margin=dict(l=70, r=30, t=50, b=50))
    return fig


_BUILDERS = {
    "series": lambda ctx, spec: _fig_series(ctx.clean.frame, spec),
    "heatmap": lambda ctx, spec: _fig_heatmap(ctx.clean.frame, spec),
    "spectrum": _fig_spectrum,
    "bar": _fig_bar,
    "box": _fig_box,
    "anomaly": _fig_anomaly,
    "nav": _fig_nav,
    "rolling": _fig_rolling,
    "fold": _fig_fold,
}


def build_figures(ctx: ScenarioContext, specs: list[dict], outdir: str) -> tuple[list[Artifact], list[str]]:
    os.makedirs(outdir, exist_ok=True)
    artifacts: list[Artifact] = []
    warnings: list[str] = []
    png_hint_given = False          # 修复建议只在首次失败时给一次，避免同样的长文本重复七遍
    built: list[dict[str, Any]] = []

    for i, spec in enumerate(specs):
        kind = spec.get("kind")
        builder = _BUILDERS.get(kind)
        if builder is None:
            warnings.append(f"图表类型未支持，已跳过：{kind}（spec={spec.get('id')}）")
            continue
        try:
            fig = builder(ctx, spec)
        except Exception as exc:  # 记录并继续，不中断整条流水线
            warnings.append(f"图表 {spec.get('id')} 渲染失败：{type(exc).__name__}: {exc}")
            continue
        base = f"fig_{i + 1:02d}_{spec.get('id', kind)}"
        html_path = os.path.join(outdir, base + ".html")
        png_path = os.path.join(outdir, base + ".png")
        try:
            fig.write_html(html_path, include_plotlyjs="cdn")
        except Exception as exc:
            warnings.append(f"交互图 {base}.html 写出失败：{exc}")
            continue
        built.append({"fig": fig, "base": base, "spec": spec,
                      "html": html_path, "png": png_path})

    if not built:
        return artifacts, warnings

    # 先尝试批量导出（一次浏览器会话）；失败再逐张重试，保留原有的清理与提示语义。
    batch_ok, batch_reason = _export_png_batch(
        [b["fig"] for b in built], [b["png"] for b in built])
    if batch_ok:
        results: list[tuple[bool, str]] = [(True, "")] * len(built)
    else:
        results = [_export_png(b["fig"], b["png"]) for b in built]

    for b, (ok, reason) in zip(built, results):
        spec = b["spec"]
        base = b["base"]
        artifacts.append(Artifact(artifact_id=f"art_fig_{spec.get('id', spec.get('kind'))}",
                                  kind="figure_interactive", path=b["html"],
                                  title=spec.get("title", spec.get("kind")),
                                  fields=spec.get("columns", [])))
        if ok:
            artifacts.append(Artifact(artifact_id=f"art_fig_{spec.get('id', spec.get('kind'))}_png",
                                      kind="figure_static", path=b["png"],
                                      title=spec.get("title", spec.get("kind")) + "（静态）",
                                      fields=spec.get("columns", [])))
        else:
            msg = f"静态图 {base}.png 导出失败（kaleido）：{reason or batch_reason}"
            if not png_hint_given:
                msg += "。" + _BROWSER_HINT
                png_hint_given = True
            warnings.append(msg)
    return artifacts, warnings
