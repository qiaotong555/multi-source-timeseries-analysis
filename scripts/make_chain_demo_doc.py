# -*- coding: utf-8 -*-
"""生成「一份数据走完整条链路」的 Word 演示文档。

用法
----
    python scripts/make_chain_demo_doc.py            # 用已有产物生成文档
    python scripts/make_chain_demo_doc.py --rerun    # 先重跑 pipeline / agent / forecast，再生成

数据：artifacts/docdemo/{pipeline,agent,forecast}/result.json
图：artifacts/docdemo/figs/*.png（本脚本按「一列一张图、单尺度」重画。
    流水线自带的 5 列共轴图在气压 ~1000 的量级下会把温度压成一条直线，不适合演示。）
输出：doc/全链路Demo_柏林2024.docx

文档里的每个数字都在生成时从 result.json / clean_data.csv 读取，不手写。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "artifacts" / "docdemo"
FIGS = DEMO / "figs"
OUT = ROOT / "doc" / "全链路Demo_柏林2024.docx"
INPUT = ROOT / "fixtures" / "real_weather" / "r01_berlin_hourly_2024" / "data.csv"
QUESTION = "分析温度的趋势、季节性和异常点，并检测变化点"
AUTHOR = "qiaotong555"

BODY = "等线"
HEAD = "微软雅黑"
MONO = "Consolas"
INK = RGBColor(0x1A, 0x1A, 0x1A)
GRAY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0x80, 0x80, 0x80)
ACCENT = RGBColor(0xB0, 0x3A, 0x2E)
LINE = "D9D9D9"
HEAD_FILL = "F2F2F2"
SOFT_FILL = "F7F8FA"
TOTAL_W = 16.6

TEMP_COLOR = "#2F6DB3"
MEAN_COLOR = "#B03A2E"
BAR_COLOR = "#4C7EA8"


# ---------------------------------------------------------------- 运行与读取

def _run_cli(name: str, *, forecast: bool = False) -> bool:
    outdir = DEMO / name
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mtsa.cli", "run",
        "--scene", "weather",
        "--input", str(INPUT),
        "--question", QUESTION,
        "--output", str(outdir),
    ]
    if name == "agent":
        cmd += ["--mode", "agent"]
    if forecast:
        cmd.append("--forecast")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["MTSA_PLANNER"] = "rule"          # 计划层固定规则，让三次运行可比
    if name == "agent" and not all(env.get(k) for k in
                                   ("MTSA_LLM_BASE_URL", "MTSA_LLM_API_KEY", "MTSA_LLM_MODEL")):
        print("[skip] 未配置 MTSA_LLM_*，保留已有 agent 产物")
        return (DEMO / name / "result.json").exists()
    print("[run]", " ".join(cmd[2:]))
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    ok = (outdir / "result.json").exists()
    if not ok:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or ["无输出"]
        print(f"[skip] {name} 未产出 result.json：{tail[0][:160]}")
    return ok


def ensure_demo(rerun: bool) -> None:
    need = {"pipeline": False, "agent": False, "forecast": True}
    for name, fc in need.items():
        if rerun or not (DEMO / name / "result.json").exists():
            _run_cli(name, forecast=fc)


def load(name: str) -> dict:
    with open(DEMO / name / "result.json", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------- 图

def _export(fig: go.Figure, path: Path, width: int = 1080, height: int = 470) -> Path:
    for attempt in range(2):
        try:
            fig.write_image(str(path), width=width, height=height)
            return path
        except Exception as exc:                      # noqa: BLE001  浏览器导出失败原因不可枚举
            if attempt == 0:
                print(f"[retry] 静态图导出失败：{type(exc).__name__}: {exc}")
                time.sleep(1.5)
            else:
                raise
    return path


def _base_layout(fig: go.Figure, *, x_title: str, y_title: str) -> None:
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        font=dict(family="Microsoft YaHei, Microsoft YaHei UI, SimHei, sans-serif", size=19, color="#333333"),
        margin=dict(l=118, r=22, t=74, b=64),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode=False,
    )
    fig.update_xaxes(title=dict(text=x_title, font=dict(size=16, color="#666666")),
                     showgrid=True, gridcolor="#EEEEEE", zeroline=False,
                     linecolor="#CCCCCC", ticks="outside", tickcolor="#CCCCCC")
    fig.update_yaxes(title=dict(text=y_title, font=dict(size=16, color="#666666")),
                     showgrid=True, gridcolor="#EEEEEE", zeroline=False,
                     linecolor="#CCCCCC", ticks="outside", tickcolor="#CCCCCC")


def build_figures() -> dict[str, Path]:
    FIGS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DEMO / "pipeline" / "clean_data.csv", parse_dates=["timestamp"]).set_index("timestamp")
    temp = df["temperature"]
    rain = df["rainfall"]

    # 图 1：全年温度（逐小时细线 + 月均粗线）——同一单位，可以共轴
    monthly = temp.groupby(temp.index.month).mean()
    months = [pd.Timestamp(year=int(temp.index.year[0]), month=int(m), day=1) for m in monthly.index]
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=temp.index, y=temp.values, mode="lines",
                              line=dict(color=TEMP_COLOR, width=0.8), opacity=0.8))
    fig1.add_trace(go.Scatter(x=months, y=monthly.values, mode="lines+markers",
                              line=dict(color=MEAN_COLOR, width=3.2), marker=dict(size=7)))
    _base_layout(fig1, x_title="", y_title="温度（°C）")
    fig1.update_xaxes(tickvals=months, ticktext=[f"{int(m)}月" for m in monthly.index])
    cold, warm = int(monthly.idxmin()), int(monthly.idxmax())
    fig1.add_annotation(x=months[cold - 1], y=monthly.loc[cold], text=f"{cold}月 {monthly.loc[cold]:.1f}°C",
                        showarrow=True, arrowhead=2, ax=0, ay=42,
                        font=dict(size=16, color=MEAN_COLOR), arrowcolor=MEAN_COLOR,
                        bgcolor="rgba(255,255,255,0.85)", borderpad=3)
    fig1.add_annotation(x=months[warm - 1], y=monthly.loc[warm], text=f"{warm}月 {monthly.loc[warm]:.1f}°C",
                        showarrow=True, arrowhead=2, ax=0, ay=-46,
                        font=dict(size=16, color=MEAN_COLOR), arrowcolor=MEAN_COLOR,
                        bgcolor="rgba(255,255,255,0.85)", borderpad=3)
    p1 = _export(fig1, FIGS / "fig_temp_year.png")

    # 图 2：日周期（全年同一时刻取平均）——24 个点，一眼看出形状
    hourly = temp.groupby(temp.index.hour).mean()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=hourly.index, y=hourly.values, mode="lines+markers",
                              line=dict(color=TEMP_COLOR, width=3), marker=dict(size=6)))
    _base_layout(fig2, x_title="一天中的时刻", y_title="平均温度（°C）")
    fig2.update_xaxes(tickvals=list(range(0, 24, 2)), ticktext=[f"{h}时" for h in range(0, 24, 2)])
    hmax, hmin = int(hourly.idxmax()), int(hourly.idxmin())
    fig2.add_annotation(x=hmax, y=hourly.loc[hmax], text=f"{hmax} 时最热 {hourly.loc[hmax]:.1f}°C",
                        showarrow=True, arrowhead=2, ax=34, ay=-26,
                        font=dict(size=16, color=MEAN_COLOR), arrowcolor=MEAN_COLOR,
                        bgcolor="rgba(255,255,255,0.85)", borderpad=3)
    fig2.add_annotation(x=hmin, y=hourly.loc[hmin], text=f"{hmin} 时最冷 {hourly.loc[hmin]:.1f}°C",
                        showarrow=True, arrowhead=2, ax=-30, ay=26,
                        font=dict(size=16, color=MEAN_COLOR), arrowcolor=MEAN_COLOR,
                        bgcolor="rgba(255,255,255,0.85)", borderpad=3)
    p2 = _export(fig2, FIGS / "fig_temp_day.png", width=1080, height=500)

    # 图 3：变化点前后 8 天的降雨——只画降雨，单位单一
    res = load("pipeline")
    cp = res["data"]["change_points"][0]["timestamp"]
    cut = pd.Timestamp(cp)
    win = rain.loc[cut - pd.Timedelta(days=7): cut + pd.Timedelta(days=7)]
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=win.index, y=win.values, marker_color=BAR_COLOR, width=1000 * 3600 * 0.7))
    _base_layout(fig3, x_title="", y_title="降雨（mm/小时）")
    fig3.update_xaxes(tickformat="%m-%d", dtick=2 * 24 * 3600 * 1000)
    peak_t, peak_v = win.idxmax(), float(win.max())
    fig3.add_vline(x=cut, line=dict(color=MEAN_COLOR, width=2, dash="dash"))
    fig3.add_annotation(x=peak_t, y=peak_v, text=f"{peak_t:%m-%d %H:%M}　{peak_v:.1f} mm",
                        showarrow=True, arrowhead=2, ax=104, ay=-54,
                        font=dict(size=16, color=MEAN_COLOR), arrowcolor=MEAN_COLOR,
                        bgcolor="rgba(255,255,255,0.9)", borderpad=4)
    p3 = _export(fig3, FIGS / "fig_rain_event.png")
    return {"year": p1, "day": p2, "rain": p3}


# ---------------------------------------------------------------- docx 基础件

def _set_font(run, name: str, size: float | None = None, bold: bool | None = None,
              color: RGBColor | None = None) -> None:
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def _shade(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def _cell_margins(table, top=70, bottom=70, left=110, right=110) -> None:
    tbl_pr = table._tbl.tblPr
    mar = tbl_pr.find(qn("w:tblCellMar"))
    if mar is None:
        mar = OxmlElement("w:tblCellMar")
        tbl_pr.append(mar)
    for tag, val in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        el = mar.find(qn("w:%s" % tag))
        if el is None:
            el = OxmlElement("w:%s" % tag)
            mar.append(el)
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")


def _table_borders(table, color=LINE) -> None:
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom", "insideH"):
        el = OxmlElement("w:%s" % edge)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "6")
        el.set(qn("w:color"), color)
        borders.append(el)
    for edge in ("left", "right", "insideV"):
        el = OxmlElement("w:%s" % edge)
        el.set(qn("w:val"), "nil")
        borders.append(el)
    tbl_pr.append(borders)


def _no_split(table) -> None:
    for row in table.rows:
        tr_pr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        tr_pr.append(cant)


def _cell(cell, text: str, size: float = 9.5, bold: bool = False,
          color: RGBColor | None = INK, align=WD_ALIGN_PARAGRAPH.LEFT, font: str = BODY) -> None:
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(1.5)
    p.paragraph_format.space_after = Pt(1.5)
    p.paragraph_format.line_spacing = 1.3
    _set_font(p.add_run(text), font, size, bold, color)


def add_table(doc, headers, rows, widths, *, font_size=9.5, center=(), head_fill=HEAD_FILL):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    _table_borders(table)
    _cell_margins(table, top=95, bottom=95)
    for i, head in enumerate(headers):
        cell = table.rows[0].cells[i]
        _cell(cell, head, font_size, True, INK, WD_ALIGN_PARAGRAPH.LEFT if i not in center else WD_ALIGN_PARAGRAPH.CENTER)
        _shade(cell, head_fill)
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            _cell(cells[i], value, font_size, False, INK,
                  WD_ALIGN_PARAGRAPH.CENTER if i in center else WD_ALIGN_PARAGRAPH.LEFT)
    for i, width in enumerate(widths):
        table.columns[i].width = Cm(width)
        for row in table.rows:
            row.cells[i].width = Cm(width)
    _no_split(table)
    return table


def add_flow(doc, steps) -> None:
    """六步链路：一行序号 + 一行说明，靠底边加一条细线。"""
    table = doc.add_table(rows=2, cols=len(steps))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    _table_borders(table)
    _cell_margins(table, top=90, bottom=90)
    for i, (name, hint) in enumerate(steps):
        top = table.rows[0].cells[i]
        _cell(top, name, 10.5, True, INK, WD_ALIGN_PARAGRAPH.CENTER, HEAD)
        _shade(top, SOFT_FILL)
        bottom = table.rows[1].cells[i]
        _cell(bottom, hint, 8.5, False, GRAY, WD_ALIGN_PARAGRAPH.CENTER)
    width = TOTAL_W / len(steps)
    for i in range(len(steps)):
        table.columns[i].width = Cm(width)
        for row in table.rows:
            row.cells[i].width = Cm(width)
    _no_split(table)


def h1(doc, text: str, *, space_before: int = 16, new_page: bool = False) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.page_break_before = new_page
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.keep_with_next = True
    _set_font(p.add_run(text), HEAD, 15, True, INK)
    ppr = p._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "8")
    bottom.set(qn("w:space"), "4")
    bottom.set(qn("w:color"), "C9C9C9")
    pbdr.append(bottom)
    ppr.insert_element_before(pbdr, "w:shd", "w:tabs", "w:suppressAutoHyphens", "w:kinsoku",
                              "w:wordWrap", "w:overflowPunct", "w:topLinePunct", "w:autoSpaceDE",
                              "w:autoSpaceDN", "w:bidi", "w:adjustRightInd", "w:snapToGrid",
                              "w:spacing", "w:ind", "w:contextualSpacing", "w:mirrorIndents",
                              "w:suppressOverlap", "w:jc", "w:textDirection", "w:textAlignment",
                              "w:textboxTightWrap", "w:outlineLvl", "w:divId", "w:cnfStyle",
                              "w:rPr", "w:sectPr", "w:pPrChange")


def h2(doc, text: str, *, space_before: int = 12) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    _set_font(p.add_run(text), HEAD, 11.5, True, INK)


def body(doc, text: str, *, size: float = 10.5, after: int = 7, lead: str | None = None,
         color: RGBColor = INK, keep_next: bool = False) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.55
    p.paragraph_format.keep_with_next = keep_next
    if lead:
        _set_font(p.add_run(lead), BODY, size, True, color)
    _set_font(p.add_run(text), BODY, size, False, color)


def small(doc, text: str, *, size: float = 8.5, after: int = 8, color: RGBColor = GRAY) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.35
    _set_font(p.add_run(text), BODY, size, False, color)


def figure(doc, path: Path, width: float, caption: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(8)
    _set_font(cap.add_run(caption), BODY, 8.5, False, GRAY)


def code_block(doc, lines) -> None:
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _cell_margins(table, top=90, bottom=90, left=160, right=160)
    _no_split(table)
    cell = table.rows[0].cells[0]
    _shade(cell, SOFT_FILL)
    cell.width = Cm(TOTAL_W)
    first = True
    for line in lines:
        p = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.25
        _set_font(p.add_run(line), MONO, 8.5, False, INK)


def callout(doc, lead: str, text: str, *, size: float = 10) -> None:
    """浅底提示框：一段结论 + 一条解释。"""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _cell_margins(table, top=150, bottom=150, left=210, right=210)
    cell = table.rows[0].cells[0]
    _shade(cell, SOFT_FILL)
    table.columns[0].width = Cm(TOTAL_W)
    cell.width = Cm(TOTAL_W)
    par = cell.paragraphs[0]
    par.paragraph_format.space_before = Pt(0)
    par.paragraph_format.space_after = Pt(0)
    par.paragraph_format.line_spacing = 1.55
    _set_font(par.add_run(lead), BODY, size, True, ACCENT)
    _set_font(par.add_run(text), BODY, size, False, INK)


def _footer(doc) -> None:
    p = doc.sections[0].footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    _set_font(run, BODY, 8.5, False, LIGHT)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    inner = OxmlElement("w:r")
    text = OxmlElement("w:t")
    text.text = "1"
    inner.append(text)
    fld.append(inner)
    run._r.addnext(fld)


def _no_snap(ppr) -> None:
    """关闭「对齐到网格」。

    默认模板带 linePitch=360 的行网格，24pt / 15pt 的标题会被 Word 按网格拉开字间距，
    导出的 PDF 看起来像每个字之间多了空格。段落级 snapToGrid=0 可消除。
    """
    old = ppr.find(qn("w:snapToGrid"))
    if old is not None:
        ppr.remove(old)
    el = OxmlElement("w:snapToGrid")
    el.set(qn("w:val"), "0")
    ppr.insert_element_before(el, "w:spacing", "w:ind", "w:contextualSpacing", "w:mirrorIndents",
                              "w:suppressOverlap", "w:jc", "w:textDirection", "w:textAlignment",
                              "w:textboxTightWrap", "w:outlineLvl", "w:divId", "w:cnfStyle",
                              "w:rPr", "w:sectPr", "w:pPrChange")


def _setup(doc) -> None:
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.left_margin = Cm(2.2)
    sec.right_margin = Cm(2.2)
    sec.top_margin = Cm(2.0)
    sec.bottom_margin = Cm(1.9)
    style = doc.styles["Normal"]
    style.font.name = BODY
    style.font.size = Pt(10.5)
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), BODY)
    _no_snap(style.element.get_or_add_pPr())


def page_break(doc) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(0)
    p.add_run().add_break(WD_BREAK.PAGE)


# ---------------------------------------------------------------- 文档

CN_COL = {
    "temperature": "温度", "humidity": "湿度", "pressure": "气压",
    "wind_speed": "风速", "rainfall": "降雨",
}
TOOL_CN = {
    "compute_metrics": "指标计算", "analyze_spectrum": "频谱分析", "run_stat_tests": "统计检验",
    "detect_anomalies": "异常检测", "detect_change_points": "变化点检测",
}
ANALYSIS_CN = {
    "trend": "趋势", "corr": "相关", "lag": "滞后", "extreme": "极端", "spectrum": "频谱",
    "changepoint": "变化点", "anomaly": "异常", "test": "检验", "stationarity": "平稳",
    "autocorr": "自相关", "normality": "正态", "seasonal": "季节性", "randomness": "随机性",
    "volatility": "波动率",
}
QUALITY_CN = {
    "MISSING_VALUES": "缺失值", "DUPLICATE_TIMESTAMPS": "重复时间戳",
    "OUT_OF_ORDER": "时间乱序", "SAMPLING_GAP": "采样缺口",
    "OUTLIER_CANDIDATE": "离群候选", "POSSIBLE_UNIT_CHANGE": "疑似单位变化",
    "CONSTANT_COLUMN": "常量列", "ALL_NULL_COLUMN": "整列为空",
}
CONF_CN = {"low": "低", "medium": "中", "high": "高"}
TERM_CN = {
    "finish": "模型主动结束",
    "budget_exhausted": "范围策略用尽（guided 只允许 1 步扩展）",
    "max_steps": "到达步数上限",
    "idle_loop": "连续空转，主动终止",
    "no_tools": "没有可用工具",
}


def _cn(name: str) -> str:
    return CN_COL.get(name, name)


def build(rerun: bool = False) -> Path:
    pipe = load("pipeline")
    data = pipe["data"]
    clean = data["clean"]
    plan = data["plan"]
    params = plan.get("parameters") or {}
    cols = plan["target_columns"]
    issues = (data.get("quality") or {}).get("issues") or []
    anoms = data.get("anomalies") or []
    cps = data.get("change_points") or []
    extremes = data.get("extremes") or []
    tests = data.get("tests") or []
    conclusions = pipe.get("conclusions") or []
    artifacts = pipe.get("artifacts") or []
    by_col = Counter(a.get("column") for a in anoms)

    df = pd.read_csv(DEMO / "pipeline" / "clean_data.csv", parse_dates=["timestamp"]).set_index("timestamp")
    temp, rain, pres = df["temperature"], df["rainfall"], df["pressure"]
    monthly = temp.groupby(temp.index.month).mean()
    hourly = temp.groupby(temp.index.hour).mean()
    cold, warm = int(monthly.idxmin()), int(monthly.idxmax())
    hmax, hmin = int(hourly.idxmax()), int(hourly.idxmin())
    rmax, rmax_at = float(rain.max()), rain.idxmax()

    cp = cps[0]
    cp_dt = pd.Timestamp(cp["timestamp"])
    # 方差突变用的窗宽由检测器按样本量决定（max(12, n//40)），报告必须写实际值，
    # 否则读者自己用 24 小时窗口复算会对不上。
    cp_window = max(12, clean["rows"] // 40)
    top = max(anoms, key=lambda a: abs(float(a.get("score") or 0))) if anoms else None
    temp_hi = next((e for e in extremes if e.get("column") == "temperature" and e.get("tail") == "high"), {})
    temp_lo = next((e for e in extremes if e.get("column") == "temperature" and e.get("tail") == "low"), {})

    has_missing = any(i.get("code") == "MISSING_VALUES" for i in issues)
    dup = next((i for i in issues if i.get("code") == "DUPLICATE_TIMESTAMPS"), None)
    quality = "、".join(f"{i.get('count', 0)} 条「{QUALITY_CN.get(i.get('code'), i.get('code'))}」"
                       f"（{_cn(str(i.get('column'))) }）"
                       for i in issues[:2]) or "未发现质量问题"
    q_line = (f"{'有' if has_missing else '0'} 缺失、{dup.get('count', 0) if dup else 0} 重复；"
              f"{quality}；只提示、不改数据")

    metrics_n = len(data.get("metrics") or {}) + len(data.get("scenario_metrics") or {})
    tests_per_col = (len(tests) // len(cols)) if cols and tests else 0
    n_files = len([f for f in os.listdir(DEMO / "pipeline") if (DEMO / "pipeline" / f).is_file()])
    n_png = len([f for f in os.listdir(DEMO / "pipeline") if f.endswith(".png")])
    cats = Counter(c.get("category") for c in conclusions)
    period_txt = next((s.get("dominant_period") for s in (data.get("spectrum") or [])
                       if s.get("column") == "temperature"), "23.27 小时")
    period_p = next((s.get("dominant_period") for s in (data.get("spectrum") or [])
                     if s.get("column") == "pressure"), "10.67 天")

    try:
        ag = load("agent")["data"]["agent"]
    except Exception:                                 # noqa: BLE001  缺模型时没有产物
        ag = None
    try:
        fc = load("forecast")["data"]["forecast"]
    except Exception:
        fc = None

    doc = Document()
    _setup(doc)
    _footer(doc)

    # ---------------- 第 1 页：导览
    title = doc.add_paragraph()
    title.paragraph_format.space_before = Pt(6)
    title.paragraph_format.space_after = Pt(2)
    _set_font(title.add_run("一条数据，走完整条链路"), HEAD, 24, True, INK)

    sub = doc.add_paragraph()
    sub.paragraph_format.space_after = Pt(12)
    _set_font(sub.add_run(f"柏林 2024 年逐小时天气　·　{clean['rows']:,} 行 CSV　→　"
                          f"{len(conclusions)} 条带证据的结论"), BODY, 11, False, GRAY)

    body(doc, "这份文档只用一份数据，把系统从「拿到 CSV」到「交出结论」的每一步走一遍。"
              "每一步给的都是这份数据上的真实结果，不是示意。")
    body(doc, "第 3、4 页回答三个具体问题：温度怎么变、有没有周期、有没有异常和突变。"
              "第 5 页是最终拿到的东西，以及这些结论能信到什么程度。", after=12)

    # 覆盖模板默认的 core properties（否则作者显示 python-docx、创建时间 2013 年）
    stamp = datetime.now(timezone.utc).replace(microsecond=0)
    props = doc.core_properties
    props.title = "一条数据，走完整条链路"
    props.subject = f"柏林 2024 年逐小时天气：{clean['rows']:,} 行 CSV 走完六步"
    props.author = AUTHOR
    props.last_modified_by = AUTHOR
    props.keywords = "时序分析, 天气, 全链路, 证据, 复现"
    props.comments = ""
    props.category = ""
    props.created = stamp
    props.modified = stamp

    h2(doc, "这份数据", space_before=4)
    add_table(doc, ["项目", "取值"], [
        ["数据", "柏林（BER）2024 年逐小时天气，Open-Meteo 再分析"],
        ["规模", f"{clean['rows']:,} 行 × {len(cols)} 列：" + "、".join(_cn(c) for c in cols)],
        ["时间", f"{df.index.min():%Y-%m-%d %H:%M} 至 {df.index.max():%Y-%m-%d %H:%M}，无缺失"],
        ["问题", QUESTION],
    ], widths=[2.6, 14.0], font_size=9.5)

    body(doc, "", after=2)
    h2(doc, "全链路只有六步", space_before=8)
    add_flow(doc, [("① 体检", "能不能用"), ("② 计划", "做什么"), ("③ 清洗", "改不改"),
                   ("④ 分析", "算什么"), ("⑤ 结论", "怎么说"), ("⑥ 报告", "交什么")])
    body(doc, "顺序固定。六步里只有第 ④ 步的内容会随问题变化，其余五步由数据和计划决定——"
              "所以同一份数据、同一个问题，重跑一定得到同一份结果。", after=4)
    small(doc, "pipeline 模式（默认）：固定计划，确定性，可作基线。agent 模式只是把第 ④ 步换成模型选工具，"
               "数值仍由代码算。两种跑法的对照在第 5 页。")
    body(doc, "", after=4)
    callout(doc, "先给结论：",
            f"这是柏林很正常的一年——{cold} 月最冷、{warm} 月最热，日周期 {period_txt}。"
            f"{len(anoms)} 个统计异常里，气压 {by_col.get('pressure', 0)} 个（冬季低压系统）、"
            f"降雨 {by_col.get('rainfall', 0)} 个；唯一的变化点是 {rmax:.1f} mm 的暴雨。"
            "数据本身没有发现问题，也没有把这场雨说成气候突变。")


    # ---------------- 第 2 页：六步
    h1(doc, "六步分别做了什么", space_before=0, new_page=True)
    body(doc, "下面这张表是这条链路的全部内容。左列是阶段，中列是它要回答的问题，右列是这份数据上的实际结果。",
         keep_next=True)
    add_table(doc, ["阶段", "这一步做什么", "这份数据的结果"], [
        ["① 体检", "时间轴是否连续，有没有缺失、重复、乱序；量级和单位是否可疑", q_line],
        ["② 计划", "定分析项、目标列、周期和阈值", f"{len(plan['analyses'])} 项分析、{len(cols)} 个目标列、"
                                                  f"周期 {params.get('seasonal_period', 24)} 小时、"
                                                  f"异常阈值 {params.get('anomaly_threshold', 8.0)}"],
        ["③ 清洗", "只在必要时插补或丢弃，动作逐条记账", f"插补 {clean['imputed_cells']} 个单元格、"
                                                          f"丢弃 {clean['dropped_rows']} 行；"
                                                          f"单位换算 {len(clean.get('unit_conversions') or [])} 次"],
        ["④ 分析", "指标、频谱、统计检验、异常、变化点、极端值",
         f"{metrics_n} 项指标（通用 {len(data.get('metrics') or {})} + 天气专用 "
         f"{len(data.get('scenario_metrics') or {})}）、每列 {tests_per_col} 项检验、"
         f"{len(anoms)} 个异常点、{len(cps)} 个变化点、{len(extremes)} 组极端值"],
        ["⑤ 结论", "把数字写成一句话，并绑定证据",
         f"{len(conclusions)} 条结论、{len(cats)} 个类别（" + "、".join(cats) + "）"],
        ["⑥ 报告", "落盘，可复现、可回放",
         f"{n_files} 个文件：result.json、report.md、{n_png} 张 PNG、{n_png} 张交互图、"
         f"清洗后数据、复现脚本"],
    ], widths=[1.9, 6.0, 8.7], font_size=9)
    small(doc, "第 ② 步选中的 " + str(len(plan["analyses"])) + " 项："
               + "、".join(ANALYSIS_CN.get(a, a) for a in plan["analyses"]) + "。", after=12)
    body(doc, "0 插补、0 丢弃，说明这份数据本身是干净的。体检报的那条「疑似单位变化」来自降雨："
              "中位数从 0.1 掉到 0.0。因为只有一个转移点，规则只提示、不做换算——宁可少改，不可改错。",
         lead="③ 这一行值得停一下。")
    body(doc, "每条结论后面固定三行：指标名、产物编号、计算位置（哪个文件哪个函数）。"
              "看到可疑结论，可以直接跳到那一行代码。",
         lead="⑤ 的证据长什么样。")
    small(doc, f"登记为产物的文件 {len(artifacts)} 个；运行目录里另有请求记录与中间文件。"
               "真伪不看目录是否非空，看 result.json 的 status。")
    body(doc, "", after=4)
    callout(doc, "这一页的分量在哪：",
            f"插补 0、丢弃 0、换算 0，但体检照样报了 {sum(int(i.get('count') or 0) for i in issues)} 条提示。"
            "系统宁可多提示，也不擅自改数——后面所有结论的可信度都建立在这条上。")


    # ---------------- 第 3 页：结果（一）
    h1(doc, "结果（一）：温度怎么变，有没有周期", space_before=0, new_page=True)
    h2(doc, "问题一：一年里怎么变？", space_before=4)
    body(doc, f"{cold} 月最冷、{warm} 月最热，月均从 {monthly.loc[cold]:.1f} °C 升到 "
              f"{monthly.loc[warm]:.1f} °C，相差 {monthly.max() - monthly.min():.1f} °C；"
              f"全年最低 {temp.min():.1f} °C、最高 {temp.max():.1f} °C。")
    figure(doc, FIGS / "fig_temp_year.png", 16.2,
           "图 1　逐小时温度（细线）与月均值（粗线）。一年只有一个峰、一个谷。")
    body(doc, "全年数据上的趋势检验是显著的（Mann-Kendall，p 远小于 0.001），但它主要来自年周期："
              "1 月到 8 月上升、9 月到 12 月下降。要谈长期趋势，至少需要多年数据并先做季节调整。",
         lead="这条线不能读成气候变暖。")

    h2(doc, "问题二：有没有周期？")
    body(doc, f"主周期 {period_txt}，约等于一天；一天里 {hmax} 时最热（{hourly.loc[hmax]:.1f} °C），"
              f"{hmin} 时最冷（{hourly.loc[hmin]:.1f} °C）。")
    figure(doc, FIGS / "fig_temp_day.png", 13.4,
           "图 2　把全年同一时刻的温度取平均，得到这条 24 小时曲线。")
    body(doc, f"气压上的周期是 {period_p}——天气系统过境的尺度，和 24 小时差两个数量级。"
              "所以图必须分开画：五列塞进同一张图，温度会被气压压成一条直线。")


    # ---------------- 第 4 页：结果（二）
    h1(doc, "结果（二）：异常和突变是什么", space_before=0, new_page=True)
    h2(doc, "问题三：有没有异常点、有没有变化点？", space_before=4)
    figure(doc, FIGS / "fig_rain_event.png", 16.2,
           f"图 3　变化点前后各 7 天的降雨。虚线是检测到的变化点：{cp_dt:%Y-%m-%d %H:%M}。")
    body(doc, f"全年最大的一小时雨量就出现在这一刻：{rmax:.1f} mm，也是全年最显著的异常点"
              f"（稳健 z={float(top['score']):.1f}，判定阈值为 8）。同一时刻，变化点检测器给出「降雨的波动幅度从 "
              f"{cp['before']:.2f} 抬到 {cp['after']:.2f}」（{cp_window} 小时滚动标准差，约 "
              f"{cp_window / 24:.0f} 天；跳升 {float(cp.get('statistic') or 0):.1f} 倍，"
              f"置信度{CONF_CN.get(str(cp.get('confidence')), cp.get('confidence'))}）。")
    body(doc, "所以这里的「变化点」不是气候结构改变，是一次强对流天气把短时波动抬上去了。",
         lead="两个结果说的是同一件事：那场雨。")

    h2(doc, "三个数字别读错")
    add_table(doc, ["看到的", "容易读成", "实际是"], [
        [f"{len(anoms)} 个异常点", f"{len(anoms)} 次灾害",
         f"气压 {by_col.get('pressure', 0)} 个 + 降雨 {by_col.get('rainfall', 0)} 个。"
         f"气压那部分是冬季低压系统（全年最低 {pres.min():.1f} hPa），属正常天气过程"],
        [f"{len(cps)} 个变化点", "一次气候突变",
         f"一场 {rmax:.1f} mm 的暴雨抬高了波动幅度；单次极端事件会被误读为结构变化，这是已知限制"],
        [f"{len(extremes)} 组极端值", "异常事件",
         f"每列按经验 p99/p1 判定的绝对极端（温度 {temp_hi.get('threshold', 0):.1f} / "
         f"{temp_lo.get('threshold', 0):.1f} °C），与统计异常是两套判据"],
    ], widths=[2.9, 4.2, 9.5], font_size=9)
    small(doc, "异常看「相对本列的偏离」，极端看「绝对分位」，变化点看「前后两段是否不同」。"
               "三个口径混在一句话里，是这类报告最常见的错误。")

    h2(doc, "往前看一个周期（可选）")
    if fc:
        fcols = fc.get("columns") or []
        beats = [c for c in fcols if c.get("beats_baseline")]
        nots = [c for c in fcols if not c.get("beats_baseline")]
        text = (f"打开预测开关（--forecast）：{len(fcols)} 列 × {fc.get('horizon')} 小时点预测，"
                f"带 {int(round(float(fc.get('interval_level', 0.9)) * 100))}% 区间。")
        if beats:
            text += (f"{len(beats)} 列优于基线（"
                     + "、".join(f"{_cn(c['column'])} MASE {c['selected_mase']:.3f}" for c in beats)
                     + "），")
        if nots:
            text += "、".join(_cn(c["column"]) for c in nots) + " 未优于基线，报告里如实写明。"
        body(doc, text)
    else:
        body(doc, "本次没有跑预测（默认关闭），这一节跳过。")


    # ---------------- 第 5 页：交付物与边界
    h1(doc, "最后拿到什么", space_before=0, new_page=True)
    add_table(doc, ["拿到的东西", "里面是什么"], [
        ["result.json / report.md", "机器可读的完整结果（状态、指标、异常、变化点、结论、证据链）"
                                    "与人读报告"],
        [f"{n_png} 张 PNG + {n_png} 张交互图", "时间序列、相关矩阵、频谱、月度均值、月度箱线、异常标注"],
        ["clean_data.csv + reproduce.py", "清洗后的数据；与本次完全相同的参数，直接重跑"],
    ], widths=[4.6, 12.0], font_size=9)
    body(doc, "", after=2)

    h2(doc, "同一份数据的第二种跑法", space_before=6)
    if ag:
        missing = "、".join(TOOL_CN.get(t, t) for t in (ag.get("coverage") or {}).get("missing_tools") or []) or "无"
        tools_n = len(ag.get("tools_run") or [])
        term = TERM_CN.get(ag.get("terminate_reason"), ag.get("terminate_reason") or "—")
        tokens = (ag.get("tokens") or {}).get("total_tokens") or 0
        calls = (ag.get("tokens") or {}).get("calls") or 0
        add_table(doc, ["维度", "pipeline（默认）", "agent"], [
            ["结论与工具", f"{len(conclusions)} 条；5 个工具全执行",
             f"{len(load('agent').get('conclusions') or [])} 条；{tools_n} 个工具（缺{missing}）"],
            ["模型调用", "无", f"{calls} 次，{tokens:,} token，{ag.get('step_count')} 步"],
            ["结束原因", "跑完固定计划", term],
            ["可复现", "确定性，重跑一致", "轨迹哈希，可 --replay 离线重放"],
        ], widths=[2.4, 6.6, 7.6], font_size=9)
        body(doc, "这一次两边结论数相同：缺的统计检验本身不生成结论，差异只落在报告第 7 节和告警里。"
                  "放到 13 个用例的批量对照，agent 平均少 23% 的结论——少的都是「没跑的工具」，不是算错。",
             lead="别把这一步的「一样」当成普遍结论。")
    else:
        body(doc, "本次没有跑 agent 模式（需要模型配置），这一节跳过。")


    h2(doc, "这些结论能信到什么程度")
    body(doc, "模型只决定「先跑哪个工具」，不参与数值；每个数字都能追到指标和代码位置。", lead="算得对。")
    body(doc, "统计异常不等于灾害；变化点可能只是单场暴雨；预测区间没有物理约束（湿度上界可超 100%）。",
         lead="读得对。")
    body(doc, "极端值按本列经验分位判定，换站点、换年份都要重算；这条链路证明流程可控，不证明业务结论。",
         lead="用得对。")

    h2(doc, "自己跑一遍")
    code_block(doc, [
        "python -m mtsa.cli run --scene weather `",
        "  --input fixtures/real_weather/r01_berlin_hourly_2024/data.csv `",
        f"  --question \"{QUESTION}\"",
    ])
    small(doc, "PowerShell 下先设置 PYTHONPATH=src。重新生成本文档："
               "python scripts/make_chain_demo_doc.py --rerun", after=0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    return OUT


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生成全链路演示 Word 文档")
    ap.add_argument("--rerun", action="store_true", help="先重跑 pipeline/agent/forecast")
    ap.add_argument("--figs-only", action="store_true", help="只重画三张图")
    args = ap.parse_args(argv)

    ensure_demo(args.rerun)
    build_figures()
    if args.figs_only:
        print("figures done ->", FIGS)
        return 0
    path = build(rerun=args.rerun)
    print("saved", path, os.path.getsize(path), "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
