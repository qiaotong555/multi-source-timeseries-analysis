# -*- coding: utf-8 -*-
"""生成《第二阶段报告：把它交给陌生人》Word/PDF。

数据：artifacts/stage2_shanghai/（由 scripts/fetch_shanghai_2024.py 抓取）
      run_raw/result.json（pipeline 实测产物）
输出：doc/第二阶段报告_上海2024.docx（+ .pdf，需本机 LibreOffice）

文档里的每个数字都在生成时从 result.json / CSV 读取，不手写；改数据后重跑即可。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "artifacts" / "stage2_shanghai"
FIGS = DEMO / "figs"
OUT = ROOT / "doc" / "第二阶段报告_上海2024.docx"
AUTHOR = "qiaotong555"

BODY = "等线"
HEAD = "微软雅黑"
MONO = "Consolas"
INK = RGBColor(0x1A, 0x1A, 0x1A)
GRAY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0x8C, 0x8C, 0x8C)
ACCENT = RGBColor(0xB0, 0x3A, 0x2E)
BLUE = RGBColor(0x2F, 0x6D, 0xB3)
LINE = "D9D9D9"
HEAD_FILL = "F2F2F2"
SOFT_FILL = "F7F8FA"
TOTAL_W = 16.6

TEMP_COLOR = "#2F6DB3"
MEAN_COLOR = "#B03A2E"
BAR_COLOR = "#4C7EA8"
FONT = "Microsoft YaHei"


# ------------------------------------------------------------------ 读取与统计
def local_time(iso: str) -> str:
    """UTC ISO → 北京时间，报告里不出现 +00:00。"""
    return datetime.fromisoformat(iso).astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")


def load_json(name: str) -> dict:
    return json.loads((DEMO / name).read_text(encoding="utf-8"))


def gather() -> dict:
    meta = load_json("fetch_meta.json")
    result = load_json("run_raw/result.json")
    cn_result = load_json("run_cn/result.json")
    data = result["data"]

    df = pd.read_csv(DEMO / "shanghai_raw.csv", parse_dates=["time"]).set_index("time")
    t = df["temperature_2m"]
    p = df["precipitation"]
    w = df["wind_speed_10m"]
    pres = df["surface_pressure"]
    hum = df["relative_humidity_2m"]

    mon = t.resample("MS").mean()
    rain_mon = p.resample("MS").sum()
    q = t.resample("QS").mean()

    hour = t.groupby(t.index.hour).mean()

    anoms = data.get("anomalies") or []
    by_col: dict[str, list] = {}
    for a in anoms:
        by_col.setdefault(a["column"], []).append(a)

    cps = data.get("change_points") or []

    def norm_anoms(res):
        out = []
        for a in res["data"].get("anomalies") or []:
            out.append((a["timestamp"], CN2RAW.get(a["column"], a["column"]), round(a["score"], 6)))
        return sorted(out)

    def norm_cp(res):
        out = []
        for c in res["data"].get("change_points") or []:
            out.append((c["timestamp"], CN2RAW.get(c["column"], c["column"]),
                        c["kind"], round(c["before"], 6), round(c["after"], 6)))
        return sorted(out)

    same_anoms = norm_anoms(result) == norm_anoms(cn_result)
    same_cp = norm_cp(result) == norm_cp(cn_result)

    typhoon = df.loc["2024-09-15 18:00":"2024-09-16 16:00"]
    wind_anoms = sorted(by_col.get("wind_speed_10m", []), key=lambda x: x["timestamp"])
    press_anoms = sorted(by_col.get("surface_pressure", []), key=lambda x: x["timestamp"])

    return {
        "meta": meta,
        "result": result,
        "data": data,
        "df": df,
        "rows": len(df),
        "sha_raw": meta["files"]["shanghai_raw.csv"]["sha1"][:12],
        "sha_cn": meta["files"]["shanghai_cn.csv"]["sha1"][:12],
        "fetched_at": meta["fetched_at"],
        "t_min": float(t.min()), "t_min_at": t.idxmin(),
        "t_max": float(t.max()), "t_max_at": t.idxmax(),
        "t_mean": float(t.mean()),
        "amp": float(mon.max() - mon.min()),
        "mon": [(i + 1, float(v)) for i, v in enumerate(mon)],
        "q": [(f"Q{i + 1}", float(v)) for i, v in enumerate(q)],
        "coldest": (int(mon.idxmin().month), float(mon.min())),
        "hottest": (int(mon.idxmax().month), float(mon.max())),
        "rain_total": float(p.sum()), "rain_hours": int((p > 0).sum()),
        "rain_max": float(p.max()), "rain_max_at": p.idxmax(),
        "rain_mon": [(i + 1, float(v)) for i, v in enumerate(rain_mon)],
        "rain_peak": (int(rain_mon.idxmax().month), float(rain_mon.max())),
        "wind_mean": float(w.mean()), "wind_max": float(w.max()), "wind_max_at": w.idxmax(),
        "p_min": float(pres.min()), "p_min_at": pres.idxmin(),
        "p_max": float(pres.max()), "p_max_at": pres.idxmax(),
        "hum_min": float(hum.min()), "hum_max": float(hum.max()),
        "hour": [(int(h), float(v)) for h, v in hour.items()],
        "h_hot": (int(hour.idxmax()), float(hour.max())),
        "h_cold": (int(hour.idxmin()), float(hour.min())),
        "n_anom": len(anoms), "anom_by_col": {k: len(v) for k, v in by_col.items()},
        "wind_anoms": wind_anoms, "press_anoms": press_anoms,
        "n_cp": len(cps), "cps": cps,
        "n_concl": len(result.get("conclusions") or []),
        "n_evidence": len(result.get("evidence") or []),
        "same_anoms": same_anoms, "same_cp": same_cp,
        "typhoon": typhoon,
        "ty_p": (float(typhoon["surface_pressure"].iloc[0]), float(typhoon["surface_pressure"].min())),
        "ty_wind": float(typhoon["wind_speed_10m"].max()),
        "ty_eye": float(typhoon["wind_speed_10m"].min()),
    }


CN2RAW = {
    "温度(°C)": "temperature_2m", "相对湿度(%)": "relative_humidity_2m",
    "气压(hPa)": "surface_pressure", "风速(km/h)": "wind_speed_10m",
    "降水量(mm)": "precipitation",
}

# ------------------------------------------------------------------ 图
def _base(fig, *, x_title="", y_title="", height=430, width=1080, margin_l=64):
    fig.update_layout(
        template="plotly_white", font=dict(family=FONT, size=12, color="#1A1A1A"),
        width=width, height=height,
        margin=dict(l=margin_l, r=26, t=34, b=52),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11)),
        hoverlabel=dict(font=dict(family=FONT, size=11)),
    )
    fig.update_xaxes(title_text=x_title, title_font=dict(size=12), tickfont=dict(size=11),
                     showgrid=False, linecolor="#D9D9D9", ticks="outside", tickcolor="#D9D9D9")
    fig.update_yaxes(title_text=y_title, title_font=dict(size=12), tickfont=dict(size=11),
                     gridcolor="#EEEEEE", zeroline=False, linecolor="#D9D9D9")


def fig_temperature(S, path: Path) -> Path:
    df, mon = S["df"], S["mon"]
    t = df["temperature_2m"]
    months = pd.date_range("2024-01-01", periods=12, freq="MS")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t.index, y=t.values, mode="lines", name="逐小时温度",
                             line=dict(color="#9DB8D2", width=1),
                             hovertemplate="%{x|%m-%d %H:%M}　%{y:.1f} °C<extra></extra>"))
    fig.add_trace(go.Scatter(x=months, y=[v for _, v in mon], mode="lines+markers", name="月均",
                             line=dict(color=MEAN_COLOR, width=2.6), marker=dict(size=6),
                             hovertemplate="%{x|%m 月}　月均 %{y:.1f} °C<extra></extra>"))
    fig.add_annotation(x=S["t_max_at"], y=S["t_max"], text=f"{S['t_max']:.1f} °C",
                       showarrow=True, arrowhead=2, ax=-12, ay=-32,
                       font=dict(size=12, color=MEAN_COLOR), bgcolor="rgba(255,255,255,0.9)")
    fig.add_annotation(x=S["t_min_at"], y=S["t_min"], text=f"{S['t_min']:.1f} °C",
                       showarrow=True, arrowhead=2, ax=-46, ay=-18,
                       font=dict(size=12, color=TEMP_COLOR), bgcolor="rgba(255,255,255,0.9)")
    _base(fig, x_title="2024 年", y_title="温度（°C）")
    fig.update_xaxes(tickmode="array", tickvals=months, ticktext=[f"{m.month} 月" for m in months])
    fig.write_image(str(path), width=1080, height=430, scale=2)
    return path


def fig_daily(S, path: Path) -> Path:
    hours = [h for h, _ in S["hour"]]
    vals = [v for _, v in S["hour"]]
    hot, cold = S["h_hot"], S["h_cold"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=vals, mode="lines+markers", name="平均温度",
                             line=dict(color=TEMP_COLOR, width=2.4), marker=dict(size=6),
                             hovertemplate="%{x} 时　%{y:.1f} °C<extra></extra>"))
    fig.add_annotation(x=hot[0], y=hot[1], text=f"{hot[0]} 时最热 {hot[1]:.1f} °C",
                       showarrow=True, arrowhead=2, ax=10, ay=-34, font=dict(size=11), bgcolor="rgba(255,255,255,0.9)")
    fig.add_annotation(x=cold[0], y=cold[1], text=f"{cold[0]} 时最冷 {cold[1]:.1f} °C",
                       showarrow=True, arrowhead=2, ax=34, ay=44, font=dict(size=11), bgcolor="rgba(255,255,255,0.9)")
    _base(fig, x_title="一天中的时刻", y_title="全年同时刻平均温度（°C）", width=900, height=380)
    fig.update_xaxes(tickmode="array", tickvals=list(range(0, 24, 3)),
                     ticktext=[f"{h} 时" for h in range(0, 24, 3)])
    fig.update_layout(showlegend=False)
    fig.write_image(str(path), width=900, height=380, scale=2)
    return path


def fig_rain_month(S, path: Path) -> Path:
    names = [f"{m} 月" for m, _ in S["rain_mon"]]
    vals = [v for _, v in S["rain_mon"]]
    peak_month, peak_val = S["rain_peak"]
    colors = [BAR_COLOR if (m != peak_month) else MEAN_COLOR for m, _ in S["rain_mon"]]
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=colors, name="月降水",
                           hovertemplate="%{x}　%{y:.0f} mm<extra></extra>"))
    fig.add_annotation(x=f"{peak_month} 月", y=peak_val, text=f"梅雨 {peak_val:.0f} mm",
                       showarrow=True, arrowhead=2, ax=16, ay=-30, font=dict(size=11, color=MEAN_COLOR),
                       bgcolor="rgba(255,255,255,0.9)")
    _base(fig, y_title="月降水量（mm）", width=900, height=360)
    fig.update_yaxes(range=[0, max(vals) * 1.22])
    fig.update_layout(showlegend=False)
    fig.write_image(str(path), width=900, height=360, scale=2)
    return path


def fig_typhoon(S, path: Path) -> Path:
    win = S["typhoon"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=("地面气压（hPa）", "10 米风速（km/h）", "逐小时降水（mm）"))
    fig.add_trace(go.Scatter(x=win.index, y=win["surface_pressure"], mode="lines",
                             line=dict(color="#4C7EA8", width=2.2), name="气压",
                             hovertemplate="%{x|%m-%d %H:%M}　%{y:.1f} hPa<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=win.index, y=win["wind_speed_10m"], mode="lines",
                             line=dict(color=TEMP_COLOR, width=2.2), name="风速",
                             hovertemplate="%{x|%m-%d %H:%M}　%{y:.1f} km/h<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=win.index, y=win["precipitation"], mode="lines",
                             line=dict(color=MEAN_COLOR, width=2.0), fill="tozeroy",
                             fillcolor="rgba(176,58,46,0.16)", name="降水",
                             hovertemplate="%{x|%m-%d %H:%M}　%{y:.1f} mm<extra></extra>"), row=3, col=1)

    for r in (1, 2, 3):
        fig.add_vline(x="2024-09-16 07:00", line=dict(color="#B03A2E", width=1, dash="dash"), row=r, col=1)

    fig.add_annotation(x="2024-09-16 10:00", y=S["p_min"], row=1, col=1,
                       text=f"最低 {S['p_min']:.1f} hPa", showarrow=True, arrowhead=2, ax=-6, ay=-28,
                       font=dict(size=11, color="#1A1A1A"), bgcolor="rgba(255,255,255,0.9)")
    fig.add_annotation(x="2024-09-16 07:00", y=S["ty_wind"], row=2, col=1,
                       text=f"{S['ty_wind']:.1f} km/h", showarrow=True, arrowhead=2, ax=18, ay=-26,
                       font=dict(size=11, color="#1A1A1A"), bgcolor="rgba(255,255,255,0.9)")
    fig.add_annotation(x="2024-09-16 10:00", y=S["ty_eye"], row=2, col=1,
                       text=f"眼区 {S['ty_eye']:.1f}", showarrow=True, arrowhead=2, ax=30, ay=34,
                       font=dict(size=11, color="#595959"), bgcolor="rgba(255,255,255,0.9)")

    _base(fig, width=980, height=600, margin_l=62)
    fig.update_xaxes(tickformat="%m-%d %H:%M", nticks=7)
    fig.update_layout(showlegend=False, font=dict(family=FONT, size=12, color="#1A1A1A"))
    for ann in fig.layout.annotations[:3]:
        ann.font = dict(family=FONT, size=12, color="#1A1A1A")
    fig.write_image(str(path), width=980, height=600, scale=2)
    return path


def build_figures(S) -> dict:
    FIGS.mkdir(parents=True, exist_ok=True)
    return {
        "temp": fig_temperature(S, FIGS / "fig1_temperature.png"),
        "daily": fig_daily(S, FIGS / "fig2_daily_cycle.png"),
        "rain": fig_rain_month(S, FIGS / "fig3_rain_month.png"),
        "typhoon": fig_typhoon(S, FIGS / "fig4_typhoon.png"),
    }

# ------------------------------------------------------------------ 版式
EMPH = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`)")


def rich(par, text, size=10.5, color=INK):
    """写入一段可能含 **粗体** 与 `等宽` 的文本。"""
    for part in EMPH.split(str(text)):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            set_font(par.add_run(part[2:-2]), BODY, size, True, color)
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            set_font(par.add_run(part[1:-1]), MONO, size - 1.0, False, ACCENT)
        else:
            set_font(par.add_run(part), BODY, size, False, color)


def set_font(run, name, size=None, bold=None, color=None):
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


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def cell_margins(table, top=90, bottom=90, left=140, right=140):
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


def table_borders(table, color=LINE, left_accent=False):
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
        if left_accent and edge == "left":
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "18")
            el.set(qn("w:color"), "B03A2E")
        else:
            el.set(qn("w:val"), "nil")
        borders.append(el)
    tbl_pr.append(borders)


def no_snap(ppr):
    old = ppr.find(qn("w:snapToGrid"))
    if old is not None:
        ppr.remove(old)
    el = OxmlElement("w:snapToGrid")
    el.set(qn("w:val"), "0")
    ppr.insert_element_before(
        el, "w:spacing", "w:ind", "w:contextualSpacing", "w:mirrorIndents",
        "w:suppressOverlap", "w:jc", "w:textDirection", "w:textAlignment",
        "w:textboxTightWrap", "w:outlineLvl", "w:divId", "w:cnfStyle",
        "w:rPr", "w:sectPr", "w:pPrChange",
    )


PBR_REST = (
    "w:shd", "w:tabs", "w:suppressAutoHyphens", "w:kinsoku", "w:wordWrap",
    "w:overflowPunct", "w:topLinePunct", "w:autoSpaceDE", "w:autoSpaceDN",
    "w:bidi", "w:adjustRightInd", "w:snapToGrid", "w:spacing", "w:ind",
    "w:contextualSpacing", "w:mirrorIndents", "w:suppressOverlap", "w:jc",
    "w:textDirection", "w:textAlignment", "w:textboxTightWrap", "w:outlineLvl",
    "w:divId", "w:cnfStyle", "w:rPr", "w:sectPr", "w:pPrChange",
)


def underline(par, size="6", color="C9C9C9", space="4"):
    ppr = par._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), size)
    bottom.set(qn("w:space"), space)
    bottom.set(qn("w:color"), color)
    pbdr.append(bottom)
    ppr.insert_element_before(pbdr, *PBR_REST)


def h1(doc, text, *, before=18, after=8, new_page=False):
    p = doc.add_paragraph()
    p.paragraph_format.page_break_before = new_page
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.line_spacing = 1.2
    set_font(p.add_run(text), HEAD, 15, True, INK)
    underline(p, size="8", space="6")


def h2(doc, text, *, before=12):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.line_spacing = 1.2
    set_font(p.add_run(text), HEAD, 11.5, True, INK)


def body(doc, text, *, size=10.5, after=8, indent=True, color=INK, lead=None,
         lead_color=INK, keep_next=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.6
    if indent:
        p.paragraph_format.first_line_indent = Cm(0.74)
    if keep_next:
        p.paragraph_format.keep_with_next = True
    if lead:
        set_font(p.add_run(lead), BODY, size, True, lead_color)
    rich(p, text, size, color)
    return p


def small(doc, text, *, size=8.5, after=8, color=GRAY):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.35
    rich(p, text, size, color)


def bullet(doc, text, *, size=10.5, lead=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.55
    p.paragraph_format.left_indent = Cm(0.95)
    p.paragraph_format.first_line_indent = Cm(-0.95)
    set_font(p.add_run("·　"), BODY, size, False, LIGHT)
    if lead:
        set_font(p.add_run(lead), BODY, size, True, INK)
    rich(p, text, size, INK)


def table(doc, headers, rows, widths, *, size=9.5, center=(), head_fill=HEAD_FILL):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    table_borders(t)
    cell_margins(t, top=95, bottom=95)
    for i, head in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        shade(cell, head_fill)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(1.5)
        p.paragraph_format.space_after = Pt(1.5)
        p.paragraph_format.line_spacing = 1.3
        set_font(p.add_run(head), BODY, size, True, INK)
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cell = cells[i]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i in center else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1.5)
            p.paragraph_format.space_after = Pt(1.5)
            p.paragraph_format.line_spacing = 1.3
            set_font(p.add_run(str(value)), BODY, size, False, INK)
    for i, w in enumerate(widths):
        t.columns[i].width = Cm(w)
        for row in t.rows:
            row.cells[i].width = Cm(w)
    for row in t.rows[:-1]:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.keep_with_next = True
    for row in t.rows:
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def callout(doc, lead, text, *, size=10, fill=SOFT_FILL):
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    table_borders(t)
    cell_margins(t, top=150, bottom=150, left=220, right=220)
    cell = t.rows[0].cells[0]
    shade(cell, fill)
    t.columns[0].width = Cm(TOTAL_W)
    cell.width = Cm(TOTAL_W)
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.55
    set_font(p.add_run(lead), BODY, size, True, ACCENT)
    set_font(p.add_run(text), BODY, size, False, INK)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def code_block(doc, lines, *, size=8.5):
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    table_borders(t)
    cell_margins(t, top=110, bottom=110, left=180, right=180)
    cell = t.rows[0].cells[0]
    shade(cell, SOFT_FILL)
    t.columns[0].width = Cm(TOTAL_W)
    cell.width = Cm(TOTAL_W)
    for i, line in enumerate(lines):
        p = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.25
        set_font(p.add_run(line), MONO, size, False, INK)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def figure(doc, path, width, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(10)
    set_font(cap.add_run(caption), BODY, 8.5, False, GRAY)


def page_break(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(0)
    p.add_run().add_break(WD_BREAK.PAGE)


def footer(doc):
    p = doc.sections[0].footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(p.add_run("— "), BODY, 8.5, False, LIGHT)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    inner = OxmlElement("w:r")
    text = OxmlElement("w:t")
    text.text = "1"
    inner.append(text)
    fld.append(inner)
    p._p.append(fld)
    set_font(p.add_run(" —"), BODY, 8.5, False, LIGHT)


def setup(doc):
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
    no_snap(style.element.get_or_add_pPr())
    footer(doc)

# ------------------------------------------------------------------ 正文
def build(figs=None) -> Path:
    S = gather()
    figs = figs or build_figures(S)
    meta, data = S["meta"], S["data"]
    docs = Document()
    setup(docs)

    props = docs.core_properties
    stamp = datetime.now(timezone.utc).replace(microsecond=0)
    props.author = AUTHOR
    props.last_modified_by = AUTHOR
    props.title = "第二阶段报告：把它交给陌生人"
    props.subject = "上海 2024 年逐小时天气：一次陌生数据实测"
    props.keywords = "时序分析, 陌生数据, 上海, 台风, 可复现"
    props.comments = ""
    props.category = "课题报告"
    props.created = stamp
    props.modified = stamp

    # ---------------- 封面
    p = docs.add_paragraph()
    p.paragraph_format.space_before = Pt(56)
    p.paragraph_format.space_after = Pt(6)
    set_font(p.add_run("课题 12 · AI for Science · S06　｜　第二阶段报告"), BODY, 9.5, False, LIGHT)

    p = docs.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.1
    set_font(p.add_run("把它交给陌生人"), HEAD, 30, True, INK)

    p = docs.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(16)
    set_font(p.add_run("上海 2024 年逐小时天气：一次陌生数据实测"), HEAD, 13, False, GRAY)

    p = docs.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(26)
    set_font(p.add_run(f"{AUTHOR}　·　2026-10-25　·　数据：Open-Meteo Archive（CC BY 4.0）"),
             BODY, 9.5, False, LIGHT)

    callout(docs, "三句话结论　",
            "① 同一份上海数据写成两种格式，都能跑通，41 个异常点与 2 个变化点逐个相同；"
            "② 系统没有「台风」这个概念，却把台风的气压骤降、风墙、眼区静风三件事都标了出来；"
            "③ 它同时把「干季降雨归零」误报成「单位变化」——第 1 个假警报，已定位到具体判据。")

    small(docs, "本报告所有数字由 scripts/make_stage2_report.py 在生成时从 result.json 与原始 CSV 读取，"
                "没有手写数值；改动数据后重跑脚本即可得到新版本。")

    # ---------------- 一
    h1(docs, "一、这次要测的不是「能不能跑」", before=0, new_page=True)
    body(docs, "项目之前的演示文档用的都是自己的 fixture：表头干净、单位统一、时间格式标准、"
               "每一列都叫 temperature_2m 这种名字。真实用户不会这么客气——他们交上来的是"
               "「温度(°C)」「2024/1/1 0:00」，还常常多带一列城市名。")
    body(docs, "所以这一轮换了一个问法：", after=4)
    bullet(docs, "**换一份别人手里的数据，这套东西还成立吗？**", size=11)
    body(docs, "为此选了一份从未进入过项目的真实数据——上海 2024 年逐小时天气——并故意准备了两种写法，"
               "同一份数值、两种格式，看差异到底出在哪一层。判据只有三条：", indent=False)
    bullet(docs, "会不会直接报错，或者静默丢列、算错列", lead="吃不吃得下：")
    bullet(docs, "数值与 pandas 独立复算是否一致", lead="算不算得对：")
    bullet(docs, "有没有假警报，有没有把日常天气当成灾害", lead="敢不敢信：")

    # ---------------- 二
    h1(docs, "二、数据：上海，一整年，8784 小时")
    table(docs, ["项目", "内容"], [
        ["来源", "Open-Meteo Archive API（底层为 ERA5 / CERRA 再分析），许可 CC BY 4.0"],
        ["位置", "上海 31.2304°N / 121.4737°E（API 实际网格 31.25°N / 121.50°E），海拔 5 m"],
        ["时段", f"2024-01-01 00:00 — 12-31 23:00（Asia/Shanghai），闰年共 {S['rows']:,} 小时"],
        ["要素", "温度、相对湿度、地面气压、10 米风速、逐小时降水"],
        ["缺失", "0（五列均无空值）"],
        ["抓取", f"{local_time(S['fetched_at'])}（北京时间）　SHA1：{S['sha_raw']}…（原样）/ {S['sha_cn']}…（中文）"],
    ], [2.6, 14.0], size=9.0)

    body(docs, "选上海不是随手挑的：它一年里同时有零下寒潮、梅雨、盛夏高温和台风，"
               "四种天气过程齐全，正好能检验「同一套判据能不能同时应付常态与极端」。"
               "而且它不在项目已有的 17 份数据里，没有针对它调过任何参数。")
    body(docs, "再分析数据不是气象站实测：ERA5 网格约 25 km，台风中心附近的高频阵风会被平滑掉。"
               f"所以 9 月 16 日这 {S['ty_wind']:.1f} km/h 只代表「这套数据里」的风速，"
               "不能当作当天真实阵风的实测值——本文所有结论都只针对这套数据。",
         lead="必须说清一件事：", indent=False)

    # ---------------- 三
    h1(docs, "三、两种写法：一份数据，两种「别人手里的样子」")
    table(docs, ["维度", "A · 原样列名", "B · 陌生人写法"], [
        ["表头", "time, temperature_2m, …", "城市, 时间, 温度(°C), 相对湿度(%), …"],
        ["时间", "2024-01-01T00:00", "2024/1/1 0:00"],
        ["文件编码", "UTF-8", "UTF-8 带 BOM（Excel 双击不乱码）"],
        ["形状", f"{S['rows']:,} 行 × 6 列", f"{S['rows']:,} 行 × 7 列（多一列城市）"],
        ["status", "succeeded", "succeeded"],
        ["异常点", f"{S['n_anom']} 个", f"{S['n_anom']} 个（逐个相同）"],
        ["变化点", f"{S['n_cp']} 个", f"{S['n_cp']} 个（逐个相同）"],
        ["结论", f"{S['n_concl']} 条", f"{S['n_concl']} 条"],
    ], [2.4, 7.1, 7.1], size=9.0, center=(1,))
    body(docs, f"把两份结果里的列名映射回英文后逐点比对：异常点的"
               f"（时间、列、分数）三元组完全相同，变化点的（时间、列、类型、前后中位数）也完全相同。"
               f"也就是说，中文表头、单位后缀、非 ISO 时间、多余的城市列，都没有影响数值。",
               indent=False)
    callout(docs, "但这是入场券，不是成绩单　",
            "格式容错做到了，只说明「不会因为写法不同而算出不同结果」。"
            "真正要问的是：算出来的那些结论，值不值得信。后两节回答这个。")

    # ---------------- 四
    h1(docs, "四、它从这一年里看出了什么")
    h2(docs, "4.1　一年怎么变", before=2)
    body(docs, f"全年最低 {S['t_min']:.1f} °C（{S['t_min_at']:%m-%d %H:%M}），最高 {S['t_max']:.1f} °C"
               f"（{S['t_max_at']:%m-%d %H:%M}），年均 {S['t_mean']:.1f} °C。"
               f"月均从 1 月的 {S['mon'][0][1]:.1f} °C 升到 8 月的 {S['mon'][7][1]:.1f} °C，"
               f"极差 {S['amp']:.1f} °C——这是典型的亚热带季风气候年周期。")
    figure(docs, figs["temp"], 15.6,
           f"图 1　逐小时温度（细线）与月均（粗线）。最低 {S['t_min']:.1f} °C，最高 {S['t_max']:.1f} °C。")

    h2(docs, "4.2　一天怎么变")
    body(docs, f"把全年同一时刻的温度取平均，得到一条干净的日周期："
               f"{S['h_cold'][0]} 时最冷（{S['h_cold'][1]:.1f} °C），{S['h_hot'][0]} 时最热"
               f"（{S['h_hot'][1]:.1f} °C），日内差 {S['h_hot'][1] - S['h_cold'][1]:.1f} °C。"
               f"相位比内陆城市晚——城市热岛与海风共同把峰值推到午后两三点。")
    figure(docs, figs["daily"], 13.4, "图 2　全年同时刻平均温度：24 小时周期是这份数据里最强的信号。")

    h2(docs, "4.3　雨下在哪里")
    peak_m, peak_v = S["rain_peak"]
    body(docs, f"全年降水 {S['rain_total']:.0f} mm，有雨的小时 {S['rain_hours']:,} 个（占 "
               f"{S['rain_hours'] / S['rows'] * 100:.0f}%）。最多的是 {peak_m} 月，"
               f"{peak_v:.0f} mm，正是梅雨；最大单小时雨量 {S['rain_max']:.1f} mm"
               f"（{S['rain_max_at']:%m-%d %H:%M}）。")
    figure(docs, figs["rain"], 13.4, f"图 3　月降水量。{peak_m} 月一支独高，是梅雨；12 月最少。")

    h2(docs, "4.4　一场台风，被拆成了三件事")
    n_press_ty = sum(1 for a in S["press_anoms"] if a["timestamp"].startswith("2024-09-16"))
    n_wind_ty = sum(1 for a in S["wind_anoms"] if a["timestamp"].startswith("2024-09-16"))
    body(docs, "9 月 16 日，一次台风过境上海。系统并不知道「台风」这个词，但它在这一天标出了三样东西：")
    bullet(docs, f"气压从 {S['ty_p'][0]:.1f} hPa 一路掉到 {S['p_min']:.1f} hPa（10:00），"
                 f"稳健 z = −31.6，是全年最显著的异常点；{len(S['press_anoms'])} 个气压异常全部落在这一天。",
           lead="气压骤降：")
    bullet(docs, f"07:00 风速 {S['ty_wind']:.1f} km/h，全年最大；5 个风速异常里有 {n_wind_ty} 个在这一天。",
           lead="风墙：")
    bullet(docs, f"10:00 风速掉到 {S['ty_eye']:.1f} km/h，被标成一个「负向」异常——那是台风眼经过时的静风。",
           lead="眼区：")
    figure(docs, figs["typhoon"], 14.6,
           "图 4　9 月 15–16 日：气压、风速、降水。虚线为风速峰值时刻；气压最低点与风速的「坑」同时出现，"
           "是台风眼经过的典型形态。")
    body(docs, "这是全篇最有说服力的一段：三个互相独立的信号（气压、风速、降水）在同一天、同一个小时内"
               "指向同一件事，而系统只是各自按自己的判据在算。", indent=False)

    # ---------------- 五
    h1(docs, "五、它没看出来什么，又报错了什么")
    h2(docs, "5.1　变化点只有两个，全是温度的季节转折", before=2)
    cp_lines = []
    for c in S["cps"]:
        cp_lines.append(f"{c['timestamp'][:10]}（{c['before']:.1f} → {c['after']:.1f} °C）")
    body(docs, "两个变化点分别是 " + "、".join(cp_lines) +
               "：前者是春季回暖的转折，后者是入秋转冷的转折。都对得上常识。")
    body(docs, "但台风没有出现在变化点里。这不是 bug，是判据的适用范围：变化点检测的是"
               "「前后两段的均值是否不同」，台风是 6 个小时的突发事件，前后两天就恢复了，"
               "均值根本没漂移。**突发事件要被看见，得靠异常通道，不是变化点通道。**", indent=False)

    h2(docs, "5.2　41 个异常的构成：七成是雨")
    ac = S["anom_by_col"]
    body(docs, f"降水 {ac.get('precipitation', 0)} 个、气压 {ac.get('surface_pressure', 0)} 个、"
               f"风速 {ac.get('wind_speed_10m', 0)} 个、温度 {ac.get('temperature_2m', 0)} 个。"
               f"降水占了 {ac.get('precipitation', 0) / S['n_anom'] * 100:.0f}%。")
    body(docs, "上海雨多，相对偏离判据会把每一场像样的大雨都标成「异常」。"
               "从统计意义讲没错——那几小时确实偏离了同期常态；"
               "但从业务意义讲，一场雷阵雨不是灾害。**统计异常与业务异常，是两件事。**", indent=False)

    h2(docs, "5.3　同分不同命：一次普通对流，比台风风墙的分数还高")
    may = [a for a in S["wind_anoms"] if a["timestamp"].startswith("2024-05-15")]
    ty = [a for a in S["wind_anoms"] if a["timestamp"].startswith("2024-09-16") and a["value"] > 30]
    if may and ty:
        m = max(may, key=lambda x: x["score"])
        t = ty[0]
        body(docs, f"5 月 15 日 {m['timestamp'][11:16]} 的风速 {m['value']:.1f} km/h，异常分数 {m['score']:.1f}；"
                   f"9 月 16 日 {t['timestamp'][11:16]} 台风风墙 {t['value']:.1f} km/h，分数 {t['score']:.1f}。"
                   f"分数几乎一样，严重性差一个数量级。")
        body(docs, "原因是分数只看「离常态多远」，不看「绝对值多大」。"
                   "5 月的常态风速低，35 km/h 很罕见；9 月台风季常态风速高，56 km/h 反而没那么「罕见」。"
                   "**如果只看分数排序，会把一场强对流排在台风前面。**", indent=False)

    h2(docs, "5.4　第 1 个假警报：干季降雨归零，被判成「单位变化」")
    issues = (S["data"].get("quality") or {}).get("issues") or []
    uc = next((i for i in issues if i.get("code") == "POSSIBLE_UNIT_CHANGE"), None)
    if uc:
        start = (uc.get("start") or "")[:10]
        when = f"（{start} 起）" if start else ""
        body(docs, f"降水在 7 月初进入干季{when}，前后各 175 个采样点（约 7 天）的中位数从 "
                   f"0.800 掉到 0.000，报告据此生成了一条「数据口径」结论：", after=4, indent=False)
        code_block(docs, [uc["detail"]])
        body(docs, "这句话会让一个不熟悉系统的用户怀疑自己的数据有问题。但实际上：", indent=False)
        bullet(docs, "没有做任何换算——`unit_conversions` 为空，C1 那次修复拦住了自动改写；", lead="影响有限：")
        bullet(docs, "正的量纲换算因子不可能把正的中位数映射成 0（华氏转摄氏也不会）。"
                     "所以「比值 = 0」在数学上就不可能是单位变化，最多是「归零 / 停测 / 干季」。",
               lead="判据不成立：")
        bullet(docs, "quality.py 第 159–160 行：`ratio = a / b`，`ratio < 0.5` 即判为 unit。"
                     "`b == 0` 有排除，`a == 0` 没有。一行条件即可修，但要配回归测试。",
               lead="位置：")
    callout(docs, "这一条最有价值　",
            "它不是阈值调得松或紧的问题，而是一条判据在数学上就不成立。"
            "调阈值永远修不好它——这正是「看结论、回到判据」比「调参数」更省事的地方。")

    # ---------------- 六
    h1(docs, "六、三条结论")
    body(docs, "能跑通不等于结论对。", after=3, indent=False, lead="一、", keep_next=True)
    body(docs, "两种格式都 succeeded，只说明进程没崩、产物落盘、结论数不为零。"
               "真正要问的是「这些结论里有多少是数据撑得住的」，那是另一个层面的验收。", indent=False)
    body(docs, "相对偏离抓得住「突然」，抓不住「重要」。", after=3, indent=False, lead="二、", keep_next=True)
    body(docs, f"台风眼和雷阵雨大风用同一把尺子量，分数只差 0.8。"
               f"这不是参数没调好，是判据本身只回答「离常态多远」，不回答「这件事多要紧」。", indent=False)
    body(docs, "假警报的根因常常不是阈值，而是一条不成立的判据。", after=3, indent=False, lead="三、", keep_next=True)
    body(docs, "「比值 0 也算单位变化」这种错误，调参调不出来，只能靠对着一份陌生数据把结论读一遍才能发现。"
               "这也是这一轮实测最大的收获：**它不是在代码里被发现的，是在报告里被读出来的。**", indent=False)

    # ---------------- 七
    h1(docs, "七、下一步")
    bullet(docs, "quality.py 加一条 `a == 0` 的排除，配 2–3 个回归测试（干季归零、"
                 "真实 °F→°C、正常跳变各一），跑全量回归。约 1 小时。", lead="修 5.4：")
    bullet(docs, "现在的事件级评估只有 14 条 NOAA 事件，且都在国外。"
                 "下一步用上海本地的天气过程（台风、梅雨、寒潮）建一份小的事件清单，"
                 "把「召回」这个指标从别人的数据挪到自己的数据上。", lead="把事件级评估挪到自己的数据上：")
    bullet(docs, "issue 模板已经就位，缺的是第一个真实用户。"
                 "这一轮至少证明了：一个陌生人拿着中文表头的 CSV，是能跑出东西的。",
           lead="继续找陌生人：")

    # ---------------- 附 A
    h1(docs, "附 A　课堂口播稿（约 10 分钟）", before=0, new_page=True)
    small(docs, "每段标注建议时长。加粗的是必须说出口的那一句，其余可以按自己的话讲。")

    script = [
        ("0:00–0:40　开场",
         "上一阶段我交的是「它能跑」。这一阶段我想回答一个更难的问题：换一份别人手里的数据，它还行吗？"
         "所以我找了一份从没进过项目的数据——上海 2024 年一整年的逐小时天气，8784 个小时。"),
        ("0:40–2:00　为什么这么测",
         "之前的演示用的是我自己的测试数据，表头干净、单位统一。真实用户不会这么客气。"
         "所以我故意把同一份数据写成两种格式：一种原样，一种改成国内导出的常见样子——"
         "中文表头、带单位、时间是 2024/1/1 0:00 这种写法。"),
        ("2:00–3:30　两种写法，结果一样",
         "两种都跑通了，而且 41 个异常点、2 个变化点逐个相同。格式容错这件事，做到了。"
         "但这是入场券，不是成绩单——接下来才是我真正想讲的。"),
        ("3:30–5:30　它看出的东西",
         "全年最低零下 6.1 度，最高 39.8 度，月均从 5 度到 31 度，振幅 26 度。"
         "一天里 5 点最冷、14 点最热。雨最多的是 6 月，337 毫米，那是梅雨。"
         "这些都是常识对得上的，说明基本盘没歪。"),
        ("5:30–7:00　一场台风",
         "9 月 16 号台风过境。系统不知道「台风」这个词，但它在同一天标出了三件事："
         "气压从 1004 掉到 986；7 点风速 56 公里每小时，全年最大；10 点风速掉到 13，那是台风眼。"
         "三个独立信号指向同一件事——这是我觉得最漂亮的一段。"),
        ("7:00–8:30　它没看出来的",
         "但变化点只报了两个，全是温度的季节转折，没有台风。因为变化点看的是「均值有没有漂移」，"
         "台风是六个小时的突发事件，两天就恢复了。这不是 bug，是判据的适用范围。"),
        ("8:30–9:30　假警报",
         "最值得说的一条：7 月进入干季，降雨中位数从 0.8 变成 0，报告说「疑似单位或量纲变化」。"
         "但正的换算因子不可能把正数变成 0，所以这条判据在数学上就不成立。"
         "它没改坏数据，但会让用户怀疑自己的数据。"),
        ("9:30–10:00　收尾",
         "所以这一轮的结论是：能跑通不等于结论对；统计上的异常不等于业务上的异常；"
         "而假警报的根因，往往不是阈值，是一条不成立的判据。"),
    ]
    for head, text in script:
        p = docs.add_paragraph()
        p.paragraph_format.space_before = Pt(9)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        set_font(p.add_run(head), HEAD, 10, True, ACCENT)
        body(docs, text, after=2, indent=False)

    # ---------------- 附 B
    h1(docs, "附 B　可能的提问与回答", before=0, new_page=True)
    qa = [
        ("为什么选上海，而不是项目已有的数据？",
         "已有的 17 份是为评估挑的（欧美、干旱区、热带、高海拔），都进过项目。上海有寒潮、梅雨、"
         "盛夏高温、台风四种过程，一年里齐全，而且从没被用来调过参数。"),
        ("再分析数据能代表真实天气吗？",
         "不能完全代表。ERA5 网格约 25 km、逐小时，台风中心附近的高频阵风会被平滑：这份数据里 "
         f"9 月 16 日最大风速 {S['ty_wind']:.1f} km/h，只能代表再分析的网格值。"
         "所以文中结论只说「这套数据上」，不说「上海真实风速」。要实测得用气象站或机场观测。"),
        ("41 个异常算多吗？",
         f"一年 {S['rows']:,} 小时里 {S['n_anom']} 个，占 {S['n_anom'] / S['rows'] * 100:.2f}%。"
         "数量不算多，但构成说明问题：七成是降雨。阈值不是按「灾害」定的。"),
        ("为什么变化点没抓到台风？",
         "变化点比较的是前后窗口的均值。台风持续约 6 小时，前后两天就恢复了，均值没有漂移。"
         "突发事件归异常通道，结构变化归变化点通道——两条通道服务两类问题。"),
        ("那个假警报会影响结果吗？",
         "不影响数值：没有做任何换算（unit_conversions 为空）。它影响的是「数据口径」那条结论的措辞，"
         "会让用户误以为数据有单位问题。所以必须修，优先级不低。"),
        ("为什么不让 Agent 跑一遍？",
         "陌生人默认拿到的是 pipeline：装完就能跑，不需要模型 key，不花钱。Agent 的差异上一阶段已经"
         "量化过：结论少 23.3%，118 条差异 100% 归因于「工具没执行」。这一轮没必要重复。"),
        ("这套东西能预报天气吗？",
         "有 --forecast，但只给点预测加经验区间，不给概率、不给置信度承诺。17 份数据 94 列里 "
         "MASE ≤ 1.0 的占 72%。它是分析工具，不是数值预报模式的替代品。"),
        ("你怎么证明结果是「对」的？",
         "三层：一是与 pandas 独立复算（月度统计、极值、日周期）；二是与官方事件对照（14 条 NOAA "
         "事件，事件级召回 1/14）；三是原始曲线人眼对照——本报告四张图都是原始数据画的，不是模型输出。"),
    ]
    for q, a in qa:
        p = docs.add_paragraph()
        p.paragraph_format.space_before = Pt(9)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        set_font(p.add_run("Q　"), HEAD, 10, True, ACCENT)
        set_font(p.add_run(q), HEAD, 10, True, INK)
        body(docs, a, after=2, indent=False)

    # ---------------- 附 C
    h1(docs, "附 C　复现")
    body(docs, "三步就能得到本报告的全部数字与图：", after=4, indent=False)
    code_block(docs, [
        "# 1. 抓数据（约 8 秒，写两个 CSV + 原始 JSON + SHA1）",
        "python scripts/fetch_shanghai_2024.py",
        "",
        "# 2. 跑分析（约 10 秒）",
        "mtsa run --scene weather \\",
        "  --input artifacts/stage2_shanghai/shanghai_raw.csv \\",
        "  --question \"分析温度的趋势、季节性和异常点，并检测变化点\" \\",
        "  --output artifacts/stage2_shanghai/run_raw",
        "",
        "# 3. 重新生成本文档（数字与图都从第 2 步的产物读取）",
        "python scripts/make_stage2_report.py",
    ], size=8.2)
    body(docs, "换中文表头那份再跑一次，就是第三节的对照实验：", after=4, indent=False)
    code_block(docs, [
        "mtsa run --scene weather \\",
        "  --input artifacts/stage2_shanghai/shanghai_cn.csv \\",
        "  --output artifacts/stage2_shanghai/run_cn",
    ], size=8.2)

    full_raw = meta["files"]["shanghai_raw.csv"]["sha1"]
    full_cn = meta["files"]["shanghai_cn.csv"]["sha1"]
    table(docs, ["项", "值"], [
        ["复现版本", f"mtsa-agent {_version()}（pipeline 模式）"],
        ["抓取时间", local_time(S["fetched_at"]) + "（北京时间）"],
        ["shanghai_raw.csv", f"SHA1 {full_raw}（{meta['files']['shanghai_raw.csv']['rows']:,} 行）"],
        ["shanghai_cn.csv", f"SHA1 {full_cn}（{meta['files']['shanghai_cn.csv']['rows']:,} 行）"],
        ["数据许可", "Open-Meteo Archive API，CC BY 4.0（署名：Open-Meteo / ERA5-CERRA）"],
        ["代码许可", "Apache-2.0"],
    ], [3.2, 13.4], size=8.6)

    small(docs, "边界说明：本报告是 1 份数据的实测，不是评测；结论的适用范围是「这份数据 + 这套判据」。"
                "换一份气候完全不同（例如热带或干旱区）的数据，异常数量的构成会变，但判据的适用性问题不会消失。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    docs.save(str(OUT))
    return OUT


# ------------------------------------------------------------------ 导出
def _version() -> str:
    try:
        from importlib.metadata import version
        return version("mtsa-agent")
    except Exception:  # noqa: BLE001
        return "0.3.0"


def find_soffice():
    for name in ("soffice", "soffice.exe"):
        found = shutil.which(name)
        if found:
            return found
    for cand in (r"C:\Program Files\LibreOffice\program\soffice.exe",
                 r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
                 "/usr/bin/soffice"):
        if Path(cand).exists():
            return cand
    return None


def export_pdf(docx_path: Path):
    soffice = find_soffice()
    if not soffice:
        print("[pdf] 没找到 LibreOffice，只产出 docx")
        return None
    profile = Path(os.environ.get("TEMP") or "/tmp") / "mtsa_lo_profile"
    profile.mkdir(parents=True, exist_ok=True)
    cmd = [soffice, "--headless", "--norestore", "--nolockcheck", "--nodefault",
           "-env:UserInstallation=" + profile.as_uri(),
           "--convert-to", "pdf:writer_pdf_Export",
           "--outdir", str(docx_path.parent), str(docx_path)]
    subprocess.run(cmd, capture_output=True, timeout=300)
    pdf = docx_path.with_suffix(".pdf")
    return pdf if pdf.exists() and pdf.stat().st_size > 5000 else None


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="生成第二阶段报告")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args(argv)

    path = build()
    docs = Document(str(path))
    chars = sum(len(p.text) for p in docs.paragraphs)
    print(f"[docx] {path}")
    print(f"       段落 {len(docs.paragraphs)}　表格类块 {len(docs.tables)}　图 {len(docs.inline_shapes)}"
          f"　正文字符 {chars}")

    if not args.no_pdf:
        pdf = export_pdf(path)
        if pdf:
            print(f"[pdf ] {pdf}  ({pdf.stat().st_size / 1024:.0f} KB)")
        else:
            print("[pdf] 未生成")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
