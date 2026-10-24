#!/usr/bin/env python
"""把对外文章 Markdown 导出为 Word（.docx）与 PDF。

用法::

    python scripts/make_article_doc.py                     # 导出默认文章
    python scripts/make_article_doc.py --md doc/articles/xxx.md
    python scripts/make_article_doc.py --no-pdf            # 只出 Word

支持：一级/二级/三级标题、正文（**粗体**、`代码`）、无序与有序列表、
引用块、分隔线、代码块、Markdown 表格（自动列宽、表头灰底）。

PDF 由本机 LibreOffice（soffice）在后台转换；未安装时只产出 .docx 并明确提示，
不会静默假装成功。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MD = ROOT / "doc" / "articles" / "可审计性是一种产品能力.md"
BYLINE = "qiaotong555"

BODY = "等线"
HEAD = "微软雅黑"
MONO = "Consolas"
INK = RGBColor(0x1A, 0x1A, 0x1A)
GRAY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0x8C, 0x8C, 0x8C)
ACCENT = RGBColor(0xB0, 0x3A, 0x2E)
LINE = "D9D9D9"
HEAD_FILL = "F2F2F2"
SOFT_FILL = "F7F8FA"
TOTAL_W = 16.6

# 导出后逐个核对的关键数字：缺一个就认为文章被截断。
REQUIRED_TOKENS = (
    "24/24",
    "0.0714",
    "0.9667",
    "53,924",
    "17,568",
    "5,977",
    "tool_not_run",
)


# ------------------------------------------------------------------ 文本与字体
INLINE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`)")


def tokenize(text):
    """把一段行内文本切成 (样式, 文本)。"""
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            yield "bold", part[2:-2]
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            yield "code", part[1:-1]
        else:
            yield "text", part


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


def rich(par, text, size, color=INK, *, font=BODY, bold=False):
    """按行内标记写入 runs：**粗体**、`等宽`。"""
    for kind, chunk in tokenize(text):
        if kind == "code":
            set_font(par.add_run(chunk), MONO, size - 1.0, False, ACCENT)
        else:
            set_font(par.add_run(chunk), font, size, bold or kind == "bold", color)

# ------------------------------------------------------------------ 版式原子
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


def no_split(table):
    for row in table.rows:
        tr_pr = row._tr.get_or_add_trPr()
        tr_pr.append(OxmlElement("w:cantSplit"))


def no_snap(ppr):
    """关掉「对齐到网格」，否则标题在固定行网格里会被拉出字距。"""
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
    """给段落加一条下边框（标题分隔线 / 水平线）。"""
    ppr = par._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), size)
    bottom.set(qn("w:space"), space)
    bottom.set(qn("w:color"), color)
    pbdr.append(bottom)
    ppr.insert_element_before(pbdr, *PBR_REST)


def rule(doc, *, before=10, after=10):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    underline(p, size="6", color="C9C9C9", space="1")


def spacer(doc, after=2):
    doc.add_paragraph().paragraph_format.space_after = Pt(after)

# ------------------------------------------------------------------ 块级渲染
def title_block(doc, title, subtitle):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(24)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    set_font(p.add_run(title), HEAD, 24, True, INK)

    if subtitle:
        s = doc.add_paragraph()
        s.paragraph_format.space_before = Pt(4)
        s.paragraph_format.space_after = Pt(10)
        s.paragraph_format.line_spacing = 1.4
        set_font(s.add_run(subtitle), HEAD, 12.5, False, GRAY)

    m = doc.add_paragraph()
    m.paragraph_format.space_before = Pt(0)
    m.paragraph_format.space_after = Pt(4)
    set_font(m.add_run(BYLINE), BODY, 9, False, LIGHT)
    set_font(m.add_run("　·　mtsa-agent 项目札记"), BODY, 9, False, LIGHT)
    rule(doc, before=2, after=16)


def heading(doc, text, level):
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.line_spacing = 1.2
    if level == 1:
        p.paragraph_format.space_before = Pt(18)
        p.paragraph_format.space_after = Pt(7)
        set_font(p.add_run(text), HEAD, 15, True, INK)
        underline(p, size="8", color="C9C9C9", space="5")
    else:
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(3)
        set_font(p.add_run(text), HEAD, 11.5, True, INK)


def para(doc, text, *, size=10.5, after=8, indent_first=True, color=INK):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.6
    if indent_first:
        p.paragraph_format.first_line_indent = Cm(0.74)
    rich(p, text, size, color)


def list_item(doc, text, marker, *, size=10.5):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.55
    p.paragraph_format.left_indent = Cm(0.95)
    p.paragraph_format.first_line_indent = Cm(-0.95)
    set_font(p.add_run(marker + "　"), BODY, size, False, GRAY)
    rich(p, text, size, INK)


def cont(doc, text, *, size=10.5):
    """列表项内部的续段：与条目文字左对齐。"""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.55
    p.paragraph_format.left_indent = Cm(0.95)
    rich(p, text, size, INK)


def quote(doc, text):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table_borders(table, left_accent=True)
    cell_margins(table, top=150, bottom=150, left=230, right=230)
    cell = table.rows[0].cells[0]
    shade(cell, SOFT_FILL)
    table.columns[0].width = Cm(TOTAL_W)
    cell.width = Cm(TOTAL_W)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.55
    rich(p, text, 10, GRAY)
    spacer(doc)


def code_block(doc, lines):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cell_margins(table, top=110, bottom=110, left=180, right=180)
    no_split(table)
    cell = table.rows[0].cells[0]
    shade(cell, SOFT_FILL)
    table.columns[0].width = Cm(TOTAL_W)
    cell.width = Cm(TOTAL_W)
    for i, line in enumerate(lines):
        p = cell.paragraphs[0] if i == 0 else cell.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.3
        set_font(p.add_run(line), MONO, 8.5, False, INK)
    spacer(doc)

def disp(text):
    """显示宽度：CJK 与全角标点按 2 计。"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def col_widths(headers, rows):
    """按各列最长内容分配宽度，最少 2.4cm，整体缩放回正文宽度。"""
    weights = []
    for i in range(len(headers)):
        cells = [headers[i]] + [r[i] for r in rows]
        longest = max(disp(c) for c in cells)
        weights.append(max(10, min(longest, 70)))
    total = sum(weights)
    ratios = [w / total for w in weights]
    # 单列不窄于 30%、不宽于 70%，免得「出处」列挤掉「数字」列
    ratios = [min(0.7, max(0.3, r)) for r in ratios]
    norm = sum(ratios)
    return [TOTAL_W * r / norm for r in ratios]


def md_table(doc, headers, rows, *, size=9.5):
    widths = col_widths(headers, rows)
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table_borders(table)
    cell_margins(table, top=100, bottom=100)

    for i, head in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        shade(cell, HEAD_FILL)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(1.5)
        p.paragraph_format.space_after = Pt(1.5)
        p.paragraph_format.line_spacing = 1.3
        rich(p, head, size, INK, bold=True)

    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cell = cells[i]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1.5)
            p.paragraph_format.space_after = Pt(1.5)
            p.paragraph_format.line_spacing = 1.3
            rich(p, value, size, INK)

    for i, width in enumerate(widths):
        table.columns[i].width = Cm(width)
        for row in table.rows:
            row.cells[i].width = Cm(width)
    no_split(table)
    # 除最后一行外都与下一行同页，避免表格被切成两页
    for row in table.rows[:-1]:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.keep_with_next = True
    spacer(doc)


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


# ------------------------------------------------------------------ Markdown 解析
SEP_CELL = re.compile(r"^:?-{2,}:?$")
ORDERED = re.compile(r"^(\d+)\.\s+(.*)$")
BULLET = ("- ", "* ")


def is_block_start(line):
    return (
        not line
        or line.startswith(("#", "|", ">", "```", "- ", "* "))
        or bool(ORDERED.match(line))
        or line in ("---", "***", "___")
    )


def take_continuation(lines, i):
    """收集列表项后以空格缩进的续段。"""
    out = []
    while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() \
            and not is_block_start(lines[i].strip()):
        out.append(("li_cont", lines[i].strip()))
        i += 1
    return out, i


def parse_blocks(md):
    lines = md.replace("\r\n", "\n").split("\n")
    blocks = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            body = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                body.append(lines[i].rstrip())
                i += 1
            i += 1
            blocks.append(("code", body))
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            blocks.append(("h", (len(m.group(1)), m.group(2).strip())))
            i += 1
            continue

        if stripped in ("---", "***", "___"):
            blocks.append(("hr", None))
            i += 1
            continue

        if stripped.startswith("|"):
            raw = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                raw.append(lines[i].strip())
                i += 1
            grid = [[c.strip() for c in r.strip("|").split("|")] for r in raw]
            grid = [r for r in grid if not all(SEP_CELL.match(c) for c in r)]
            if grid:
                blocks.append(("table", (grid[0], grid[1:])))
            continue

        if stripped.startswith(">"):
            body = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                body.append(lines[i].strip().lstrip(">").strip())
                i += 1
            blocks.append(("quote", " ".join(b for b in body if b)))
            continue

        m = ORDERED.match(stripped)
        if m:
            blocks.append(("ol", (m.group(1) + ".", m.group(2).strip())))
            extra, i = take_continuation(lines, i + 1)
            blocks.extend(extra)
            continue

        if stripped.startswith(BULLET):
            blocks.append(("ul", ("•", stripped[2:].strip())))
            extra, i = take_continuation(lines, i + 1)
            blocks.extend(extra)
            continue

        body = [stripped]
        i += 1
        while i < len(lines) and not is_block_start(lines[i].strip()):
            body.append(lines[i].strip())
            i += 1
        blocks.append(("p", " ".join(body)))
    return blocks

def build(md_path, out_path):
    blocks = parse_blocks(md_path.read_text(encoding="utf-8"))

    doc = Document()
    setup(doc)

    start = 0
    title = md_path.stem
    subtitle = None
    if blocks and blocks[0][0] == "h" and blocks[0][1][0] == 1:
        title = blocks[0][1][1]
        start = 1
        if len(blocks) > 1 and blocks[1][0] == "h" and blocks[1][1][0] == 2:
            subtitle = blocks[1][1][1]
            start = 2
    title_block(doc, title, subtitle)

    # 默认模板的 core properties 是 python-docx / 2013 年，交付前必须改掉
    stamp = datetime.now(timezone.utc).replace(microsecond=0)
    props = doc.core_properties
    props.author = BYLINE
    props.last_modified_by = BYLINE
    props.title = title
    props.subject = subtitle or ""
    props.keywords = "时序分析; Agent; 可审计性"
    props.comments = ""
    props.category = ""
    props.created = stamp
    props.modified = stamp

    stats = {"table": 0, "code": 0, "h1": 0, "h2": 0, "p": 0, "li": 0, "hr": 0}
    for kind, payload in blocks[start:]:
        if kind == "h":
            level, text = payload
            if level <= 2:
                heading(doc, text, 1)
                stats["h1"] += 1
            else:
                heading(doc, text, 2)
                stats["h2"] += 1
        elif kind == "p":
            para(doc, payload)
            stats["p"] += 1
        elif kind in ("ol", "ul"):
            list_item(doc, payload[1], payload[0])
            stats["li"] += 1
        elif kind == "li_cont":
            cont(doc, payload)
            stats["p"] += 1
        elif kind == "quote":
            quote(doc, payload)
        elif kind == "code":
            code_block(doc, payload)
            stats["code"] += 1
        elif kind == "table":
            md_table(doc, payload[0], payload[1])
            stats["table"] += 1
        elif kind == "hr":
            rule(doc, before=8, after=12)
            stats["hr"] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    stats["title"] = title
    stats["blocks"] = len(blocks)
    return stats


# ------------------------------------------------------------------ PDF 与自检
def find_soffice():
    for name in ("soffice", "soffice.exe"):
        found = shutil.which(name)
        if found:
            return found
    for cand in (
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        "/usr/bin/soffice",
        "/usr/lib/libreoffice/program/soffice",
    ):
        if Path(cand).exists():
            return cand
    return None


def export_pdf(docx_path):
    soffice = find_soffice()
    if soffice is None:
        print("[pdf] 未找到 LibreOffice（soffice），跳过 PDF。")
        return None

    profile = Path(os.environ.get("TEMP") or "/tmp") / "mtsa_lo_profile"
    profile.mkdir(parents=True, exist_ok=True)
    cmd = [
        soffice, "--headless", "--norestore", "--nolockcheck", "--nodefault",
        "-env:UserInstallation=" + profile.as_uri(),
        "--convert-to", "pdf:writer_pdf_Export",
        "--outdir", str(docx_path.parent), str(docx_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    pdf = docx_path.with_suffix(".pdf")
    if not pdf.exists() or pdf.stat().st_size < 5000:
        raw = (proc.stdout or b"") + (proc.stderr or b"")
        try:
            detail = raw.decode("utf-8")
        except UnicodeDecodeError:
            detail = raw.decode("gbk", "replace")
        print("[pdf] 转换失败：" + detail.strip())
        return None
    return pdf


def plain(text):
    return re.sub(r"\s+", "", text.replace("**", "").replace("`", ""))


def verify(docx_path, md_path):
    doc = Document(str(docx_path))
    chunks = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                chunks.append(cell.text)
    text = "\n".join(chunks)
    flat = plain(text)

    problems = []
    for token in REQUIRED_TOKENS:
        if token not in text:
            problems.append("缺少关键数字：" + token)

    source = md_path.read_text(encoding="utf-8")
    missing = []
    for kind, payload in parse_blocks(source):
        if kind in ("p", "quote", "li_cont"):
            target = plain(payload)
        elif kind in ("ol", "ul"):
            target = plain(payload[1])
        elif kind == "h":
            target = plain(payload[1])
        else:
            continue
        if target and target not in flat:
            missing.append(target[:40])
    if missing:
        problems.append("有 %d 个段落未进入文档，例如：%s" % (len(missing), missing[:3]))
    return not problems, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description="把文章 Markdown 导出为 Word/PDF")
    ap.add_argument("--md", default=str(DEFAULT_MD), help="Markdown 源文件")
    ap.add_argument("--out", default=None, help="输出 .docx 路径（默认与 md 同名）")
    ap.add_argument("--no-pdf", action="store_true", help="只生成 Word")
    args = ap.parse_args(argv)

    md_path = Path(args.md).resolve()
    if not md_path.exists():
        print("找不到源文件：%s" % md_path)
        return 2
    docx_path = Path(args.out).resolve() if args.out else md_path.with_suffix(".docx")

    stats = build(md_path, docx_path)
    print("[docx] %s" % docx_path)
    print("       标题=%s  一级=%d  二级=%d  段落=%d  列表=%d  表格=%d  代码块=%d  分隔线=%d"
          % (stats["title"], stats["h1"], stats["h2"], stats["p"],
             stats["li"], stats["table"], stats["code"], stats["hr"]))

    ok, problems = verify(docx_path, md_path)
    for item in problems:
        print("[warn] %s" % item)
    print("[docx] 自检：" + ("通过" if ok else "%d 项待查" % len(problems)))

    if args.no_pdf:
        return 0 if ok else 1

    pdf = export_pdf(docx_path)
    if pdf is None:
        print("[pdf] 未生成（见上方原因），Word 版可用。")
        return 0 if ok else 1
    print("[pdf ] %s  (%d KB)" % (pdf, pdf.stat().st_size / 1024))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
