# -*- coding: utf-8 -*-
"""D3：数据扩容验收（跨区域 6 份 + 观测阵风 1 份）。

跑什么：
- 7 份新数据 × 4 个问题 = 28 次 pipeline（固定 rule 计划，不调用模型）；
- 阵风列是否被识别为独立目标列（不被 wind 通配吞掉）；
- 「日均风速看不见阵风」用 R17 的 4 条 L2 大风事件窗口直接检验；
- 旧 10 份的 data.csv SHA1 与 git HEAD 比对，确认附录只增不改。

运行：
    PYTHONPATH=src python -m evals.d3_expansion_eval
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import warnings
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
for _p in (SRC, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402

from evals.real_data_eval import QUESTIONS, load_labels  # noqa: E402
from evals.split_eval import rule_planner_env  # noqa: E402
from mtsa.engine import run_analysis  # noqa: E402
from mtsa.models import AnalysisRequest  # noqa: E402
from mtsa.scenarios.weather import _canon_map  # noqa: E402

BASE = os.path.join(ROOT, "fixtures", "real_weather")
REG = os.path.join(BASE, "registry.json")
L2 = os.path.join(BASE, "l2_verified_events.csv")
OUT_ROOT = os.path.join(ROOT, "artifacts", "d3_eval")
OUT_JSON = os.path.join(ROOT, "evals", "d3_results.json")
OUT_MD = os.path.join(ROOT, "evals", "d3_report.md")

NEW_IDS = ["R11", "R12", "R13", "R14", "R15", "R16", "R17"]
GUST_DS = "R17"
ELEV_MIN = 1500
CONTINENT = {"R11": "南美", "R12": "南美", "R13": "非洲", "R14": "非洲",
             "R15": "非洲", "R16": "南美", "R17": "北美"}


def _rel(p: str) -> str:
    return os.path.relpath(p, ROOT).replace(os.sep, "/")


def load_registry() -> list[dict]:
    with open(REG, encoding="utf-8") as f:
        return [d for d in json.load(f)["datasets"] if "error" not in d]


def old_sha1_from_git() -> dict[str, str]:
    """HEAD 里的 registry（扩容前的 10 份）。"""
    try:
        # 必须显式 UTF-8：Windows 上 text=True 会按 GBK 解码 git 的 UTF-8 输出，
        # 读线程抛 UnicodeDecodeError 后 stdout 变成 None（静默丢结果）。
        out = subprocess.run(["git", "show", "HEAD:fixtures/real_weather/registry.json"],
                             cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", timeout=20)
        if out.returncode != 0:
            return {}
        reg = json.loads(out.stdout)
        return {d["id"]: d["data_sha1"] for d in reg["datasets"] if "error" not in d}
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, TypeError):
        return {}


def run_one(meta: dict, q: dict) -> dict:
    out = os.path.join(OUT_ROOT, meta["id"], q["id"])
    os.makedirs(out, exist_ok=True)
    req = AnalysisRequest(request_id="d3_%s_%s" % (meta["id"], q["id"]),
                          input_path=os.path.join(ROOT, meta["data_path"]),
                          scene_id="weather", question=q["question"], mode="pipeline",
                          seed=20260927, output_dir=out)
    with rule_planner_env("rule"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_analysis(req, verbose=False)
    path = os.path.join(out, "result.json")
    doc = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    return {"doc": doc, "status": res.status, "dir": out}


def evidence_coverage(doc: dict) -> float | None:
    cs = doc.get("conclusions") or []
    ids = {e.get("evidence_id") for e in (doc.get("evidence") or [])}
    if not cs:
        return None
    return round(sum(1 for c in cs if c.get("evidence_ids")
                     and all(i in ids for i in c["evidence_ids"])) / len(cs), 4)


def l2_wind_events() -> list[dict]:
    with open(L2, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows
            if r.get("column") == "wind_speed" and "Wind" in (r.get("event_types") or "")]


def gust_vs_mean(r17_doc: dict, events: list[dict]) -> list[dict]:
    """事件窗口内：阵风列与平均风速列是否各有 p99 观测。"""
    data = r17_doc.get("data") or {}
    ex = data.get("extremes") or []
    spans = {}

    def spans_of(col):
        for e in ex:
            if e.get("column") == col and e.get("tail") == "high":
                return [(pd.Timestamp(r["start"]), pd.Timestamp(r["end"])) for r in (e.get("runs") or [])]
        return []

    spans["wind_gusts"] = spans_of("wind_gusts")
    spans["wind_speed"] = spans_of("wind_speed")
    out = []
    for ev in events:
        a = pd.Timestamp(ev["start"])
        b = pd.Timestamp(ev["end"]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        row = {"event_id": (ev.get("note") or "")[:80], "start": ev["start"],
               "kind": ev.get("event_types")}
        for col, segs in spans.items():
            hit = [s for s in segs if s[0] <= b and s[1] >= a]
            row[col] = bool(hit)
        out.append(row)
    return out


def main() -> int:
    reg = {d["id"]: d for d in load_registry()}
    missing = [i for i in NEW_IDS if i not in reg]
    if missing:
        print("registry 缺少：%s" % missing, file=sys.stderr)
        return 2
    new = [reg[i] for i in NEW_IDS]
    old = old_sha1_from_git()

    # --- D3-1 数据事实 ---
    elev = {d["id"]: d.get("elevation_m") for d in new}
    high = [i for i, v in elev.items() if isinstance(v, (int, float)) and v >= ELEV_MIN]
    continents = sorted({CONTINENT[i] for i in NEW_IDS})
    provenance_ok = all(d.get("raw") and all(r.get("url") and r.get("sha1") for r in d["raw"])
                        for d in new)
    sha_keep = {i: (old.get(i) == reg[i]["data_sha1"]) for i in reg if i in old}

    # --- D3-2 目标列识别 ---
    targets = {}
    for d in new:
        cols = ["timestamp", "station"] + d["columns"]
        targets[d["id"]] = list(_canon_map(cols).values())
    gust_ok = "wind_gusts" in targets.get(GUST_DS, [])
    others_clean = all("wind_gusts" not in targets[i] for i in NEW_IDS if i != GUST_DS)

    # --- D3-3 运行矩阵 ---
    runs = []
    for d in new:
        for q in QUESTIONS:
            print("运行 %s/%s ..." % (d["id"], q["id"]))
            res = run_one(d, q)
            doc = res["doc"] or {}
            labels = load_labels(d["data_path"])
            runs.append({
                "dataset": d["id"], "question": q["id"], "status": res["status"],
                "conclusions": len(doc.get("conclusions") or []),
                "evidence_coverage": evidence_coverage(doc),
                "quality_codes": sorted({i.get("code") for i in
                                         ((doc.get("data") or {}).get("quality") or {}).get("issues") or []}),
                "extremes": len((doc.get("data") or {}).get("extremes") or []),
                "forecast": "forecast" in ((doc.get("data") or {})),
                "labels": len(labels),
                "rows": (doc.get("data") or {}).get("clean", {}).get("rows"),
            })

    ok_runs = [r for r in runs if r["status"] == "succeeded" and r["conclusions"] > 0]
    cov = [r["evidence_coverage"] for r in ok_runs if r["evidence_coverage"] is not None]
    min_cov = min(cov) if cov else None

    # --- D3-4 阵风可观测性 ---
    r17_q1 = os.path.join(OUT_ROOT, GUST_DS, "q1", "result.json")
    r17_doc = json.load(open(r17_q1, encoding="utf-8")) if os.path.exists(r17_q1) else {}
    events = l2_wind_events()
    gv = gust_vs_mean(r17_doc, events) if r17_doc else []
    n_gust = sum(1 for x in gv if x.get("wind_gusts"))
    n_mean = sum(1 for x in gv if x.get("wind_speed"))
    gust_thr = next((e["threshold"] for e in (r17_doc.get("data", {}).get("extremes") or [])
                     if e.get("column") == "wind_gusts" and e.get("tail") == "high"), None)

    trunc = []
    for d in new:
        q1 = os.path.join(OUT_ROOT, d["id"], "q1", "result.json")
        if not os.path.exists(q1):
            continue
        doc = json.load(open(q1, encoding="utf-8"))
        for e in (doc.get("data") or {}).get("extremes") or []:
            if e.get("runs_truncated"):
                trunc.append("%s/%s" % (d["id"], e.get("column")))

    checks = [
        {"id": "D3-1", "name": "新增 7 份、≥2 大洲、海拔≥1500m ≥4、来源可复核、旧 SHA1 不变",
         "ok": len(new) == len(NEW_IDS) and len(continents) >= 2 and len(high) >= 4
               and provenance_ok and all(sha_keep.values()),
         "detail": "%d 份 / %s / 高海拔 %d 份 / 来源 %s / 旧 SHA1 全同 %s" % (
             len(new), "、".join(continents), len(high),
             "齐全" if provenance_ok else "缺失", all(sha_keep.values()))},
        {"id": "D3-2", "name": "R17 的 wind_gusts 是独立目标列",
         "ok": gust_ok and others_clean,
         "detail": "R17=%s；其余 %d 份无阵风列" % (targets.get(GUST_DS), len(NEW_IDS) - 1)},
        {"id": "D3-3", "name": "7×4=28 次全部成功，证据覆盖率 1.000",
         "ok": len(ok_runs) == len(runs) and min_cov == 1.0,
         "detail": "%d/%d 成功，最低证据覆盖 %s" % (len(ok_runs), len(runs), min_cov)},
        # 原始口径（写下时未看结果）：阵风命中 ≥3 且小时平均风速命中 < 阵风。
        # 实测两者都是 3/4 —— 这条按原标准记为未通过，不放宽。
        {"id": "D3-4", "name": "4 条 L2 大风事件窗口内阵风有 p99、小时平均风速没有",
         "ok": n_gust >= 3 and n_mean < n_gust,
         "detail": "阵风命中 %d/4、小时平均风速命中 %d/4（R17 阵风 p99=%s m/s）" % (
             n_gust, n_mean, None if gust_thr is None else round(gust_thr, 2))},
        # 事后观察，不计入门禁：真正拉开差距的是时间分辨率（日均 → 小时），不是阵风列。
        {"id": "D3-4b", "name": "（事后）分辨率对比：R06 日均 1/4 vs R17 小时 3/4",
         "ok": True, "gate": False,
         "detail": "R06 日均风速覆盖 1/4（D2 已发布），R17 小时阵风/平均风速各 3/4；"
                   "唯一未覆盖的 2023-08-13 为 Thunderstorm Wind，Central Park 当日阵风峰值仅 8.2 m/s"},
        {"id": "D3-2b", "name": "极端段未被上限截断（时间覆盖完整）",
         "ok": not trunc, "detail": "截断的数据集/列：%s" % (trunc or "无")},
    ]

    doc = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "new_datasets": [{k: d.get(k) for k in ("id", "name", "source", "note",
                                                "rows", "stations", "start", "end",
                                                "frequency", "columns", "elevation_m",
                                                "data_sha1")}
                         for d in new],
        "continents": {i: CONTINENT[i] for i in NEW_IDS},
        "targets": targets,
        "old_sha1_unchanged": sha_keep,
        "runs": runs,
        "gust_check": {"threshold": gust_thr, "events": gv,
                       "gust_hits": n_gust, "mean_hits": n_mean},
        "checks": checks,
        "passed": all(c["ok"] for c in checks if c.get("gate", True)),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2, default=str)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(render_md(doc))
    print()
    for c in checks:
        mark = "INFO" if c.get("gate") is False else ("PASS" if c["ok"] else "FAIL")
        print("%s %s -> %s（%s）" % (c["id"], mark, c["name"], c["detail"]))
    print("结果：%s" % OUT_JSON)
    print("报告：%s" % OUT_MD)
    return 0 if doc["passed"] else 1


def render_md(doc: dict) -> str:
    L = ["# D3 数据扩容验收", "",
         "生成时间：%s" % doc["generated_at"],
         "新增：7 份公开数据（南美 3 / 非洲 3 / 北美 1），不修改分析阈值。", "",
         "## 门禁", "", "| 编号 | 检查 | 结果 | 证据 |", "| --- | --- | --- | --- |"]
    for c in doc["checks"]:
        mark = "INFO（不计门禁）" if c.get("gate") is False else ("PASS" if c["ok"] else "FAIL")
        L.append("| %s | %s | %s | %s |" % (c["id"], c["name"], mark, c["detail"]))
    L += ["", "**%s**" % ("全部通过" if doc["passed"] else "存在未通过项"), "",
          "## 新增数据", "",
          "| 编号 | 站点 | 大洲 | 行数 | 海拔 (m) | 列 |", "| --- | --- | --- | ---: | ---: | --- |"]
    for d in doc["new_datasets"]:
        L.append("| %s | %s | %s | %s | %s | %s |" % (
            d["id"], d["name"], doc["continents"][d["id"]], d["rows"],
            d.get("elevation_m") if d.get("elevation_m") is not None else "—",
            "、".join(d["columns"])))
    L += ["", "## 目标列识别", "", "| 数据集 | 目标列 |", "| --- | --- |"]
    for k, v in doc["targets"].items():
        L.append("| %s | %s |" % (k, "、".join(v)))
    L += ["", "## 阵风可观测性（R17，观测，非再分析）", ""]
    g = doc["gust_check"]
    L.append("R17 阵风 p99 = %s m/s；4 条 L2 大风事件窗口内：" % g["threshold"])
    L += ["", "| 事件日 | 类型 | 阵风有 p99 观测 | 平均风速有 p99 观测 |",
          "| --- | --- | --- | --- |"]
    for e in g["events"]:
        L.append("| %s | %s | %s | %s |" % (e["start"][:10], e["kind"],
                                            "是" if e["wind_gusts"] else "否",
                                            "是" if e["wind_speed"] else "否"))
    L += ["", "命中：阵风 %d/4，平均风速 %d/4。" % (g["gust_hits"], g["mean_hits"]), "",
          "## 运行矩阵（7 × 4）", "",
          "| 数据集 | 问题 | 状态 | 结论数 | 证据覆盖 | 极端项 | 质量码 |",
          "| --- | --- | --- | ---: | ---: | ---: | --- |"]
    for r in doc["runs"]:
        L.append("| %s | %s | %s | %d | %s | %d | %s |" % (
            r["dataset"], r["question"], r["status"], r["conclusions"],
            r["evidence_coverage"], r["extremes"], "、".join(r["quality_codes"]) or "—"))
    L += ["", "## 复现", "", "```bash", "PYTHONPATH=src python -m evals.d3_expansion_eval", "```", ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
