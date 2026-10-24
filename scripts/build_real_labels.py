# -*- coding: utf-8 -*-
"""按 B2 协议从真实天气数据生成 L1 客观质量事件标签。

只读 data.csv，不运行分析引擎、不调用模型。标签是数据自身属性的确定性函数，
重跑得到同一份结果，便于复核。

规则（2026-10-24 定稿，先于 80 次正式运行）：
- missing_gap：同一 (station, column) 连续缺失 >= 3 个采样点；
- sampling_gap：同一 station 时间轴缺口 expected_missing >= 3；
- duplicate_timestamp：同一 (station, timestamp) 出现 > 1 次；
- out_of_order：同一 station 内 timestamp 下降；
- constant_run：同一 (station, column) 连续 >= 24 个相同非空值，但排除物理上
  可解释的饱和（湿度 >= 99.9、降雨 = 0）。无法与真故障区分的恒定段不贴标签，
  避免把「连续大雾」当成「传感器卡死」；
- sentinel：数值等于 -999 / -9999（转换阶段应已清除，此处用于核验）；
- unit_change：只在有外部证据时人工写入，本脚本不自动生成；
- verified_anomaly：L2，来自 l2_verified_events.csv，本脚本只搬运、不编造。

L3（无标签的异常候选）不写入 labels.csv，评估时只报数量，不计 FP/FN。

运行：
    python scripts/build_real_labels.py
    python scripts/build_real_labels.py --only R06,R08
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "fixtures" / "real_weather"

MISSING_RUN_MIN = 3          # 连续缺失采样点数
SAMPLING_GAP_MIN = 3         # 时间轴缺口折算的缺失采样点数
CONSTANT_RUN_MIN = 24        # 恒定段最小长度
SENTINELS = (-999.0, -9999.0)
L2_FILE = BASE / "l2_verified_events.csv"

FIELDS = ["kind", "column", "station", "start", "end", "count", "source", "note"]

RULE_TEXT = {
    "missing_gap": "同一 (station, column) 连续缺失 >= %d 个采样点" % MISSING_RUN_MIN,
    "sampling_gap": "同一 station 时间轴缺口 expected_missing >= %d" % SAMPLING_GAP_MIN,
    "duplicate_timestamp": "同一 (station, timestamp) 出现 > 1 次",
    "out_of_order": "同一 station 内 timestamp 下降",
    "constant_run": ("连续 >= %d 个相同非空值，排除湿度 >= 99.9 与降雨 = 0 的物理饱和"
                     % CONSTANT_RUN_MIN),
    "sentinel": "残留缺测哨兵值 -999 / -9999",
    "unit_change": "仅人工写入，脚本不生成",
    "verified_anomaly": "L2 外部清单，脚本只搬运",
}


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iso(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _missing_runs(series: pd.Series) -> list[tuple[int, int]]:
    na = series.isna().to_numpy()
    runs, i = [], 0
    while i < len(na):
        if na[i]:
            j = i
            while j < len(na) and na[j]:
                j += 1
            if j - i >= MISSING_RUN_MIN:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _constant_runs(series: pd.Series, column: str) -> list[tuple[int, int, float]]:
    v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    runs, i = [], 0
    while i < len(v):
        if np.isnan(v[i]):
            i += 1
            continue
        j = i
        while j < len(v) and not np.isnan(v[j]) and v[j] == v[i]:
            j += 1
        if j - i >= CONSTANT_RUN_MIN and not _physically_plausible(column, v[i]):
            runs.append((i, j, float(v[i])))
        i = j
    return runs


def _physically_plausible(column: str, value: float) -> bool:
    """恒定段是否可能是真实天气而不是传感器卡死。"""
    if column == "humidity" and value >= 99.9:
        return True          # 连续饱和（雾、降水）
    if column == "rainfall" and value == 0.0:
        return True          # 干旱期无降水
    if column == "wind_speed" and value == 0.0:
        return True          # 静风
    return False


def build_dataset(meta: dict) -> tuple[list[dict], dict]:
    path = ROOT / meta["data_path"]
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["station", "timestamp"])
    value_cols = [c for c in df.columns if c not in ("timestamp", "station")]
    labels: list[dict] = []
    detail: dict = {"rows": int(len(df)), "columns": value_cols,
                    "stations": sorted(str(x) for x in df["station"].unique()),
                    "counts": {}}

    for station, g in df.groupby("station", sort=True):
        g = g.reset_index(drop=True)
        ts = g["timestamp"]
        step = ts.diff().dt.total_seconds().dropna()
        step_s = float(step.median()) if len(step) else 0.0

        # 时间轴：重复 / 乱序 / 缺口
        dup_mask = ts.duplicated(keep=False)
        if dup_mask.any():
            dup = ts[dup_mask]
            labels.append({
                "kind": "duplicate_timestamp", "column": "timestamp", "station": station,
                "start": _iso(dup.min()), "end": _iso(dup.max()),
                "count": int(dup_mask.sum()),
                "source": meta["data_path"],
                "note": "同一站点内重复时间戳 %d 行" % int(dup_mask.sum())})

        drops = int((ts.diff().dt.total_seconds() < 0).sum())
        if drops:
            labels.append({
                "kind": "out_of_order", "column": "timestamp", "station": station,
                "start": _iso(ts.min()), "end": _iso(ts.max()), "count": drops,
                "source": meta["data_path"], "note": "时间戳下降 %d 次" % drops})

        if step_s > 0:
            dif = ts.diff().dt.total_seconds()
            for idx in np.where(dif.to_numpy() > step_s * 1.5)[0]:
                expected = int(round(float(dif.iloc[idx]) / step_s)) - 1
                if expected >= SAMPLING_GAP_MIN:
                    labels.append({
                        "kind": "sampling_gap", "column": "timestamp", "station": station,
                        "start": _iso(ts.iloc[idx - 1]), "end": _iso(ts.iloc[idx]),
                        "count": expected,
                        "source": meta["data_path"],
                        "note": "预计缺失 %d 个采样点（间隔 %.0f 秒）" % (expected, step_s)})

        for col in value_cols:
            s = pd.to_numeric(g[col], errors="coerce")
            for a, b in _missing_runs(s):
                labels.append({
                    "kind": "missing_gap", "column": col, "station": station,
                    "start": _iso(ts.iloc[a]), "end": _iso(ts.iloc[b - 1]),
                    "count": b - a, "source": meta["data_path"],
                    "note": "连续缺失 %d 个采样点" % (b - a)})
            for a, b, v in _constant_runs(s, col):
                labels.append({
                    "kind": "constant_run", "column": col, "station": station,
                    "start": _iso(ts.iloc[a]), "end": _iso(ts.iloc[b - 1]),
                    "count": b - a, "source": meta["data_path"],
                    "note": "连续 %d 点恒为 %g" % (b - a, v)})
            vals = s.dropna()
            sent = vals[vals.isin(SENTINELS)]
            if len(sent):
                labels.append({
                    "kind": "sentinel", "column": col, "station": station,
                    "start": _iso(ts.iloc[int(sent.index[0])]),
                    "end": _iso(ts.iloc[int(sent.index[-1])]), "count": int(len(sent)),
                    "source": meta["data_path"], "note": "残留缺测哨兵值"})

    detail["counts"] = dict(Counter(x["kind"] for x in labels))
    return labels, detail


def load_l2(only_ids: set[str]) -> list[dict]:
    if not L2_FILE.exists():
        return []
    with L2_FILE.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if not (r.get("dataset") or "").strip():
            continue
        if only_ids and r["dataset"].strip().upper() not in only_ids:
            continue
        if not (r.get("source") or "").strip():
            raise ValueError("L2 事件必须有外部来源：%s" % r)
        out.append({
            "kind": "verified_anomaly", "column": (r.get("column") or "").strip(),
            "station": (r.get("station") or "").strip(),
            "start": (r.get("start") or "").strip(), "end": (r.get("end") or "").strip(),
            "count": int(r.get("count") or 1), "source": r["source"].strip(),
            "note": (r.get("note") or "").strip(), "_dataset": r["dataset"].strip().upper()})
    return out


def write_labels(path: Path, labels: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for x in sorted(labels, key=lambda y: (y["kind"], y["column"], y["station"], y["start"])):
            w.writerow({k: x.get(k, "") for k in FIELDS})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args(argv)
    wanted = {x.strip().upper() for x in args.only.split(",") if x.strip()}

    registry = json.loads((BASE / "registry.json").read_text(encoding="utf-8"))
    l2 = load_l2(wanted)
    summary = {"generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
               "rules": RULE_TEXT, "datasets": [], "totals": {}}
    totals = Counter()

    for meta in sorted(registry["datasets"], key=lambda x: x["id"]):
        if "error" in meta:
            summary["datasets"].append({"id": meta["id"], "error": meta["error"]})
            continue
        if wanted and meta["id"] not in wanted:
            continue
        labels, detail = build_dataset(meta)
        for r in l2:
            if r["_dataset"] == meta["id"]:
                labels.append({k: v for k, v in r.items() if k != "_dataset"})
        out = ROOT / Path(meta["data_path"]).parent / "labels.csv"
        write_labels(out, labels)
        entry = {"id": meta["id"], "name": meta["name"], "labels": len(labels),
                 "data_sha1": meta["data_sha1"],
                 "labels_sha1": sha1_file(out),
                 "counts": dict(Counter(x["kind"] for x in labels)),
                 "detail": detail, "path": str(out.relative_to(ROOT)).replace("\\", "/")}
        summary["datasets"].append(entry)
        totals.update(entry["counts"])
        print("%s %-34s labels=%d %s" % (meta["id"], meta["name"], len(labels),
                                         entry["counts"]))

    summary["totals"] = dict(totals)
    (BASE / "labels_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("合计：%s" % dict(totals))
    print("L1 标签总数：%d" % sum(v for k, v in totals.items() if k != "verified_anomaly"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())