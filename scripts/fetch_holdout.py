# -*- coding: utf-8 -*-
"""抓取并冻结最终保留集（C1）：3 份此前没用过的真实天气数据。

为什么另开一个目录
------------------
`fixtures/real_weather/` 的 17 份数据已经用于 80 次真实数据评估——R08 的单位误判、
R05 的重复误报都是看着它们的结果修掉的。它们不能再充当"最后一次不看"的保留集。
保留集单独放 `fixtures/holdout/`，`evals/real_data_eval.py` 只读
`fixtures/real_weather/registry.json`，扫不到这里，结构上避免日后被顺手复用。

职责边界
--------
只做下载 / 转换 / 记录 SHA1，不改分析逻辑。已有数据默认拒绝覆盖：覆盖会改变
SHA1，而保留集的全部价值就在于 SHA1 不变。

运行：
    PYTHONPATH=src python -m scripts.fetch_holdout           # 首次抓取
    PYTHONPATH=src python -m scripts.fetch_holdout --check   # 只校验 SHA1，不下载
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fetch_real_weather import convert_openmeteo  # noqa: E402

OUT = ROOT / "fixtures" / "holdout"

# 三个气候带刻意只挑 17 份里没有的类型：
#   地中海（Csa）—— 夏季极端高温 + 冬季温和多雨，无对应数据；
#   热带季风（Am）—— 6–9 月暴雨，与新加坡的全年湿润（Af）不同；
#   极地苔原（ET）—— 极昼极夜，温度长期贴近 0，与 Reykjavik / Ushuaia 的海洋性亚寒带不同。
DATASETS = [
    {"id": "H01", "name": "athens_hourly_2024", "kind": "openmeteo_hourly",
     "lat": 37.98, "lon": 23.73, "station": "ATH",
     "start": "2024-01-01", "end": "2024-12-31",
     "source": "Open-Meteo Archive API",
     "climate": "地中海气候 Csa",
     "note": "夏季极端高温、冬季温和多雨、年降水集中在冷半年"},
    {"id": "H02", "name": "mumbai_hourly_2024", "kind": "openmeteo_hourly",
     "lat": 19.08, "lon": 72.88, "station": "BOM",
     "start": "2024-01-01", "end": "2024-12-31",
     "source": "Open-Meteo Archive API",
     "climate": "热带季风气候 Am",
     "note": "6–9 月季风暴雨、湿度近饱和、旱雨季振幅极大"},
    {"id": "H03", "name": "longyearbyen_hourly_2024", "kind": "openmeteo_hourly",
     "lat": 78.22, "lon": 15.63, "station": "LYR",
     "start": "2024-01-01", "end": "2024-12-31",
     "source": "Open-Meteo Archive API",
     "climate": "极地苔原气候 ET",
     "note": "78°N、极昼极夜、温度长期贴近 0、风速高"},
]

REGISTRY = OUT / "registry.json"


def sha1_file(p: Path) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry() -> dict:
    if not REGISTRY.exists():
        return {"generated_at": None, "datasets": []}
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def build(ds: dict) -> dict:
    ddir = OUT / ("%s_%s" % (ds["id"].lower(), ds["name"]))
    data_path = ddir / "data.csv"
    meta_path = ddir / "metadata.json"
    if data_path.exists() and meta_path.exists():
        # 不覆盖：保留集的 SHA1 必须稳定
        return json.loads(meta_path.read_text(encoding="utf-8"))

    df, raws = convert_openmeteo(ds)
    df = df.sort_values(["timestamp", "station"]).reset_index(drop=True)
    df = df.dropna(subset=["timestamp"])
    value_cols = [c for c in df.columns if c not in ("timestamp", "station")]
    df = df[~df[value_cols].isna().all(axis=1)].reset_index(drop=True)
    if len(df) < 8000:
        raise ValueError("转换后有效行数过少：%d" % len(df))

    ddir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False, encoding="utf-8-sig")

    (ddir / "raw").mkdir(exist_ok=True)
    for i, raw in enumerate(raws):
        (ddir / "raw" / ("source_%02d.json" % (i + 1))).write_text(
            json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    meta = {
        "id": ds["id"], "name": ds["name"], "source": ds["source"],
        "climate": ds["climate"], "note": ds["note"],
        "lat": ds["lat"], "lon": ds["lon"], "station": ds["station"],
        "retrieved_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "start": str(df["timestamp"].min()), "end": str(df["timestamp"].max()),
        "frequency": "hourly", "columns": value_cols,
        "data_sha1": sha1_file(data_path),
        "raw": raws,
        "data_path": str(data_path.relative_to(ROOT)).replace(os.sep, "/"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    (ddir / "sha1.txt").write_text(meta["data_sha1"] + "  data.csv\n", encoding="utf-8")
    return meta


def check() -> int:
    reg = load_registry()
    if not reg["datasets"]:
        print("还没有保留集数据，先跑一次抓取")
        return 1
    bad = 0
    for meta in reg["datasets"]:
        p = ROOT / meta["data_path"]
        if not p.exists():
            print("%s 缺失：%s" % (meta["id"], meta["data_path"]))
            bad += 1
            continue
        now = sha1_file(p)
        ok = now == meta["data_sha1"]
        print("%s %s %s" % (meta["id"], "OK " if ok else "CHANGED", now))
        bad += 0 if ok else 1
    return 0 if bad == 0 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验 SHA1，不下载")
    args = ap.parse_args(argv)
    if args.check:
        return check()

    OUT.mkdir(parents=True, exist_ok=True)
    metas, failed = [], 0
    for ds in DATASETS:
        print("下载/转换 %s %s ..." % (ds["id"], ds["name"]))
        try:
            meta = build(ds)
            metas.append(meta)
            print("  rows=%d %s → %s  sha1=%s" % (
                meta["rows"], meta["start"], meta["end"], meta["data_sha1"]))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("  FAILED: %s" % exc, file=sys.stderr)

    if failed:
        print("有 %d 份抓取失败，registry 未更新" % failed, file=sys.stderr)
        return 1

    reg = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "note": ("最终保留集（C1）。与 fixtures/real_weather/ 分开存放，"
                    "不参与任何调参；SHA1 一经冻结不再变动。"),
           "datasets": metas}
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print("完成：%d 份；registry=%s" % (len(metas), REGISTRY.relative_to(ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())