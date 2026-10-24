# -*- coding: utf-8 -*-
"""下载并转换 10 份公开真实天气数据为项目统一长表。

只做下载/转换/记录 SHA1，不修改分析逻辑。
运行：python scripts/fetch_real_weather.py [--only R01,R02] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fixtures" / "real_weather"
UA = {"User-Agent": "mtsa-real-data-eval/1.0 (research; contact: local)"}

OM_HOURLY = "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,precipitation"
OM_DAILY = ("temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
            "relative_humidity_2m_mean,surface_pressure_mean,"
            "wind_speed_10m_max,precipitation_sum")

DATASETS = [
    {"id": "R01", "name": "berlin_hourly_2024", "kind": "openmeteo_hourly",
     "lat": 52.52, "lon": 13.41, "start": "2024-01-01", "end": "2024-12-31",
     "station": "BER", "source": "Open-Meteo Archive API",
     "note": "温带、完整五要素、小时级"},
    {"id": "R02", "name": "singapore_hourly_2024", "kind": "openmeteo_hourly",
     "lat": 1.35, "lon": 103.82, "start": "2024-01-01", "end": "2024-12-31",
     "station": "SIN", "source": "Open-Meteo Archive API",
     "note": "热带、高温高湿、年振幅小"},
    {"id": "R03", "name": "phoenix_daily_2015_2024", "kind": "openmeteo_daily",
     "lat": 33.45, "lon": -112.07, "start": "2015-01-01", "end": "2024-12-31",
     "station": "PHX", "source": "Open-Meteo Archive API",
     "note": "干旱、极端高温、日级、10 年"},
    {"id": "R04", "name": "reykjavik_hourly_2024", "kind": "openmeteo_hourly",
     "lat": 64.15, "lon": -21.94, "start": "2024-01-01", "end": "2024-12-31",
     "station": "REK", "source": "Open-Meteo Archive API",
     "note": "寒带、风大、温度低"},
    {"id": "R05", "name": "three_cities_hourly_2024", "kind": "openmeteo_multi",
     "start": "2024-01-01", "end": "2024-12-31",
     "stations": [("SYD", -33.87, 151.21), ("TYO", 35.68, 139.69), ("SIN", 1.35, 103.82)],
     "source": "Open-Meteo Archive API", "note": "多站点长表、跨气候带"},
    {"id": "R06", "name": "nyc_central_park_daily_2018_2024", "kind": "ghcn",
     "station_id": "USW00094728", "station": "NYC", "start": "2018-01-01", "end": "2024-12-31",
     "source": "NOAA GHCN-Daily", "note": "官方 CSV、日值、缺测哨兵"},
    {"id": "R07", "name": "melbourne_daily_2018_2024", "kind": "ghcn",
     "station_id": "ASN00086282", "station": "MEL", "start": "2018-01-01", "end": "2024-12-31",
     "source": "NOAA GHCN-Daily", "note": "南半球（澳大利亚）、官方 CSV、日值"},
    {"id": "R08", "name": "berlin_tegel_hourly_2024", "kind": "dwd",
     "station_id": "00433", "station": "BER_TEGEL", "start": "2024-01-01", "end": "2024-12-31",
     "source": "DWD Open Data", "note": "德语列名、zip 内文本、-999 缺测"},
    {"id": "R09", "name": "munich_hourly_2024", "kind": "dwd",
     "station_id": "01270", "station": "MUC", "start": "2024-01-01", "end": "2024-12-31",
     "source": "DWD Open Data", "note": "同源不同站"},
    {"id": "R10", "name": "toronto_hourly_2024", "kind": "eccc",
     "station_id": "51459", "station": "YYZ", "year": 2024,
     "source": "ECCC Bulk Data", "note": "月度 CSV 拼接、BOM、空白缺测"},
]


def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def sha1_file(p: Path) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, retries: int = 3) -> bytes:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (i + 1))
    raise RuntimeError("下载失败：%s（%s）" % (url, last))


def _frame(ts, station, **cols) -> pd.DataFrame:
    df = pd.DataFrame({"timestamp": pd.to_datetime(ts), "station": station})
    for k, v in cols.items():
        # 关键：reset_index，避免按原索引对齐导致整列 NaN
        df[k] = pd.to_numeric(pd.Series(v).reset_index(drop=True), errors="coerce")
    return df


def openmeteo_url(ds: dict, lat: float, lon: float) -> str:
    ent = "daily" if ds["kind"] == "openmeteo_daily" else "hourly"
    params = OM_HOURLY if ent == "hourly" else OM_DAILY
    return ("https://archive-api.open-meteo.com/v1/archive?latitude=%s&longitude=%s"
            "&start_date=%s&end_date=%s&%s=%s&timezone=UTC" % (
                lat, lon, ds["start"], ds["end"], ent, params))


def convert_openmeteo(ds: dict) -> tuple[pd.DataFrame, list[dict]]:
    frames, raws = [], []
    stations = ds.get("stations") or [(ds["station"], ds["lat"], ds["lon"])]
    for name, lat, lon in stations:
        url = openmeteo_url(ds, lat, lon)
        payload = fetch(url)
        raws.append({"url": url, "bytes": len(payload), "sha1": sha1_bytes(payload)})
        js = json.loads(payload.decode("utf-8"))
        if ds["kind"] != "openmeteo_daily":
            h = js["hourly"]
            frames.append(_frame(h["time"], name,
                                 temperature=h["temperature_2m"],
                                 humidity=h["relative_humidity_2m"],
                                 pressure=h["surface_pressure"],
                                 wind_speed=h["wind_speed_10m"],
                                 rainfall=h["precipitation"]))
        else:
            d = js["daily"]
            tmax = pd.to_numeric(pd.Series(d["temperature_2m_max"]), errors="coerce")
            tmin = pd.to_numeric(pd.Series(d["temperature_2m_min"]), errors="coerce")
            frames.append(_frame(d["time"], name,
                                 temperature=(tmax + tmin) / 2.0,
                                 humidity=d["relative_humidity_2m_mean"],
                                 pressure=d["surface_pressure_mean"],
                                 wind_speed=d["wind_speed_10m_max"],
                                 rainfall=d["precipitation_sum"]))
    return pd.concat(frames, ignore_index=True), raws


def convert_ghcn(ds: dict) -> tuple[pd.DataFrame, list[dict]]:
    url = ("https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/"
           "access/%s.csv" % ds["station_id"])
    payload = fetch(url)
    raw = {"url": url, "bytes": len(payload), "sha1": sha1_bytes(payload)}
    import io as _io
    df = pd.read_csv(_io.BytesIO(payload), dtype=str)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    lo = pd.Timestamp(ds["start"])
    hi = pd.Timestamp(ds["end"]) + pd.Timedelta(days=1)
    df = df[(df["DATE"] >= lo) & (df["DATE"] < hi)].copy()

    def num(col):
        if col not in df:
            return pd.Series([float("nan")] * len(df))
        s = pd.to_numeric(df[col], errors="coerce")
        return s.replace([-9999, -999.0, -9999.0], float("nan"))

    tmax, tmin = num("TMAX"), num("TMIN")
    out = _frame(df["DATE"].values, ds["station"],
                 temperature=(tmax + tmin) / 20.0,
                 rainfall=num("PRCP") / 10.0,
                 wind_speed=num("AWND") / 10.0)
    return out, [raw]


def convert_dwd(ds: dict) -> tuple[pd.DataFrame, list[dict]]:
    url = ("https://opendata.dwd.de/climate_environment/CDC/observations_germany/"
           "climate/hourly/air_temperature/historical/"
           "stundenwerte_TU_%s_19510101_20251231_hist.zip" % ds["station_id"])
    payload = fetch(url)
    raw = {"url": url, "bytes": len(payload), "sha1": sha1_bytes(payload)}
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        name = next(n for n in zf.namelist() if n.startswith("produkt_tu_stunde_"))
        text = zf.read(name).decode("latin-1")
    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    df = df[df["MESS_DATUM"].str.fullmatch(r"\d{10}")].copy()
    df["timestamp"] = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d%H", errors="coerce")
    lo = pd.Timestamp(ds["start"])
    hi = pd.Timestamp(ds["end"]) + pd.Timedelta(days=1)
    df = df[(df["timestamp"] >= lo) & (df["timestamp"] < hi)].copy()

    def num(col):
        s = pd.to_numeric(df[col], errors="coerce")
        return s.replace([-999, -999.0], float("nan"))

    out = _frame(df["timestamp"].values, ds["station"],
                 temperature=num("TT_TU"), humidity=num("RF_TU"))
    return out, [raw]


def convert_eccc(ds: dict) -> tuple[pd.DataFrame, list[dict]]:
    frames, raws = [], []
    year = ds["year"]
    for month in range(1, 13):
        url = ("https://climate.weather.gc.ca/climate_data/bulk_data_e.html?"
               "format=csv&stationID=%s&Year=%d&Month=%d&Day=1&timeframe=1&submit=Download+Data"
               % (ds["station_id"], year, month))
        payload = fetch(url)
        raws.append({"url": url, "bytes": len(payload), "sha1": sha1_bytes(payload)})
        frames.append(pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig", dtype=str))
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
    ts = pd.to_datetime(df["Date/Time (LST)"], errors="coerce")

    def num(col):
        if col not in df:
            return pd.Series([float("nan")] * len(df))
        return pd.to_numeric(df[col], errors="coerce")

    out = _frame(ts.values, ds["station"],
                 temperature=num("Temp (°C)"),
                 humidity=num("Rel Hum (%)"),
                 pressure=num("Stn Press (kPa)") * 10.0,
                 wind_speed=num("Wind Spd (km/h)") / 3.6,
                 rainfall=num("Precip. Amount (mm)"))
    out = out[out["timestamp"].notna()].copy()
    return out, raws


CONVERTERS = {
    "openmeteo_hourly": convert_openmeteo,
    "openmeteo_daily": convert_openmeteo,
    "openmeteo_multi": convert_openmeteo,
    "ghcn": convert_ghcn,
    "dwd": convert_dwd,
    "eccc": convert_eccc,
}


def build(ds: dict, force: bool = False) -> dict:
    ddir = OUT / ("%s_%s" % (ds["id"].lower(), ds["name"]))
    ddir.mkdir(parents=True, exist_ok=True)
    data_path = ddir / "data.csv"
    meta_path = ddir / "metadata.json"
    if data_path.exists() and meta_path.exists() and not force:
        return json.loads(meta_path.read_text(encoding="utf-8"))

    df, raws = CONVERTERS[ds["kind"]](ds)
    df = df.sort_values(["timestamp", "station"]).reset_index(drop=True)
    df = df.dropna(subset=["timestamp"])
    # 整列为空的字段说明该源根本不报送这一要素（例如 GHCN 站点无 AWND、
    # ECCC 小时表不报降水量）。保留假列会污染后续质量检查，因此删除并在
    # 元数据留痕；报告里按「源不提供」解释，而不是当成缺失事件。
    all_null = [c for c in df.columns
                if c not in ("timestamp", "station") and not df[c].notna().any()]
    if all_null:
        df = df.drop(columns=all_null)
    # 保留真实重复时间戳；只删除完全空白的行
    value_cols = [c for c in df.columns if c not in ("timestamp", "station")]
    df = df[~df[value_cols].isna().all(axis=1)].reset_index(drop=True)
    if len(df) < 100:
        raise ValueError("转换后有效行数过少：%d" % len(df))
    if not df[value_cols].notna().any().any():
        raise ValueError("所有数值列均为空，转换失败")
    df.to_csv(data_path, index=False, encoding="utf-8-sig")

    (ddir / "raw").mkdir(exist_ok=True)
    for i, raw in enumerate(raws):
        (ddir / "raw" / ("source_%02d.txt" % (i + 1))).write_text(
            json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    meta = {
        "id": ds["id"], "name": ds["name"], "source": ds["source"],
        "note": ds.get("note", ""),
        "retrieved_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "stations": sorted(str(x) for x in df["station"].unique()),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "frequency": "daily" if ds["kind"] == "openmeteo_daily" or ds["kind"] == "ghcn" else "hourly",
        "columns": value_cols,
        "dropped_all_null_columns": all_null,
        "data_sha1": sha1_file(data_path),
        "raw": raws,
        "data_path": str(data_path.relative_to(ROOT)).replace(os.sep, "/"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ddir / "sha1.txt").write_text(meta["data_sha1"] + "  data.csv\n", encoding="utf-8")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    wanted = {x.strip().upper() for x in args.only.split(",") if x.strip()}
    selected = [d for d in DATASETS if not wanted or d["id"] in wanted]
    OUT.mkdir(parents=True, exist_ok=True)
    metas = []
    for ds in selected:
        print("下载/转换 %s %s ..." % (ds["id"], ds["name"]))
        try:
            meta = build(ds, force=args.force)
            metas.append(meta)
            print("  rows=%d stations=%s %s → %s" % (
                meta["rows"], ",".join(meta["stations"]), meta["start"], meta["end"]))
        except Exception as exc:  # noqa: BLE001
            print("  FAILED: %s" % exc, file=sys.stderr)
            metas.append({"id": ds["id"], "name": ds["name"], "error": str(exc)})
    reg_path = OUT / "registry.json"
    existing = {}
    if reg_path.exists():
        try:
            existing = {x["id"]: x for x in json.loads(reg_path.read_text(encoding="utf-8")).get("datasets", [])}
        except Exception:  # noqa: BLE001
            existing = {}
    for m in metas:
        existing[m["id"]] = m
    reg = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "datasets": [existing[k] for k in sorted(existing)]}
    reg_path.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ok = sum(1 for m in metas if "error" not in m)
    print("完成：%d/%d；registry=%s" % (ok, len(metas), reg_path))
    return 0 if ok == len(metas) else 1


if __name__ == "__main__":
    raise SystemExit(main())
