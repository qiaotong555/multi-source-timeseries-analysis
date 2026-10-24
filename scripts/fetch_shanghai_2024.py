# -*- coding: utf-8 -*-
"""抓取上海 2024 年逐小时天气，并落两种「别人手里的」CSV 形态。

用途：N1 陌生人可用性测试。同一份数据、两种列名与时间写法，看分析引擎
能不能都吃下去；差异本身就是结论。

数据：Open-Meteo Archive API（ERA5 / CERRA 再分析），CC BY 4.0，
      署名要求见 DATA_SOURCES.md。
用法：python scripts/fetch_shanghai_2024.py [--out artifacts/stage2_shanghai]

产出：
  shanghai_raw.json     接口原始响应（含 sha1）
  shanghai_raw.csv      原样列名：time,temperature_2m,...（ISO 时间）
  shanghai_cn.csv       陌生人形态：中文表头 + 单位后缀 + "2024/1/1 0:00" + city 列
  fetch_meta.json        来源 URL、抓取时间、两个 CSV 的 sha1 与行数
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LAT, LON = 31.2304, 121.4737          # 人民广场附近
START, END = "2024-01-01", "2024-12-31"
TZ = "Asia/Shanghai"
HOURLY = "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,precipitation"

# 陌生人形态：中文表头 + 单位后缀（国内导出的 CSV 常见）
CN_COLUMNS = {
    "temperature_2m": "温度(°C)",
    "relative_humidity_2m": "相对湿度(%)",
    "surface_pressure": "气压(hPa)",
    "wind_speed_10m": "风速(km/h)",
    "precipitation": "降水量(mm)",
}


def build_url() -> str:
    return (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={START}&end_date={END}"
        f"&hourly={HOURLY}&timezone={TZ}"
    )


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def cn_time(iso: str) -> str:
    """2024-01-01T00:00 -> 2024/1/1 0:00（Windows 导出常见写法，无前导零）。"""
    dt = datetime.fromisoformat(iso)
    return f"{dt.year}/{dt.month}/{dt.day} {dt.hour}:{dt.minute:02d}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "stage2_shanghai"))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    url = build_url()
    print("[fetch]", url)
    req = urllib.request.Request(url, headers={"User-Agent": "mtsa-n1-shanghai/1.0"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        raw = resp.read()
    (out / "shanghai_raw.json").write_bytes(raw)

    payload = json.loads(raw.decode("utf-8"))
    hourly = payload["hourly"]
    times = hourly["time"]
    series = {k: hourly[k] for k in HOURLY.split(",")}
    n = len(times)
    print(f"[shape] {n} 行；纬度 {payload.get('latitude')} 经度 {payload.get('longitude')}；"
          f"时区 {payload.get('timezone')}；海拔 {payload.get('elevation')} m")

    # 形态 A：原样列名 + ISO 时间
    a_path = out / "shanghai_raw.csv"
    with a_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", *HOURLY.split(",")])
        for i, t in enumerate(times):
            w.writerow([t, *[series[k][i] for k in HOURLY.split(",")]])

    # 形态 B：中文表头 + 单位 + 非 ISO 时间 + 多一个 city 列
    b_path = out / "shanghai_cn.csv"
    keys = HOURLY.split(",")
    with b_path.open("w", encoding="utf-8-sig", newline="") as fh:   # utf-8-sig：Excel 打开不乱码
        w = csv.writer(fh)
        w.writerow(["城市", "时间", *[CN_COLUMNS[k] for k in keys]])
        for i, t in enumerate(times):
            w.writerow(["上海", cn_time(t), *[series[k][i] for k in keys]])

    meta = {
        "source": "Open-Meteo Archive API（ERA5 / CERRA 再分析）",
        "license": "CC BY 4.0",
        "url": url,
        "city": "上海（31.2304N, 121.4737E，海拔 %s m）" % payload.get("elevation"),
        "timezone": payload.get("timezone"),
        "fetched_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "rows": n,
        "columns": keys + ["time"],
        "files": {},
    }
    for path in (a_path, b_path):
        data = path.read_bytes()
        meta["files"][path.name] = {
            "sha1": sha1_bytes(data), "bytes": len(data),
            "rows": sum(1 for _ in path.open("r", encoding="utf-8-sig")) - 1,
        }
    meta["raw_json_sha1"] = sha1_bytes(raw)
    (out / "fetch_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[write]", a_path, meta["files"][a_path.name]["rows"], "行")
    print("[write]", b_path, meta["files"][b_path.name]["rows"], "行")
    print("[sha1] raw", meta["raw_json_sha1"][:16], "| raw.csv", meta["files"][a_path.name]["sha1"][:16],
          "| cn.csv", meta["files"][b_path.name]["sha1"][:16])
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
