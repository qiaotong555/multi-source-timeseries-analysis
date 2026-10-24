# -*- coding: utf-8 -*-
"""D3：数据扩容（跨区域 6 份 + 观测阵风 1 份）与两处被它暴露的缺陷。

接缝：
- `fixtures/real_weather/registry.json`：新数据的来源/SHA1/海拔；
- `weather._canon_map`：阵风必须单列，不能被 wind 通配吞掉；
- `anomaly.detect_extremes`：插补排除必须按列（行级并集会把别列的缺失连坐）；
- 极端段上限：长序列不能被静默截断（D2 39.6 的教训）。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from mtsa import anomaly
from mtsa.models import CleanDataset
from mtsa.scenarios.weather import _canon_map

from .helpers import make_plan

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = os.path.join(ROOT, "fixtures", "real_weather", "registry.json")
NEW_IDS = ["R11", "R12", "R13", "R14", "R15", "R16", "R17"]


def _registry() -> dict[str, dict]:
    with open(REG, encoding="utf-8") as f:
        reg = json.load(f)
    return {d["id"]: d for d in reg["datasets"] if "error" not in d}


def test_new_datasets_have_provenance_and_elevation():
    reg = _registry()
    assert len(reg) >= 17
    high = 0
    for i in NEW_IDS:
        d = reg[i]
        assert d["raw"], i
        assert all(r.get("url") and r.get("sha1") for r in d["raw"]), i
        assert d["rows"] > 1000, i
        if isinstance(d.get("elevation_m"), (int, float)) and d["elevation_m"] >= 1500:
            high += 1
    assert high >= 4, high                     # 高海拔站点至少 4 份


def test_gust_column_is_its_own_target():
    cols = ["timestamp", "station", "temperature", "humidity", "pressure",
            "wind_speed", "wind_gusts", "rainfall"]
    mapping = _canon_map(cols)
    assert mapping["wind"] == "wind_speed"     # 平均风速不被阵风抢走
    assert mapping["wind_gust"] == "wind_gusts"
    assert len(set(mapping.values())) == len(mapping)


def test_only_gust_column_does_not_duplicate_targets():
    mapping = _canon_map(["temperature", "wind_gusts"])
    assert mapping["wind_gust"] == "wind_gusts"
    assert "wind" not in mapping                # 没有平均风速时不能硬指到阵风
    assert len(set(mapping.values())) == len(mapping)


def _frame(n: int = 600):
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    gust = np.linspace(0, 10, n)
    press = np.linspace(0, 10, n)
    gust[-1] = 100.0
    press[-1] = 100.0
    return pd.DataFrame({"wind_gusts": gust, "pressure": press}, index=idx)


def test_extreme_masks_imputation_per_column():
    """同一时刻：阵风是观测、气压是插补 —— 阵风极值必须保留。"""
    df = _frame()
    ts = str(df.index[-1])
    clean = CleanDataset(frame=df, imputed_index=[ts],
                         imputed_columns={"pressure": [ts]})
    out = anomaly.detect_extremes(clean, make_plan("weather", ["wind_gusts", "pressure"]))
    gust = next(e for e in out if e["column"] == "wind_gusts" and e["tail"] == "high")
    press = next(e for e in out if e["column"] == "pressure" and e["tail"] == "high")
    assert any(r["end"] == ts for r in gust["runs"]), "阵风极值被别列的插补连坐"
    assert not any(r["end"] == ts for r in press["runs"]), "被插补的气压极值不能计入"


def test_row_level_mask_still_drops_it_for_legacy_callers():
    """只给行级 imputed_index 时退回旧语义（D2 行为），证明差异来自逐列信息。"""
    df = _frame()
    ts = str(df.index[-1])
    clean = CleanDataset(frame=df, imputed_index=[ts])
    out = anomaly.detect_extremes(clean, make_plan("weather", ["wind_gusts", "pressure"]))
    gust = next(e for e in out if e["column"] == "wind_gusts" and e["tail"] == "high")
    assert not any(r["end"] == ts for r in gust["runs"])


def test_extreme_runs_are_not_truncated_on_long_series():
    """400 段极端不能被上限砍掉：D2 的 100 段上限在 R17 上砍掉了 3/4 的时间轴。"""
    n, runs_wanted = 8000, 300
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    vals = np.zeros(n)
    pos = np.arange(5, n - 5, n // runs_wanted)[:runs_wanted]
    vals[pos] = 50.0
    df = pd.DataFrame({"wind_gusts": vals}, index=idx)
    out = anomaly.detect_extremes(CleanDataset(frame=df),
                                  make_plan("weather", ["wind_gusts"]))
    high = next(e for e in out if e["tail"] == "high")
    assert high["run_count"] > 100
    assert len(high["runs"]) == high["run_count"]
    assert high.get("runs_truncated") is False
