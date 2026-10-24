# -*- coding: utf-8 -*-
"""C4：质量问题的可定位区间（runs）与报告/结论引用。

接缝：
- `quality.check_quality`：MISSING_VALUES / SAMPLING_GAP / DUPLICATE / OUT_OF_ORDER 的区间；
- `report._issue_range_cell`：报告第 2 节的「区间」列；
- `scenarios.weather._issue_range_text`：`数据口径` 结论里的区间；
- `QualityIssue` 的向后兼容：旧 payload 没有 runs 也能解析。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mtsa import profiler, quality
from mtsa.io_loaders import build_spec
from mtsa.models import QualityIssue, RawDataset
from mtsa.report import _issue_range_cell
from mtsa.scenarios.weather import _issue_range_text

from .helpers import make_weather_frame


def _profile(df: pd.DataFrame):
    raw = RawDataset(frame=df, spec=build_spec("mem.csv"), file_hash="0" * 40,
                     row_count=len(df),
                     column_dtypes={c: str(t) for c, t in df.dtypes.items()})
    return raw, *profiler.profile_dataset(raw)


def _issues(rep, code):
    return [i for i in rep.issues if i.code == code]


def _single_series(n: int = 600) -> pd.DataFrame:
    """去掉 station 列，得到单实体序列，便于断言精确起止。"""
    return make_weather_frame(n=n, stations=1).drop(columns=["station"])


def test_missing_runs_carry_start_end_and_count():
    df = _single_series()
    df.loc[100:160, "temperature"] = np.nan
    df.loc[300:304, "temperature"] = np.nan
    raw, profile, parsed = _profile(df)
    rep = quality.check_quality(raw, profile, parsed)
    issues = _issues(rep, "MISSING_VALUES")
    assert len(issues) == 1
    iss = issues[0]
    assert iss.run_count == 2
    assert iss.runs_truncated is False
    assert [r["count"] for r in iss.runs] == [61, 5]
    assert iss.runs[0]["start"] == str(df.loc[100, "timestamp"])
    assert iss.runs[0]["end"] == str(df.loc[160, "timestamp"])
    assert iss.start == iss.runs[0]["start"]
    assert iss.end == iss.runs[-1]["end"]


def test_runs_are_capped_but_total_is_reported():
    df = _single_series()
    for k in range(25):
        df.loc[20 + 20 * k, "temperature"] = np.nan
    raw, profile, parsed = _profile(df)
    rep = quality.check_quality(raw, profile, parsed)
    iss = _issues(rep, "MISSING_VALUES")[0]
    assert iss.run_count == 25
    assert len(iss.runs) == quality.MAX_ISSUE_RUNS == 20
    assert iss.runs_truncated is True


def test_grouped_missing_runs_stay_per_station():
    df = make_weather_frame(n=120, stations=2)
    idx0 = df.index[(df["station"] == "STA_0")][10:13].tolist()
    idx1 = df.index[(df["station"] == "STA_1")][50:53].tolist()
    df.loc[idx0 + idx1, "temperature"] = np.nan
    raw, profile, parsed = _profile(df)
    rep = quality.check_quality(raw, profile, parsed)
    iss = _issues(rep, "MISSING_VALUES")[0]
    assert iss.run_count == 2
    assert {r["group"] for r in iss.runs} == {"STA_0", "STA_1"}
    assert sorted(r["count"] for r in iss.runs) == [3, 3]


def test_sampling_gap_lists_every_gap():
    df = _single_series()
    df = df.drop(index=range(200, 260)).reset_index(drop=True)
    raw, profile, parsed = _profile(df)
    rep = quality.check_quality(raw, profile, parsed)
    iss = _issues(rep, "SAMPLING_GAP")[0]
    assert iss.run_count == rep.gap_count >= 1
    assert len(iss.runs) == rep.gap_count
    assert iss.runs[0]["start"] and iss.runs[0]["end"]
    assert iss.runs_truncated is False


def test_duplicate_and_out_of_order_have_positions():
    df = _single_series()
    df = pd.concat([df, df.iloc[200:206]], ignore_index=True)
    block = df.iloc[300:306]
    rest = pd.concat([df.iloc[:300], df.iloc[306:]], ignore_index=True)
    df = pd.concat([rest.iloc[:350], block, rest.iloc[350:]], ignore_index=True)
    raw, profile, parsed = _profile(df)
    rep = quality.check_quality(raw, profile, parsed)
    dup = _issues(rep, "DUPLICATE_TIMESTAMPS")[0]
    assert dup.start and dup.end and dup.row_label is not None
    ooo = _issues(rep, "OUT_OF_ORDER")[0]
    assert ooo.start and ooo.end


def test_report_and_conclusion_render_runs_compactly():
    iss = QualityIssue(
        code="MISSING_VALUES", severity="warn", column="temperature", count=42,
        runs=[{"start": "2024-01-01 00:00:00", "end": "2024-01-01 02:00:00", "count": 3},
              {"start": "2024-02-01 00:00:00", "end": "2024-02-01 00:00:00", "count": 1},
              {"start": "2024-03-01 00:00:00", "end": "2024-03-01 00:00:00", "count": 1},
              {"start": "2024-04-01 00:00:00", "end": "2024-04-01 00:00:00", "count": 1},
              {"start": "2024-05-01 00:00:00", "end": "2024-05-01 00:00:00", "count": 1}],
        run_count=6, runs_truncated=False,
    )
    cell = _issue_range_cell(iss)
    assert "2024-01-01 00:00:00~2024-01-01 02:00:00（3 点）" in cell
    assert "共 6 段" in cell
    text = _issue_range_text(iss)
    assert text.startswith("，区间 ")
    assert "另有 3 段" in text


def test_old_payload_without_runs_still_parses():
    iss = QualityIssue.model_validate({"code": "MISSING_VALUES", "severity": "warn",
                                       "column": "temperature", "count": 3,
                                       "detail": "缺失率 1.00%"})
    assert iss.runs == [] and iss.run_count == 0 and iss.runs_truncated is False
    assert _issue_range_cell(iss) == "—"
    assert _issue_range_text(iss) == ""
