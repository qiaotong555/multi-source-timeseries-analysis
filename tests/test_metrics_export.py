# -*- coding: utf-8 -*-
"""A4：指标表单独导出 CSV。

导出件必须与 `result.json` 里的行**逐字段一致**——否则它不是证据，只是一份长得像的副本。
这里锁五条：

1. 两个 CSV 都作为 `table_csv` 产物登记，且在磁盘上存在；
2. 行数与 `data.metrics` / `data.scenario_metrics` 相等；
3. 表头就是 `MetricRow` 的字段（顺序固定）；
4. 抽样比对 `metric_id` 与 `value`（浮点按 1e-12 容差）；
5. 空表不落盘——`_write_metric_table` 对空列表返回 None。
"""
from __future__ import annotations

import csv
import os

import mtsa.engine as engine
import mtsa.report as report_mod
from mtsa.models import AnalysisRequest

FIX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures")


def _run(tmp_path):
    req = AnalysisRequest(request_id="a4",
                          input_path=os.path.join(FIX, "weather_v1.csv"),
                          scene_id="weather", question="分析温度的趋势与异常",
                          output_dir=str(tmp_path), mode="pipeline", planner="rule")
    res = engine.run_analysis(req, verbose=False)
    assert res.status == "succeeded", res.error
    return res


def _aids(res):
    return sorted(a.artifact_id for a in res.artifacts)


def _rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_metric_csvs_are_artifacts(tmp_path):
    res = _run(tmp_path)
    by_id = {a.artifact_id: a for a in res.artifacts}
    for aid in ("art_metrics_csv", "art_scenario_metrics_csv"):
        assert aid in by_id, _aids(res)
        assert by_id[aid].kind == "table_csv"
        assert os.path.exists(by_id[aid].path)


def test_rows_match_result_json(tmp_path):
    res = _run(tmp_path)
    for name, key in (("metrics.csv", "metrics"), ("scenario_metrics.csv", "scenario_metrics")):
        path = os.path.join(str(tmp_path), name)
        rows = _rows(path)
        expected = res.data[key]
        assert len(rows) == len(expected), (name, len(rows), len(expected))
        for row, item in zip(rows, expected):
            assert row["metric_id"] == item["metric_id"]
            if item["value"] is None:
                assert row["value"] == ""
            else:
                assert abs(float(row["value"]) - float(item["value"])) < 1e-12


def test_header_is_metric_row_fields(tmp_path):
    res = _run(tmp_path)
    path = os.path.join(str(tmp_path), "metrics.csv")
    with open(path, encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    assert header == report_mod.METRIC_CSV_FIELDS


def test_empty_table_writes_nothing(tmp_path):
    assert report_mod._write_metric_table([], str(tmp_path), "x.csv", "art_x", "X") is None
    assert not os.path.exists(os.path.join(str(tmp_path), "x.csv"))
