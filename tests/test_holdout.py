# -*- coding: utf-8 -*-
"""C1 最终保留集：冻结、校验、只跑一次的约束不能靠自觉。"""
from __future__ import annotations

import hashlib
import json

import pytest

from evals import holdout


def _meta(i: int, climate: str, sha: str = "x" * 40) -> dict:
    return {"id": "H%02d" % i, "name": "n%d" % i, "source": "Open-Meteo Archive API",
            "climate": climate, "station": "S%d" % i,
            "data_path": "fixtures/holdout/h%02d/data.csv" % i, "data_sha1": sha,
            "rows": 8784, "start": "2024-01-01 00:00:00", "end": "2024-12-31 23:00:00",
            "columns": ["temperature"]}


def test_manifest_requires_three_distinct_climates():
    reg = {"datasets": [_meta(1, "地中海"), _meta(2, "热带季风"), _meta(3, "极地苔原")]}
    m = holdout.build_manifest(reg)
    assert len(m["datasets"]) == 3
    assert len({d["climate"] for d in m["datasets"]}) == 3
    assert m["purpose"] and m["declaration"] and m["protocol"]["questions"]

    with pytest.raises(holdout.HoldoutError):
        holdout.build_manifest({"datasets": [_meta(1, "地中海"), _meta(2, "地中海"),
                                             _meta(3, "极地苔原")]})
    with pytest.raises(holdout.HoldoutError):
        holdout.build_manifest({"datasets": [_meta(1, "地中海"), _meta(2, "热带季风")]})


def test_verify_catches_changed_data(tmp_path):
    p = tmp_path / "data.csv"
    p.write_bytes(b"timestamp,station,temperature\n2024-01-01 00:00,S,1.0\n")
    good = hashlib.sha1(p.read_bytes()).hexdigest()
    m = {"datasets": [dict(_meta(1, "地中海", good), data_path="data.csv")]}
    assert holdout.verify(m, root=tmp_path) == []

    m["datasets"][0]["data_sha1"] = "0" * 40
    problems = holdout.verify(m, root=tmp_path)
    assert problems and "SHA1 不符" in problems[0]
    assert "H01" in problems[0]


def test_verify_reports_missing_file(tmp_path):
    m = {"datasets": [dict(_meta(1, "地中海"), data_path="nope.csv")]}
    problems = holdout.verify(m, root=tmp_path)
    assert problems and "缺失" in problems[0]


def test_second_run_is_refused_unless_forced_with_reason():
    existing = {"finished_at": "2026-10-25T10:00:00+08:00", "reruns": 0}
    with pytest.raises(holdout.HoldoutError) as e:
        holdout.next_run_meta(existing, force=False, reason="")
    assert "只跑一次" in str(e.value)

    with pytest.raises(holdout.HoldoutError):
        holdout.next_run_meta(existing, force=True, reason="   ")

    meta = holdout.next_run_meta(existing, force=True, reason="修复崩溃后重跑")
    assert meta["reruns"] == 1
    assert meta["previous"]["finished_at"] == existing["finished_at"]

    assert holdout.next_run_meta(None, force=False, reason="")["reruns"] == 0


def test_protocol_pins_pipeline_rule_and_no_threshold_override():
    """保留集不许改阈值：协议里不能出现 param_overrides，模式必须固定。"""
    m = holdout.build_manifest({"datasets": [_meta(1, "地中海"), _meta(2, "热带季风"),
                                             _meta(3, "极地苔原")]})
    proto = m["protocol"]
    assert proto["mode"] == "pipeline" and proto["planner"] == "rule"
    assert "param_overrides" not in proto
    assert [q["id"] for q in proto["questions"]] == ["q1", "q2", "q3", "q4"]


@pytest.mark.skipif(not holdout.MANIFEST.exists(), reason="保留集尚未冻结")
def test_frozen_manifest_still_matches_the_files_on_disk():
    manifest = json.loads(holdout.MANIFEST.read_text(encoding="utf-8"))
    assert holdout.verify(manifest) == [], "保留集数据被改动过，SHA1 校验失败"
    assert len({d["climate"] for d in manifest["datasets"]}) == 3