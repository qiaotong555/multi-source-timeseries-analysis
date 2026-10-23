# -*- coding: utf-8 -*-
"""比较两份 result.json 的语义内容是否一致（C 任务使用）。

不比较 run_id / created_at / request_id / 产物路径 / 证据编号等运行期字段；
比较结论、证据、警告、计划、指标、检验、异常、变化点与质量报告。
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "evals", "short_repro_results.json")

DATA_FIELDS = ("metrics", "scenario_metrics", "spectrum", "tests",
               "anomalies", "change_points", "quality", "plan")


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    checks = {}
    for f in ("conclusions", "evidence", "warnings"):
        checks[f] = a.get(f) == b.get(f)
    for f in DATA_FIELDS:
        checks[f"data.{f}"] = (a.get("data") or {}).get(f) == (b.get("data") or {}).get(f)
    diffs = [k for k, v in checks.items() if not v]
    return {"identical": not diffs, "checks": checks, "diffs": diffs,
            "conclusions_count": [len(a.get("conclusions") or []),
                                  len(b.get("conclusions") or [])]}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="比较两份 result.json 的语义一致性")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    a = json.load(open(args.a, encoding="utf-8"))
    b = json.load(open(args.b, encoding="utf-8"))
    res = {"generated_at": datetime.now().isoformat(timespec="seconds"),
           "a": os.path.relpath(args.a, ROOT), "b": os.path.relpath(args.b, ROOT),
           **compare(a, b)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())