# -*- coding: utf-8 -*-
"""实验：把 `extreme` 的覆盖归因从 compute_metrics 改成「引擎」，指标会动多少。

背景（#29 遗留）：`TOOL_COVERS["compute_metrics"]` 含 `extreme`，但极值实际由
`engine` 调 `anomaly.detect_extremes` 统一计算，没有任何工具负责它。
agent 模式下只要 `agent_info is not None` 就无条件计算极值（#29 的判据）。

两种归因方案：
- 现状：`extreme` 由 compute_metrics 覆盖 → compute_metrics 没跑时，即使极值已算，也不算覆盖。
- 方案：`extreme` 由「引擎」覆盖 → 只要计划里有 extreme 就算覆盖（与实现一致）。

用法：
    python scripts/exp_coverage_owner.py [artifacts/t212_compare]
"""
from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from mtsa.observation import TOOL_COVERS  # noqa: E402

ENGINE_COVERS = ("extreme",)


def judge(cov: dict, planned: list[str], executed: list[str], scheme: str) -> dict:
    """按两种归因分别算 covered / coverage。"""
    tool_covers = {t: tuple(a for a in TOOL_COVERS.get(t, ()) if a in planned)
                   for t in executed}
    if scheme == "engine":
        # 工具不再声明覆盖 extreme；改由引擎无条件覆盖（agent 模式必算极值）
        covered = {a for t in executed for a in tool_covers[t] if a != "extreme"}
        covered |= {a for a in ENGINE_COVERS if a in planned}
    else:
        covered = {a for t in executed for a in tool_covers[t] if a in planned}
    return {
        "covered": sorted(covered),
        "coverage": round(len(covered) / len(planned), 4) if planned else 1.0,
    }


def main(argv: list[str]) -> int:
    root = argv[1] if len(argv) > 1 else os.path.join(ROOT, "artifacts", "t212_compare")
    # 两种目录命名都有：<case>_agent/result.json 与 <case>/result.json
    paths = sorted(glob.glob(os.path.join(root, "*_agent", "result.json")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(root, "*", "result.json")))
    if not paths:
        print("没有找到 agent 结果：", root)
        return 2

    print("| 用例 | 计划含 extreme | compute_metrics | 现状 coverage | 引擎归因 | 差 |")
    print("|---|---|---|---|---|---|")
    changed, total = 0, 0
    sum_old = sum_new = 0.0
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        cov = (d.get("data") or {}).get("agent", {}).get("coverage") or {}
        if not cov:
            continue
        planned = cov.get("planned_analyses") or []
        executed = cov.get("executed_tools") or []
        old = judge(cov, planned, executed, "tool")
        new = judge(cov, planned, executed, "engine")
        total += 1
        sum_old += old["coverage"]
        sum_new += new["coverage"]
        if old["coverage"] != new["coverage"]:
            changed += 1
        print("| %s | %s | %s | %.4f | %.4f | %+.4f |" % (
            os.path.basename(os.path.dirname(p)), "是" if "extreme" in planned else "否",
            "是" if "compute_metrics" in executed else "否",
            old["coverage"], new["coverage"], new["coverage"] - old["coverage"]))

    print()
    print("用例 %d，覆盖率发生变化的 %d 个" % (total, changed))
    if total:
        print("平均 analysis_coverage：现状 %.4f → 引擎归因 %.4f（%+.4f）" % (
            sum_old / total, sum_new / total, (sum_new - sum_old) / total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
