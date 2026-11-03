"""实验：plan cache 的键对哪些改动敏感、对哪些不敏感。

背景：2026-11-03 把 `TOOL_SPECS["changepoint"]` 从「均值与方差突变」改成「均值突变」
（方差通道停用）。如果缓存键不含提示词指纹，改完之后同一条问题仍会命中旧计划，
用户看到 `plan_source=cache` 就以为是最新结果。

本脚本只读 src/，不写任何产物。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402

from mtsa import planner as PL  # noqa: E402


class FakeProfile:
    numeric_columns = ["temperature", "humidity", "pressure"]


class FakeClient:
    def __init__(self, model: str, base_url: str = "https://api.deepseek.com/v1") -> None:
        self.model = model
        self.base_url = base_url


def key(client, question: str, frame: pd.DataFrame, scene: str = "weather") -> str:
    p = PL.LLMPlanner(client, cache_dir=None)
    return p._cache_key(question, FakeProfile(), frame, scene)


def main() -> int:
    frame = pd.DataFrame({"a": range(100)})
    question = "分析温度与降水的趋势和异常，并给出未来一天的预测"
    base_client = FakeClient("deepseek-chat")

    k0 = key(base_client, question, frame)
    print("基线 key        :", k0)

    # 1) 改提示词内容（模拟 A3 那次的改动）
    old_desc = PL.TOOL_SPECS["changepoint"]
    PL.TOOL_SPECS["changepoint"] = "变化点检测：均值突变（水平跳变）。"
    k1 = key(base_client, question, frame)
    print("改 TOOL_SPECS 后:", k1, "→", "**同**" if k1 == k0 else "不同")

    # 2) 往提示词里塞一条实质规则
    original_prompt = PL.LLMPlanner._system_prompt
    PL.LLMPlanner._system_prompt = staticmethod(lambda: original_prompt() + "\n7. 新增规则。")
    k2 = key(base_client, question, frame)
    print("改 system prompt:", k2, "→", "**同**" if k2 == k0 else "不同")
    PL.LLMPlanner._system_prompt = staticmethod(original_prompt)

    # 3) 换模型
    k3 = key(FakeClient("qwen-max"), question, frame)
    print("换模型          :", k3, "→", "**同**" if k3 == k0 else "不同")

    # 4) 换 base_url（同模型不同供应商，行为可能不同）
    k4 = key(FakeClient("deepseek-chat", "https://api.moonshot.cn/v1"), question, frame)
    print("换 base_url     :", k4, "→", "**同**" if k4 == k0 else "不同")

    # 5) 对照组：真该变的东西
    k5 = key(base_client, question + "（换个问法）", frame)
    k6 = key(base_client, question, pd.DataFrame({"a": range(101)}))
    print("改问题          :", k5, "→", "**同**" if k5 == k0 else "不同（预期）")
    print("改行数          :", k6, "→", "**同**" if k6 == k0 else "不同（预期）")

    PL.TOOL_SPECS["changepoint"] = old_desc

    stale = [k1, k2, k3, k4]
    n_stale = sum(1 for k in stale if k == k0)
    print()
    print("结论：%d / 4 项「应当失效」的改动没有改变键。" % n_stale)
    return 0 if n_stale else 1


if __name__ == "__main__":
    raise SystemExit(main())
