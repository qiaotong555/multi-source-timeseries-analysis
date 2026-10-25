# -*- coding: utf-8 -*-
"""E2 成本语义：token 用量 → 金额（估算），以及价目表的严格加载。

三条约束，与项目其他部分保持一致：

1. **不内置任何供应商价格**。价格与计价单位会变，内置一张表等于把过期数据当事实；
   价目表必须由使用者显式提供（`--price-table` 或 `MTSA_PRICE_TABLE`）。
2. **没有表就明确失败，不静默返回 0**。0 会被读成「免费」，比报错危险得多。
3. **金额永远是估算**，返回体带 `estimated=True`、价目表路径与生效日期；
   金额只用于展示与预算提示，不参与任何统计计算或结论判定。

价目表格式（JSON）：

    {
      "currency": "CNY",
      "unit_tokens": 1000000,
      "input": 2.0,
      "output": 8.0,
      "cached_input": 0.5,
      "effective_date": "2026-10-01",
      "source": "供应商公开价目页（填写链接）"
    }

`input` / `output` / `cached_input` 的单位是「每 `unit_tokens` 个 token 的价格」。
`unit_tokens` 必须显式给出：按 1K 还是 1M 报价差 1000 倍，猜错就是灾难。
`cached_input` 可省略；省略时缓存命中的 token 按 `input` 计价，并在结果里标注。

`model` / `tier` / `note` 可选，但强烈建议填：

- `model`：这份价格对应哪个模型。填了就会与实际调用的模型比对，不一致时在结果里
  标 `model_match=false` 并进 warnings——价格挂错模型比不填更危险。
- `tier`：同一模型的分档（如 `peak` / `off_peak`）。峰谷价必须拆成两份表，
  不能取平均，也不能只写一档然后假装通用。
- `note`：一句话说明（如「高峰价，空闲时段约为其一半」）。

预算：`--max-cost` / `MTSA_MAX_COST` 只**标记**超支，不中止运行。与之相对，
token 预算（`--agent-max-tokens`）会在下一次调用前硬停——两者的语义不同，
不要混用：token 预算防跑飞，金额预算做对账提示。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional

ENV_PRICE_TABLE = "MTSA_PRICE_TABLE"
ENV_MAX_COST = "MTSA_MAX_COST"


class PriceTableError(ValueError):
    """价目表缺失或非法。调用方必须把这条错误暴露给用户，不得吞掉。"""


class CostError(ValueError):
    """用量数据不足以计价。宁可报错也不估算。"""


@dataclass(frozen=True)
class PriceTable:
    currency: str
    unit_tokens: int
    input_per_unit: float
    output_per_unit: float
    cached_input_per_unit: Optional[float]
    source: str
    effective_date: Optional[str]
    path: str
    model: Optional[str] = None
    tier: Optional[str] = None
    note: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "unit_tokens": self.unit_tokens,
            "input": self.input_per_unit,
            "output": self.output_per_unit,
            "cached_input": self.cached_input_per_unit,
            "source": self.source,
            "effective_date": self.effective_date,
            "path": self.path,
            "model": self.model,
            "tier": self.tier,
            "note": self.note,
        }


def _require_number(doc: dict[str, Any], key: str, path: str, *, positive: bool) -> float:
    if key not in doc:
        raise PriceTableError(
            f"价目表缺少字段 {key}：{path}。计价单位与单价必须显式写出，不做默认值猜测。")
    raw = doc[key]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise PriceTableError(f"价目表字段 {key} 必须是数字，实际为 {type(raw).__name__}：{path}")
    value = float(raw)
    if positive and value <= 0:
        raise PriceTableError(f"价目表字段 {key} 必须大于 0，实际为 {value}：{path}")
    if not positive and value < 0:
        raise PriceTableError(f"价目表字段 {key} 不能为负数，实际为 {value}：{path}")
    return value


def load_price_table(path: str) -> PriceTable:
    """读取并校验价目表；任何问题都抛 `PriceTableError`，绝不返回默认价。"""
    if not path or not str(path).strip():
        raise PriceTableError(
            f"价目表路径为空。请用 --price-table 或环境变量 {ENV_PRICE_TABLE} 指定；"
            "未配置价格时不做金额换算。")
    p = os.path.abspath(os.path.expanduser(str(path).strip()))
    if not os.path.isfile(p):
        raise PriceTableError(
            f"价目表不存在：{p}。未配置价格时不做金额换算，也不按 0 计价。")
    try:
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
    except json.JSONDecodeError as exc:
        raise PriceTableError(f"价目表不是合法 JSON：{p}：{exc}") from exc
    except OSError as exc:
        raise PriceTableError(f"价目表无法读取：{p}：{exc}") from exc
    if not isinstance(doc, dict):
        raise PriceTableError(f"价目表顶层必须是对象，实际为 {type(doc).__name__}：{p}")

    currency = str(doc.get("currency") or "").strip()
    if not currency:
        raise PriceTableError(f"价目表缺少 currency（如 CNY、USD）：{p}")

    if "unit_tokens" not in doc:
        raise PriceTableError(
            f"价目表缺少 unit_tokens（按多少 token 报价，如 1000000）：{p}")
    unit_tokens = doc["unit_tokens"]
    if isinstance(unit_tokens, bool) or not isinstance(unit_tokens, int) or unit_tokens <= 0:
        raise PriceTableError(f"价目表 unit_tokens 必须是正整数，实际为 {unit_tokens!r}：{p}")

    input_per_unit = _require_number(doc, "input", p, positive=False)
    output_per_unit = _require_number(doc, "output", p, positive=False)
    cached = doc.get("cached_input")
    if cached is None:
        cached_input_per_unit = None
    else:
        cached_input_per_unit = _require_number(doc, "cached_input", p, positive=False)

    source = str(doc.get("source") or "").strip()
    if not source:
        raise PriceTableError(
            f"价目表缺少 source（价格出处，如供应商价目页链接）：{p}。"
            "没有出处的价格无法复核，不能用于对外承诺。")
    effective = doc.get("effective_date")

    def _opt_str(key: str) -> Optional[str]:
        raw = doc.get(key)
        text = str(raw).strip() if raw is not None else ""
        return text or None

    return PriceTable(
        currency=currency,
        unit_tokens=int(unit_tokens),
        input_per_unit=input_per_unit,
        output_per_unit=output_per_unit,
        cached_input_per_unit=cached_input_per_unit,
        source=source,
        effective_date=str(effective).strip() if effective else None,
        path=p,
        model=_opt_str("model"),
        tier=_opt_str("tier"),
        note=_opt_str("note"),
    )


def price_table_path(env: Optional[dict[str, str]] = None) -> Optional[str]:
    """从环境变量取价目表路径；未配置返回 None（不是空字符串）。"""
    env = os.environ if env is None else env
    raw = (env.get(ENV_PRICE_TABLE, "") or "").strip()
    return raw or None


def _tokens(usage: dict[str, Any], key: str) -> Optional[int]:
    raw = usage.get(key)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = int(raw)
    return value if value >= 0 else None


def estimate_cost(usage: dict[str, Any], table: PriceTable) -> dict[str, Any]:
    """按用量分项计价。

    `prompt_tokens` / `completion_tokens` 缺一不可：只有总数时分不了输入输出价，
    这里选择报错，而不是拿总数乘一个平均数冒充精确值。
    """
    if not isinstance(usage, dict):
        raise CostError("usage 不是字典，无法计价")
    prompt = _tokens(usage, "prompt_tokens")
    completion = _tokens(usage, "completion_tokens")
    if prompt is None or completion is None:
        raise CostError(
            "usage 缺少 prompt_tokens 或 completion_tokens，无法分项计价（不做估算）")
    cached = _tokens(usage, "cached_tokens") or 0
    cached = min(cached, prompt)
    billable_input = prompt - cached

    unit = float(table.unit_tokens)
    cached_price = (table.input_per_unit if table.cached_input_per_unit is None
                    else table.cached_input_per_unit)
    input_cost = billable_input / unit * table.input_per_unit
    cached_cost = cached / unit * cached_price
    output_cost = completion / unit * table.output_per_unit
    total = input_cost + cached_cost + output_cost

    breakdown = {
        "input": round(input_cost, 6),
        "cached_input": round(cached_cost, 6),
        "output": round(output_cost, 6),
    }
    return {
        "estimated": True,
        "currency": table.currency,
        "unit_tokens": table.unit_tokens,
        "tokens": {"prompt": prompt, "cached": cached, "completion": completion},
        "breakdown": breakdown,
        "total": round(total, 6),
        "price": table.to_dict(),
        "cached_price_fallback": table.cached_input_per_unit is None,
        "note": "按 token 用量与给定价目表估算，不等于供应商账单；金额不参与任何统计结论",
    }


# ---------------- 预算与模型绑定 ----------------

def max_cost_from_env(env: Optional[dict[str, str]] = None) -> Optional[float]:
    """从环境变量取金额上限；未配置返回 None，格式错误直接报错（不静默忽略）。"""
    env = os.environ if env is None else env
    raw = (env.get(ENV_MAX_COST, "") or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise CostError(f"{ENV_MAX_COST} 不是数字：{raw!r}") from None
    if value <= 0:
        raise CostError(f"{ENV_MAX_COST} 必须大于 0，实际为 {value}")
    return value


def model_matches(table: PriceTable, model: Optional[str]) -> Optional[bool]:
    """价目表声明的模型与实际调用的模型是否一致；任一方缺失时返回 None（未判定）。"""
    if not table.model or not model:
        return None
    return table.model.strip().lower() == str(model).strip().lower()


def check_budget(cost: dict, limit: float) -> dict:
    """金额预算检查：只标记，不中止。

    与 token 预算的区别要写进报告：token 预算在下一次调用前硬停，金额预算在
    跑完之后对账。用金额去硬停会逼出一个「花到多少就中断」的运行状态，
    报告不完整反而更难解释；用户明确选择了「跑完再标记」。
    """
    if limit is None or float(limit) <= 0:
        raise CostError(f"金额预算必须大于 0，实际为 {limit!r}")
    total = cost.get("total")
    if total is None:
        return {"enabled": True, "checked": False, "reason": "no_amount",
                "limit": round(float(limit), 6), "policy": "mark_only"}
    limit_f = float(limit)
    total_f = float(total)
    exceeded = total_f > limit_f
    return {
        "enabled": True,
        "checked": True,
        "limit": round(limit_f, 6),
        "currency": cost.get("currency"),
        "exceeded": exceeded,
        "over_by": round(max(0.0, total_f - limit_f), 6),
        "remaining": round(max(0.0, limit_f - total_f), 6),
        "policy": "mark_only",
    }
