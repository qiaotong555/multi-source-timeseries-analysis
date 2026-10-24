# -*- coding: utf-8 -*-
"""OpenAI 兼容的最小 LLM 客户端。

只为「计划层」服务：一次请求、一个 JSON 对象、失败即抛错交给上层回退。
因此不引入 openai / httpx 等依赖，直接用标准库 urllib。

设计取舍
--------
1. **不做重试**。重试逻辑属于计划层的兜底链，客户端只负责「发一次、拿结果」。
   客户端内部偷偷重试会让上层无法统计真实的失败次数。
2. **只接受 JSON 对象**。请求时声明 response_format=json_object，避免模型输出
   带 markdown 代码块的自由文本；解析失败按失败处理，由上层回退到规则。
3. **超时必设**。无超时的外部调用会把整条流水线挂在网络问题上。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional


class LLMError(RuntimeError):
    """LLM 调用失败：网络、鉴权、超时、返回非 JSON，均归入此类。"""


class OpenAICompatClient:
    """OpenAI 兼容的对话补全客户端（DeepSeek、通义、Kimi、智谱同属此格式）。"""

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = 30.0, temperature: float = 0.0) -> None:
        if not base_url:
            raise ValueError("base_url 不能为空")
        if not api_key:
            raise ValueError("api_key 不能为空")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        # 温度固定为 0：计划层要的是可复现，不是多样性
        self.temperature = temperature
        # token 用量：只做累加，不改请求。评估要报成本就必须有真实来源，
        # 不能靠字数估算冒充账单。
        self.calls = 0
        self.usage_totals = {"prompt_tokens": 0, "completion_tokens": 0,
                             "total_tokens": 0, "cached_tokens": 0}
        # 单次调用的最大 token 数：预算前置检查要用它估计「下一次调用大概要花多少」，
        # 用均值会低估长上下文那一步，进而超限。
        self.max_call_tokens = 0

    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        """发起一次对话，返回解析后的 JSON 对象；任何异常都转成 LLMError。"""
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=raw,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:                      # noqa: BLE001
                pass
            raise LLMError(f"HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMError(f"网络或超时：{exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMError(f"响应不是合法 JSON：{exc}") from exc

        self._record_usage(payload)
        return self._extract_json(payload)

    def _record_usage(self, payload: dict[str, Any]) -> None:
        """累计服务端返回的 usage；缺失时只记调用次数，不猜 token。"""
        self.calls += 1
        usage = payload.get("usage") if isinstance(payload, dict) else None
        if not isinstance(usage, dict):
            return
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            try:
                self.usage_totals[key] += int(usage.get(key) or 0)
            except (TypeError, ValueError):
                continue
        try:
            call_total = int(usage.get("total_tokens") or 0)
        except (TypeError, ValueError):
            call_total = 0
        self.max_call_tokens = max(self.max_call_tokens, call_total)
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            try:
                self.usage_totals["cached_tokens"] += int(details.get("cached_tokens") or 0)
            except (TypeError, ValueError):
                pass

    def usage_summary(self) -> Optional[dict[str, Any]]:
        """返回本次客户端生命周期内的累计用量；服务端没给 usage 时返回 None。"""
        if not self.calls or not self.usage_totals["total_tokens"]:
            return None
        return {"calls": self.calls, "max_call_tokens": self.max_call_tokens,
                **self.usage_totals}

    @staticmethod
    def _extract_json(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"响应结构不符合预期：{str(payload)[:200]}") from exc
        if not isinstance(content, str):
            raise LLMError("响应 content 不是字符串")
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"content 不是合法 JSON：{content[:200]}") from exc
        if not isinstance(data, dict):
            raise LLMError(f"content 顶层必须是对象，实际为 {type(data).__name__}")
        return data


def client_from_env(env: Optional[dict[str, str]] = None) -> Optional["OpenAICompatClient"]:
    """按环境变量构造客户端；配置不全返回 None，由调用方回退到规则计划。"""
    import os

    env = env if env is not None else os.environ
    base_url = env.get("MTSA_LLM_BASE_URL", "").strip()
    api_key = env.get("MTSA_LLM_API_KEY", "").strip()
    if not base_url or not api_key:
        return None
    return OpenAICompatClient(
        base_url=base_url,
        api_key=api_key,
        model=env.get("MTSA_LLM_MODEL", "deepseek-chat").strip() or "deepseek-chat",
        timeout=float(env.get("MTSA_LLM_TIMEOUT", "30") or 30),
    )
