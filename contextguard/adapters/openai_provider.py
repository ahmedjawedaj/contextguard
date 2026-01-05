"""
OpenAI provider for `LLMJudge` (implements `LLMProvider` protocol).

Design (strategy pattern):
- Implements the `LLMProvider` protocol used by `LLMJudge`.
- Minimal JSON-only call to OpenAI Chat Completions.
- `build_messages` is overrideable to customize system/user prompts.

Usage:
    from contextguard.adapters.openai_provider import OpenAIProvider
    from contextguard import LLMJudge
    llm = OpenAIProvider(model="gpt-4o-mini")
    judge = LLMJudge(llm)

Customization:
- Subclass and override `build_messages` to inject domain/system prompts.
- You can also wrap this provider with your own retry/backoff/debias layer.

Notes:
- Optional dependency: requires `openai>=1.0.0`.
- Network calls are not retried here; wrap externally if needed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..verify.judges import LLMProviderBase


class OpenAIProvider(LLMProviderBase):
    """
    Thin wrapper over the OpenAI chat completion API.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        max_prompt_chars: Optional[int] = None,
    ):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("OpenAIProvider requires `openai` package >=1.0.0") from e

        self.client = OpenAI(api_key=api_key, base_url=base_url, default_headers=extra_headers)
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.max_prompt_chars = max_prompt_chars

    def complete_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Returns parsed JSON according to the judge's schema.
        """
        if self.max_prompt_chars and len(prompt) > self.max_prompt_chars:
            raise ValueError(f"Prompt exceeds max_prompt_chars={self.max_prompt_chars}")
        messages = self.build_messages(prompt)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            response_format={"type": "json_object"},
            timeout=self.timeout,
            max_tokens=self.max_output_tokens,
        )
        content = resp.choices[0].message.content or "{}"
        import json

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}

    def build_messages(self, prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": "You are a careful, JSON-only function."},
            {"role": "user", "content": prompt},
        ]

