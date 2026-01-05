"""
Budgeted provider for `LLMJudge` (decorator over `LLMProvider`).

Features:
- Enforces max prompt length (in characters) and max output tokens before
  calling the underlying provider.
- Optional logging for budget violations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import logging

from ..verify.judges import LLMProviderBase


class BudgetedProvider(LLMProviderBase):
    """
    Wraps an `LLMProvider` and enforces prompt/output budgets.
    """

    def __init__(
        self,
        provider: LLMProviderBase,
        *,
        max_prompt_chars: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.provider = provider
        self.max_prompt_chars = max_prompt_chars
        self.max_output_tokens = max_output_tokens
        self.logger = logger or logging.getLogger(__name__)

    def complete_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        if self.max_prompt_chars and len(prompt) > self.max_prompt_chars:
            msg = f"Prompt length {len(prompt)} exceeds max_prompt_chars={self.max_prompt_chars}"
            self.logger.warning(msg)
            raise ValueError(msg)

        # For providers that accept max_output_tokens, attach via schema hint or attr
        # If the underlying provider exposes `max_output_tokens`, set attribute temporarily.
        if hasattr(self.provider, "max_output_tokens") and self.max_output_tokens:
            prev = getattr(self.provider, "max_output_tokens", None)
            try:
                setattr(self.provider, "max_output_tokens", self.max_output_tokens)
                return self.provider.complete_json(prompt, schema, temperature)
            finally:
                setattr(self.provider, "max_output_tokens", prev)

        return self.provider.complete_json(prompt, schema, temperature)

