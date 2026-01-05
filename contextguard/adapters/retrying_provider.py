"""
Retrying/logging wrapper for LLM providers.

Patterns:
- Decorator/strategy: wraps any `LLMProvider` and adds retry with exponential
  backoff and jitter, plus structured logging.
- Composable: can be stacked with other providers (e.g., OpenAIProvider, your
  custom provider).

Usage:
    base = OpenAIProvider(model="gpt-4o-mini")
    llm = RetryingProvider(base, max_attempts=3, base_delay=0.5)
    judge = LLMJudge(llm)

Customization:
- Override `_sleep` for testability.
- Override `_log` to integrate with your observability stack.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, Optional

from ..verify.judges import LLMProviderBase


class RetryingProvider(LLMProviderBase):
    """
    Wraps an `LLMProvider` with retry/backoff and logging.
    """

    def __init__(
        self,
        provider: LLMProviderBase,
        *,
        max_attempts: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 4.0,
        logger: Optional[logging.Logger] = None,
    ):
        self.provider = provider
        self.max_attempts = max(1, max_attempts)
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.logger = logger or logging.getLogger(__name__)

    def complete_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        attempt = 0
        last_error: Optional[Exception] = None
        while attempt < self.max_attempts:
            attempt += 1
            try:
                self._log("info", f"LLM call attempt {attempt}/{self.max_attempts}")
                return self.provider.complete_json(prompt, schema, temperature)
            except Exception as e:  # pragma: no cover - defensive
                last_error = e
                if attempt >= self.max_attempts:
                    self._log("error", f"LLM call failed after {attempt} attempts: {e}")
                    raise
                delay = self._compute_delay(attempt)
                self._log("warning", f"LLM call failed (attempt {attempt}), retrying in {delay:.2f}s: {e}")
                self._sleep(delay)
        # Should not reach here
        if last_error:
            raise last_error
        return {}

    # ------------------------------------------------------------------
    # Internal helpers (override for testing/customization)
    # ------------------------------------------------------------------
    def _compute_delay(self, attempt: int) -> float:
        exp = self.base_delay * (2 ** (attempt - 1))
        jitter = random.uniform(0, self.base_delay)
        return min(exp + jitter, self.max_delay)

    def _sleep(self, delay: float) -> None:
        time.sleep(delay)

    def _log(self, level: str, msg: str) -> None:
        log_fn = getattr(self.logger, level, self.logger.info)
        log_fn(msg)

