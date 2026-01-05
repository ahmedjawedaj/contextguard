"""
Circuit-breaker wrapper for LLM providers.

Design:
- Decorator over any `LLMProviderBase`.
- Trips the circuit after N consecutive failures; half-open after a cool-down.
- Optional rate-limit guard via asyncio semaphore for concurrent calls.
- Override-friendly hooks for logging and time.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional

from ..verify.judges import LLMProviderBase


class CircuitBreakerProvider(LLMProviderBase):
    """Adds circuit-breaker (and optional rate-limit) to an LLM provider."""

    def __init__(
        self,
        provider: LLMProviderBase,
        *,
        failure_threshold: int = 5,
        reset_timeout: float = 5.0,
        max_in_flight: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        time_fn: Optional[Callable[[], float]] = None,
    ):
        self.provider = provider
        self.failure_threshold = max(1, failure_threshold)
        self.reset_timeout = reset_timeout
        self.logger = logger or logging.getLogger(__name__)
        self.time_fn = time_fn or time.time
        self.state = "closed"
        self.fail_count = 0
        self.opened_at: Optional[float] = None
        self.semaphore = asyncio.Semaphore(max_in_flight) if max_in_flight else None

    def _now(self) -> float:
        return float(self.time_fn())

    def _log(self, level: str, msg: str):
        fn = getattr(self.logger, level, self.logger.info)
        fn(msg)

    def _enter(self):
        if self.state == "open":
            if self.opened_at is not None and (self._now() - self.opened_at) >= self.reset_timeout:
                self.state = "half_open"
                self._log("info", "LLM circuit half-open")
            else:
                raise RuntimeError("LLM circuit open")

    def _on_success(self):
        self.fail_count = 0
        if self.state in ("half_open", "open"):
            self._log("info", "LLM circuit closed")
        self.state = "closed"
        self.opened_at = None

    def _on_failure(self):
        self.fail_count += 1
        if self.fail_count >= self.failure_threshold:
            if self.state != "open":
                self._log("warning", "LLM circuit opened")
            self.state = "open"
            self.opened_at = self._now()

    def complete_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        self._enter()

        async def _bounded_call():
            if self.semaphore:
                async with self.semaphore:
                    return self.provider.complete_json(prompt, schema, temperature)
            return self.provider.complete_json(prompt, schema, temperature)

        try:
            if self.semaphore:
                result = asyncio.get_event_loop().run_until_complete(_bounded_call())
            else:
                result = self.provider.complete_json(prompt, schema, temperature)
            self._on_success()
            return result
        except Exception as e:  # pragma: no cover - defensive
            self._on_failure()
            self._log("error", f"LLM provider failure: {e}")
            raise

