"""
Circuit-breaker and rate-limit wrapper for retrievers.

Design:
- Decorator over any `Retriever`.
- Circuit-breaker trips after N consecutive failures; half-open after a cool-down.
- Optional rate-limit: guard max in-flight calls.
- Override hooks for logging and time (testable).
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional
import asyncio

from ..retrieve.protocols import Retriever, CanonicalFilters


class CircuitBreakerRetriever(Retriever):
    """Adds circuit-breaker and optional rate-limit to a Retriever."""

    def __init__(
        self,
        retriever: Retriever,
        *,
        failure_threshold: int = 5,
        reset_timeout: float = 5.0,
        max_in_flight: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        time_fn: Optional[Callable[[], float]] = None,
    ):
        self.retriever = retriever
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
                self._log("info", "circuit half-open")
            else:
                raise RuntimeError("circuit open")

    def _on_success(self):
        self.fail_count = 0
        if self.state in ("half_open", "open"):
            self._log("info", "circuit closed")
        self.state = "closed"
        self.opened_at = None

    def _on_failure(self):
        self.fail_count += 1
        if self.fail_count >= self.failure_threshold:
            if self.state != "open":
                self._log("warning", "circuit opened")
            self.state = "open"
            self.opened_at = self._now()

    def search(self, query: str, *, filters: Optional[CanonicalFilters] = None, k: int = 10):
        self._enter()

        async def _run():
            if self.semaphore:
                async with self.semaphore:
                    return self.retriever.search(query, filters=filters, k=k)
            return self.retriever.search(query, filters=filters, k=k)

        try:
            if self.semaphore:
                # run in event loop if semaphore is set (rate limit)
                result = asyncio.get_event_loop().run_until_complete(_run())
            else:
                result = self.retriever.search(query, filters=filters, k=k)
            self._on_success()
            return result
        except Exception as e:  # pragma: no cover - defensive
            self._on_failure()
            self._log("error", f"retriever failure: {e}")
            raise

