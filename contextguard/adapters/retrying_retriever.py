"""
Retrying wrapper for retrievers.

Design:
- Decorator around any `Retriever` to add retry/backoff/jitter and optional logging.
- Keeps Retriever protocol intact; all customization via constructor args or subclassing.

Customization:
- Override `_sleep` for testing.
- Override `_log` to integrate with your logging stack.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Optional

from ..retrieve.protocols import Retriever, CanonicalFilters


class RetryingRetriever(Retriever):
    """Adds retry/backoff to any Retriever."""

    def __init__(
        self,
        retriever: Retriever,
        *,
        max_attempts: int = 3,
        base_delay: float = 0.2,
        max_delay: float = 2.0,
        logger: Optional[logging.Logger] = None,
    ):
        self.retriever = retriever
        self.max_attempts = max(1, max_attempts)
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.logger = logger or logging.getLogger(__name__)

    def search(self, query: str, *, filters: Optional[CanonicalFilters] = None, k: int = 10):
        attempt = 0
        last_error: Optional[Exception] = None
        while attempt < self.max_attempts:
            attempt += 1
            try:
                self._log("info", f"retriever attempt {attempt}/{self.max_attempts}")
                return self.retriever.search(query, filters=filters, k=k)
            except Exception as e:  # pragma: no cover - defensive
                last_error = e
                if attempt >= self.max_attempts:
                    self._log("error", f"retriever failed after {attempt} attempts: {e}")
                    raise
                delay = self._compute_delay(attempt)
                self._log("warning", f"retriever failed (attempt {attempt}), retrying in {delay:.2f}s: {e}")
                self._sleep(delay)
        if last_error:
            raise last_error
        return []

    def _compute_delay(self, attempt: int) -> float:
        exp = self.base_delay * (2 ** (attempt - 1))
        jitter = random.uniform(0, self.base_delay)
        return min(exp + jitter, self.max_delay)

    def _sleep(self, delay: float) -> None:
        time.sleep(delay)

    def _log(self, level: str, msg: str) -> None:
        log_fn = getattr(self.logger, level, self.logger.info)
        log_fn(msg)

