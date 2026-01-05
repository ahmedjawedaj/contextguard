"""
Lightweight instrumentation hooks for logging and metrics.

Design:
- Protocol-style sink interfaces for logs and metrics.
- `Instrumentation` can be passed through pipeline stages; each stage emits
  structured events without depending on a specific logging/metrics backend.

Customization:
- Implement `LoggerSink` and/or `MetricsSink` with your logging/metrics stack.
- Extend event names/payloads as needed; this is intentionally minimal and
  non-intrusive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Optional


class LoggerSink(Protocol):
    """A sink that accepts structured log events."""

    def emit(self, event: str, payload: Dict[str, Any]) -> None:
        ...


class MetricsSink(Protocol):
    """A sink that accepts metric increments/timing events."""

    def increment(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        ...

    def timing(self, name: str, value_ms: float, tags: Optional[Dict[str, str]] = None) -> None:
        ...


@dataclass
class Instrumentation:
    """
    Container for optional logger/metrics sinks.

    All sinks are optional; stages should check before emitting.
    """

    logger: Optional[LoggerSink] = None
    metrics: Optional[MetricsSink] = None

    def log(self, event: str, payload: Dict[str, Any]) -> None:
        if self.logger:
            self.logger.emit(event, payload)

    def inc(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        if self.metrics:
            self.metrics.increment(name, value=value, tags=tags)

    def timing(self, name: str, value_ms: float, tags: Optional[Dict[str, str]] = None) -> None:
        if self.metrics:
            self.metrics.timing(name, value_ms=value_ms, tags=tags)

