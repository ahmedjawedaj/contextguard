# Operations Runbook

Logging & metrics
- Use `Instrumentation` to pass a `logger` (LoggerSink) and `metrics` (MetricsSink).
- Async runner emits events: `plan.built`, `retrieve.batch.ms`, `gate.results`, `judge.score_batch.ms`, `aggregate.overall`, plus error counters.
- Add your own sinks to forward to stdout/JSON logs or metrics backends (Datadog, Prometheus).

Retries and budgets
- LLM: wrap providers with `BudgetedProvider` (prompt/output caps) and `RetryingProvider` (limited retries, backoff).
- Retrieval/judge errors: async runner logs and re-raises; wrap at call site for circuit-breaker/rate-limit if needed.

Resource limits
- Configure planner total_k and judge chunk/text limits via settings; profile-specific configs tighten gating/aggregation.

Tracing
- Use `TraceBuilder` to collect a run; export DOT for audit.

Deployment tips
- Keep optional deps isolated (llm/qdrant/chroma/cloud). Only install what you use.
- Externalize credentials (env/secret manager); do not log PII/secrets. Add scrubbing in your sinks.

