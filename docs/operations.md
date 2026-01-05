# Operations & Hardening

## Budgets and guardrails
- Retrieval: clamp claims and total_k; enforce MAX_JUDGE_CHUNKS_PER_CLAIM and MAX_JUDGE_TEXT_LEN.
- LLM: enforce prompt/output budgets with `BudgetedProvider`; wrap with `RetryingProvider` for retry/backoff/jitter.
- Planner/gate/judge respect domain profiles for stricter defaults.

## Logging/metrics
- Async runner: pass `logger` or `Instrumentation` to `async_run_verification`; emits events for plan/retrieve/gate/judge/aggregate plus errors.
- Wrap providers with your own logger or subclass `RetryingProvider`/`BudgetedProvider` to emit metrics.
- Trace: use `TraceGraph`/`TraceBuilder` to capture every decision; export DOT for audits.

## Error handling
- Retrieval/judge exceptions in async runner are logged (if logger provided) and re-raised.
- Circuit-breaker wrappers available: `CircuitBreakerProvider`, `CircuitBreakerRetriever`; `RetryingProvider`/`RetryingRetriever` for retry/backoff.
- Keep provider wrappers thin; layer retries/backoff externally where possible.

## Security notes
- Prompt hardening in `LLMJudge` prompts; still apply your own red-team/guardrails at provider edges.
- Enforce source policy via `SourcePolicy` and gating; populate provenance to avoid silent acceptance.

## Performance
- Use async runner to parallelize retrieval + judge; limit concurrency via `max_concurrent_tasks`.
- Use backend-side filtering via `CanonicalFilters` to reduce post-filter load; optional async retriever avoids threadpools.
- Cache retrieval if backend supports it (see RetrieverBase cache flag); dedup/rerank with `retrieve.rerank` helpers.

