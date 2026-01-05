# Security & Compliance

Scope
- Verification engine only; generation guardrails are minimal (LLM prompts are hardened, but no full red-team layer).
- Provenance is required for trust; populate `provenance.source_id/source_type`, and prefer primary/secondary sources.

Reporting vulnerabilities
- See `SECURITY.md` for responsible disclosure process.

Practices and guidance
- Prompt safety: LLM prompts wrap content in inert tags and request JSON. Add your own provider-side guardrails/red-team filters.
- Source policy: Use `SourcePolicy` and gating to block TERTIARY or disallowed domains.
- PII/secrets: Do not feed secrets; scrub logs and traces if you include sensitive data. Add scrubbing in your LoggerSink/MetricsSink if needed.
- Provenance integrity: Ensure adapters set `retrieved_at`, `chunk_id`, and stable `source_id`.
- Rate limits/budgets: Use `BudgetedProvider` and `RetryingProvider` for LLM calls; use `max_concurrent_tasks` in async runner to avoid overload; apply source policy to reduce risky inputs.

Optional dependencies and isolation
- LLM/DB backends are pluggable; keep credentials out of code/config and inject via env/secret manager.

