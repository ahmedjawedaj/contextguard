# FAQ

**Q: Why is evidence rejected?**  
Check `GateDecision.reasons` (ReasonCode). Common: CTXT_ENTITY_MISMATCH, CTXT_TIME_MISMATCH, CTXT_SOURCE_POLICY_VIOLATION, EVIDENCE_DUPLICATE.

**Q: My adapter returns chunks but gating rejects them.**  
Populate `entity_ids`, `year`, `provenance.source_type`, and (optionally) `metadata.doc_type`. Use `CanonicalFilters.from_state_spec(state)` to push constraints into your backend.

**Q: How do I control LLM cost/latency?**  
Use `BudgetedProvider` (prompt/output caps), `RetryingProvider` (limited retries), judge budgets (`MAX_JUDGE_CHUNKS_PER_CLAIM`, `MAX_JUDGE_TEXT_LEN`).

**Q: Can I use my own LLM?**  
Yes. Implement `LLMProviderBase.complete_json` and pass to `LLMJudge`. Decorate with budget/retry wrappers.

**Q: How do I see why a verdict happened?**  
Use `TraceBuilder` to capture the run; export DOT via `TraceGraph.to_dot()` and inspect accepted/rejected evidence nodes.

**Q: Optional dependencies missing (e.g., qdrant_client/chromadb/boto3)?**  
Install extras: `contextguard[qdrant]`, `contextguard[chroma]`, `contextguard[cloud]`, `contextguard[llm]`.

