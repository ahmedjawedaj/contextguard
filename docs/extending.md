# Extending ContextGuard

## LLM providers
- Implement `LLMProviderBase.complete_json(prompt, schema, temperature)` and pass to `LLMJudge`.
- Decorate with `BudgetedProvider` (budgets) and `RetryingProvider` (backoff/logging) as needed.

## Retrievers
- Implement `Retriever.search(query, filters, k)` returning `Chunk` with populated `provenance`, `entity_ids`, `year`, optional `metadata.doc_type`.
- Reuse `CanonicalFilters.from_state_spec(state)` to translate constraints to your backend.
- For template-method convenience, subclass `RetrieverBase` (see LangChain/LlamaIndex/Chroma/Qdrant adapters).

## Stores
- Implement `Store` protocol/abstract base for state, facts, runs, and traces using your database of choice.

## Numeric/units
- Extend `verify.numeric` with domain-specific units/scales; add to `_UNIT_TOKENS` or add new normalization functions.

## Profiles
- Add/adjust domain profiles by extending `GatingConfig.from_profile` and `AggregationConfig.from_profile`.

## Generation
- Implement `Generator` or subclass `LLMGenerator` to add streaming, richer schema, or guardrails.

