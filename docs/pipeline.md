# Pipeline (plan → retrieve → gate → judge → aggregate → report/context pack)

## Planner
- `plan_retrieval` generates support + counter queries per claim.
- Respects budgets (MAX_TOTAL_K, MAX_CLAIMS) and domain profiles.
- Emits trace nodes for plan/steps.

## Retriever
- Interface: `Retriever.search(query, filters, k) -> List[Chunk]`.
- Optional async interface: `AsyncRetriever.asearch(...)`; async runner uses it when available.
- Filters: `CanonicalFilters` derived from `StateSpec` (entity_ids, year, source types, doc_type).
- Adapters provided: LangChain, LlamaIndex, Chroma, Qdrant; implement your own by returning `Chunk` with `provenance` populated.

## Gate
- Hard eligibility: entity/time match, source policy, freshness.
- Quality/noise: length, boilerplate.
- Diversity: per-source, per-domain, per-doc-type caps.
- Reason codes for every rejection; accepted/rejected evidence recorded in trace.

## Judge
- Rule-based (`RuleBasedJudge`), LLM (`LLMJudge`), NLI (`NLIJudge`).
- LLM provider is pluggable (`LLMProviderBase`); budgets and retries via `BudgetedProvider` + `RetryingProvider`.
- Caps chunks per claim and text length to control cost/latency.

## Aggregate
- Per-claim: support/contradict scores, coverage, reasons, confidence; primary-source contradictions prioritized.
- Overall: weighted roll-up, critical claim handling, warnings.
- Coverage now tracks doc_types when present.

## Trace
- `TraceBuilder` emits nodes for plan, retrieval, gate decisions, judge calls, evidence assessments, claim verdicts, report/context pack.
- `TraceGraph.to_dot()` for Graphviz; rejected vs accepted visible.

## Outputs
- VerdictReport: JSON + Markdown (citations, accepted/rejected evidence, retrieval plan, metadata).
- ContextPack: facts-first payload for guarded generation.

