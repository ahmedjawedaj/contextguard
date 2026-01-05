# Concepts

## Contracts
- **StateSpec**: constraint contract (entities, time, metric, units, source policy, language). Drives retrieval filters and gating.
- **Claim**: atomic fact to verify (text, entities, time, units, weight, critical).
- **Chunk**: retrieved evidence with `provenance` (source_id/source_type, url/title/domain, timestamps) and extracted metadata (entity_ids, year, doc_type).
- **ReasonCode**: machine-readable reasons for rejections/warnings (CTXT_*, EVIDENCE_*, CLAIM_*, SYS_*).

## Pipeline stages
1) Planner: builds support + counter-evidence queries from claims/state.
2) Retriever: returns `Chunk` objects; can be any backend (vector/search) via `Retriever.search`.
3) Gate: hard eligibility checks (entity/time/source policy), quality filters, diversity controls, reason codes.
4) Judge: support/contradict scoring (rule/LLM/NLI); budgets for evidence count/text length.
5) Aggregate: per-claim and overall verdicts, coverage, confidence; primary-source contradictions prioritized.
6) Trace: micrograd-style DAG with all decisions (retrieval, gate, judge, verdict); exportable to DOT.
7) Outputs: verdict report (JSON/Markdown), context pack for generation.

## Profiles
- DomainProfile presets (finance, policy, enterprise) adjust gating/aggregation strictness and diversity thresholds.

## Budgets and guardrails
- Retrieval: max claims, total_k, chunks per claim.
- Judge: max chunks per claim, max text length.
- LLM: prompt/output budgets via `BudgetedProvider`; retries/backoff via `RetryingProvider`.

## Explainability
- TraceBuilder records nodes for plan, retrieval, gate decisions, judge calls, evidence assessments, claim verdicts, and final report/context pack.
- DOT export for visualization; accepted/rejected evidence visible with reasons.

