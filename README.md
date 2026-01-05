ContextGuard: State-Contracted Verification for Agentic RAG
===========================================================
[![CI](https://img.shields.io/github/actions/workflow/status/ahmedjawedaj/contextguard/ci.yml?branch=main)](https://github.com/ahmedjawedaj/contextguard/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue)](https://ahmedjawedaj.github.io/contextguard/)
[![PyPI](https://img.shields.io/pypi/v/llm-contextguard)](https://pypi.org/project/llm-contextguard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

ContextGuard is a text-only verification and consistency engine. It treats multi-turn RAG and fact-checking like a compiler:

- **StateSpec** = your constraint contract (entities, time, metric, units, source policy).
- **Planner** = builds support + counter-evidence queries.
- **Gate** = hard admission control (reject wrong-year/entity/source chunks with reason codes).
- **Judge** = support/contradict scoring for each claim–evidence pair.
- **Aggregate** = per-claim + overall verdicts with confidence.
- **Trace DAG** = micrograd-style execution graph for full explainability.
- **Report** = SUPPORTED / CONTRADICTED / INSUFFICIENT / MIXED + citations.

Why this exists
---------------
Multi-turn RAG fails because similarity ≠ relevance under constraints. Benchmarks like MTRAG/CORAL (multi-turn drift) and FEVER/SciFact (evidence-required verification) show strong systems still pull wrong-year/entity/source chunks and answer confidently. ContextGuard fixes this by making constraints first-class and rejecting ineligible evidence before generation.

What’s included (v0.1)
----------------------
- **Core contracts:** `StateSpec`, `Claim`, `Chunk`, `Verdict`, `ReasonCode`.
- **Merge engine:** carryover + reset semantics with conflict detection.
- **Planner:** coverage-first retrieval with mandatory counter-evidence queries.
- **Gate:** hard constraint checks (entity, time, source policy), diversity control, noise filtering, reason codes.
- **Judges:** rule-based, LLM-based, and NLI-ready interfaces for support/contradict scoring.
- **Aggregation:** per-claim + overall verdict logic with confidence and coverage signals.
- **Reports:** JSON/Markdown/HTML rendering, plus a facts-first context pack for safe RAG generation.
- **Trace DAG:** micrograd-style execution graph; export to Graphviz DOT/SVG.
- **Storage:** SQLite-backed state/fact/run store (zero-ops).
- **Hero demo:** `examples/05_trace_graphviz.py` generates a report + DOT trace.
- **Resilience:** Retry/budget and circuit-breaker wrappers for LLM providers/retrievers; optional async retriever support; dedup/rerank helpers.

Quick start (local)
-------------------
Clone the repo and run the hero demo (uses only standard lib + pydantic).

```bash
cd contextguard
python examples/05_trace_graphviz.py
```

Outputs (in `examples/output/`):
- `report.md` — verdict report with citations and stats.
- `trace.dot` / `trace.svg` — Graphviz diagram of the full decision DAG.

Install (when published)
------------------------
Standard install (runtime only):
```bash
pip install llm-contextguard
```

From source:
```bash
pip install -e .
```

Optional extras:
```bash
pip install llm-contextguard[demo]        # graphviz for DOT->SVG/PNG rendering
pip install llm-contextguard[nli]         # sentence-transformers for NLIJudge
pip install llm-contextguard[dev]         # ruff + mypy + pytest
```

Programmatic use
----------------
Minimal end-to-end flow (rule-based components):

```python
from contextguard import (
    StateSpec, StateDelta, EntityRef, TimeConstraint,
    merge_state, plan_retrieval, gate_chunks,
    RuleBasedClaimSplitter, RuleBasedJudge,
    ClaimAggregator, OverallAggregator, build_report
)

# 1) Start state and merge user constraints
state = StateSpec(thread_id="t1")
delta = StateDelta(
    entities_add=[EntityRef(entity_id="AAPL")],
    time=TimeConstraint(year=2024),
    metric="revenue",
)
merge_result = merge_state(state, delta, turn_id=1)
state = merge_result.state

# 2) Split claims (rule-based or LLM)
claims = RuleBasedClaimSplitter().split("Apple 2024 revenue will be $400B.")

# 3) Plan retrieval (support + counter)
plan = plan_retrieval(claims, state, total_k=20)

# 4) Retrieve with your own retriever implementing `Retriever.search()`
#    Here, you would call your backend and get `Chunk` objects back.
#    chunks = my_retriever.search(...)

# 5) Gate evidence (hard constraints)
# gated = gate_chunks(chunks, state)

# 6) Judge + aggregate
# judge_results = RuleBasedJudge().score_batch(claims[0], accepted_chunks, state)
# claim_verdict = ClaimAggregator().aggregate(claims[0], judge_results)
# overall_label, overall_conf, warnings = OverallAggregator().aggregate([claim_verdict])

# 7) Build report
# report = build_report(thread_id="t1", state=state,
#                       claim_verdicts=[claim_verdict],
#                       overall_label=overall_label,
#                       overall_confidence=overall_conf)
```

Key concepts
------------
- **StateSpec**: persistent constraints (entities, time, metric, units, source policy). This is the “contract” that gates retrieval.
- **Planner**: issues both support and counter-evidence queries to avoid confirmation bias.
- **Gate**: rejects chunks that violate constraints; enforces diversity; emits reason codes.
- **Judge**: scores claim–evidence pairs for support/contradiction; LLM or rule-based/NLI.
- **Aggregate**: decides SUPPORTED / CONTRADICTED / INSUFFICIENT / MIXED with coverage-aware confidence.
- **Trace DAG**: every step is recorded; exportable to Graphviz for “show me why this fact got in.”

Hero demo (recommended)
-----------------------
`python examples/05_trace_graphviz.py`
- Simulates a 3-turn conversation:
  - “Compare Apple and Microsoft revenue”
  - “Now do 2024 projections”
  - “Only use primary sources”
- Demonstrates constraint carryover, gating, counter-evidence, verdicts, and trace visualization.

Quick eval harness (smoke)
--------------------------
A tiny JSONL fixture is provided: `tests/fixtures/eval.jsonl`

Run baseline:
```bash
python -m contextguard.eval.harness --data tests/fixtures/eval.jsonl --k 5
```

Ablations:
- Disable gating: `--disable-gating`
- Disable counter-evidence: `--disable-counter`

Example (no gating, no counter):
```bash
python -m contextguard.eval.harness --data tests/fixtures/eval.jsonl --k 5 --disable-gating --disable-counter
```

Results (placeholder – fill with real numbers in CI):
- verdict_accuracy: …
- evidence_precision/recall: …
- fever_score: …

Adapters & extensibility
------------------------
- **Retrievers:** implement `Retriever.search(query, filters, k) -> List[Chunk]` for any vector DB / search backend.
- Provided adapters:
  - `LangChainRetrieverAdapter` — wrap any LangChain retriever; override `doc_to_chunk` or subclass to customize provenance/metadata and filter matching.
  - `LlamaIndexRetrieverAdapter` — wrap any LlamaIndex retriever/query engine; override `node_to_chunk` or subclass for richer metadata handling.
- **Judges:** plug your own LLM via `LLMJudge` (structured JSON prompts) or an NLI model via `NLIJudge`.
- **LLM providers:** `OpenAIProvider` implements the `LLMProvider` protocol; override `build_messages` or wrap with your own retry/guard layers. `RetryingProvider` decorates any provider with backoff + logging (strategy/decorator pattern).
- **Budgets:** `BudgetedProvider` enforces prompt/output limits before calling the underlying provider (pair with `RetryingProvider`).
- **Generation (optional):** `LLMGenerator` turns a `ContextPack` + user prompt into a guarded JSON answer using any `LLMProvider`. Override `build_prompt` / `build_schema` or implement the `Generator` protocol for domain-specific pipelines.
- **Stores:** SQLite by default; `S3Store` is provided for S3-compatible buckets; add Postgres/Redis by implementing the `Store` protocol.
- **Async pipeline:** `async_run_verification` runs plan → retrieve → gate → judge → aggregate with asyncio (wrapping sync retrievers/judges via threadpool).
- **Frameworks:** LangChain/LlamaIndex adapters are provided; wrap your retriever to feed `Chunk` objects.

Docs
----
- Built with MkDocs + mkdocstrings. To serve locally:
```bash
pip install llm-contextguard[docs]
mkdocs serve
```

How to integrate a retriever (metadata expectations)
----------------------------------------------------
- Implement `Retriever.search(query, filters, k)` and return `Chunk` objects with **provenance.source_id** and **provenance.source_type** (`PRIMARY`/`SECONDARY`/`TERTIARY`). Default source policy rejects `TERTIARY`.
- Fill structured metadata: `chunk.entity_ids` (list of canonical IDs), `chunk.year` (int), and `metadata.doc_type` if available. Gating relies on `entity_ids`, `year`, and `source_type`; aggregation gives higher weight to primary contradictions.
- Translate filters: use `CanonicalFilters.from_state_spec(state)` to map to your backend (entity/year/source filters). Respect `filters.allowed_source_types`, `filters.year`, and `filters.entity_ids`.
- Domain/profile strictness: `GatingConfig.from_profile(...)` and `AggregationConfig.from_profile(...)` tighten rules (e.g., finance prefers primary + >=2 sources; policy expects primary; enterprise is moderate). Pass the profile to planner/gate/aggregate if you want domain-tuned behavior.
- Provenance timestamps: set `provenance.retrieved_at` (ISO, timezone-aware) and `provenance.chunk_id` if you have stable chunk IDs; they improve reproducibility and trace output.

What’s next (roadmap)
---------------------
- Eval harness on FEVER/SciFact (and multi-turn sets like MTRAG/CORAL).
- Domain profiles (finance/news/policy/enterprise) with pre-tuned gating thresholds.
- Confidence calibration and better rationale spans.
- CI + release automation to PyPI/TestPyPI on tagged releases.

License
-------
MIT — see `LICENSE`.

Contacts
--------
Contributors welcome. Open issues/PRs with trace screenshots and repro steps. 

