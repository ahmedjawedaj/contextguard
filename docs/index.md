# ContextGuard

State-contracted verification and explainable retrieval gating for multi-turn RAG.

What it is
- A constraint-first verification engine: plan → retrieve → gate → judge → aggregate → report/context pack.
- Contracts and types: `StateSpec`, `Claim`, `Chunk`/`Provenance`, `ReasonCode`.
- Outputs: verdict report (primary), context pack (secondary), trace DAG (explainability).

Why it exists
- Similarity ≠ relevance under constraints. Wrong-year/entity/source chunks cause confident but wrong answers.
- ContextGuard hard-gates evidence using state constraints and reason codes, making failures visible and auditable.

Key capabilities
- Planner (support + counter), hard gating (entity/time/source policy/diversity), judges (rule/LLM/NLI), aggregation with source priority, numeric/time safeguards, trace DAG, rich reports.
- Adapters: LLM providers (OpenAI + budget/retry), retrievers (LangChain, LlamaIndex, Chroma, Qdrant), stores (SQLite, S3), async pipeline, generation scaffold.

Who it’s for
- Teams building RAG/agent systems that must be constraint-aware, auditable, and reproducible.

