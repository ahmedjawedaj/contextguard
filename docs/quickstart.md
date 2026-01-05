# Quickstart

## Install
Base:
```bash
pip install llm-contextguard
```
Optional extras:
```bash
pip install llm-contextguard[llm]      # OpenAI provider
pip install llm-contextguard[qdrant]   # Qdrant adapter
pip install llm-contextguard[chroma]   # Chroma adapter
pip install llm-contextguard[cloud]    # S3 store
```

## Minimal sync example
```python
from contextguard import (
    StateSpec, EntityRef, TimeConstraint,
    Claim, MockRetriever, RuleBasedJudge,
    plan_retrieval, gate_chunks, aggregate_claim, aggregate_overall,
)

state = StateSpec(thread_id="t1", entities=[EntityRef(entity_id="acme")], time=TimeConstraint(year=2024))
claim = Claim(claim_id="c1", text="ACME 2024 revenue was $200M.", entities=["acme"], time=TimeConstraint(year=2024))

retriever = MockRetriever()
retriever.add_chunk("ACME 2024 revenue was $200M according to its audited annual report.",
    source_id="annual_report_2024", source_type="PRIMARY", entity_ids=["acme"], year=2024)

plan = plan_retrieval([claim], state, total_k=5)
chunks = []
for step in plan.steps:
    chunks.extend(retriever.search(step.query, filters=step.filters, k=step.k))

gated = gate_chunks(chunks, state)
accepted = [g.chunk for g in gated if g.accepted]

judge = RuleBasedJudge()
jr = judge.score_batch(claim, accepted, state)
claim_verdict = aggregate_claim(claim, jr)
overall_label, overall_conf, _ = aggregate_overall([claim_verdict])
```

## Async example (plan → retrieve → gate → judge → aggregate)
```python
import asyncio
from contextguard import async_run_verification, StateSpec, EntityRef, TimeConstraint, Claim, MockRetriever, RuleBasedJudge

async def main():
    state = StateSpec(thread_id="async", entities=[EntityRef(entity_id="acme")], time=TimeConstraint(year=2024))
    claim = Claim(claim_id="c1", text="ACME 2024 revenue was $200M.", entities=["acme"], time=TimeConstraint(year=2024))
    retriever = MockRetriever()
    retriever.add_chunk("ACME 2024 revenue was $200M according to its audited annual report.",
        source_id="annual_report_2024", source_type="PRIMARY", entity_ids=["acme"], year=2024)
    judge = RuleBasedJudge()
    overall_label, overall_conf, cvs = await async_run_verification([claim], state, retriever, judge=judge)
    print(overall_label, overall_conf)

asyncio.run(main())
```

## LLM judge with budgets + retries
```python
from contextguard import OpenAIProvider, BudgetedProvider, RetryingProvider, LLMJudge

base = OpenAIProvider(model="gpt-4o-mini", max_prompt_chars=8000, max_output_tokens=300, timeout=30)
llm = RetryingProvider(BudgetedProvider(base, max_prompt_chars=8000, max_output_tokens=300), max_attempts=3)
judge = LLMJudge(llm)
```

