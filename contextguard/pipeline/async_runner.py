"""
Async runner for the ContextGuard pipeline (plan → retrieve → gate → judge → aggregate).

Design:
- Uses asyncio to parallelize retrieval across plan steps while keeping the
  existing synchronous components unchanged (wrapped via `asyncio.to_thread`).
- Provides a single entry point `async_run_verification` that mirrors the
  synchronous flow.

Customization / extension points:
- Swap in any `Retriever` that has a synchronous `search`; async wrapper handles
  concurrency via thread pool. For fully async retrievers, override
  `_aretrieve` to call native async methods.
- Override `build_judge` to change judge type (LLMJudge/NLI/etc.) or inject
  domain-specific judges.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional, Tuple
import time

from ..core.specs import Claim, ClaimVerdict, StateSpec, VerdictLabel
from ..retrieve.planner import plan_retrieval
from ..retrieve.gating import gate_chunks, filter_accepted
from ..retrieve.protocols import Retriever, AsyncRetriever
from ..verify.judges import Judge, RuleBasedJudge
from ..verify.aggregate import aggregate_claim, aggregate_overall
from ..core.trace import TraceBuilder
from ..core.instrumentation import Instrumentation


async def _aretrieve(retriever: Retriever, query: str, filters, k: int):
    """
    Async wrapper:
    - If retriever supports `asearch`, call it.
    - Otherwise, run synchronous `search` in a threadpool.
    """
    if isinstance(retriever, AsyncRetriever) or hasattr(retriever, "asearch"):
        return await retriever.asearch(query, filters=filters, k=k)  # type: ignore[attr-defined]
    return await asyncio.to_thread(retriever.search, query, filters=filters, k=k)


def _build_judge(judge: Optional[Judge]) -> Judge:
    """Hook to inject a judge; defaults to RuleBasedJudge."""
    return judge or RuleBasedJudge()


async def async_run_verification(
    claims: List[Claim],
    state: StateSpec,
    retriever: Retriever,
    *,
    judge: Optional[Judge] = None,
    total_k: int = 20,
    trace: Optional[TraceBuilder] = None,
    profile=None,
    logger: Optional[Any] = None,
    instrumentation: Optional[Instrumentation] = None,
    max_concurrent_tasks: Optional[int] = None,
) -> Tuple[VerdictLabel, float, List[ClaimVerdict]]:
    """
    Asynchronous end-to-end verification runner.

    Returns:
        overall_label, overall_confidence, claim_verdicts
    """
    judge_impl = _build_judge(judge)
    plan = plan_retrieval(claims, state, total_k=total_k, trace=trace, profile=profile)

    if instrumentation:
        instrumentation.log("plan.built", {"steps": len(plan.steps), "total_k": total_k})
        instrumentation.inc("plan.count")

    # Concurrent retrieval per step
    semaphore = asyncio.Semaphore(max_concurrent_tasks or len(plan.steps) or 1)

    async def _bounded_retrieve(query: str, filters, k: int):
        async with semaphore:
            return await _aretrieve(retriever, query, filters, k)

    retrieve_tasks = [
        _bounded_retrieve(step.query, step.filters, step.k) for step in plan.steps
    ]
    try:
        t0 = time.time()
        step_results = await asyncio.gather(*retrieve_tasks)
        if instrumentation:
            instrumentation.timing("retrieve.batch.ms", (time.time() - t0) * 1000.0)
    except Exception as e:
        if logger:
            logger.error(f"Retrieval failed: {e}")
        if instrumentation:
            instrumentation.inc("retrieve.errors")
        raise
    all_chunks = [c for step_list in step_results for c in step_list]

    gated = gate_chunks(all_chunks, state, trace=trace)
    accepted = filter_accepted(gated)

    if instrumentation:
        instrumentation.log(
            "gate.results",
            {"accepted": len(accepted), "total": len(gated)},
        )
        instrumentation.inc("gate.count")

    claim_verdicts: List[ClaimVerdict] = []
    for claim in claims:
        relevant_chunks = [c for c in accepted]  # naive; could be filtered per-claim if needed
        try:
            t1 = time.time()
            jr = await asyncio.to_thread(judge_impl.score_batch, claim, relevant_chunks, state)
            if instrumentation:
                instrumentation.timing("judge.score_batch.ms", (time.time() - t1) * 1000.0)
                instrumentation.inc("judge.count")
        except Exception as e:
            if logger:
                logger.error(f"Judge failed for claim {claim.claim_id}: {e}")
            if instrumentation:
                instrumentation.inc("judge.errors")
            raise
        cv = aggregate_claim(claim, jr, trace=trace)
        claim_verdicts.append(cv)

    overall_label, overall_conf, _ = aggregate_overall(claim_verdicts, trace=trace)

    if instrumentation:
        instrumentation.log(
            "aggregate.overall",
            {"label": overall_label.value, "confidence": overall_conf},
        )
        instrumentation.inc("aggregate.count")
    return overall_label, overall_conf, claim_verdicts

