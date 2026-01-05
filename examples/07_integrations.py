"""
Integration snippets for adapters and async runner.

Demonstrates:
- RetryingProvider wrapping an LLM provider (dummy here to avoid network).
- S3Store wiring (skipped if boto3 not installed).
- async_run_verification using MockRetriever + RuleBasedJudge.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from contextguard import (
    StateSpec,
    EntityRef,
    TimeConstraint,
    Claim,
    MockRetriever,
    RuleBasedJudge,
    LLMJudge,
    RetryingProvider,
    async_run_verification,
    S3Store,
)

# -----------------------------------------------------------------------------
# RetryingProvider + LLMJudge (dummy provider to avoid network)
# -----------------------------------------------------------------------------


class DummyLLMProvider:
    """Implements LLMProvider.complete_json for demo purposes."""

    def __init__(self, name: str = "dummy"):
        self.name = name

    def complete_json(self, prompt: str, schema: Dict[str, Any], temperature: float = 0.0) -> Dict[str, Any]:
        return {
            "schema_version": "v0.1",
            "support": 0.7,
            "contradict": 0.1,
            "rationale": f"[{self.name}] pseudo-eval of prompt length={len(prompt)}",
            "reasons": [],
            "confidence": 0.8,
            "evidence_quality": {
                "entity_match": True,
                "time_match": True,
                "metric_match": False,
                "unit_match": False,
            },
        }


def demo_retrying_provider() -> None:
    """
    Wrap an LLMProvider with retries/backoff/logging; usable with LLMJudge.
    """
    base = DummyLLMProvider("dummy")
    llm = RetryingProvider(base, max_attempts=2, base_delay=0.1, max_delay=0.2)
    judge = LLMJudge(llm)
    # The actual judging is exercised in async demo below; here we just show construction.
    print("RetryingProvider + LLMJudge wired (dummy provider).")


# -----------------------------------------------------------------------------
# S3Store wiring (no network call; skipped if boto3 missing)
# -----------------------------------------------------------------------------


def demo_s3_store() -> None:
    """
    Show how to construct S3Store. This does not perform network calls.
    """
    try:
        store = S3Store(bucket="your-bucket-name", prefix="contextguard-demo/")
    except ImportError:
        print("boto3 not installed; skipping S3Store demo.")
        return

    print("S3Store ready (no calls made). Example keys:")
    print(" state key:", store.state_key("thread123"))
    print(" fact key :", store.fact_key("fact123"))
    print(" run key  :", store.run_key("run123"))


# -----------------------------------------------------------------------------
# Async verification demo
# -----------------------------------------------------------------------------


async def demo_async_runner() -> None:
    """
    Run async verification: plan → retrieve → gate → judge → aggregate.
    Uses MockRetriever + RuleBasedJudge.
    """
    state = StateSpec(
        thread_id="async-demo",
        entities=[EntityRef(entity_id="acme")],
        time=TimeConstraint(year=2024),
    )

    claim = Claim(
        claim_id="c1",
        text="ACME 2024 revenue was $200M.",
        entities=["acme"],
        time=TimeConstraint(year=2024),
    )

    retriever = MockRetriever()
    retriever.add_chunk(
        "ACME 2024 revenue was $200M according to its audited annual report.",
        source_id="acme_annual_report_2024",
        source_type="PRIMARY",
        entity_ids=["acme"],
        year=2024,
        metadata={"doc_type": "annual_report"},
    )
    retriever.add_chunk(
        "A blog claims ACME 2024 revenue was $500M.",
        source_id="random_blog_post",
        source_type="TERTIARY",
        entity_ids=["acme"],
        year=2024,
        metadata={"doc_type": "blog"},
    )

    judge = RuleBasedJudge()
    overall_label, overall_conf, claim_verdicts = await async_run_verification(
        claims=[claim],
        state=state,
        retriever=retriever,
        judge=judge,
        total_k=5,
    )

    print(f"Async overall: {overall_label.value} (conf {overall_conf:.2f})")
    for cv in claim_verdicts:
        print(f"- {cv.claim.text} → {cv.label.value} (conf {cv.confidence:.2f}) reasons={[r.value for r in cv.reasons]}")


if __name__ == "__main__":
    demo_retrying_provider()
    demo_s3_store()
    asyncio.run(demo_async_runner())

