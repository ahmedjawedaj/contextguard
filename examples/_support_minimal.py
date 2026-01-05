"""
Minimal proof runner used by tests/test_minimal_proof.py.
Uses the real pipeline (retriever → plan → gate → judge → aggregate) with mock data.
"""

import re
from typing import List

from contextguard import (
    StateSpec,
    EntityRef,
    TimeConstraint,
    Claim,
    plan_retrieval,
    gate_chunks,
    RuleBasedJudge,
    aggregate_claim,
    aggregate_overall,
    MockRetriever,
    VerdictLabel,
)


def _parse_year(text: str) -> int:
    m = re.search(r"(20\d{2})", text)
    return int(m.group(1)) if m else 2024


def _build_retriever() -> MockRetriever:
    r = MockRetriever()
    # Primary supporting
    r.add_chunk(
        "ACME 2024 revenue was $200M according to its audited annual report.",
        source_id="annual_report_2024",
        source_type="PRIMARY",
        entity_ids=["acme"],
        year=2024,
    )
    # Primary prior-year (should be rejected on time)
    r.add_chunk(
        "ACME 2023 revenue was $180M (previous year).",
        source_id="annual_report_2023",
        source_type="PRIMARY",
        entity_ids=["acme"],
        year=2023,
    )
    # Tertiary blog (should be rejected by policy)
    r.add_chunk(
        "A blog claims ACME 2024 revenue was $500M, but provides no citations.",
        source_id="random_blog_post",
        source_type="TERTIARY",
        entity_ids=["acme"],
        year=2024,
    )
    return r


def run_minimal_proof(claim_text: str):
    year = _parse_year(claim_text)

    state = StateSpec(
        thread_id="minimal",
        entities=[EntityRef(entity_id="acme")],
        time=TimeConstraint(year=year),
    )

    retriever = _build_retriever()

    claim = Claim(
        claim_id="c1",
        text=claim_text,
        entities=["acme"],
        time=TimeConstraint(year=year),
    )

    plan = plan_retrieval([claim], state, total_k=5)
    chunks = []
    for step in plan.steps:
        chunks.extend(retriever.search(step.query, filters=step.filters, k=step.k))

    gated = gate_chunks(chunks, state)
    accepted = [g.chunk for g in gated if g.accepted]

    judge = RuleBasedJudge()
    jr = judge.score_batch(claim, accepted, state)
    claim_verdict = aggregate_claim(claim, jr)
    overall_label, _, _ = aggregate_overall([claim_verdict])

    class SimpleReport:
        pass

    rep = SimpleReport()
    rep.overall_label = overall_label.value if isinstance(overall_label, VerdictLabel) else str(overall_label)
    # Capture rejection reasons for debugging
    rep.rejected_reasons: List[str] = []
    for g in gated:
        if not g.accepted:
            rep.rejected_reasons.extend(
                [getattr(r, "value", str(r)) for r in g.decision.reasons]
            )
    return rep

