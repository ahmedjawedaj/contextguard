"""
Minimal end-to-end proof for ContextGuard using the real pipeline
(planner → retriever → gate → judge → aggregate) and a Graphviz trace.

Run:
  python examples/06_minimal_proof.py

Outputs:
  - Console verdict summary
  - Graphviz trace at examples/output/minimal_trace.dot
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from contextguard import (
    Claim,
    EntityRef,
    MockRetriever,
    RuleBasedJudge,
    StateSpec,
    TimeConstraint,
    aggregate_claim,
    aggregate_overall,
    gate_chunks,
    plan_retrieval,
)
from contextguard.core.specs import SourceType
from contextguard.core.trace import TraceBuilder


def _parse_year(text: str) -> int:
    m = re.search(r"(20\d{2})", text)
    return int(m.group(1)) if m else datetime.now(timezone.utc).year


def _build_retriever(now: str) -> MockRetriever:
    r = MockRetriever()
    r.add_chunk(
        "ACME 2024 revenue was $200M according to its audited annual report.",
        source_id="acme_annual_report_2024",
        source_type=SourceType.PRIMARY,
        entity_ids=["acme"],
        year=2024,
        metadata={"doc_type": "annual_report"},
    )
    r.add_chunk(
        "ACME 2023 revenue was $180M (previous year).",
        source_id="acme_annual_report_2023",
        source_type=SourceType.PRIMARY,
        entity_ids=["acme"],
        year=2023,
        metadata={"doc_type": "annual_report"},
    )
    r.add_chunk(
        "A blog claims ACME 2024 revenue was $500M, but provides no citations.",
        source_id="random_blog_post",
        source_type=SourceType.TERTIARY,
        entity_ids=["acme"],
        year=2024,
        metadata={"doc_type": "blog"},
    )
    # Stamp retrieved_at for provenance deterministically
    for ch in r.chunks:
        ch.provenance.retrieved_at = now
    return r


def run_demo(claim_text: str = "ACME 2024 revenue was $200M.") -> None:
    now = datetime.now(timezone.utc).isoformat()
    year = _parse_year(claim_text)

    trace = TraceBuilder(run_id="minimal-proof")
    user_node = trace.add_user_turn(claim_text)

    state = StateSpec(
        thread_id="minimal-proof",
        entities=[EntityRef(entity_id="acme")],
        time=TimeConstraint(year=year),
    )
    state_node = trace.add_state_merge(
        state_dict=state.model_dump(),
        conflicts=[],
        parents=[user_node],
    )

    claim = Claim(
        claim_id="c1",
        text=claim_text,
        entities=["acme"],
        time=TimeConstraint(year=year),
    )
    claim_node = trace.add_claim(claim_text=claim.text, claim_id=claim.claim_id, parents=[user_node])

    retriever = _build_retriever(now)

    plan = plan_retrieval([claim], state, total_k=5, trace=trace, trace_parents=[claim_node])

    gated_all = []
    for step in plan.steps:
        step_parent = [step.trace_node_id] if step.trace_node_id else [claim_node]
        results = retriever.search(step.query, filters=step.filters, k=step.k)
        gated = gate_chunks(results, state, trace=trace, parents=step_parent)
        gated_all.extend(gated)

    accepted_chunks = [g.chunk for g in gated_all if g.accepted]
    judge = RuleBasedJudge()
    judge_results = judge.score_batch(claim, accepted_chunks, state)

    claim_verdict = aggregate_claim(claim, judge_results, trace=trace, trace_parents=[claim_node])
    overall_label, overall_conf, warnings = aggregate_overall([claim_verdict], trace=trace, trace_parents=[state_node])

    rejected_reasons: List[str] = []
    for g in gated_all:
        if not g.accepted:
            rejected_reasons.extend([r.value if hasattr(r, "value") else str(r) for r in g.decision.reasons])

    print(f"Overall verdict: {overall_label.value} (conf {overall_conf:.2f})")
    if warnings:
        print(f"Warnings: {[w.value for w in warnings]}")

    print("\nClaim verdicts:")
    print(f"- {claim.text}")
    print(f"  Verdict: {claim_verdict.label.value} (conf {claim_verdict.confidence:.2f}) reasons={[r.value for r in claim_verdict.reasons]}")
    for ev in claim_verdict.evidence:
        src = ev.chunk.provenance.source_id
        print(f"    * {ev.role.value} src={src} accepted={ev.decision.accepted} reasons={[r.value for r in ev.decision.reasons]}")
        print(f"      rationale: {ev.rationale}")

    if rejected_reasons:
        print(f"\nRejected evidence reasons: {rejected_reasons}")

    out_dir = Path("examples/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    dot_path = out_dir / "minimal_trace.dot"
    dot_path.write_text(trace.graph.to_dot(rankdir="LR"), encoding="utf-8")
    print(f"\nWrote trace graph to {dot_path} (render with: dot -Tpng {dot_path} -o examples/output/minimal_trace.png)")


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
