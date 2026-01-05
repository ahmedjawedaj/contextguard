"""
Example 02: Multi-turn verification with state carryover.

What it does:
- Simulates three turns updating state (entities, time, source policy)
- Runs planner → retrieve → gate → judge → aggregate → report
- Prints verdict and saves outputs to examples/output
"""

from pathlib import Path

from contextguard import (
    StateSpec,
    StateDelta,
    EntityRef,
    TimeConstraint,
    SourcePolicy,
    SourceType,
    Claim,
    plan_retrieval,
    gate_chunks,
    RuleBasedJudge,
    aggregate_claim,
    aggregate_overall,
    build_report,
    MockRetriever,
)


def main():
    state = StateSpec(thread_id="conversation-demo")

    # Turn 1: set entities and metric (implicit)
    delta1 = StateDelta(entities_add=[EntityRef(entity_id="acme"), EntityRef(entity_id="beta")])
    state = delta1.model_copy(update=state.model_dump())

    # Turn 2: add time constraint
    delta2 = StateDelta(time=TimeConstraint(year=2024))
    state = delta2.model_copy(update=state.model_dump())

    # Turn 3: restrict to primary sources
    delta3 = StateDelta(source_policy=SourcePolicy(allowed_source_types=[SourceType.PRIMARY]))
    state = delta3.model_copy(update=state.model_dump())

    # Mock evidence
    retriever = MockRetriever()
    retriever.add_chunk(
        "Acme 2024 revenue guidance is $5B.",
        source_id="acme_guidance",
        source_type=SourceType.PRIMARY,
        entity_ids=["acme"],
        year=2024,
    )
    retriever.add_chunk(
        "Beta 2024 revenue projection is $2.5B.",
        source_id="beta_guidance",
        source_type=SourceType.PRIMARY,
        entity_ids=["beta"],
        year=2024,
    )

    claims = [
        Claim(claim_id="c1", text="Acme revenue will be $5B in 2024", entities=["acme"], time=TimeConstraint(year=2024)),
        Claim(claim_id="c2", text="Beta revenue will be $2.5B in 2024", entities=["beta"], time=TimeConstraint(year=2024)),
    ]

    plan = plan_retrieval(claims, state, total_k=10)
    all_chunks = []
    for step in plan.steps:
        all_chunks.extend(retriever.search(step.query, filters=step.filters, k=step.k))
    gated = gate_chunks(all_chunks, state)
    accepted = [g.chunk for g in gated if g.accepted]

    judge = RuleBasedJudge()
    claim_verdicts = []
    for claim in claims:
        relevant = [c for c in accepted if any(e in c.entity_ids for e in claim.entities)]
        jr = judge.score_batch(claim, relevant, state)
        claim_verdicts.append(aggregate_claim(claim, jr))

    overall_label, overall_conf, warnings = aggregate_overall(claim_verdicts)
    report = build_report(
        thread_id="conversation-demo",
        state=state,
        claim_verdicts=claim_verdicts,
        overall_label=overall_label,
        overall_confidence=overall_conf,
        warnings=warnings,
        retrieval_stats={
            "total": len(all_chunks),
            "accepted": len(accepted),
            "rejected": len(all_chunks) - len(accepted),
        },
    )

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "02_report.md").write_text(report.render_markdown if hasattr(report, "render_markdown") else "", encoding="utf-8")
    print(f"Overall: {report.overall_label.value} (conf {report.overall_confidence:.0%})")


if __name__ == "__main__":
    main()
