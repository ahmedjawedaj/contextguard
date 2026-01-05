"""
Example 03: Enterprise corpus-only verification.

What it does:
- Disables web, uses only internal corpus sources
- Adds a region-specific entity and policy constraint
- Runs end-to-end on mock internal docs
"""

from pathlib import Path

from contextguard import (
    StateSpec,
    StateDelta,
    EntityRef,
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
    state = StateSpec(thread_id="enterprise-demo")
    delta = StateDelta(
        entities_add=[EntityRef(entity_id="product_alpha")],
        source_policy=SourcePolicy(
            allow_web=False,
            allow_corpus=True,
            allowed_source_types=[SourceType.PRIMARY],
        ),
        scope_note="Region: EU",
    )
    state = delta.model_copy(update=state.model_dump())

    retriever = MockRetriever()
    retriever.add_chunk(
        "Internal policy for product_alpha in EU requires GDPR compliance.",
        source_id="internal_policy_1",
        source_type=SourceType.PRIMARY,
        entity_ids=["product_alpha"],
        metadata={"region": "EU"},
    )
    retriever.add_chunk(
        "Outdated policy from 2020 (do not use).",
        source_id="internal_policy_old",
        source_type=SourceType.PRIMARY,
        entity_ids=["product_alpha"],
    )

    claim = Claim(
        claim_id="c1",
        text="Product Alpha complies with EU GDPR per internal policy",
        entities=["product_alpha"],
    )

    plan = plan_retrieval([claim], state, total_k=5)
    all_chunks = []
    for step in plan.steps:
        all_chunks.extend(retriever.search(step.query, filters=step.filters, k=step.k))
    gated = gate_chunks(all_chunks, state)
    accepted = [g.chunk for g in gated if g.accepted]

    judge = RuleBasedJudge()
    jr = judge.score_batch(claim, accepted, state)
    claim_verdict = aggregate_claim(claim, jr)
    overall_label, overall_conf, warnings = aggregate_overall([claim_verdict])

    report = build_report(
        thread_id="enterprise-demo",
        state=state,
        claim_verdicts=[claim_verdict],
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
    (output_dir / "03_report.md").write_text(report.render_markdown if hasattr(report, "render_markdown") else "", encoding="utf-8")
    print(f"Enterprise verdict: {report.overall_label.value} (conf {report.overall_confidence:.0%})")


if __name__ == "__main__":
    main()
