"""
Example 01: Verify a single-article claim and emit a verdict report.

What it does:
- Builds a simple state with an entity and year constraint
- Loads a mock "article" as evidence
- Runs planner → retrieve → gate → judge → aggregate → report
- Prints the verdict and saves a markdown/JSON report to examples/output
"""

from pathlib import Path

from contextguard import (
    StateSpec,
    StateDelta,
    EntityRef,
    TimeConstraint,
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
    # 1) State: entity + year constraint
    state = StateSpec(thread_id="article-demo")
    state_delta = StateDelta(
        entities_add=[EntityRef(entity_id="acme", display_name="Acme Corp")],
        time=TimeConstraint(year=2024),
    )
    state = state_delta.model_copy(update=state.model_dump())  # simple merge for demo

    # 2) Mock retriever with an article chunk
    retriever = MockRetriever()
    retriever.add_chunk(
        text="Acme Corp reported revenue growth of 10% in 2024, driven by widgets.",
        source_id="article_1",
        entity_ids=["acme"],
        year=2024,
    )

    # 3) Claim to verify
    claim = Claim(
        claim_id="c1",
        text="Acme Corp revenue grew 10% in 2024",
        entities=["acme"],
        time=TimeConstraint(year=2024),
    )

    # 4) Plan → retrieve → gate
    plan = plan_retrieval([claim], state, total_k=5)
    all_chunks = []
    for step in plan.steps:
        chunks = retriever.search(step.query, filters=step.filters, k=step.k)
        all_chunks.extend(chunks)
    gated = gate_chunks(all_chunks, state)
    accepted = [g.chunk for g in gated if g.accepted]

    # 5) Judge + aggregate
    judge = RuleBasedJudge()
    judge_results = judge.score_batch(claim, accepted, state)
    claim_verdict = aggregate_claim(claim, judge_results)
    overall_label, overall_conf, warnings = aggregate_overall([claim_verdict])

    # 6) Build report
    report = build_report(
        thread_id="article-demo",
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
    (output_dir / "01_report.md").write_text(report.render_markdown if hasattr(report, "render_markdown") else "", encoding="utf-8")
    print(f"Verdict: {report.overall_label.value} (confidence {report.overall_confidence:.0%})")


if __name__ == "__main__":
    main()
