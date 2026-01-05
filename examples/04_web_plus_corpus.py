"""
Example 04: Federated (web + corpus) verification.

What it does:
- Demonstrates dual retrieval: corpus-first plus web supplement (simulated)
- Uses FederatedRetriever to combine sources
"""

from pathlib import Path

from contextguard import (
    StateSpec,
    StateDelta,
    EntityRef,
    Claim,
    plan_retrieval,
    gate_chunks,
    RuleBasedJudge,
    aggregate_claim,
    aggregate_overall,
    build_report,
    MockRetriever,
    FederatedRetriever,
)


def main():
    state = StateSpec(thread_id="federated-demo")
    delta = StateDelta(entities_add=[EntityRef(entity_id="acme")])
    state = delta.model_copy(update=state.model_dump())

    # Corpus retriever
    corpus = MockRetriever(name="corpus")
    corpus.add_chunk(
        "Internal corpus: Acme posted 9% growth in 2024.",
        source_id="corpus_doc",
        entity_ids=["acme"],
        year=2024,
    )
    # Web retriever (simulated)
    web = MockRetriever(name="web")
    web.add_chunk(
        "Web article: Analysts expect Acme to grow 10% in 2024.",
        source_id="web_doc",
        entity_ids=["acme"],
        year=2024,
    )

    retriever = FederatedRetriever([corpus, web], merge_strategy="score_sort")

    claim = Claim(
        claim_id="c1",
        text="Acme will grow around 10% in 2024",
        entities=["acme"],
    )

    plan = plan_retrieval([claim], state, total_k=6)
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
        thread_id="federated-demo",
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
    (output_dir / "04_report.md").write_text(report.render_markdown if hasattr(report, "render_markdown") else "", encoding="utf-8")
    print(f"Federated verdict: {report.overall_label.value} (conf {report.overall_confidence:.0%})")


if __name__ == "__main__":
    main()
