import re
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
    ReasonCode,
)


def extract_year(text: str) -> int:
    m = re.search(r"(20\d{2})", text)
    return int(m.group(1)) if m else None


def run_case(claim_text: str, evidence_items):
    state = StateSpec(
        thread_id="t",
        entities=[EntityRef(entity_id="acme")],
        time=TimeConstraint(year=extract_year(claim_text)),
    )
    claim = Claim(
        claim_id="c1",
        text=claim_text,
        entities=["acme"],
        time=TimeConstraint(year=state.time.year),
    )
    retriever = MockRetriever()
    for item in evidence_items:
        retriever.add_chunk(
            item["text"],
            source_id=item["id"],
            source_type=item.get("source_type", "PRIMARY"),
            entity_ids=item.get("entities", ["acme"]),
            year=item.get("year"),
        )
    plan = plan_retrieval([claim], state, total_k=10, enable_counter=False)
    chunks = []
    for step in plan.steps:
        chunks.extend(retriever.search(step.query, filters=None, k=step.k))
    gated = gate_chunks(chunks, state)
    accepted = []
    seen_ids = set()
    rejected = []
    for g in gated:
        sid = g.chunk.provenance.source_id
        if g.accepted and sid not in seen_ids:
            accepted.append(g.chunk)
            seen_ids.add(sid)
        else:
            rejected.append(g)

    judge = RuleBasedJudge()
    jr = judge.score_batch(claim, accepted, state)
    cv = aggregate_claim(claim, jr)
    overall_label, _, _ = aggregate_overall([cv])

    return {
        "overall": overall_label,
        "claim_verdict": cv,
        "accepted_ids": [c.provenance.source_id for c in accepted],
        "rejected": rejected,
    }


def reasons_of(rejected):
    out = []
    for g in rejected:
        out.extend([getattr(r, "value", str(r)) for r in g.decision.reasons])
    return out


def test_supported_primary():
    evidence = [
        {"id": "annual_report_2024", "text": "ACME 2024 revenue was $200M according to its audited annual report.", "source_type": "PRIMARY", "year": 2024},
        {"id": "blog", "text": "A blog claims ACME 2024 revenue was $500M, but provides no citations.", "source_type": "TERTIARY", "year": 2024},
        {"id": "annual_report_2023", "text": "ACME 2023 revenue was $180M (previous year).", "source_type": "PRIMARY", "year": 2023},
    ]
    result = run_case("ACME 2024 revenue was $200M.", evidence)
    assert result["overall"] == VerdictLabel.SUPPORTED
    assert result["accepted_ids"] == ["annual_report_2024"]
    rs = reasons_of(result["rejected"])
    assert ReasonCode.CTXT_TIME_MISMATCH.value in rs
    assert ReasonCode.CTXT_SOURCE_POLICY_VIOLATION.value in rs


def test_contradicted_primary():
    evidence = [
        {"id": "annual_report_2024", "text": "ACME 2024 revenue was $200M according to its audited annual report.", "source_type": "PRIMARY", "year": 2024},
        {"id": "blog", "text": "A blog claims ACME 2024 revenue was $500M, but provides no citations.", "source_type": "TERTIARY", "year": 2024},
    ]
    result = run_case("ACME 2024 revenue was $500M.", evidence)
    assert result["overall"] == VerdictLabel.CONTRADICTED
    assert result["accepted_ids"] == ["annual_report_2024"]


def test_insufficient_wrong_year():
    evidence = [
        {"id": "annual_report_2024", "text": "ACME 2024 revenue was $200M according to its audited annual report.", "source_type": "PRIMARY", "year": 2024},
        {"id": "annual_report_2023", "text": "ACME 2023 revenue was $180M (previous year).", "source_type": "PRIMARY", "year": 2023},
    ]
    result = run_case("ACME 2025 revenue was $200M.", evidence)
    assert result["overall"] == VerdictLabel.INSUFFICIENT
    assert result["accepted_ids"] == []
    rs = reasons_of(result["rejected"])
    assert ReasonCode.CTXT_TIME_MISMATCH.value in rs


def test_entity_mismatch_rejected():
    evidence = [
        {"id": "typo_entity", "text": "ACNE 2024 revenue was $200M.", "source_type": "PRIMARY", "year": 2024, "entities": ["acne"]},
        {"id": "other_entity", "text": "BETA 2024 revenue was $200M.", "source_type": "PRIMARY", "year": 2024, "entities": ["beta"]},
    ]
    result = run_case("ACME 2024 revenue was $200M.", evidence)
    assert result["overall"] == VerdictLabel.INSUFFICIENT
    rs = reasons_of(result["rejected"])
    assert ReasonCode.CTXT_ENTITY_MISMATCH.value in rs


def test_normalization_equivalence():
    evidence = [
        {"id": "annual_report_2024", "text": "ACME 2024 revenue was $0.2B.", "source_type": "PRIMARY", "year": 2024},
    ]
    result = run_case("ACME 2024 revenue was $200M.", evidence)
    assert result["overall"] == VerdictLabel.SUPPORTED
    assert result["accepted_ids"] == ["annual_report_2024"]

