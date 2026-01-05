from contextguard.verify.report import ReportRenderer
from contextguard.core.specs import (
    VerdictReport,
    VerdictLabel,
    StateSpec,
    Claim,
    ClaimVerdict,
    EvidenceAssessment,
    EvidenceRole,
    Chunk,
    Provenance,
    GateDecision,
)


def make_minimal_report():
    state = StateSpec(thread_id="t1")
    claim = Claim(claim_id="c1", text="Test claim", entities=["acme"])
    prov = Provenance(source_id="doc1", source_type="PRIMARY", url="http://example.com")
    chunk = Chunk(text="evidence text", provenance=prov, score=1.0)
    decision = GateDecision(accepted=True, reasons=[], constraint_matches={"entity": True})
    ea = EvidenceAssessment(
        chunk=chunk,
        decision=decision,
        role=EvidenceRole.SUPPORTING,
        support_score=0.9,
        contradict_score=0.0,
        rationale="quoted rationale",
    )
    cv = ClaimVerdict(
        claim=claim,
        label=VerdictLabel.SUPPORTED,
        confidence=0.9,
        reasons=[],
        summary="supported",
        evidence=[ea],
        coverage_sources=1,
        coverage_doc_types=1,
        support_score=0.9,
        contradict_score=0.0,
        coverage_score=1.0,
    )
    report = VerdictReport(
        thread_id="t1",
        state=state,
        overall_label=VerdictLabel.SUPPORTED,
        overall_confidence=0.9,
        claims=[cv],
        warnings=[],
        executive_summary="supported",
        total_chunks_retrieved=1,
        chunks_accepted=1,
        chunks_rejected=0,
        context_pack=None,
    )
    return report


def test_report_json_canonical_shape():
    report = make_minimal_report()
    as_dict = ReportRenderer.to_dict(report)
    assert as_dict["schema_version"] == "v0.1"
    assert as_dict["overall"]["label"] == "SUPPORTED"
    assert as_dict["claims"][0]["verdict"] == "SUPPORTED"
    assert as_dict["claims"][0]["evidence"][0]["source_id"] == "doc1"


def test_report_markdown_contains_tables():
    report = make_minimal_report()
    md = ReportRenderer.to_markdown(report)
    assert "| Source |" in md
    assert "Rejected evidence" in md or "Evidence (accepted)" in md

