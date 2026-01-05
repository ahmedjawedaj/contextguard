
from contextguard.retrieve.gating import GatingConfig, EvidenceGate
from contextguard.core.specs import StateSpec, EntityRef, TimeConstraint, Chunk, Provenance, DomainProfile


def make_chunk(text: str, year: int, quarter=None, start_date=None, end_date=None):
    return Chunk(
        text=text,
        provenance=Provenance(source_id="doc", source_type="PRIMARY"),
        score=1.0,
        entity_ids=["acme"],
        year=year,
        metadata={"quarter": quarter, "start_date": start_date, "end_date": end_date},
    )


def test_finance_quarter_straddle_rejected_without_adjacent():
    state = StateSpec(
        thread_id="t",
        entities=[EntityRef(entity_id="acme")],
        time=TimeConstraint(year=2024, quarter=4, fiscal=True),
    )
    cfg = GatingConfig.from_profile(DomainProfile.FINANCE)
    gate = EvidenceGate(cfg)
    chunk = make_chunk("Q4 results", year=2025, quarter=4)
    ok, matched = gate._check_time(chunk, state)
    assert not ok


def test_policy_effective_date_range():
    state = StateSpec(
        thread_id="t",
        entities=[EntityRef(entity_id="law")],
        time=TimeConstraint(start_date="2023-01-01", end_date="2023-12-31"),
    )
    cfg = GatingConfig.from_profile(DomainProfile.POLICY)
    gate = EvidenceGate(cfg)
    chunk = make_chunk("Effective mid 2023", year=2023, start_date="2023-06-01", end_date="2023-06-30")
    ok, matched = gate._check_time(chunk, state)
    assert ok and matched


def test_enterprise_adjacent_year_soft_match():
    state = StateSpec(
        thread_id="t",
        entities=[EntityRef(entity_id="policy")],
        time=TimeConstraint(year=2024),
    )
    cfg = GatingConfig.from_profile(DomainProfile.ENTERPRISE)
    cfg.allow_adjacent_years = True
    gate = EvidenceGate(cfg)
    chunk = make_chunk("late 2023 version", year=2023)
    ok, matched = gate._check_time(chunk, state)
    assert ok  # soft accept with allow_adjacent_years default True

