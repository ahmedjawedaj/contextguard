from contextguard.retrieve.gating import GatingConfig
from contextguard.verify.aggregate import AggregationConfig
from contextguard.core.specs import DomainProfile


def test_finance_profile_gating():
    cfg = GatingConfig.from_profile(DomainProfile.FINANCE)
    assert cfg.max_chunks_per_source == 2
    assert cfg.require_time_match is True
    assert cfg.allow_adjacent_years is False


def test_finance_profile_aggregation():
    cfg = AggregationConfig.from_profile(DomainProfile.FINANCE)
    assert cfg.min_sources_for_support == 2
    assert cfg.support_threshold >= 0.7

