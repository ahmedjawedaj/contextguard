from contextguard.retrieve.gating import GatingConfig
from contextguard.verify.aggregate import AggregationConfig
from contextguard.core.specs import DomainProfile


def test_policy_profile_gating():
    cfg = GatingConfig.from_profile(DomainProfile.POLICY)
    assert cfg.strict_source_policy is True
    assert cfg.allow_adjacent_years is False
    assert cfg.time_match_tolerance_days == 30


def test_policy_profile_aggregation():
    cfg = AggregationConfig.from_profile(DomainProfile.POLICY)
    assert cfg.min_sources_for_support == 1
    assert cfg.support_threshold >= 0.7

