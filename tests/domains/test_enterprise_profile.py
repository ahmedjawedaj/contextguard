from contextguard.retrieve.gating import GatingConfig
from contextguard.verify.aggregate import AggregationConfig
from contextguard.core.specs import DomainProfile


def test_enterprise_profile_gating():
    cfg = GatingConfig.from_profile(DomainProfile.ENTERPRISE)
    assert cfg.strict_source_policy is True
    assert cfg.max_chunks_per_source == 2


def test_enterprise_profile_aggregation():
    cfg = AggregationConfig.from_profile(DomainProfile.ENTERPRISE)
    assert cfg.min_sources_for_support == 1

