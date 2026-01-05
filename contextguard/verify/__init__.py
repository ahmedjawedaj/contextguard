# ContextGuard Verify Package
# Contains: claim_splitter, judges, aggregate, report

from .claim_splitter import (
    split_claims,
    RuleBasedClaimSplitter,
    LLMClaimSplitter,
    filter_verifiable,
    get_claim_summary,
)

from .judges import (
    Judge,
    RuleBasedJudge,
    LLMJudge,
    NLIJudge,
    JudgeResult,
    create_judge,
    judge_claim,
    best_evidence,
)

from .aggregate import (
    aggregate_claim,
    aggregate_overall,
    ClaimAggregator,
    OverallAggregator,
    AggregationConfig,
    verdict_summary,
)

from .report import (
    build_report,
    render_report,
    save_report,
    ReportBuilder,
    ReportRenderer,
)
