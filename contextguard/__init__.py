"""
ContextGuard: State-Contracted Verification for Agentic RAG

A verification/consistency engine that solves the "context detachment" problem
in multi-turn RAG and fact-checking systems.

Key Features:
- StateSpec: Persistent constraints that filter retrieval (entities, time, source policy)
- Evidence Gating: Hard rejection of ineligible chunks with reason codes
- Counter-Evidence: Always search for contradictions (anti-confirmation-bias)
- Trace DAG: Micrograd-style explainability for every decision
- Verdict Reports: SUPPORTED | CONTRADICTED | INSUFFICIENT | MIXED

Quick Start:
    from contextguard import (
        StateSpec, StateDelta, Claim,
        merge_state, gate_chunks, plan_retrieval,
        build_report, TraceGraph
    )
    
    # Create state
    state = StateSpec(thread_id="t1")
    
    # Merge constraints from user input
    delta = StateDelta(entities_add=[...], time=TimeConstraint(year=2024))
    result = merge_state(state, delta, turn_id=1)
    
    # Plan and execute retrieval
    plan = plan_retrieval(claims, result.state)
    
    # Gate chunks
    gated = gate_chunks(chunks, result.state)
    
    # Build report
    report = build_report(...)

For more examples, see the examples/ directory.
"""

__version__ = "0.1.0"
__author__ = "ContextGuard Contributors"

# Core specs
from .core.specs import (
    # Enums
    VerdictLabel,
    EvidenceRole,
    SourceType,
    ReasonCode,
    # State contract
    StateSpec,
    StateDelta,
    TimeConstraint,
    UnitConstraint,
    SourcePolicy,
    EntityRef,
    MergeResult,
    MergeConflict,
    # Evidence
    Chunk,
    Provenance,
    GateDecision,
    EvidenceAssessment,
    # Claims and verdicts
    Claim,
    ClaimVerdict,
    VerdictReport,
    ContextPack,
)
from .core.instrumentation import Instrumentation, LoggerSink, MetricsSink

# State management
from .core.merge import (
    merge_state,
    apply_delta,
    create_initial_state,
    MergeConfig,
)

# Trace/explainability
from .core.trace import (
    TraceGraph,
    TraceNode,
    TraceBuilder,
    NodeKind,
    NodeOp,
)

# Retrieval
from .retrieve.protocols import (
    Retriever,
    RetrieverBase,
    MockRetriever,
    FederatedRetriever,
    CanonicalFilters,
)
from .adapters.langchain import LangChainRetrieverAdapter
from .adapters.llamaindex import LlamaIndexRetrieverAdapter
from .adapters.openai_provider import OpenAIProvider
from .adapters.retrying_provider import RetryingProvider
from .adapters.budgeted_provider import BudgetedProvider
from .adapters.retrying_retriever import RetryingRetriever
from .adapters.circuit_breaker_retriever import CircuitBreakerRetriever
from .adapters.circuit_breaker_provider import CircuitBreakerProvider

# Optional adapters (guarded imports)
try:
    from .adapters.chroma import ChromaRetrieverAdapter
except ImportError:  # pragma: no cover - optional dependency
    ChromaRetrieverAdapter = None  # type: ignore

try:
    from .adapters.qdrant import QdrantRetrieverAdapter
except ImportError:  # pragma: no cover - optional dependency
    QdrantRetrieverAdapter = None  # type: ignore

from .retrieve.planner import (
    plan_retrieval,
    RetrievalPlan,
    RetrievalStep,
    RetrievalPlanner,
    QueryType,
)

from .retrieve.gating import (
    gate_chunks,
    filter_accepted,
    filter_rejected,
    summarize_gating,
    EvidenceGate,
    GatingConfig,
    GatedChunk,
)

# Verification
from .verify.claim_splitter import (
    split_claims,
    RuleBasedClaimSplitter,
    LLMClaimSplitter,
)

from .verify.judges import (
    Judge,
    RuleBasedJudge,
    LLMJudge,
    JudgeResult,
    LLMProviderBase,
    create_judge,
)
from .generate.generator import Generator, LLMGenerator
from .pipeline.async_runner import async_run_verification

from .verify.aggregate import (
    aggregate_claim,
    aggregate_overall,
    ClaimAggregator,
    OverallAggregator,
    AggregationConfig,
)

from .verify.report import (
    build_report,
    render_report,
    save_report,
    ReportBuilder,
    ReportRenderer,
)

# Storage
from .stores.sqlite import (
    SQLiteStore,
    create_store,
)
from .stores.cloud import S3Store
from .stores.postgres import PostgresStore

__all__ = [
    # Version
    "__version__",
    # Enums
    "VerdictLabel",
    "EvidenceRole",
    "SourceType",
    "ReasonCode",
    "QueryType",
    "NodeKind",
    "NodeOp",
    # State
    "StateSpec",
    "StateDelta",
    "TimeConstraint",
    "UnitConstraint",
    "SourcePolicy",
    "EntityRef",
    "MergeResult",
    "MergeConflict",
    "MergeConfig",
    "merge_state",
    "apply_delta",
    "create_initial_state",
    # Evidence
    "Chunk",
    "Provenance",
    "GateDecision",
    "EvidenceAssessment",
    "GatedChunk",
    # Claims
    "Claim",
    "ClaimVerdict",
    "split_claims",
    "RuleBasedClaimSplitter",
    "LLMClaimSplitter",
    # Verdict
    "VerdictReport",
    "ContextPack",
    "build_report",
    "render_report",
    "save_report",
    "ReportBuilder",
    "ReportRenderer",
    # Trace
    "TraceGraph",
    "TraceNode",
    "TraceBuilder",
    # Retrieval
    "Retriever",
    "RetrieverBase",
    "MockRetriever",
    "FederatedRetriever",
    "CanonicalFilters",
    "LangChainRetrieverAdapter",
    "LlamaIndexRetrieverAdapter",
    "ChromaRetrieverAdapter",
    "QdrantRetrieverAdapter",
    "RetryingRetriever",
    "CircuitBreakerRetriever",
    "CircuitBreakerProvider",
    "CircuitBreakerRetriever",
    "OpenAIProvider",
    "RetryingProvider",
    "BudgetedProvider",
    "plan_retrieval",
    "RetrievalPlan",
    "RetrievalStep",
    "RetrievalPlanner",
    # Gating
    "gate_chunks",
    "filter_accepted",
    "filter_rejected",
    "summarize_gating",
    "EvidenceGate",
    "GatingConfig",
    # Judges
    "Judge",
    "RuleBasedJudge",
    "LLMJudge",
    "JudgeResult",
    "LLMProviderBase",
    "create_judge",
    "OpenAIProvider",
    "RetryingProvider",
    # Generation
    "Generator",
    "LLMGenerator",
    # Async runner
    "async_run_verification",
    # Aggregation
    "aggregate_claim",
    "aggregate_overall",
    "ClaimAggregator",
    "OverallAggregator",
    "AggregationConfig",
    # Storage
    "SQLiteStore",
    "create_store",
    "S3Store",
    "PostgresStore",
]
