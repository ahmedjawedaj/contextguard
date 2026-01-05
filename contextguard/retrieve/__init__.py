# ContextGuard Retrieve Package
# Contains: protocols, planner, gating

from .protocols import (
    Retriever,
    RetrieverBase,
    MockRetriever,
    FederatedRetriever,
    CanonicalFilters,
)

from .planner import (
    plan_retrieval,
    RetrievalPlan,
    RetrievalStep,
    RetrievalPlanner,
    QueryType,
)

from .gating import (
    gate_chunks,
    filter_accepted,
    filter_rejected,
    summarize_gating,
    EvidenceGate,
    GatingConfig,
    GatedChunk,
)
