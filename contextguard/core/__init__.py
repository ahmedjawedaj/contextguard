# ContextGuard Core Package
# Contains: specs, merge, trace

from .specs import (
    VerdictLabel,
    EvidenceRole,
    SourceType,
    ReasonCode,
    StateSpec,
    StateDelta,
    TimeConstraint,
    UnitConstraint,
    SourcePolicy,
    EntityRef,
    MergeResult,
    MergeConflict,
    Chunk,
    Provenance,
    GateDecision,
    EvidenceAssessment,
    Claim,
    ClaimVerdict,
    VerdictReport,
    ContextPack,
)

from .merge import (
    merge_state,
    apply_delta,
    create_initial_state,
    MergeConfig,
)

from .trace import (
    TraceGraph,
    TraceNode,
    TraceBuilder,
    NodeKind,
    NodeOp,
)
