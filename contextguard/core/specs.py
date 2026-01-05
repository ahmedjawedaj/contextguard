"""
ContextGuard Core Specifications

This module defines the fundamental data structures (contracts) that power ContextGuard:
- StateSpec: Persistent constraints that filter retrieval and enforce consistency
- ClaimSpec: Atomic claims to be verified
- Evidence: Retrieved chunks with provenance
- Verdict: Per-claim and overall verification results
- ReasonCode: Machine-readable explanation codes

These are the "types" of the ContextGuard compiler.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict
import hashlib


# =============================================================================
# Domains / Profiles
# =============================================================================


class DomainProfile(str, Enum):
    GENERIC = "generic"
    FINANCE = "finance"
    POLICY = "policy"
    ENTERPRISE = "enterprise"


# =============================================================================
# ENUMS: Verdict Labels, Evidence Roles, Source Types, Reason Codes
# =============================================================================


class VerdictLabel(str, Enum):
    """Final verdict for a claim or overall report."""
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT = "INSUFFICIENT"
    MIXED = "MIXED"


class EvidenceRole(str, Enum):
    """Role of evidence in supporting or contradicting a claim."""
    SUPPORTING = "SUPPORTING"
    CONTRADICTING = "CONTRADICTING"
    BACKGROUND = "BACKGROUND"  # Relevant but not directly supporting/contradicting


class SourceType(str, Enum):
    """Classification of evidence sources by reliability tier."""
    PRIMARY = "PRIMARY"       # Official filings, laws, original statements, internal docs
    SECONDARY = "SECONDARY"   # News articles, analyses, third-party reports
    TERTIARY = "TERTIARY"     # Social media, blogs, forums, user-generated content


class ReasonCode(str, Enum):
    """
    Machine-readable reason codes for every decision in the pipeline.
    These appear in gate decisions, verdicts, and warnings.
    
    Organized by category:
    - CTXT_*: Context/constraint failures (the core problem we're solving)
    - EVIDENCE_*: Evidence quality issues
    - CLAIM_*: Claim formulation issues
    - SYS_*: System/execution issues
    """
    # --- Context / constraint failures (THE CORE FAILURE MODES) ---
    CTXT_ENTITY_MISMATCH = "CTXT_ENTITY_MISMATCH"          # Wrong entity in evidence
    CTXT_ENTITY_AMBIGUOUS = "CTXT_ENTITY_AMBIGUOUS"        # Can't resolve entity
    CTXT_TIME_MISMATCH = "CTXT_TIME_MISMATCH"              # Wrong year/quarter/date
    CTXT_TIME_AMBIGUOUS = "CTXT_TIME_AMBIGUOUS"            # Can't determine time scope
    CTXT_METRIC_MISMATCH = "CTXT_METRIC_MISMATCH"          # Wrong metric (revenue vs profit)
    CTXT_UNIT_SCALE_MISMATCH = "CTXT_UNIT_SCALE_MISMATCH"  # Currency/scale mismatch
    CTXT_SOURCE_POLICY_VIOLATION = "CTXT_SOURCE_POLICY_VIOLATION"  # Source not allowed
    CTXT_SCOPE_MISMATCH = "CTXT_SCOPE_MISMATCH"            # Wrong scope (subsidiary, region)
    CTXT_FRESHNESS_VIOLATION = "CTXT_FRESHNESS_VIOLATION"  # Evidence too old

    # --- Evidence quality failures ---
    EVIDENCE_DUPLICATE = "EVIDENCE_DUPLICATE"              # Too many from same source
    EVIDENCE_LOW_RELEVANCE = "EVIDENCE_LOW_RELEVANCE"      # Low similarity score
    EVIDENCE_NO_PROVENANCE = "EVIDENCE_NO_PROVENANCE"      # Can't trace origin
    EVIDENCE_TOO_OLD = "EVIDENCE_TOO_OLD"                  # Stale evidence
    EVIDENCE_TOO_THIN = "EVIDENCE_TOO_THIN"                # No claim-bearing statement
    EVIDENCE_BOILERPLATE = "EVIDENCE_BOILERPLATE"          # Nav text, headers, noise
    EVIDENCE_CONFLICTING_SOURCES = "EVIDENCE_CONFLICTING_SOURCES"  # Sources disagree
    EVIDENCE_LOW_COVERAGE = "EVIDENCE_LOW_COVERAGE"        # Not enough independent sources

    # --- Claim issues ---
    CLAIM_TOO_VAGUE = "CLAIM_TOO_VAGUE"                    # Not specific enough to verify
    CLAIM_NOT_ATOMIC = "CLAIM_NOT_ATOMIC"                  # Should be split
    CLAIM_REQUIRES_PRIMARY = "CLAIM_REQUIRES_PRIMARY"      # Only secondary evidence found
    CLAIM_NEEDS_CLARIFICATION = "CLAIM_NEEDS_CLARIFICATION"  # Ambiguous, needs user input
    CLAIM_SUBJECTIVE = "CLAIM_SUBJECTIVE"                  # Opinion, not fact

    # --- System issues ---
    SYS_RETRIEVAL_FAILED = "SYS_RETRIEVAL_FAILED"          # Retriever error
    SYS_JUDGE_FAILED = "SYS_JUDGE_FAILED"                  # LLM judge error
    SYS_TIMEOUT = "SYS_TIMEOUT"                            # Operation timed out
    SYS_RATE_LIMITED = "SYS_RATE_LIMITED"                  # API rate limit


# =============================================================================
# STATE SPECIFICATION: The Constraint Contract
# =============================================================================


class TimeConstraint(BaseModel):
    """
    Time-based constraints for retrieval and verification.
    
    Supports:
    - Specific year/quarter (fiscal or calendar)
    - Date ranges
    - Both can be combined (e.g., Q1 2024 with specific start/end dates)
    """
    model_config = ConfigDict(extra="forbid")
    
    year: Optional[int] = None
    quarter: Optional[Literal[1, 2, 3, 4]] = None
    start_date: Optional[str] = None  # ISO date: "YYYY-MM-DD"
    end_date: Optional[str] = None    # ISO date: "YYYY-MM-DD"
    fiscal: bool = False              # True = fiscal year, False = calendar year
    
    def matches(self, other: "TimeConstraint") -> bool:
        """Check if another time constraint is compatible."""
        if self.year is not None and other.year is not None:
            if self.year != other.year:
                return False
        if self.quarter is not None and other.quarter is not None:
            if self.quarter != other.quarter:
                return False
        # Date range overlap check
        if self.start_date and other.end_date:
            if other.end_date < self.start_date:
                return False
        if self.end_date and other.start_date:
            if other.start_date > self.end_date:
                return False
        return True
    
    def is_empty(self) -> bool:
        """Check if no time constraints are set."""
        return (
            self.year is None 
            and self.quarter is None 
            and self.start_date is None 
            and self.end_date is None
        )


class UnitConstraint(BaseModel):
    """
    Unit and scale constraints for numeric verification.
    
    Critical for financial data where:
    - "200" could mean 200, 200K, 200M, or 200B
    - USD vs EUR matters
    - Nominal vs real (inflation-adjusted) differs
    """
    model_config = ConfigDict(extra="forbid")
    
    currency: Optional[str] = None  # ISO 4217: "USD", "EUR", "GBP"
    scale: Optional[Literal["raw", "thousand", "million", "billion"]] = None
    basis: Optional[Literal["nominal", "real", "adjusted"]] = None
    
    def is_empty(self) -> bool:
        return self.currency is None and self.scale is None and self.basis is None


class SourcePolicy(BaseModel):
    """
    Source filtering policy.
    
    Controls what evidence is admissible based on:
    - Source type (primary/secondary/tertiary)
    - Specific domains (allow/block lists)
    - Recency requirements
    - Corpus vs web access
    """
    model_config = ConfigDict(extra="forbid")
    
    # Access controls
    allow_web: bool = True
    allow_corpus: bool = True
    
    # Source type filtering
    allowed_source_types: List[SourceType] = Field(
        default_factory=lambda: [SourceType.PRIMARY, SourceType.SECONDARY]
    )
    preferred_source_types: List[SourceType] = Field(
        default_factory=lambda: [SourceType.PRIMARY]
    )
    
    # Domain filtering
    allowed_domains: Optional[List[str]] = None   # If set, only these domains
    blocked_domains: Optional[List[str]] = None   # These domains are rejected
    
    # Freshness
    max_age_days: Optional[int] = None  # Reject evidence older than this
    
    def allows_source_type(self, source_type: SourceType) -> bool:
        return source_type in self.allowed_source_types
    
    def allows_domain(self, domain: str) -> bool:
        if self.blocked_domains and domain in self.blocked_domains:
            return False
        if self.allowed_domains is not None:
            return domain in self.allowed_domains
        return True


class EntityRef(BaseModel):
    """
    Canonical entity reference.
    
    Entities are the "who" of verification: companies, people, organizations.
    The entity_id should be a stable canonical identifier (ticker, LEI, internal ID).
    """
    model_config = ConfigDict(extra="forbid")
    
    entity_id: str                              # Canonical ID (e.g., "AAPL", "LEI:123")
    display_name: Optional[str] = None          # Human-readable name
    aliases: List[str] = Field(default_factory=list)  # Alternative names
    entity_type: Optional[str] = None           # "company", "person", "org", etc.
    
    def matches_text(self, text: str) -> bool:
        """Check if text mentions this entity (case-insensitive)."""
        text_lower = text.lower()
        if self.entity_id.lower() in text_lower:
            return True
        if self.display_name and self.display_name.lower() in text_lower:
            return True
        return any(alias.lower() in text_lower for alias in self.aliases)


class StateSpec(BaseModel):
    """
    The State Contract: persistent constraints that control retrieval and verification.
    
    This is THE core abstraction of ContextGuard. It represents:
    - WHAT entities we're talking about
    - WHEN (time constraints)
    - WHAT metric/topic
    - HOW to normalize units
    - WHICH sources are allowed
    
    The StateSpec persists across turns and filters retrieval.
    A chunk that violates any constraint is rejected with reason codes.
    """
    model_config = ConfigDict(extra="forbid")
    
    # Identity
    thread_id: str
    
    # Entity constraints (WHO)
    entities: List[EntityRef] = Field(default_factory=list)
    
    # Semantic constraints (WHAT)
    topic: Optional[str] = None     # Domain: "finance", "policy", "news", "enterprise"
    metric: Optional[str] = None    # Specific metric: "revenue", "projection", "guidance"
    
    # Time constraints (WHEN)
    time: TimeConstraint = Field(default_factory=TimeConstraint)
    
    # Unit constraints (HOW)
    units: UnitConstraint = Field(default_factory=UnitConstraint)
    
    # Source policy (WHERE FROM)
    source_policy: SourcePolicy = Field(default_factory=SourcePolicy)
    
    # Scoping
    scope_note: Optional[str] = None  # e.g., "exclude subsidiaries", "global only"
    language: Optional[str] = "en"
    
    # Metadata for debugging and reproducibility
    spec_version: str = "v0.1"
    last_updated_turn: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def get_entity_ids(self) -> List[str]:
        """Get all entity IDs for filter construction."""
        return [e.entity_id for e in self.entities]
    
    def has_constraints(self) -> bool:
        """Check if any meaningful constraints are set."""
        return (
            len(self.entities) > 0
            or self.metric is not None
            or not self.time.is_empty()
            or not self.units.is_empty()
        )


# =============================================================================
# STATE DELTA: Changes extracted from user input
# =============================================================================


class StateDelta(BaseModel):
    """
    Partial state update extracted from new user input.
    
    This is NOT the full state; it's what changed in the current turn.
    The merge algorithm combines StateDelta with existing StateSpec.
    """
    model_config = ConfigDict(extra="forbid")
    
    # Entity changes
    entities_add: List[EntityRef] = Field(default_factory=list)
    entities_reset: bool = False  # If True, replace all entities with entities_add
    
    # Semantic changes
    metric: Optional[str] = None
    topic: Optional[str] = None
    
    # Time changes
    time: Optional[TimeConstraint] = None
    
    # Unit changes
    units: Optional[UnitConstraint] = None
    
    # Source policy changes
    source_policy: Optional[SourcePolicy] = None
    
    # Scope changes
    scope_note: Optional[str] = None
    
    # Extraction quality signals
    needs_clarification: List[ReasonCode] = Field(default_factory=list)
    extraction_confidence: float = 1.0


class MergeConflict(BaseModel):
    """Record of a conflict detected during state merge."""
    model_config = ConfigDict(extra="forbid")
    
    field: str
    old_value: Any
    new_value: Any
    reason: ReasonCode
    resolution: str  # "kept_old", "used_new", "needs_clarification"


class MergeResult(BaseModel):
    """Result of merging StateDelta into StateSpec."""
    model_config = ConfigDict(extra="forbid")
    
    state: StateSpec
    conflicts: List[MergeConflict] = Field(default_factory=list)
    warnings: List[ReasonCode] = Field(default_factory=list)
    changes_applied: List[str] = Field(default_factory=list)  # Fields that changed


# =============================================================================
# PROVENANCE: Where evidence came from
# =============================================================================


class Provenance(BaseModel):
    """
    Complete provenance chain for a piece of evidence.
    
    This is critical for:
    - Audit trails
    - Reproducibility
    - Trust calibration
    - Citations in reports
    """
    model_config = ConfigDict(extra="forbid")
    
    # Source identification
    source_id: str                  # Document ID, URL hash, or internal ID
    source_type: SourceType
    
    # Source metadata
    title: Optional[str] = None
    url: Optional[str] = None
    domain: Optional[str] = None
    author: Optional[str] = None
    
    # Temporal metadata
    published_at: Optional[str] = None   # ISO datetime
    retrieved_at: Optional[str] = None   # ISO datetime
    
    # Chunk-level provenance
    chunk_id: Optional[str] = None
    chunk_index: Optional[int] = None
    span: Optional[Tuple[int, int]] = None  # Character span in chunk text
    
    # Retrieval provenance
    retrieval_query: Optional[str] = None
    retrieval_score: Optional[float] = None


# =============================================================================
# EVIDENCE: Retrieved chunks with metadata
# =============================================================================


class Chunk(BaseModel):
    """
    A retrieved chunk of text with full metadata.
    
    This is the universal representation that works across all vector DBs.
    Adapters convert backend-specific formats to/from Chunk.
    """
    model_config = ConfigDict(extra="allow")  # Allow backend-specific metadata
    
    # Content
    text: str
    
    # Scoring
    score: Optional[float] = None  # Similarity score from retriever
    
    # Provenance (required for traceability)
    provenance: Provenance
    
    # Structured metadata for filtering
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Extracted facets (populated by gating/enrichment)
    entity_ids: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    quarter: Optional[int] = None
    
    def get_source_id(self) -> str:
        return self.provenance.source_id
    
    def get_domain(self) -> Optional[str]:
        return self.provenance.domain


class GateDecision(BaseModel):
    """
    Decision from the evidence gating layer.
    
    Every chunk gets a GateDecision explaining:
    - Was it accepted or rejected?
    - Why? (reason codes)
    - Which constraints did it match/violate?
    """
    model_config = ConfigDict(extra="forbid")
    
    accepted: bool
    reasons: List[ReasonCode] = Field(default_factory=list)
    relevance_score: Optional[float] = None
    
    # Detailed constraint matching (for debugging)
    constraint_matches: Dict[str, bool] = Field(default_factory=dict)
    # Example: {"entity": True, "time": False, "source_policy": True}


class EvidenceAssessment(BaseModel):
    """
    Full assessment of a chunk as evidence for a claim.
    
    Combines:
    - The chunk itself
    - Gate decision (why it was accepted/rejected)
    - Judge scores (support/contradict)
    - Extracted rationale
    """
    model_config = ConfigDict(extra="forbid")
    
    chunk: Chunk
    decision: GateDecision
    
    # Role determination
    role: EvidenceRole = EvidenceRole.BACKGROUND
    
    # Judge scores (0-1)
    support_score: Optional[float] = None
    contradict_score: Optional[float] = None
    
    # Extracted rationale (minimal quote that justifies verdict)
    rationale: Optional[str] = None
    rationale_span: Optional[Tuple[int, int]] = None


# =============================================================================
# CLAIMS: Atomic propositions to verify
# =============================================================================


class Claim(BaseModel):
    """
    An atomic, verifiable claim.
    
    Claims are the "program" that ContextGuard verifies.
    Each claim should be:
    - Atomic: one fact per claim
    - Testable: can be supported or contradicted by evidence
    - Specific: has clear entities, time, metrics
    """
    model_config = ConfigDict(extra="forbid")
    
    claim_id: str
    text: str
    
    # Extracted facets (for targeted retrieval)
    entities: List[str] = Field(default_factory=list)  # Entity IDs
    metric: Optional[str] = None
    time: Optional[TimeConstraint] = None
    units: Optional[UnitConstraint] = None
    
    # Claim properties
    weight: float = 1.0       # Importance weight for aggregation
    critical: bool = False    # If True, contradiction → overall contradiction
    
    # Quality flags
    is_vague: bool = False
    is_subjective: bool = False
    needs_split: bool = False
    
    @classmethod
    def generate_id(cls, text: str) -> str:
        """Generate stable ID from claim text."""
        normalized = text.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:12]


class ClaimVerdict(BaseModel):
    """
    Verdict for a single claim.
    
    Contains:
    - The claim
    - Label (SUPPORTED/CONTRADICTED/INSUFFICIENT/MIXED)
    - Confidence score
    - Reason codes explaining the verdict
    - Evidence that led to the verdict
    """
    model_config = ConfigDict(extra="forbid")
    
    claim: Claim
    label: VerdictLabel
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Explanation
    reasons: List[ReasonCode] = Field(default_factory=list)
    summary: Optional[str] = None  # Human-readable summary
    
    # Evidence
    evidence: List[EvidenceAssessment] = Field(default_factory=list)
    
    # Coverage metrics (for confidence calibration)
    coverage_sources: int = 0       # Number of unique sources
    coverage_doc_types: int = 0     # Number of unique document types
    
    # Scores used for decision (for debugging)
    support_score: Optional[float] = None
    contradict_score: Optional[float] = None
    coverage_score: Optional[float] = None


# =============================================================================
# VERDICT REPORT: The primary output
# =============================================================================


class VerdictReport(BaseModel):
    """
    The complete verification report.
    
    This is the PRIMARY OUTPUT of ContextGuard:
    - Overall verdict with confidence
    - Per-claim verdicts with citations
    - Warnings and issues
    - State used for verification
    """
    model_config = ConfigDict(extra="forbid")
    
    # Identification
    report_id: Optional[str] = None
    thread_id: str
    created_at: Optional[str] = None
    
    # State at verification time
    state: StateSpec
    
    # Overall verdict
    overall_label: VerdictLabel
    overall_confidence: float = Field(ge=0.0, le=1.0)
    
    # Per-claim verdicts
    claims: List[ClaimVerdict] = Field(default_factory=list)
    
    # Issues and warnings
    warnings: List[ReasonCode] = Field(default_factory=list)
    
    # Human-readable summary
    executive_summary: str = ""
    
    # Retrieval statistics
    total_chunks_retrieved: int = 0
    chunks_accepted: int = 0
    chunks_rejected: int = 0
    
    # Secondary output: context pack for generation
    context_pack: Optional[Dict[str, Any]] = None
    # Provenance / reproducibility
    llm_model: Optional[str] = None
    llm_prompt_version: Optional[str] = None
    llm_temperature: Optional[float] = None
    retrieval_plan: Optional[List[Dict[str, Any]]] = None
    seed: Optional[str] = None
    
    def get_supported_claims(self) -> List[ClaimVerdict]:
        return [c for c in self.claims if c.label == VerdictLabel.SUPPORTED]
    
    def get_contradicted_claims(self) -> List[ClaimVerdict]:
        return [c for c in self.claims if c.label == VerdictLabel.CONTRADICTED]
    
    def has_critical_failure(self) -> bool:
        return any(
            c.claim.critical and c.label == VerdictLabel.CONTRADICTED
            for c in self.claims
        )


# =============================================================================
# CONTEXT PACK: Secondary output for safe RAG generation
# =============================================================================


class ContextPack(BaseModel):
    """
    Safe context pack for LLM generation.
    
    This is the SECONDARY OUTPUT: a curated set of verified facts
    that can be safely fed to an LLM for generation.
    
    Only includes evidence from SUPPORTED claims.
    """
    model_config = ConfigDict(extra="forbid")
    
    # Facts-first context
    facts: List[Dict[str, Any]] = Field(default_factory=list)
    # Each fact: {"text": ..., "citation": ..., "confidence": ...}
    
    # Minimal supporting quotes
    supporting_quotes: List[Dict[str, Any]] = Field(default_factory=list)
    # Each quote: {"text": ..., "source": ..., "provenance": ...}
    
    # Constraints applied
    constraints_applied: Dict[str, Any] = Field(default_factory=dict)
    
    # Statistics
    total_facts: int = 0
    token_estimate: int = 0
    rejected_count: int = 0
    
    def to_prompt_text(self, max_tokens: int = 2000) -> str:
        """Convert to text suitable for LLM prompt."""
        lines = ["## Verified Facts\n"]
        
        for fact in self.facts:
            lines.append(f"- {fact['text']} [{fact.get('citation', 'no citation')}]")
        
        if self.supporting_quotes:
            lines.append("\n## Supporting Evidence\n")
            for quote in self.supporting_quotes[:5]:  # Limit quotes
                lines.append(f"> {quote['text']}\n> — {quote.get('source', 'unknown')}\n")
        
        result = "\n".join(lines)
        
        # Simple token estimation (rough)
        if len(result) // 4 > max_tokens:
            result = result[:max_tokens * 4] + "\n[truncated]"
        
        return result
