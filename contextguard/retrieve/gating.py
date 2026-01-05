"""
ContextGuard Evidence Gating

This module implements the evidence gating layer that:
1. Enforces constraint eligibility (hard rejection)
2. Filters noise and boilerplate
3. Enforces diversity (prevents top-k monoculture)
4. Produces reason codes for every decision

Gating is the mechanism that prevents "plausible but wrong" chunks
from reaching the verification stage.

Key insight: Similarity ≠ Relevance under constraints.
A chunk with 0.95 cosine similarity can be COMPLETELY WRONG
if it violates a time or entity constraint.

Design principle: HARD GATES, not soft penalties.
Rejected chunks are rejected with reason codes, not downranked.
This makes the system explainable and debuggable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import re

from ..core.specs import (
    Chunk,
    StateSpec,
    GateDecision,
    ReasonCode,
    DomainProfile,
)
from ..core.trace import TraceBuilder


@dataclass
class GatingConfig:
    """Configuration for the gating layer."""
    
    # Relevance thresholds
    min_relevance_score: float = 0.0  # Minimum similarity score (0 = accept all)
    
    # Diversity controls
    max_chunks_per_source: int = 3    # Max chunks from same source_id
    max_chunks_per_domain: int = 5    # Max chunks from same domain
    max_chunks_per_doc_type: Optional[int] = None  # Max per doc_type if provided in metadata
    
    # Content filters
    min_chunk_length: int = 100       # Reject chunks shorter than this
    max_chunk_length: int = 5000      # Reject chunks longer than this
    
    # Boilerplate detection
    boilerplate_patterns: List[str] = field(default_factory=lambda: [
        r"^\s*navigation\s*$",
        r"^\s*menu\s*$",
        r"^\s*copyright\s*©",
        r"^\s*all rights reserved",
        r"^\s*privacy policy",
        r"^\s*terms of service",
        r"^\s*cookie policy",
        r"^\s*subscribe to",
        r"^\s*sign up for",
        r"^\s*follow us on",
        r"^\s*share this",
        r"^\s*related articles",
        r"^\s*you may also like",
        r"^\s*advertisement",
    ])
    
    # Entity matching
    require_entity_match: bool = True  # Reject if no entity matches
    entity_match_is_soft: bool = True  # If True, missing entity info doesn't reject
    
    # Time matching
    require_time_match: bool = True    # Reject if time doesn't match
    time_match_is_soft: bool = False   # If True, missing time info doesn't reject
    allow_adjacent_years: bool = False  # Permit adjacent-year mentions
    time_match_tolerance_days: int = 0  # Allowed overlap tolerance for ranges
    fiscal_year_start_month: int = 1   # For fiscal computations (1=Jan)
    
    # Source policy
    strict_source_policy: bool = True  # Reject on source policy violation

    @classmethod
    def from_profile(cls, profile: "DomainProfile") -> "GatingConfig":
        """
        Factory presets for different domains.
        """
        base = cls()
        if profile == DomainProfile.FINANCE:
            base.max_chunks_per_source = 2
            base.allow_adjacent_years = False
            base.require_time_match = True
            base.time_match_is_soft = False
            base.fiscal_year_start_month = 2  # typical FY starting Feb for some firms
        elif profile == DomainProfile.POLICY:
            base.strict_source_policy = True
            base.require_entity_match = True
            base.allow_adjacent_years = False
            base.time_match_tolerance_days = 30  # effective vs publication
        elif profile == DomainProfile.ENTERPRISE:
            base.max_chunks_per_source = 2
            base.max_chunks_per_domain = 3
            base.strict_source_policy = True
        return base


@dataclass
class GatedChunk:
    """A chunk with its gating decision."""
    chunk: Chunk
    decision: GateDecision
    
    @property
    def accepted(self) -> bool:
        return self.decision.accepted


class EvidenceGate:
    """
    The evidence gating layer.
    
    Evaluates each chunk against:
    1. StateSpec constraints (entity, time, source policy)
    2. Quality filters (length, boilerplate)
    3. Diversity requirements (max per source)
    
    Returns GateDecision with reason codes for every chunk.
    """
    
    def __init__(self, config: Optional[GatingConfig] = None):
        self.config = config or GatingConfig()
        
        # Compile boilerplate patterns
        self._boilerplate_re = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.config.boilerplate_patterns
        ]
    
    def gate(
        self,
        chunks: List[Chunk],
        state: StateSpec,
        trace: Optional[TraceBuilder] = None,
        parents: Optional[List[str]] = None,
    ) -> List[GatedChunk]:
        """
        Gate a list of chunks against the current state.
        
        Returns GatedChunk objects with accept/reject decisions and reason codes.
        """
        results: List[GatedChunk] = []
        
        # Track diversity
        source_counts: Dict[str, int] = {}
        domain_counts: Dict[str, int] = {}
        doc_type_counts: Dict[str, int] = {}
        parent_list = parents or [None] * len(chunks)
        
        for idx, chunk in enumerate(chunks):
            chunk_parent: List[str] = []
            chunk_node: Optional[str] = None
            if trace is not None:
                if parent_list and len(parent_list) > idx and parent_list[idx]:
                    chunk_parent = [parent_list[idx]]
                chunk_node = trace.add_chunk(
                    chunk.text[:100],
                    chunk.get_source_id(),
                    chunk.score,
                    parents=chunk_parent,
                )
            decision = self._gate_single(
                chunk=chunk,
                state=state,
                source_counts=source_counts,
                domain_counts=domain_counts,
                doc_type_counts=doc_type_counts,
            )
            
            results.append(GatedChunk(chunk=chunk, decision=decision))
            
            # Update diversity counts if accepted
            if decision.accepted:
                source_id = chunk.get_source_id()
                domain = chunk.get_domain()
                doc_type = chunk.metadata.get("doc_type") if chunk.metadata else None
                
                source_counts[source_id] = source_counts.get(source_id, 0) + 1
                if domain:
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                if doc_type:
                    doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
        
            if trace is not None:
                trace.add_gate_decision(
                    accepted=decision.accepted,
                    reasons=[r.value if hasattr(r, "value") else str(r) for r in decision.reasons],
                    constraint_matches=decision.constraint_matches,
                    parents=[pid for pid in [chunk_node] if pid] or chunk_parent,
                )
        return results
    
    def _gate_single(
        self,
        chunk: Chunk,
        state: StateSpec,
        source_counts: Dict[str, int],
        domain_counts: Dict[str, int],
        doc_type_counts: Dict[str, int],
    ) -> GateDecision:
        """Gate a single chunk."""
        
        reasons: List[ReasonCode] = []
        constraint_matches: Dict[str, bool] = {}
        
        # 1. Check relevance score
        if chunk.score is not None and chunk.score < self.config.min_relevance_score:
            reasons.append(ReasonCode.EVIDENCE_LOW_RELEVANCE)
        
        # 2. Check content quality
        quality_ok, quality_reasons = self._check_quality(chunk)
        if not quality_ok:
            reasons.extend(quality_reasons)
        
        # 3. Check entity constraints
        entity_ok, entity_match = self._check_entity(chunk, state)
        constraint_matches["entity"] = entity_ok
        if not entity_ok:
            reasons.append(ReasonCode.CTXT_ENTITY_MISMATCH)
        
        # 4. Check time constraints
        time_ok, time_match = self._check_time(chunk, state)
        constraint_matches["time"] = time_ok
        if not time_ok:
            reasons.append(ReasonCode.CTXT_TIME_MISMATCH)
        
        # 5. Check source policy
        policy_ok, policy_reasons = self._check_source_policy(chunk, state)
        constraint_matches["source_policy"] = policy_ok
        if not policy_ok:
            reasons.extend(policy_reasons)
        
        # 6. Check diversity
        diversity_ok, diversity_reasons = self._check_diversity(
            chunk, source_counts, domain_counts, doc_type_counts
        )
        constraint_matches["diversity"] = diversity_ok
        if not diversity_ok:
            reasons.extend(diversity_reasons)
        
        # Decision: accept if no hard rejections
        # (Some reasons are warnings, not rejections)
        hard_rejections = {
            ReasonCode.CTXT_ENTITY_MISMATCH,
            ReasonCode.CTXT_TIME_MISMATCH,
            ReasonCode.CTXT_SOURCE_POLICY_VIOLATION,
            ReasonCode.CTXT_FRESHNESS_VIOLATION,
            ReasonCode.EVIDENCE_BOILERPLATE,
            ReasonCode.EVIDENCE_DUPLICATE,
        }
        
        has_hard_rejection = any(r in hard_rejections for r in reasons)
        accepted = not has_hard_rejection
        
        return GateDecision(
            accepted=accepted,
            reasons=reasons,
            relevance_score=chunk.score,
            constraint_matches=constraint_matches,
        )
    
    def _check_quality(self, chunk: Chunk) -> Tuple[bool, List[ReasonCode]]:
        """Check content quality (length, boilerplate)."""
        reasons = []
        
        text = chunk.text
        
        # Length checks
        if len(text) < self.config.min_chunk_length:
            reasons.append(ReasonCode.EVIDENCE_TOO_THIN)
        
        if len(text) > self.config.max_chunk_length:
            # Don't reject, just warn
            pass
        
        # Boilerplate detection
        text_lower = text.lower().strip()
        for pattern in self._boilerplate_re:
            if pattern.search(text_lower):
                reasons.append(ReasonCode.EVIDENCE_BOILERPLATE)
                break
        
        # Check for very low alpha ratio (likely garbage)
        alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
        if alpha_ratio < 0.3:
            reasons.append(ReasonCode.EVIDENCE_TOO_THIN)
        
        return len(reasons) == 0, reasons
    
    def _check_entity(
        self,
        chunk: Chunk,
        state: StateSpec,
    ) -> Tuple[bool, bool]:
        """
        Check entity constraint.
        
        Returns (passed, matched):
        - passed: whether the check passed (may pass even without match if soft)
        - matched: whether an entity actually matched
        """
        if not state.entities:
            # No entity constraint
            return True, False
        
        if not self.config.require_entity_match:
            return True, False
        
        # Check if chunk has entity info
        chunk_entities = set(chunk.entity_ids)
        state_entities = set(e.entity_id for e in state.entities)
        
        if chunk_entities:
            # Chunk has entity info - check for overlap
            matched = bool(chunk_entities & state_entities)
            return matched or not self.config.require_entity_match, matched
        
        # Chunk has no entity info - do text matching
        for entity in state.entities:
            if entity.matches_text(chunk.text):
                return True, True
        
        # No match found
        if self.config.entity_match_is_soft:
            # Soft mode: don't reject if entity info is missing
            return True, False
        else:
            return False, False
    
    def _check_time(
        self,
        chunk: Chunk,
        state: StateSpec,
    ) -> Tuple[bool, bool]:
        """
        Check time constraint.
        
        Returns (passed, matched).
        """
        if state.time.is_empty():
            # No time constraint
            return True, False
        
        if not self.config.require_time_match:
            return True, False
        
        tc = state.time

        # Helper: parse date string
        def parse_date(val: Optional[str]) -> Optional[datetime]:
            if not val:
                return None
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            except Exception:
                return None

        # Extract chunk temporal info
        c_year = chunk.year
        c_quarter = getattr(chunk, "quarter", None) or chunk.metadata.get("quarter")
        c_start = parse_date(chunk.metadata.get("start_date"))
        c_end = parse_date(chunk.metadata.get("end_date"))

        # TimeConstraint components
        t_year = tc.year
        t_quarter = tc.quarter
        t_start = parse_date(tc.start_date)
        t_end = parse_date(tc.end_date)

        # Quarter match if specified
        if t_quarter is not None:
            if c_quarter is not None:
                if c_quarter != t_quarter:
                    return False, False
            else:
                # quarter missing
                if not self.config.time_match_is_soft:
                    return False, False

        # Date range overlap if provided
        if t_start or t_end:
            # If chunk has no dates, fallback to soft handling
            if not c_start and c_year:
                c_start = datetime(c_year, 1, 1)
            if not c_end and c_year:
                c_end = datetime(c_year, 12, 31)
            if c_start and c_end:
                tol = timedelta(days=self.config.time_match_tolerance_days)
                if t_start and c_end + tol < t_start:
                    return False, False
                if t_end and c_start - tol > t_end:
                    return False, False
                return True, True
            return (self.config.time_match_is_soft, False)

        # Fiscal vs calendar year handling
        if t_year is not None and c_year is not None:
            if tc.fiscal:
                # For fiscal, require same fiscal year unless explicitly allowed
                if c_year != t_year:
                    return False, False
                return True, True
            else:
                if c_year == t_year:
                    return True, True
                if self.config.allow_adjacent_years and c_year in {t_year - 1, t_year + 1}:
                    return True, False
                return False, False

        # If we reach here, no strong signal
        return (self.config.time_match_is_soft, False)
    
    def _check_source_policy(
        self,
        chunk: Chunk,
        state: StateSpec,
    ) -> Tuple[bool, List[ReasonCode]]:
        """Check source policy constraints."""
        reasons = []
        policy = state.source_policy
        
        # Check source type
        if policy.allowed_source_types:
            if chunk.provenance.source_type not in policy.allowed_source_types:
                if self.config.strict_source_policy:
                    reasons.append(ReasonCode.CTXT_SOURCE_POLICY_VIOLATION)
        
        # Check domain
        domain = chunk.provenance.domain
        if domain:
            if policy.blocked_domains and domain in policy.blocked_domains:
                reasons.append(ReasonCode.CTXT_SOURCE_POLICY_VIOLATION)
            
            if policy.allowed_domains is not None:
                if domain not in policy.allowed_domains:
                    reasons.append(ReasonCode.CTXT_SOURCE_POLICY_VIOLATION)
        
        # Check freshness
        if policy.max_age_days is not None:
            published = chunk.provenance.published_at
            if published:
                try:
                    pub_date = datetime.fromisoformat(published.replace('Z', '+00:00'))
                    age = datetime.now(pub_date.tzinfo) - pub_date
                    if age.days > policy.max_age_days:
                        reasons.append(ReasonCode.CTXT_FRESHNESS_VIOLATION)
                except (ValueError, TypeError):
                    pass  # Can't parse date - don't reject
        
        return len(reasons) == 0, reasons
    
    def _check_diversity(
        self,
        chunk: Chunk,
        source_counts: Dict[str, int],
        domain_counts: Dict[str, int],
        doc_type_counts: Dict[str, int],
    ) -> Tuple[bool, List[ReasonCode]]:
        """Check diversity constraints."""
        reasons = []
        
        source_id = chunk.get_source_id()
        domain = chunk.get_domain()
        doc_type = chunk.metadata.get("doc_type") if chunk.metadata else None
        
        # Check per-source limit
        if source_counts.get(source_id, 0) >= self.config.max_chunks_per_source:
            reasons.append(ReasonCode.EVIDENCE_DUPLICATE)
        
        # Check per-domain limit
        if domain and domain_counts.get(domain, 0) >= self.config.max_chunks_per_domain:
            reasons.append(ReasonCode.EVIDENCE_DUPLICATE)

        # Check per-doc-type limit
        if self.config.max_chunks_per_doc_type is not None and doc_type:
            if doc_type_counts.get(doc_type, 0) >= self.config.max_chunks_per_doc_type:
                reasons.append(ReasonCode.EVIDENCE_DUPLICATE)
        
        return len(reasons) == 0, reasons


# =============================================================================
# Convenience functions
# =============================================================================


def gate_chunks(
    chunks: List[Chunk],
    state: StateSpec,
    config: Optional[GatingConfig] = None,
    trace: Optional[TraceBuilder] = None,
    parents: Optional[List[str]] = None,
) -> List[GatedChunk]:
    """
    Convenience function to gate chunks.
    
    Returns list of GatedChunk with accept/reject decisions.
    """
    gate = EvidenceGate(config=config)
    return gate.gate(chunks, state, trace=trace, parents=parents)


def filter_accepted(gated: List[GatedChunk]) -> List[Chunk]:
    """Get only accepted chunks."""
    return [g.chunk for g in gated if g.accepted]


def filter_rejected(gated: List[GatedChunk]) -> List[GatedChunk]:
    """Get only rejected chunks with their decisions."""
    return [g for g in gated if not g.accepted]


def summarize_gating(gated: List[GatedChunk]) -> Dict[str, Any]:
    """
    Summarize gating results.
    
    Useful for debugging and reporting.
    """
    accepted = [g for g in gated if g.accepted]
    rejected = [g for g in gated if not g.accepted]
    
    # Count reasons
    reason_counts: Dict[str, int] = {}
    for g in rejected:
        for reason in g.decision.reasons:
            reason_name = reason.value if hasattr(reason, 'value') else str(reason)
            reason_counts[reason_name] = reason_counts.get(reason_name, 0) + 1
    
    # Source diversity for accepted
    sources = set(g.chunk.get_source_id() for g in accepted)
    
    return {
        "total": len(gated),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "acceptance_rate": len(accepted) / max(len(gated), 1),
        "unique_sources": len(sources),
        "rejection_reasons": reason_counts,
    }


def explain_rejection(gated_chunk: GatedChunk) -> str:
    """
    Generate human-readable explanation for a rejection.
    """
    if gated_chunk.accepted:
        return "Chunk was accepted."
    
    reasons = gated_chunk.decision.reasons
    constraints = gated_chunk.decision.constraint_matches
    
    lines = ["Chunk was REJECTED:"]
    
    for reason in reasons:
        reason_name = reason.value if hasattr(reason, 'value') else str(reason)
        explanation = _reason_explanations.get(reason_name, reason_name)
        lines.append(f"  - {explanation}")
    
    if constraints:
        lines.append("\nConstraint check results:")
        for constraint, passed in constraints.items():
            status = "✓" if passed else "✗"
            lines.append(f"  {status} {constraint}")
    
    return "\n".join(lines)


_reason_explanations = {
    "CTXT_ENTITY_MISMATCH": "Chunk doesn't mention the required entity/entities",
    "CTXT_TIME_MISMATCH": "Chunk is from the wrong time period",
    "CTXT_SOURCE_POLICY_VIOLATION": "Chunk source violates the source policy",
    "CTXT_FRESHNESS_VIOLATION": "Chunk is too old (exceeds max age)",
    "EVIDENCE_LOW_RELEVANCE": "Chunk has low relevance score",
    "EVIDENCE_DUPLICATE": "Too many chunks already accepted from this source",
    "EVIDENCE_BOILERPLATE": "Chunk appears to be boilerplate/navigation text",
    "EVIDENCE_TOO_THIN": "Chunk is too short or lacks substantive content",
}
