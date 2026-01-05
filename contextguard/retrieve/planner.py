"""
ContextGuard Retrieval Planner

This module generates retrieval plans that enforce:
1. Coverage: at least one query per entity × claim combination
2. Counter-evidence: always search for contradictions (anti-confirmation-bias)
3. Constraint injection: queries include state constraints

The planner is the difference between "top-k once" and "systematic evidence gathering."

Key insight: Most RAG systems fail because they retrieve once and hope.
ContextGuard retrieves systematically based on what needs to be verified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib

from ..core.specs import (
    StateSpec,
    Claim,
    DomainProfile,
)
from ..core import settings
from ..core.trace import TraceBuilder
from .protocols import CanonicalFilters


class QueryType(str, Enum):
    """Types of retrieval queries."""
    SUPPORT = "support"           # Looking for supporting evidence
    COUNTER = "counter"           # Looking for contradicting evidence
    BACKGROUND = "background"     # General context/background
    PRIMARY = "primary"           # Primary source only


@dataclass
class RetrievalStep:
    """
    A single step in the retrieval plan.
    
    Each step is a query with:
    - The query text
    - Filters to apply
    - Query type (support/counter/background)
    - Target claim (optional)
    """
    step_id: str
    query: str
    query_type: QueryType
    filters: CanonicalFilters
    
    # Targeting
    claim_id: Optional[str] = None
    entity_id: Optional[str] = None
    
    # Execution parameters
    k: int = 10
    priority: int = 0  # Higher = execute first
    
    # Metadata for tracing
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_node_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "query": self.query,
            "query_type": self.query_type.value,
            "filters": self.filters.to_dict(),
            "claim_id": self.claim_id,
            "entity_id": self.entity_id,
            "k": self.k,
            "priority": self.priority,
        }


@dataclass
class RetrievalPlan:
    """
    A complete retrieval plan with ordered steps.
    """
    plan_id: str
    steps: List[RetrievalStep] = field(default_factory=list)
    
    # Source plan
    state_id: Optional[str] = None
    claim_ids: List[str] = field(default_factory=list)
    trace_node_id: Optional[str] = None
    
    # Execution hints
    total_k: int = 50  # Target total chunks
    enable_counter: bool = True
    
    def get_steps_for_claim(self, claim_id: str) -> List[RetrievalStep]:
        """Get all steps targeting a specific claim."""
        return [s for s in self.steps if s.claim_id == claim_id]
    
    def get_support_steps(self) -> List[RetrievalStep]:
        """Get support query steps."""
        return [s for s in self.steps if s.query_type == QueryType.SUPPORT]
    
    def get_counter_steps(self) -> List[RetrievalStep]:
        """Get counter-evidence query steps."""
        return [s for s in self.steps if s.query_type == QueryType.COUNTER]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "steps": [s.to_dict() for s in self.steps],
            "claim_ids": self.claim_ids,
            "total_k": self.total_k,
            "enable_counter": self.enable_counter,
        }


class RetrievalPlanner:
    """
    Plans retrieval based on claims and state constraints.
    
    The planner ensures:
    1. Every claim gets at least one support query
    2. Every claim gets at least one counter query (if enabled)
    3. Queries are constrained by StateSpec (entity, time, source policy)
    4. Multi-entity claims get per-entity queries
    """
    
    def __init__(
        self,
        default_k_per_step: int = 10,
        max_steps: int = 20,
        enable_counter: bool = True,
        counter_keywords: Optional[List[str]] = None,
        profile: Optional["DomainProfile"] = None,
    ):
        self.default_k = default_k_per_step
        self.max_steps = max_steps
        self.enable_counter = enable_counter
        self.counter_keywords = counter_keywords or [
            "false", "not true", "denied", "disputed", 
            "incorrect", "misleading", "refuted", "debunked",
            "controversy", "criticism", "opposite"
        ]
        self.profile = profile
    
    def plan(
        self,
        claims: List[Claim],
        state: StateSpec,
        total_k: int = 50,
        trace: Optional[TraceBuilder] = None,
        trace_parents: Optional[List[str]] = None,
    ) -> RetrievalPlan:
        """
        Generate a retrieval plan for the given claims and state.
        
        Strategy:
        1. For each claim, generate support + counter queries
        2. Distribute k across steps based on claim weight
        3. Apply state constraints to all queries
        """
        # Clamp budgets
        claims = claims[: settings.MAX_CLAIMS]
        total_k = min(total_k, settings.MAX_TOTAL_K)

        plan_id = self._generate_plan_id(claims, state)
        steps: List[RetrievalStep] = []
        
        # Calculate per-claim k allocation
        total_weight = sum(c.weight for c in claims) or 1.0
        base_k_per_claim = total_k / len(claims) if claims else total_k
        
        for claim in claims:
            # Weight-adjusted k for this claim
            claim_k = int(base_k_per_claim * (claim.weight / (total_weight / len(claims))))
            claim_k = max(1, min(claim_k, settings.MAX_CHUNKS_PER_CLAIM))
            
            # Generate steps for this claim
            claim_steps = self._plan_for_claim(
                claim=claim,
                state=state,
                target_k=claim_k,
            )
            steps.extend(claim_steps)
        
        # Limit total steps
        if len(steps) > self.max_steps:
            # Prioritize support over counter, higher weight claims first
            steps.sort(key=lambda s: (
                0 if s.query_type == QueryType.SUPPORT else 1,
                -s.priority,
            ))
            steps = steps[:self.max_steps]
        
        plan_node_id = None
        if trace is not None:
            plan_node_id = trace.add_plan(plan_id, len(steps), parents=trace_parents or [])
            for step in steps:
                step.trace_node_id = trace.add_plan_step(
                    step_id=step.step_id,
                    query=step.query,
                    query_type=step.query_type.value,
                    k=step.k,
                    parents=[plan_node_id],
                )
        return RetrievalPlan(
            plan_id=plan_id,
            steps=steps,
            state_id=state.thread_id,
            claim_ids=[c.claim_id for c in claims],
            total_k=total_k,
            enable_counter=self.enable_counter,
            trace_node_id=plan_node_id,
        )
    
    def _plan_for_claim(
        self,
        claim: Claim,
        state: StateSpec,
        target_k: int,
    ) -> List[RetrievalStep]:
        """Generate retrieval steps for a single claim."""
        steps = []
        
        # Determine entities to query
        # Use claim entities if specified, otherwise use state entities
        entities = claim.entities or [e.entity_id for e in state.entities]
        
        # Build base filters from state
        base_filters = CanonicalFilters.from_state_spec(state)
        
        # Override with claim-specific constraints
        if claim.time:
            if claim.time.year:
                base_filters.year = claim.time.year
            if claim.time.quarter:
                base_filters.quarter = claim.time.quarter
        
        # Decide query strategy based on entities
        if len(entities) <= 2:
            # Few entities: one query per entity
            per_entity_k = target_k // (len(entities) or 1)
            for entity_id in entities:
                # Support query
                support_query = self._build_support_query(claim, entity_id, state)
                entity_filters = base_filters.model_copy()
                entity_filters.entity_ids = [entity_id]
                
                steps.append(RetrievalStep(
                    step_id=self._step_id(claim.claim_id, entity_id, "support"),
                    query=support_query,
                    query_type=QueryType.SUPPORT,
                    filters=entity_filters,
                    claim_id=claim.claim_id,
                    entity_id=entity_id,
                    k=per_entity_k,
                    priority=10,  # Support queries are higher priority
                ))
                
                # Counter query (if enabled)
                if self.enable_counter:
                    counter_query = self._build_counter_query(claim, entity_id, state)
                    steps.append(RetrievalStep(
                        step_id=self._step_id(claim.claim_id, entity_id, "counter"),
                        query=counter_query,
                        query_type=QueryType.COUNTER,
                        filters=entity_filters,
                        claim_id=claim.claim_id,
                        entity_id=entity_id,
                        k=per_entity_k // 2,  # Fewer counter results needed
                        priority=5,
                    ))
        else:
            # Many entities: one combined query
            support_query = self._build_support_query(claim, None, state)
            steps.append(RetrievalStep(
                step_id=self._step_id(claim.claim_id, "all", "support"),
                query=support_query,
                query_type=QueryType.SUPPORT,
                filters=base_filters,
                claim_id=claim.claim_id,
                k=target_k,
                priority=10,
            ))
            
            if self.enable_counter:
                counter_query = self._build_counter_query(claim, None, state)
                steps.append(RetrievalStep(
                    step_id=self._step_id(claim.claim_id, "all", "counter"),
                    query=counter_query,
                    query_type=QueryType.COUNTER,
                    filters=base_filters,
                    claim_id=claim.claim_id,
                    k=target_k // 2,
                    priority=5,
                ))
        
        return steps
    
    def _build_support_query(
        self,
        claim: Claim,
        entity_id: Optional[str],
        state: StateSpec,
    ) -> str:
        """
        Build a support query for a claim.
        
        Strategy:
        - Start with claim text
        - Add entity name if targeting specific entity
        - Add time context if specified
        - Add metric context if specified
        """
        parts = [claim.text]
        
        # Add entity context
        if entity_id:
            # Try to get display name
            for entity in state.entities:
                if entity.entity_id == entity_id:
                    if entity.display_name:
                        parts.append(entity.display_name)
                    break
            else:
                parts.append(entity_id)
        
        # Add time context
        if state.time.year:
            parts.append(str(state.time.year))
        if state.time.quarter:
            parts.append(f"Q{state.time.quarter}")
        
        # Add metric context
        if state.metric:
            parts.append(state.metric)
        
        return " ".join(parts)
    
    def _build_counter_query(
        self,
        claim: Claim,
        entity_id: Optional[str],
        state: StateSpec,
    ) -> str:
        """
        Build a counter-evidence query for a claim.
        
        Strategy:
        - Start with claim text
        - Add negation/contradiction keywords
        - Keep entity/time context
        
        This is critical for avoiding confirmation bias.
        """
        # Start with base support query
        base = self._build_support_query(claim, entity_id, state)
        
        # Add counter keywords
        # Use a few relevant ones based on claim content
        keywords = self._select_counter_keywords(claim.text)
        
        return f"{base} {' OR '.join(keywords)}"
    
    def _select_counter_keywords(self, claim_text: str) -> List[str]:
        """Select appropriate counter keywords based on claim content."""
        # Simple heuristic: use 3 keywords
        # In production, you might use NLI or claim type classification
        
        claim_lower = claim_text.lower()
        
        # If claim is about numbers/financials
        if any(word in claim_lower for word in ["revenue", "profit", "sales", "growth", "$", "million", "billion"]):
            return ["incorrect", "misleading", "restated"]
        
        # If claim is about statements/quotes
        if any(word in claim_lower for word in ["said", "announced", "claimed", "stated"]):
            return ["denied", "retracted", "clarified"]
        
        # Default set
        return ["false", "disputed", "not true"]
    
    def _generate_plan_id(self, claims: List[Claim], state: StateSpec) -> str:
        """Generate a stable plan ID."""
        content = f"{state.thread_id}:{','.join(c.claim_id for c in claims)}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def _step_id(self, claim_id: str, entity: str, query_type: str) -> str:
        """Generate a step ID."""
        content = f"{claim_id}:{entity}:{query_type}"
        return hashlib.sha256(content.encode()).hexdigest()[:8]


# =============================================================================
# Convenience functions
# =============================================================================


def plan_retrieval(
    claims: List[Claim],
    state: StateSpec,
    total_k: int = 50,
    enable_counter: bool = True,
    trace: Optional[TraceBuilder] = None,
    trace_parents: Optional[List[str]] = None,
    profile: Optional["DomainProfile"] = None,
) -> RetrievalPlan:
    """
    Convenience function to create a retrieval plan.
    
    Uses default planner settings.
    """
    planner = RetrievalPlanner(enable_counter=enable_counter, profile=profile)
    return planner.plan(claims, state, total_k=total_k, trace=trace, trace_parents=trace_parents)


def estimate_plan_cost(plan: RetrievalPlan) -> Dict[str, Any]:
    """
    Estimate the cost/size of executing a retrieval plan.
    
    Useful for budgeting and planning.
    """
    total_k = sum(step.k for step in plan.steps)
    
    return {
        "total_steps": len(plan.steps),
        "support_steps": len(plan.get_support_steps()),
        "counter_steps": len(plan.get_counter_steps()),
        "total_k": total_k,
        "estimated_chunks": total_k,
        "unique_claims": len(set(s.claim_id for s in plan.steps if s.claim_id)),
        "unique_entities": len(set(s.entity_id for s in plan.steps if s.entity_id)),
    }
