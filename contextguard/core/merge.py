"""
ContextGuard State Merge Algorithm

This module implements the state merge logic that handles:
- Constraint carryover across turns
- Explicit reset semantics
- Conflict detection and resolution
- Merge rule application

The merge algorithm is THE mechanism that prevents "context drift."

Merge Principles:
1. PERSIST by default: if not mentioned, carry forward
2. RESET is explicit: entities only reset when entities_reset=True
3. CONFLICTS are recorded: don't silently overwrite ambiguous fields
4. HARD CONSTRAINTS: source policy + time are sticky unless explicitly overridden
"""

from __future__ import annotations

from typing import List, Optional
from copy import deepcopy

from .specs import (
    StateSpec,
    StateDelta,
    MergeResult,
    MergeConflict,
    ReasonCode,
    EntityRef,
    TimeConstraint,
    UnitConstraint,
    SourcePolicy,
)


class MergeConfig:
    """Configuration for merge behavior."""
    
    def __init__(
        self,
        # Conflict thresholds
        low_confidence_threshold: float = 0.5,
        # Entity handling
        allow_entity_merge: bool = True,
        max_entities: int = 10,
        # Time handling
        time_conflict_requires_clarification: bool = True,
        # Metric handling
        metric_change_is_override: bool = True,
        # Debug
        verbose: bool = False,
    ):
        self.low_confidence_threshold = low_confidence_threshold
        self.allow_entity_merge = allow_entity_merge
        self.max_entities = max_entities
        self.time_conflict_requires_clarification = time_conflict_requires_clarification
        self.metric_change_is_override = metric_change_is_override
        self.verbose = verbose


DEFAULT_CONFIG = MergeConfig()


def merge_state(
    prev: StateSpec,
    delta: StateDelta,
    turn_id: int,
    config: MergeConfig = DEFAULT_CONFIG,
) -> MergeResult:
    """
    Merge a StateDelta into an existing StateSpec.
    
    This is the core algorithm that maintains constraint continuity.
    
    Args:
        prev: Previous state (from prior turn)
        delta: Changes extracted from current user input
        turn_id: Current turn number
        config: Merge configuration
        
    Returns:
        MergeResult containing:
        - new state
        - list of conflicts detected
        - list of warnings
        - list of changes applied
    """
    # Start with a deep copy of previous state
    new_state = deepcopy(prev)
    conflicts: List[MergeConflict] = []
    warnings: List[ReasonCode] = []
    changes: List[str] = []
    
    # Update turn metadata
    new_state.last_updated_turn = turn_id
    
    # Step A: Merge entities
    entity_result = _merge_entities(
        prev_entities=prev.entities,
        delta_entities=delta.entities_add,
        reset=delta.entities_reset,
        config=config,
    )
    new_state.entities = entity_result["entities"]
    if entity_result.get("changed"):
        changes.append("entities")
    if entity_result.get("conflict"):
        conflicts.append(entity_result["conflict"])
    if entity_result.get("warning"):
        warnings.append(entity_result["warning"])
    
    # Step B: Merge metric
    metric_result = _merge_metric(
        prev_metric=prev.metric,
        delta_metric=delta.metric,
        confidence=delta.extraction_confidence,
        config=config,
    )
    if metric_result.get("value") is not None:
        new_state.metric = metric_result["value"]
        changes.append("metric")
    elif metric_result.get("keep_prev"):
        pass  # Keep previous
    if metric_result.get("conflict"):
        conflicts.append(metric_result["conflict"])
    
    # Step C: Merge topic
    if delta.topic is not None:
        new_state.topic = delta.topic
        changes.append("topic")
    
    # Step D: Merge time constraints
    time_result = _merge_time(
        prev_time=prev.time,
        delta_time=delta.time,
        config=config,
    )
    new_state.time = time_result["time"]
    if time_result.get("changed"):
        changes.append("time")
    if time_result.get("conflict"):
        conflicts.append(time_result["conflict"])
    if time_result.get("warning"):
        warnings.append(time_result["warning"])
    
    # Step E: Merge unit constraints
    unit_result = _merge_units(
        prev_units=prev.units,
        delta_units=delta.units,
        config=config,
    )
    new_state.units = unit_result["units"]
    if unit_result.get("changed"):
        changes.append("units")
    if unit_result.get("conflict"):
        conflicts.append(unit_result["conflict"])
    
    # Step F: Merge source policy
    policy_result = _merge_source_policy(
        prev_policy=prev.source_policy,
        delta_policy=delta.source_policy,
        config=config,
    )
    new_state.source_policy = policy_result["policy"]
    if policy_result.get("changed"):
        changes.append("source_policy")
    
    # Step G: Merge scope note
    if delta.scope_note is not None:
        if prev.scope_note and prev.scope_note != delta.scope_note:
            # Scope changed - record as warning
            warnings.append(ReasonCode.CTXT_SCOPE_MISMATCH)
        new_state.scope_note = delta.scope_note
        changes.append("scope_note")
    
    # Add any clarification needs from delta
    warnings.extend(delta.needs_clarification)
    
    # Final validation
    if not new_state.entities and prev.entities:
        # Entities were cleared without explicit reset
        if not delta.entities_reset:
            warnings.append(ReasonCode.CLAIM_NEEDS_CLARIFICATION)
    
    return MergeResult(
        state=new_state,
        conflicts=conflicts,
        warnings=list(set(warnings)),  # Dedupe
        changes_applied=changes,
    )


def _merge_entities(
    prev_entities: List[EntityRef],
    delta_entities: List[EntityRef],
    reset: bool,
    config: MergeConfig,
) -> dict:
    """Merge entity lists according to reset/add semantics."""
    result: dict = {"changed": False}
    
    if reset:
        # Full reset: replace with new entities
        result["entities"] = delta_entities
        result["changed"] = True
        
        if not delta_entities:
            # Reset to empty - needs clarification
            result["warning"] = ReasonCode.CTXT_ENTITY_AMBIGUOUS
            
        return result
    
    if not delta_entities:
        # No change
        result["entities"] = prev_entities
        return result
    
    if not config.allow_entity_merge:
        # Replace mode
        result["entities"] = delta_entities
        result["changed"] = True
        return result
    
    # Merge mode: union by entity_id
    existing_ids = {e.entity_id for e in prev_entities}
    merged = list(prev_entities)
    
    for new_entity in delta_entities:
        if new_entity.entity_id not in existing_ids:
            merged.append(new_entity)
            existing_ids.add(new_entity.entity_id)
            result["changed"] = True
        else:
            # Entity exists - check for alias updates
            for i, existing in enumerate(merged):
                if existing.entity_id == new_entity.entity_id:
                    # Merge aliases
                    new_aliases = set(existing.aliases) | set(new_entity.aliases)
                    if new_aliases != set(existing.aliases):
                        merged[i] = EntityRef(
                            entity_id=existing.entity_id,
                            display_name=new_entity.display_name or existing.display_name,
                            aliases=list(new_aliases),
                            entity_type=new_entity.entity_type or existing.entity_type,
                        )
                        result["changed"] = True
                    break
    
    # Enforce max entities
    if len(merged) > config.max_entities:
        result["warning"] = ReasonCode.CTXT_ENTITY_AMBIGUOUS
        merged = merged[:config.max_entities]
    
    result["entities"] = merged
    return result


def _merge_metric(
    prev_metric: Optional[str],
    delta_metric: Optional[str],
    confidence: float,
    config: MergeConfig,
) -> dict:
    """Merge metric field with conflict detection."""
    result: dict = {}
    
    if delta_metric is None:
        result["keep_prev"] = True
        return result
    
    if prev_metric is None:
        result["value"] = delta_metric
        return result
    
    # Both exist - check for conflict
    if prev_metric.lower() != delta_metric.lower():
        if config.metric_change_is_override:
            # Accept the new metric
            result["value"] = delta_metric
            
            # But if confidence is low, record a conflict
            if confidence < config.low_confidence_threshold:
                result["conflict"] = MergeConflict(
                    field="metric",
                    old_value=prev_metric,
                    new_value=delta_metric,
                    reason=ReasonCode.CTXT_METRIC_MISMATCH,
                    resolution="used_new",
                )
        else:
            # Keep old and record conflict
            result["keep_prev"] = True
            result["conflict"] = MergeConflict(
                field="metric",
                old_value=prev_metric,
                new_value=delta_metric,
                reason=ReasonCode.CTXT_METRIC_MISMATCH,
                resolution="kept_old",
            )
    else:
        # Same metric (case-insensitive match)
        result["keep_prev"] = True
    
    return result


def _merge_time(
    prev_time: TimeConstraint,
    delta_time: Optional[TimeConstraint],
    config: MergeConfig,
) -> dict:
    """Merge time constraints with field-level granularity."""
    result: dict = {"changed": False}
    
    if delta_time is None:
        result["time"] = prev_time
        return result
    
    # Create new time from merging fields
    new_time = TimeConstraint(
        year=delta_time.year if delta_time.year is not None else prev_time.year,
        quarter=delta_time.quarter if delta_time.quarter is not None else prev_time.quarter,
        start_date=delta_time.start_date if delta_time.start_date is not None else prev_time.start_date,
        end_date=delta_time.end_date if delta_time.end_date is not None else prev_time.end_date,
        fiscal=delta_time.fiscal if delta_time.fiscal else prev_time.fiscal,
    )
    
    # Detect conflicts
    if prev_time.year is not None and delta_time.year is not None:
        if prev_time.year != delta_time.year:
            if config.time_conflict_requires_clarification:
                result["conflict"] = MergeConflict(
                    field="time.year",
                    old_value=prev_time.year,
                    new_value=delta_time.year,
                    reason=ReasonCode.CTXT_TIME_MISMATCH,
                    resolution="used_new",
                )
    
    # Check if date range conflicts with year
    if new_time.start_date and new_time.year:
        # If start_date year doesn't match, warn
        try:
            start_year = int(new_time.start_date[:4])
            if start_year != new_time.year:
                result["warning"] = ReasonCode.CTXT_TIME_AMBIGUOUS
        except (ValueError, IndexError):
            pass
    
    result["time"] = new_time
    result["changed"] = new_time != prev_time
    
    return result


def _merge_units(
    prev_units: UnitConstraint,
    delta_units: Optional[UnitConstraint],
    config: MergeConfig,
) -> dict:
    """Merge unit constraints."""
    result: dict = {"changed": False}
    
    if delta_units is None:
        result["units"] = prev_units
        return result
    
    # Create new units from merging fields
    new_units = UnitConstraint(
        currency=delta_units.currency if delta_units.currency is not None else prev_units.currency,
        scale=delta_units.scale if delta_units.scale is not None else prev_units.scale,
        basis=delta_units.basis if delta_units.basis is not None else prev_units.basis,
    )
    
    # Detect currency change (important for finance)
    if prev_units.currency and delta_units.currency:
        if prev_units.currency != delta_units.currency:
            result["conflict"] = MergeConflict(
                field="units.currency",
                old_value=prev_units.currency,
                new_value=delta_units.currency,
                reason=ReasonCode.CTXT_UNIT_SCALE_MISMATCH,
                resolution="used_new",
            )
    
    result["units"] = new_units
    result["changed"] = new_units != prev_units
    
    return result


def _merge_source_policy(
    prev_policy: SourcePolicy,
    delta_policy: Optional[SourcePolicy],
    config: MergeConfig,
) -> dict:
    """Merge source policy with careful handling of allow/block lists."""
    result: dict = {"changed": False}
    
    if delta_policy is None:
        result["policy"] = prev_policy
        return result
    
    # For source policy, we typically want full override when specified
    # because partial merging of allow/block lists is confusing
    
    new_policy = SourcePolicy(
        allow_web=delta_policy.allow_web if delta_policy.allow_web is not None else prev_policy.allow_web,
        allow_corpus=delta_policy.allow_corpus if delta_policy.allow_corpus is not None else prev_policy.allow_corpus,
        allowed_source_types=delta_policy.allowed_source_types if delta_policy.allowed_source_types else prev_policy.allowed_source_types,
        preferred_source_types=delta_policy.preferred_source_types if delta_policy.preferred_source_types else prev_policy.preferred_source_types,
        allowed_domains=delta_policy.allowed_domains if delta_policy.allowed_domains is not None else prev_policy.allowed_domains,
        blocked_domains=_merge_domain_lists(prev_policy.blocked_domains, delta_policy.blocked_domains),
        max_age_days=delta_policy.max_age_days if delta_policy.max_age_days is not None else prev_policy.max_age_days,
    )
    
    result["policy"] = new_policy
    result["changed"] = True  # Any policy change is considered a change
    
    return result


def _merge_domain_lists(
    prev: Optional[List[str]],
    delta: Optional[List[str]],
) -> Optional[List[str]]:
    """Merge domain block lists (union)."""
    if prev is None and delta is None:
        return None
    if prev is None:
        return delta
    if delta is None:
        return prev
    # Union of blocked domains
    return list(set(prev) | set(delta))


# =============================================================================
# Convenience functions
# =============================================================================


def apply_delta(
    state: StateSpec,
    delta: StateDelta,
    turn_id: int,
) -> StateSpec:
    """
    Simple merge that returns just the new state.
    Ignores conflicts and warnings (use merge_state for full control).
    """
    result = merge_state(state, delta, turn_id)
    return result.state


def create_initial_state(thread_id: str) -> StateSpec:
    """Create a fresh state for a new thread."""
    return StateSpec(thread_id=thread_id)


def detect_reset_intent(message: str) -> bool:
    """
    Heuristic to detect if user wants to reset context.
    
    Looks for phrases like:
    - "start over"
    - "new topic"
    - "forget that"
    - "actually, let's talk about..."
    """
    reset_phrases = [
        "start over",
        "new topic",
        "forget that",
        "forget about",
        "never mind",
        "let's talk about something else",
        "different question",
        "change of topic",
        "actually, ",
        "wait, let me ask",
    ]
    message_lower = message.lower()
    return any(phrase in message_lower for phrase in reset_phrases)


def summarize_state_changes(result: MergeResult) -> str:
    """
    Generate a human-readable summary of what changed.
    Useful for debugging and user feedback.
    """
    lines = []
    
    if result.changes_applied:
        lines.append(f"Updated: {', '.join(result.changes_applied)}")
    
    if result.conflicts:
        for conflict in result.conflicts:
            lines.append(
                f"Conflict in {conflict.field}: "
                f"'{conflict.old_value}' → '{conflict.new_value}' "
                f"({conflict.resolution})"
            )
    
    if result.warnings:
        warning_names = [w.value for w in result.warnings]
        lines.append(f"Warnings: {', '.join(warning_names)}")
    
    if not lines:
        return "No changes detected."
    
    return "\n".join(lines)
