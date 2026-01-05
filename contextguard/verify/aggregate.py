"""
ContextGuard Verdict Aggregation

This module implements the verdict aggregation logic:
1. Per-claim aggregation: combine multiple evidence assessments into a claim verdict
2. Overall aggregation: combine multiple claim verdicts into an overall verdict

The aggregation layer is the "linker" of the verification compiler.
It produces the final executable (verdict report).

Key design decisions:
- Critical claim contradiction → overall contradiction
- Low coverage → lower confidence
- Mixed evidence → MIXED or INSUFFICIENT verdict
- Weighted claims affect overall verdict proportionally
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import math

from ..core.specs import (
    Claim,
    ClaimVerdict,
    VerdictLabel,
    EvidenceAssessment,
    ReasonCode,
    Chunk,
    Provenance,
    SourceType,
    GateDecision,
    DomainProfile,
)
from ..core.trace import TraceBuilder
from .judges import JudgeResult


# =============================================================================
# AGGREGATION CONFIG
# =============================================================================


@dataclass
class AggregationConfig:
    """Configuration for verdict aggregation."""
    
    # Per-claim thresholds
    support_threshold: float = 0.7     # Min support score for SUPPORTED
    contradict_threshold: float = 0.7  # Min contradict score for CONTRADICTED
    margin_threshold: float = 0.3      # Min margin between support/contradict for clear verdict
    
    # Coverage requirements
    min_sources_for_support: int = 1   # Min unique sources for SUPPORTED
    min_sources_for_high_confidence: int = 2  # For confidence boost
    
    # Overall aggregation
    contradict_ratio_for_overall: float = 0.3  # If >30% contradicted → overall CONTRADICTED
    support_ratio_for_overall: float = 0.7     # If >70% supported → overall SUPPORTED
    
    # Critical claims
    critical_claim_weight: float = 3.0  # Weight multiplier for critical claims

    @classmethod
    def from_profile(cls, profile: "DomainProfile") -> "AggregationConfig":
        cfg = cls()
        if profile == DomainProfile.FINANCE:
            cfg.min_sources_for_support = 2
            cfg.min_sources_for_high_confidence = 2
            cfg.support_threshold = 0.7
            cfg.contradict_threshold = 0.6
        elif profile == DomainProfile.POLICY:
            cfg.min_sources_for_support = 1  # primary source expected
            cfg.support_threshold = 0.7
            cfg.contradict_threshold = 0.6
        elif profile == DomainProfile.ENTERPRISE:
            cfg.min_sources_for_support = 1
            cfg.support_threshold = 0.7
            cfg.contradict_threshold = 0.6
        return cfg


class ClaimAggregator:
    """
    Aggregates evidence assessments into a claim verdict.
    
    Strategy:
    1. Find best support and contradict scores
    2. Calculate coverage (unique sources)
    3. Apply decision rules
    4. Compute confidence
    """
    
    def __init__(self, config: Optional[AggregationConfig] = None):
        self.config = config or AggregationConfig()
    
    def aggregate(
        self,
        claim: Claim,
        judge_results: List[JudgeResult],
        accepted_chunks: int = 0,
        rejected_chunks: int = 0,
        trace: Optional[TraceBuilder] = None,
        trace_parents: Optional[List[str]] = None,
    ) -> ClaimVerdict:
        """
        Aggregate judge results into a claim verdict.
        
        Args:
            claim: The claim being verified
            judge_results: Results from judging claim against evidence
            accepted_chunks: Number of chunks that passed gating
            rejected_chunks: Number of chunks that failed gating
            
        Returns:
            ClaimVerdict with label, confidence, and evidence
        """
        
        if not judge_results:
            # No evidence at all
            return ClaimVerdict(
                claim=claim,
                label=VerdictLabel.INSUFFICIENT,
                confidence=0.0,
                reasons=[ReasonCode.EVIDENCE_LOW_COVERAGE],
                summary="No evidence found for this claim.",
                evidence=[],
                coverage_sources=0,
                coverage_doc_types=0,
            )
        
        # Calculate aggregate scores
        support_score, contradict_score = self._calculate_scores(judge_results)
        
        # Calculate coverage
        coverage_sources, coverage_doc_types = self._calculate_coverage(judge_results)
        
        # Determine label
        label, reasons = self._determine_label(
            support_score=support_score,
            contradict_score=contradict_score,
            coverage_sources=coverage_sources,
            judge_results=judge_results,
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            label=label,
            support_score=support_score,
            contradict_score=contradict_score,
            coverage_sources=coverage_sources,
            judge_results=judge_results,
        )
        
        # Generate summary
        summary = self._generate_summary(
            label=label,
            support_score=support_score,
            contradict_score=contradict_score,
            coverage_sources=coverage_sources,
        )
        
        # Convert judge results to evidence assessments so verdicts keep citations/rationales
        evidence: List[EvidenceAssessment] = []
        for jr in judge_results:
            # Minimal provenance when only chunk_id is known
            prov = Provenance(
                source_id=jr.chunk_id,
                source_type=SourceType.SECONDARY,  # best-effort default
            )
            chunk = Chunk(
                text="",  # unknown here; real pipeline should supply full chunk
                provenance=prov,
                score=None,
            )
            decision = GateDecision(
                accepted=True,
                reasons=[],
                relevance_score=None,
                constraint_matches={},
            )
            evidence.append(
                EvidenceAssessment(
                    chunk=chunk,
                    decision=decision,
                    role=jr.get_role(),
                    support_score=jr.support_score,
                    contradict_score=jr.contradict_score,
                    rationale=jr.rationale,
                )
            )
        
        claim_verdict = ClaimVerdict(
            claim=claim,
            label=label,
            confidence=confidence,
            reasons=reasons,
            summary=summary,
            evidence=evidence,
            coverage_sources=coverage_sources,
            coverage_doc_types=coverage_doc_types,
            support_score=support_score,
            contradict_score=contradict_score,
            coverage_score=coverage_sources / max(self.config.min_sources_for_high_confidence, 1),
        )

        # Emit trace nodes for evidence assessments and claim verdict
        if trace is not None:
            evidence_parent_ids: List[str] = trace_parents or []
            for ea in evidence:
                trace.add_evidence_assessment(
                    role=ea.role.value,
                    support_score=ea.support_score,
                    contradict_score=ea.contradict_score,
                    rationale=ea.rationale,
                    parents=evidence_parent_ids,
                )
            trace.add_claim_verdict(
                claim_id=claim.claim_id,
                label=label.value,
                confidence=confidence,
                reasons=[r.value for r in reasons],
                parents=trace_parents or [],
            )

        return claim_verdict
    
    def _calculate_scores(
        self,
        results: List[JudgeResult],
    ) -> Tuple[float, float]:
        """
        Calculate aggregate support and contradict scores.
        
        Strategy: Use max score (strongest evidence).
        Alternative: weighted average, could be configurable.
        """
        if not results:
            return 0.0, 0.0
        
        support_score = max(r.support_score for r in results)
        contradict_score = max(r.contradict_score for r in results)
        
        return support_score, contradict_score
    
    def _calculate_coverage(
        self,
        results: List[JudgeResult],
    ) -> Tuple[int, int]:
        """
        Calculate coverage metrics.
        
        Returns (unique_sources, unique_doc_types).
        """
        sources: Set[str] = set()
        doc_types: Set[str] = set()

        for result in results:
            sources.add(result.chunk_id)
            if result.doc_type:
                doc_types.add(result.doc_type)

        return len(sources), len(doc_types)
    
    def _determine_label(
        self,
        support_score: float,
        contradict_score: float,
        coverage_sources: int,
        judge_results: List[JudgeResult],
    ) -> Tuple[VerdictLabel, List[ReasonCode]]:
        """
        Determine the verdict label based on scores and coverage.
        """
        reasons: List[ReasonCode] = []
        
        # Check for low coverage
        if coverage_sources < self.config.min_sources_for_support:
            reasons.append(ReasonCode.EVIDENCE_LOW_COVERAGE)
        
        # Primary-source contradictions win unless a clearly stronger primary support exists.
        primary_contra = max(
            (r.contradict_score for r in judge_results if r.source_type == SourceType.PRIMARY),
            default=0.0,
        )
        primary_support = max(
            (r.support_score for r in judge_results if r.source_type == SourceType.PRIMARY),
            default=0.0,
        )
        if primary_contra >= self.config.contradict_threshold:
            primary_support_clear = (
                primary_support >= self.config.support_threshold
                and (primary_support - primary_contra) >= self.config.margin_threshold
            )
            if not primary_support_clear:
                if support_score >= self.config.support_threshold:
                    reasons.append(ReasonCode.EVIDENCE_CONFLICTING_SOURCES)
                return VerdictLabel.CONTRADICTED, reasons

        # Calculate margin
        margin = abs(support_score - contradict_score)
        
        # Decision logic
        if contradict_score >= self.config.contradict_threshold:
            if support_score >= self.config.support_threshold and margin < self.config.margin_threshold:
                # Both high → MIXED
                reasons.append(ReasonCode.EVIDENCE_CONFLICTING_SOURCES)
                return VerdictLabel.MIXED, reasons
            else:
                # Clear contradiction
                return VerdictLabel.CONTRADICTED, reasons
        
        if support_score >= self.config.support_threshold:
            if coverage_sources >= self.config.min_sources_for_support:
                return VerdictLabel.SUPPORTED, reasons
            else:
                # Support but low coverage
                return VerdictLabel.INSUFFICIENT, reasons
        
        if support_score > 0.3 or contradict_score > 0.3:
            # Some signal but not enough
            return VerdictLabel.INSUFFICIENT, reasons
        
        # No clear signal
        reasons.append(ReasonCode.EVIDENCE_TOO_THIN)
        return VerdictLabel.INSUFFICIENT, reasons
    
    def _calculate_confidence(
        self,
        label: VerdictLabel,
        support_score: float,
        contradict_score: float,
        coverage_sources: int,
        judge_results: List[JudgeResult],
    ) -> float:
        """
        Calculate confidence in the verdict.
        
        Factors:
        - Strength of winning score
        - Coverage (more sources = more confidence)
        - Agreement among evidence
        - Individual judge confidence
        """
        # Base confidence from winning score
        if label == VerdictLabel.SUPPORTED:
            base = support_score
        elif label == VerdictLabel.CONTRADICTED:
            base = contradict_score
        elif label == VerdictLabel.MIXED:
            base = 0.5  # MIXED has medium confidence by design
        else:
            # INSUFFICIENT: clamp low to avoid false certainty
            if coverage_sources < self.config.min_sources_for_support:
                return 0.15
            base = 0.25  # low ceiling otherwise
        
        # Coverage factor
        coverage_factor = min(
            coverage_sources / self.config.min_sources_for_high_confidence,
            1.0
        )
        
        # Agreement factor (low variance = high agreement)
        if len(judge_results) > 1:
            if label in [VerdictLabel.SUPPORTED, VerdictLabel.CONTRADICTED]:
                scores = [r.support_score for r in judge_results] if label == VerdictLabel.SUPPORTED else [r.contradict_score for r in judge_results]
                mean = sum(scores) / len(scores)
                variance = sum((s - mean) ** 2 for s in scores) / len(scores)
                agreement_factor = 1.0 - min(math.sqrt(variance), 0.5)
            else:
                agreement_factor = 0.7
        else:
            agreement_factor = 0.8  # Single source has medium-high agreement
        
        # Judge confidence factor
        avg_judge_confidence = sum(r.confidence for r in judge_results) / max(len(judge_results), 1)
        
        # Combine factors
        confidence = base * 0.4 + coverage_factor * 0.2 + agreement_factor * 0.2 + avg_judge_confidence * 0.2
        
        return min(max(confidence, 0.0), 1.0)
    
    def _generate_summary(
        self,
        label: VerdictLabel,
        support_score: float,
        contradict_score: float,
        coverage_sources: int,
    ) -> str:
        """Generate a human-readable summary."""
        
        if label == VerdictLabel.SUPPORTED:
            return f"Claim is supported by {coverage_sources} source(s) with confidence {support_score:.0%}."
        elif label == VerdictLabel.CONTRADICTED:
            return f"Claim is contradicted by evidence with confidence {contradict_score:.0%}."
        elif label == VerdictLabel.MIXED:
            return f"Evidence is mixed: {support_score:.0%} support vs {contradict_score:.0%} contradiction."
        else:
            return f"Insufficient evidence to verify claim ({coverage_sources} source(s) found)."


class OverallAggregator:
    """
    Aggregates claim verdicts into an overall verdict.
    
    Strategy:
    1. Weight claims by importance (weight + critical flag)
    2. Check for critical contradictions
    3. Calculate weighted verdict distribution
    4. Apply decision rules
    """
    
    def __init__(self, config: Optional[AggregationConfig] = None):
        self.config = config or AggregationConfig()
    
    def aggregate(
        self,
        claim_verdicts: List[ClaimVerdict],
        trace: Optional[TraceBuilder] = None,
        trace_parents: Optional[List[str]] = None,
    ) -> Tuple[VerdictLabel, float, List[ReasonCode]]:
        """
        Aggregate claim verdicts into overall verdict.
        
        Returns:
            (overall_label, overall_confidence, warnings)
        """
        
        if not claim_verdicts:
            return VerdictLabel.INSUFFICIENT, 0.0, [ReasonCode.CLAIM_NEEDS_CLARIFICATION]
        
        warnings: List[ReasonCode] = []
        
        # Check for critical contradictions
        for cv in claim_verdicts:
            if cv.claim.critical and cv.label == VerdictLabel.CONTRADICTED:
                warnings.append(ReasonCode.EVIDENCE_CONFLICTING_SOURCES)
                # Critical contradiction → overall contradiction
                confidence = cv.confidence * 0.8 + 0.2  # Boost confidence for critical
                return VerdictLabel.CONTRADICTED, confidence, warnings
        
        # Calculate weighted counts
        total_weight = 0.0
        supported_weight = 0.0
        contradicted_weight = 0.0
        insufficient_weight = 0.0
        mixed_weight = 0.0
        
        confidence_sum = 0.0
        
        for cv in claim_verdicts:
            weight = cv.claim.weight
            if cv.claim.critical:
                weight *= self.config.critical_claim_weight
            
            total_weight += weight
            
            if cv.label == VerdictLabel.SUPPORTED:
                supported_weight += weight
            elif cv.label == VerdictLabel.CONTRADICTED:
                contradicted_weight += weight
            elif cv.label == VerdictLabel.INSUFFICIENT:
                insufficient_weight += weight
            elif cv.label == VerdictLabel.MIXED:
                mixed_weight += weight
            
            confidence_sum += cv.confidence * weight
        
        # Calculate ratios
        support_ratio = supported_weight / total_weight if total_weight > 0 else 0
        contradict_ratio = contradicted_weight / total_weight if total_weight > 0 else 0
        insufficient_ratio = insufficient_weight / total_weight if total_weight > 0 else 0
        mixed_ratio = mixed_weight / total_weight if total_weight > 0 else 0
        
        # Weighted average confidence
        avg_confidence = confidence_sum / total_weight if total_weight > 0 else 0
        
        # Decision logic
        if contradict_ratio >= self.config.contradict_ratio_for_overall:
            label = VerdictLabel.CONTRADICTED
            conf = avg_confidence
        elif support_ratio >= self.config.support_ratio_for_overall and contradict_ratio == 0:
            label = VerdictLabel.SUPPORTED
            conf = avg_confidence
        elif support_ratio > 0 and contradict_ratio > 0:
            warnings.append(ReasonCode.EVIDENCE_CONFLICTING_SOURCES)
            label = VerdictLabel.MIXED
            conf = avg_confidence * 0.8
        elif insufficient_ratio > 0.5:
            warnings.append(ReasonCode.EVIDENCE_LOW_COVERAGE)
            label = VerdictLabel.INSUFFICIENT
            conf = avg_confidence * 0.5
        elif mixed_ratio > 0.3:
            label = VerdictLabel.MIXED
            conf = avg_confidence * 0.7
        else:
            label = VerdictLabel.INSUFFICIENT
            conf = avg_confidence * 0.5

        if trace is not None:
            trace.add_verdict_report(
                label.value,
                conf,
                parents=trace_parents or [],
            )

        return label, conf, warnings


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def aggregate_claim(
    claim: Claim,
    judge_results: List[JudgeResult],
    config: Optional[AggregationConfig] = None,
    trace: Optional[TraceBuilder] = None,
    trace_parents: Optional[List[str]] = None,
) -> ClaimVerdict:
    """
    Convenience function to aggregate a single claim.
    """
    aggregator = ClaimAggregator(config=config)
    return aggregator.aggregate(claim, judge_results, trace=trace, trace_parents=trace_parents)


def aggregate_overall(
    claim_verdicts: List[ClaimVerdict],
    config: Optional[AggregationConfig] = None,
    trace: Optional[TraceBuilder] = None,
    trace_parents: Optional[List[str]] = None,
) -> Tuple[VerdictLabel, float, List[ReasonCode]]:
    """
    Convenience function to aggregate overall verdict.
    """
    aggregator = OverallAggregator(config=config)
    overall_label, overall_conf, warnings = aggregator.aggregate(claim_verdicts)
    if trace is not None:
        trace.add_verdict_report(
            overall_label.value,
            overall_conf,
            parents=trace_parents or [],
        )
    return overall_label, overall_conf, warnings


def verdict_summary(
    claim_verdicts: List[ClaimVerdict],
) -> Dict[str, Any]:
    """
    Generate a summary of claim verdicts.
    """
    by_label = {}
    for cv in claim_verdicts:
        label = cv.label.value
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(cv.claim.text[:50] + "...")
    
    total = len(claim_verdicts)
    
    return {
        "total_claims": total,
        "supported": len(by_label.get("SUPPORTED", [])),
        "contradicted": len(by_label.get("CONTRADICTED", [])),
        "insufficient": len(by_label.get("INSUFFICIENT", [])),
        "mixed": len(by_label.get("MIXED", [])),
        "claims_by_label": by_label,
        "average_confidence": sum(cv.confidence for cv in claim_verdicts) / max(total, 1),
    }
