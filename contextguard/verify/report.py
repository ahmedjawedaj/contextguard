"""
ContextGuard Report Generation

This module generates the final verdict report in multiple formats:
- JSON: For programmatic access
- Markdown: For human reading
- Context Pack: For safe RAG generation

The report is the PRIMARY OUTPUT of ContextGuard.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import hashlib

from ..core.specs import (
    StateSpec,
    VerdictReport,
    VerdictLabel,
    ClaimVerdict,
    ReasonCode,
    ContextPack,
    EvidenceRole,
)


class ReportBuilder:
    """
    Builds VerdictReport from aggregated results.
    """
    
    def __init__(
        self,
        thread_id: str,
        state: StateSpec,
    ):
        self.thread_id = thread_id
        self.state = state
        self.claim_verdicts: List[ClaimVerdict] = []
        self.warnings: List[ReasonCode] = []
        
        # Statistics
        self.total_chunks_retrieved = 0
        self.chunks_accepted = 0
        self.chunks_rejected = 0
    
    def add_claim_verdict(self, verdict: ClaimVerdict) -> None:
        """Add a claim verdict to the report."""
        self.claim_verdicts.append(verdict)
    
    def add_warning(self, warning: ReasonCode) -> None:
        """Add a warning to the report."""
        if warning not in self.warnings:
            self.warnings.append(warning)
    
    def set_retrieval_stats(
        self,
        total: int,
        accepted: int,
        rejected: int,
    ) -> None:
        """Set retrieval statistics."""
        self.total_chunks_retrieved = total
        self.chunks_accepted = accepted
        self.chunks_rejected = rejected
    
    def build(
        self,
        overall_label: VerdictLabel,
        overall_confidence: float,
        *,
        report_id: Optional[str] = None,
        created_at: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_prompt_version: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        retrieval_plan: Optional[List[Dict[str, Any]]] = None,
        seed: Optional[str] = None,
    ) -> VerdictReport:
        """
        Build the final report.
        """
        # Generate executive summary
        summary = self._generate_summary(overall_label, overall_confidence)
        
        # Build context pack (secondary output)
        context_pack = self._build_context_pack()
        
        now_ts = created_at or datetime.now(timezone.utc).isoformat()
        rid = report_id or hashlib.sha256(now_ts.encode()).hexdigest()[:16]
        
        return VerdictReport(
            report_id=rid,
            thread_id=self.thread_id,
            created_at=now_ts,
            state=self.state,
            overall_label=overall_label,
            overall_confidence=overall_confidence,
            claims=self.claim_verdicts,
            warnings=self.warnings,
            executive_summary=summary,
            total_chunks_retrieved=self.total_chunks_retrieved,
            chunks_accepted=self.chunks_accepted,
            chunks_rejected=self.chunks_rejected,
            context_pack=context_pack.model_dump() if context_pack else None,
            llm_model=llm_model,
            llm_prompt_version=llm_prompt_version,
            llm_temperature=llm_temperature,
            retrieval_plan=retrieval_plan,
            seed=seed,
        )
    
    def _generate_summary(
        self,
        label: VerdictLabel,
        confidence: float,
    ) -> str:
        """Generate executive summary."""
        
        total = len(self.claim_verdicts)
        supported = len([c for c in self.claim_verdicts if c.label == VerdictLabel.SUPPORTED])
        contradicted = len([c for c in self.claim_verdicts if c.label == VerdictLabel.CONTRADICTED])
        insufficient = len([c for c in self.claim_verdicts if c.label == VerdictLabel.INSUFFICIENT])
        
        lines = []
        
        # Overall verdict
        if label == VerdictLabel.SUPPORTED:
            lines.append(f"**SUPPORTED** (confidence: {confidence:.0%})")
            lines.append("The content is supported by the available evidence.")
        elif label == VerdictLabel.CONTRADICTED:
            lines.append(f"**CONTRADICTED** (confidence: {confidence:.0%})")
            lines.append("The content is contradicted by the available evidence.")
        elif label == VerdictLabel.MIXED:
            lines.append(f"**MIXED** (confidence: {confidence:.0%})")
            lines.append("The evidence presents conflicting information.")
        else:
            lines.append(f"**INSUFFICIENT EVIDENCE** (confidence: {confidence:.0%})")
            lines.append("Not enough evidence to verify the content.")
        
        # Breakdown
        lines.append("")
        lines.append(f"Claims analyzed: {total}")
        if supported > 0:
            lines.append(f"- Supported: {supported}")
        if contradicted > 0:
            lines.append(f"- Contradicted: {contradicted}")
        if insufficient > 0:
            lines.append(f"- Insufficient evidence: {insufficient}")
        
        # Retrieval stats
        if self.total_chunks_retrieved > 0:
            lines.append("")
            lines.append(f"Evidence retrieved: {self.total_chunks_retrieved}")
            lines.append(f"- Accepted: {self.chunks_accepted}")
            lines.append(f"- Rejected: {self.chunks_rejected}")
        
        return "\n".join(lines)
    
    def _build_context_pack(self) -> Optional[ContextPack]:
        """Build context pack from supported claims."""
        
        supported_verdicts = [
            cv for cv in self.claim_verdicts
            if cv.label == VerdictLabel.SUPPORTED
        ]
        
        if not supported_verdicts:
            return None
        
        facts = []
        quotes = []
        
        for cv in supported_verdicts:
            # Add fact
            facts.append({
                "text": cv.claim.text,
                "citation": f"[{cv.coverage_sources} source(s)]",
                "confidence": cv.confidence,
            })
            
            # Add supporting quotes from evidence
            for ea in cv.evidence:
                if ea.role == EvidenceRole.SUPPORTING and ea.rationale:
                    quotes.append({
                        "text": ea.rationale,
                        "source": ea.chunk.provenance.source_id,
                        "provenance": ea.chunk.provenance.model_dump(),
                    })
        
        return ContextPack(
            facts=facts,
            supporting_quotes=quotes[:10],  # Limit quotes
            constraints_applied={
                "entities": [e.entity_id for e in self.state.entities],
                "time": self.state.time.model_dump() if self.state.time else None,
                "source_policy": self.state.source_policy.model_dump(),
            },
            total_facts=len(facts),
            token_estimate=sum(len(f["text"]) // 4 for f in facts),
            rejected_count=self.chunks_rejected,
        )


class ReportRenderer:
    """
    Renders VerdictReport to various formats with a stable schema.
    """

    SCHEMA_VERSION = "v0.1"

    @classmethod
    def canonical_dict(cls, report: VerdictReport) -> Dict[str, Any]:
        """Canonical, stable JSON-ready structure."""
        return {
            "schema_version": cls.SCHEMA_VERSION,
            "report_id": report.report_id,
            "thread_id": report.thread_id,
            "created_at": report.created_at,
            "overall": {
                "label": report.overall_label.value,
                "confidence": report.overall_confidence,
            },
            "llm_model": report.llm_model,
            "llm_prompt_version": report.llm_prompt_version,
            "llm_temperature": report.llm_temperature,
            "retrieval_plan": report.retrieval_plan,
            "seed": report.seed,
            "warnings": [w.value if hasattr(w, "value") else str(w) for w in report.warnings],
            "retrieval": {
                "total": report.total_chunks_retrieved,
                "accepted": report.chunks_accepted,
                "rejected": report.chunks_rejected,
            },
            "claims": [
                {
                    "claim_id": cv.claim.claim_id,
                    "text": cv.claim.text,
                    "verdict": cv.label.value,
                    "confidence": cv.confidence,
                    "reasons": [r.value if hasattr(r, "value") else str(r) for r in cv.reasons],
                    "evidence": [
                        {
                            "source_id": ea.chunk.provenance.source_id,
                            "role": ea.role.value,
                            "citation": ea.rationale,
                            "provenance": ea.chunk.provenance.model_dump(),
                            "support_score": ea.support_score,
                            "contradict_score": ea.contradict_score,
                        }
                        for ea in cv.evidence
                    ],
                    "rejected": [
                        {
                            "source_id": ea.chunk.provenance.source_id,
                            "reason": ea.decision.reasons[0].value
                            if ea.decision.reasons
                            else "UNKNOWN",
                        }
                        for ea in cv.evidence
                        if not ea.decision.accepted
                    ],
                }
                for cv in report.claims
            ],
        }

    @classmethod
    def to_json(cls, report: VerdictReport, indent: int = 2) -> str:
        """Render report as JSON (canonical schema)."""
        return json.dumps(cls.canonical_dict(report), indent=indent)

    @classmethod
    def to_dict(cls, report: VerdictReport) -> Dict[str, Any]:
        """Render report as dictionary (canonical schema)."""
        return cls.canonical_dict(report)
    
    @staticmethod
    def to_markdown(report: VerdictReport) -> str:
        """Render report as Markdown with evidence and rejected tables."""
        lines = []
        
        # Header
        lines.append("# Verification Report")
        lines.append("")
        lines.append(f"**Report ID:** `{report.report_id}`")
        lines.append(f"**Generated:** {report.created_at}")
        lines.append("")
        
        # Overall verdict
        label_emoji = {
            VerdictLabel.SUPPORTED: "✅",
            VerdictLabel.CONTRADICTED: "❌",
            VerdictLabel.MIXED: "⚠️",
            VerdictLabel.INSUFFICIENT: "❓",
        }
        
        emoji = label_emoji.get(report.overall_label, "")
        lines.append(f"## {emoji} Overall Verdict: {report.overall_label.value}")
        lines.append(f"**Confidence:** {report.overall_confidence:.0%}")
        lines.append("")
        
        # Executive summary
        if report.executive_summary:
            lines.append("### Summary")
            lines.append(report.executive_summary)
            lines.append("")
        
        # Warnings
        if report.warnings:
            lines.append("### ⚠️ Warnings")
            for warning in report.warnings:
                lines.append(f"- {warning.value}")
            lines.append("")
        
        # Claims
        lines.append("## Claims")
        lines.append("")
        
        for i, cv in enumerate(report.claims, 1):
            claim_emoji = label_emoji.get(cv.label, "")
            lines.append(f"### {i}. {claim_emoji} {cv.label.value}")
            lines.append(f"**Claim:** {cv.claim.text}")
            lines.append(f"**Confidence:** {cv.confidence:.0%}")
            
            if cv.summary:
                lines.append(f"**Summary:** {cv.summary}")
            
            if cv.reasons:
                lines.append(f"**Reasons:** {', '.join(r.value for r in cv.reasons)}")
            
            # Evidence table
            if cv.evidence:
                lines.append("")
                lines.append("**Evidence (accepted):**")
                lines.append("")
                lines.append("| # | Role | Source | Rationale | Provenance |")
                lines.append("|---|------|--------|-----------|------------|")
                for j, ea in enumerate(cv.evidence[:5], 1):  # limit rows for brevity
                    role_icon = "🟢" if ea.role == EvidenceRole.SUPPORTING else "🔴" if ea.role == EvidenceRole.CONTRADICTING else "⚪"
                    prov = ea.chunk.provenance
                    prov_str = prov.url or prov.source_id
                    rationale = ea.rationale or ""
                    lines.append(f"| {j} | {role_icon} | `{prov.source_id}` | {rationale} | {prov_str} |")

            # Rejected evidence (if any)
            rejected = [ea for ea in cv.evidence if not ea.decision.accepted]
            if rejected:
                lines.append("")
                lines.append("**Rejected evidence:**")
                lines.append("")
                lines.append("| Source | Reason |")
                lines.append("|--------|--------|")
                for ea in rejected[:5]:
                    reason = ea.decision.reasons[0].value if ea.decision.reasons else "UNKNOWN"
                    lines.append(f"| `{ea.chunk.provenance.source_id}` | {reason} |")
            
            lines.append("")
        
        # State constraints
        lines.append("## Constraints Applied")
        lines.append("")
        
        if report.state.entities:
            entities = ", ".join(e.entity_id for e in report.state.entities)
            lines.append(f"**Entities:** {entities}")
        
        if report.state.time and not report.state.time.is_empty():
            if report.state.time.year:
                lines.append(f"**Year:** {report.state.time.year}")
            if report.state.time.quarter:
                lines.append(f"**Quarter:** Q{report.state.time.quarter}")
        
        if report.state.metric:
            lines.append(f"**Metric:** {report.state.metric}")
        
        lines.append("")
        
        # Retrieval stats
        lines.append("## Retrieval Statistics")
        lines.append("")
        lines.append(f"- Total chunks retrieved: {report.total_chunks_retrieved}")
        lines.append(f"- Chunks accepted: {report.chunks_accepted}")
        lines.append(f"- Chunks rejected: {report.chunks_rejected}")
        
        if report.total_chunks_retrieved > 0:
            rate = report.chunks_accepted / report.total_chunks_retrieved * 100
            lines.append(f"- Acceptance rate: {rate:.1f}%")
        
        return "\n".join(lines)
    
    @staticmethod
    def to_html(report: VerdictReport) -> str:
        """Render report as HTML (basic)."""
        md = ReportRenderer.to_markdown(report)
        
        # Very basic markdown to HTML conversion
        # In production, use a proper markdown library
        html = md
        html = html.replace("# Verification Report", "<h1>Verification Report</h1>")
        html = html.replace("## ", "<h2>").replace("\n\n", "</h2>\n", 1)
        html = html.replace("### ", "<h3>").replace("\n\n", "</h3>\n", 1)
        html = html.replace("**", "<strong>").replace("**", "</strong>")
        html = html.replace("\n\n", "<br><br>")
        html = html.replace("- ", "<li>").replace("\n<li>", "</li>\n<li>")
        
        return f"<html><body>{html}</body></html>"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def build_report(
    thread_id: str,
    state: StateSpec,
    claim_verdicts: List[ClaimVerdict],
    overall_label: VerdictLabel,
    overall_confidence: float,
    warnings: Optional[List[ReasonCode]] = None,
    retrieval_stats: Optional[Dict[str, int]] = None,
) -> VerdictReport:
    """
    Convenience function to build a report.
    """
    builder = ReportBuilder(thread_id=thread_id, state=state)
    
    for cv in claim_verdicts:
        builder.add_claim_verdict(cv)
    
    if warnings:
        for w in warnings:
            builder.add_warning(w)
    
    if retrieval_stats:
        builder.set_retrieval_stats(
            total=retrieval_stats.get("total", 0),
            accepted=retrieval_stats.get("accepted", 0),
            rejected=retrieval_stats.get("rejected", 0),
        )
    
    return builder.build(overall_label, overall_confidence)


def render_report(
    report: VerdictReport,
    format: str = "markdown",
) -> str:
    """
    Convenience function to render a report.
    
    Args:
        report: The report to render
        format: "markdown", "json", or "html"
    """
    if format == "json":
        return ReportRenderer.to_json(report)
    elif format == "html":
        return ReportRenderer.to_html(report)
    else:
        return ReportRenderer.to_markdown(report)


def save_report(
    report: VerdictReport,
    filepath: str,
    format: Optional[str] = None,
) -> None:
    """
    Save report to file.
    
    Format is inferred from file extension if not specified.
    """
    if format is None:
        if filepath.endswith(".json"):
            format = "json"
        elif filepath.endswith(".html"):
            format = "html"
        else:
            format = "markdown"
    
    content = render_report(report, format)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
