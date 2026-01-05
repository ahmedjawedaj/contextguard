"""
ContextGuard Trace Graph (Micrograd-style Explainability)

This module implements the execution DAG that tracks every decision
in the verification pipeline, enabling:

- Reverse-walk explanation ("why did this verdict happen?")
- Graphviz visualization (DOT export)
- Audit trails for compliance
- Debugging and A/B testing

Inspired by Andrej Karpathy's micrograd, which builds a computation graph
for backpropagation. ContextGuard builds a decision graph for explanation.

Key difference from micrograd:
- micrograd: tracks gradients for differentiation
- ContextGuard: tracks reason codes and provenance for explanation
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import hashlib



class NodeKind(str, Enum):
    """Types of nodes in the trace graph."""
    
    # Input nodes
    USER_TURN = "USER_TURN"
    CONTENT = "CONTENT"
    
    # State nodes
    STATE_SPEC = "STATE_SPEC"
    STATE_DELTA = "STATE_DELTA"
    STATE_MERGE = "STATE_MERGE"
    
    # Claim nodes
    CLAIM = "CLAIM"
    CLAIM_SPLIT = "CLAIM_SPLIT"
    
    # Retrieval nodes
    PLAN = "PLAN"
    PLAN_STEP = "PLAN_STEP"
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
    CHUNK = "CHUNK"
    
    # Gating nodes
    GATE_DECISION = "GATE_DECISION"
    
    # Verification nodes
    JUDGE_CALL = "JUDGE_CALL"
    EVIDENCE_ASSESSMENT = "EVIDENCE_ASSESSMENT"
    CLAIM_VERDICT = "CLAIM_VERDICT"
    
    # Aggregation nodes
    AGGREGATION = "AGGREGATION"
    VERDICT_REPORT = "VERDICT_REPORT"
    
    # Output nodes
    CONTEXT_PACK = "CONTEXT_PACK"


class NodeOp(str, Enum):
    """Operations that produce trace nodes."""
    
    # Extraction
    EXTRACT_STATE = "extract_state"
    EXTRACT_CLAIMS = "extract_claims"
    
    # State management
    MERGE_STATE = "merge_state"
    RESET_STATE = "reset_state"
    
    # Planning
    PLAN_RETRIEVAL = "plan_retrieval"
    GENERATE_QUERY = "generate_query"
    
    # Retrieval
    RETRIEVE = "retrieve"
    
    # Gating
    CHECK_ELIGIBILITY = "check_eligibility"
    CHECK_DIVERSITY = "check_diversity"
    FILTER_NOISE = "filter_noise"
    
    # Verification
    JUDGE_SUPPORT = "judge_support"
    JUDGE_CONTRADICT = "judge_contradict"
    EXTRACT_RATIONALE = "extract_rationale"
    
    # Aggregation
    AGGREGATE_EVIDENCE = "aggregate_evidence"
    AGGREGATE_CLAIMS = "aggregate_claims"
    
    # Output
    BUILD_REPORT = "build_report"
    BUILD_CONTEXT_PACK = "build_context_pack"


@dataclass
class TraceNode:
    """
    A node in the execution trace graph.
    
    Like a Value in micrograd, but instead of storing gradients,
    stores reason codes and decision metadata.
    """
    
    # Identity
    id: str
    
    # Classification
    op: str           # What operation produced this node
    kind: str         # What type of data this node represents
    
    # Human-readable
    summary: str      # Short description for display
    
    # Payload (the actual data)
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # Graph structure (parents = dependencies)
    parents: List[str] = field(default_factory=list)
    
    # Explanation metadata
    reasons: List[str] = field(default_factory=list)  # Reason codes
    confidence: Optional[float] = None
    
    # Timing
    ts: float = field(default_factory=time.time)
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "op": self.op,
            "kind": self.kind,
            "summary": self.summary,
            "payload": self.payload,
            "parents": self.parents,
            "reasons": self.reasons,
            "confidence": self.confidence,
            "ts": self.ts,
            "duration_ms": self.duration_ms,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceNode":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            op=data["op"],
            kind=data["kind"],
            summary=data["summary"],
            payload=data.get("payload", {}),
            parents=data.get("parents", []),
            reasons=data.get("reasons", []),
            confidence=data.get("confidence"),
            ts=data.get("ts", time.time()),
            duration_ms=data.get("duration_ms"),
        )


class TraceGraph:
    """
    The execution graph for a verification run.
    
    Like micrograd's computational graph, but for verification decisions.
    Enables "explain backward" - walking from verdict to evidence to retrieval.
    """
    
    def __init__(self, run_id: Optional[str] = None, seed: Optional[str] = None, time_fn: Optional[Callable[[], float]] = None):
        # Deterministic run_id if seed provided
        if seed is not None:
            self.run_id = hashlib.sha256(seed.encode()).hexdigest()[:12]
        else:
            self.run_id = run_id or uuid.uuid4().hex[:12]
        self.nodes: Dict[str, TraceNode] = {}
        self._id_counter = 0
        self._time_fn = time_fn or time.time
    
    def _next_id(self) -> str:
        """Generate next node ID."""
        self._id_counter += 1
        return f"n{self._id_counter:04d}"
    
    def add(
        self,
        op: str,
        kind: str,
        summary: str,
        payload: Optional[Dict[str, Any]] = None,
        parents: Optional[List[str]] = None,
        reasons: Optional[List[str]] = None,
        confidence: Optional[float] = None,
    ) -> str:
        """
        Add a node to the graph.
        
        Returns the node ID for wiring to children.
        """
        node_id = self._next_id()
        
        node = TraceNode(
            id=node_id,
            op=op,
            kind=kind,
            summary=summary,
            payload=payload or {},
            parents=parents or [],
            reasons=reasons or [],
            confidence=confidence,
            ts=self._time_fn(),
        )
        
        self.nodes[node_id] = node
        return node_id
    
    def add_node(self, node: TraceNode) -> str:
        """Add a pre-constructed node."""
        if not node.id:
            node.id = self._next_id()
        self.nodes[node.id] = node
        return node.id
    
    def get(self, node_id: str) -> Optional[TraceNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_parents(self, node_id: str) -> List[TraceNode]:
        """Get parent nodes."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[pid] for pid in node.parents if pid in self.nodes]
    
    def get_children(self, node_id: str) -> List[TraceNode]:
        """Get child nodes (nodes that have this as a parent)."""
        return [
            node for node in self.nodes.values()
            if node_id in node.parents
        ]
    
    def get_roots(self) -> List[TraceNode]:
        """Get root nodes (no parents)."""
        return [node for node in self.nodes.values() if not node.parents]
    
    def get_leaves(self) -> List[TraceNode]:
        """Get leaf nodes (no children)."""
        all_parents = set()
        for node in self.nodes.values():
            all_parents.update(node.parents)
        return [node for node in self.nodes.values() if node.id not in all_parents]
    
    # =========================================================================
    # Explanation (the "backward" of micrograd)
    # =========================================================================
    
    def explain(
        self,
        node_id: str,
        max_depth: int = 10,
        include_payload: bool = False,
    ) -> str:
        """
        Walk backward from a node, producing a human-readable explanation.
        
        Like backward() in micrograd, but produces text instead of gradients.
        """
        lines: List[str] = []
        visited: Set[str] = set()
        
        def walk(nid: str, depth: int):
            if depth > max_depth or nid in visited:
                return
            visited.add(nid)
            
            node = self.nodes.get(nid)
            if not node:
                return
            
            indent = "  " * depth
            
            # Format node info
            line = f"{indent}[{node.kind}] {node.op}: {node.summary}"
            if node.confidence is not None:
                line += f" (conf={node.confidence:.2f})"
            lines.append(line)
            
            # Show reasons if present
            if node.reasons:
                lines.append(f"{indent}  reasons: {', '.join(node.reasons)}")
            
            # Optionally show payload
            if include_payload and node.payload:
                payload_str = json.dumps(node.payload, indent=2, default=str)
                for pline in payload_str.split("\n"):
                    lines.append(f"{indent}  {pline}")
            
            # Recurse to parents
            for parent_id in node.parents:
                walk(parent_id, depth + 1)
        
        walk(node_id, 0)
        return "\n".join(lines)
    
    def explain_verdict(
        self,
        verdict_node_id: Optional[str] = None,
        max_depth: int = 8,
    ) -> str:
        """
        Explain a verdict by walking its evidence and decision chain.
        
        If no verdict_node_id is provided, finds the VERDICT_REPORT node.
        """
        if verdict_node_id is None:
            # Find the verdict report node
            for node in self.nodes.values():
                if node.kind == NodeKind.VERDICT_REPORT.value:
                    verdict_node_id = node.id
                    break
        
        if verdict_node_id is None:
            return "No verdict found in trace."
        
        return self.explain(verdict_node_id, max_depth=max_depth)
    
    def get_evidence_chain(self, verdict_node_id: str) -> List[TraceNode]:
        """
        Get the chain of evidence nodes that led to a verdict.
        """
        chain: List[TraceNode] = []
        visited: Set[str] = set()
        
        def collect(nid: str):
            if nid in visited:
                return
            visited.add(nid)
            
            node = self.nodes.get(nid)
            if not node:
                return
            
            # Collect evidence-related nodes
            if node.kind in [
                NodeKind.CHUNK.value,
                NodeKind.GATE_DECISION.value,
                NodeKind.EVIDENCE_ASSESSMENT.value,
                NodeKind.CLAIM_VERDICT.value,
            ]:
                chain.append(node)
            
            for parent_id in node.parents:
                collect(parent_id)
        
        collect(verdict_node_id)
        return chain
    
    def get_rejected_chunks(self) -> List[TraceNode]:
        """Get all chunks that were rejected by gating."""
        rejected = []
        for node in self.nodes.values():
            if node.kind == NodeKind.GATE_DECISION.value:
                if node.payload.get("accepted") is False:
                    rejected.append(node)
        return rejected
    
    def get_accepted_chunks(self) -> List[TraceNode]:
        """Get all chunks that were accepted by gating."""
        accepted = []
        for node in self.nodes.values():
            if node.kind == NodeKind.GATE_DECISION.value:
                if node.payload.get("accepted") is True:
                    accepted.append(node)
        return accepted
    
    # =========================================================================
    # Visualization (Graphviz DOT export)
    # =========================================================================
    
    def to_dot(
        self,
        title: Optional[str] = None,
        show_payloads: bool = False,
        highlight_rejected: bool = True,
        rankdir: str = "TB",  # TB = top-to-bottom, LR = left-to-right
    ) -> str:
        """
        Export the graph as Graphviz DOT format.
        
        This is the visual "micrograd moment" - a diagram showing
        how the verdict was derived from evidence through decisions.
        """
        lines = [
            'digraph ContextGuard {',
            f'  label="{title or f"ContextGuard Run {self.run_id}"}"',
            '  labelloc="t"',
            '  fontsize="16"',
            f'  rankdir="{rankdir}"',
            '  node [shape=box, fontname="Helvetica", fontsize="10"]',
            '  edge [fontname="Helvetica", fontsize="8"]',
            '',
        ]
        
        # Color scheme by node kind
        colors = {
            NodeKind.USER_TURN.value: "#E8F5E9",      # Light green
            NodeKind.CONTENT.value: "#E8F5E9",
            NodeKind.STATE_SPEC.value: "#E3F2FD",     # Light blue
            NodeKind.STATE_DELTA.value: "#E3F2FD",
            NodeKind.STATE_MERGE.value: "#E3F2FD",
            NodeKind.CLAIM.value: "#FFF3E0",          # Light orange
            NodeKind.CLAIM_SPLIT.value: "#FFF3E0",
            NodeKind.PLAN.value: "#F3E5F5",           # Light purple
            NodeKind.PLAN_STEP.value: "#F3E5F5",
            NodeKind.RETRIEVAL_QUERY.value: "#F3E5F5",
            NodeKind.CHUNK.value: "#FFFDE7",          # Light yellow
            NodeKind.GATE_DECISION.value: "#FFEBEE",  # Light red (can be rejected)
            NodeKind.JUDGE_CALL.value: "#FCE4EC",     # Light pink
            NodeKind.EVIDENCE_ASSESSMENT.value: "#FCE4EC",
            NodeKind.CLAIM_VERDICT.value: "#C8E6C9",  # Medium green
            NodeKind.VERDICT_REPORT.value: "#4CAF50", # Green (final)
            NodeKind.CONTEXT_PACK.value: "#81C784",   # Light green
        }
        
        # Add nodes
        for nid, node in self.nodes.items():
            color = colors.get(node.kind, "#FAFAFA")
            
            # Check if rejected (for highlighting)
            if highlight_rejected and node.kind == NodeKind.GATE_DECISION.value:
                if node.payload.get("accepted") is False:
                    color = "#FFCDD2"  # Red for rejected
            
            # Build label
            label = f"{node.op}\\n{node.summary}"
            if node.confidence is not None:
                label += f"\\n(conf: {node.confidence:.2f})"
            if node.reasons:
                reasons_str = ", ".join(node.reasons[:3])
                if len(node.reasons) > 3:
                    reasons_str += "..."
                label += f"\\n[{reasons_str}]"
            
            # Escape quotes in label
            label = label.replace('"', '\\"')
            
            lines.append(
                f'  "{nid}" [label="{label}", style="filled", fillcolor="{color}"]'
            )
        
        lines.append('')
        
        # Add edges
        for nid, node in self.nodes.items():
            for parent_id in node.parents:
                if parent_id in self.nodes:
                    lines.append(f'  "{parent_id}" -> "{nid}"')
        
        lines.append('}')
        return '\n'.join(lines)
    
    def save_dot(self, filepath: str, **kwargs) -> None:
        """Save DOT output to a file."""
        dot = self.to_dot(**kwargs)
        with open(filepath, 'w') as f:
            f.write(dot)
    
    def render_svg(self, filepath: str, **kwargs) -> Optional[str]:
        """
        Render to SVG using graphviz (if installed).
        Returns the SVG filepath on success, None on failure.
        """
        try:
            import subprocess
            
            dot_content = self.to_dot(**kwargs)
            
            # Write temporary DOT file
            dot_path = filepath.replace('.svg', '.dot')
            with open(dot_path, 'w') as f:
                f.write(dot_content)
            
            # Render with graphviz
            subprocess.run(
                ['dot', '-Tsvg', dot_path, '-o', filepath],
                check=True,
                capture_output=True,
            )
            
            return filepath
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    
    # =========================================================================
    # Serialization
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire graph."""
        return {
            "run_id": self.run_id,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceGraph":
        """Deserialize from dictionary."""
        graph = cls(run_id=data.get("run_id"))
        for nid, node_data in data.get("nodes", {}).items():
            node = TraceNode.from_dict(node_data)
            node.id = nid
            graph.nodes[nid] = node
        graph._id_counter = len(graph.nodes)
        return graph
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> "TraceGraph":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the trace graph."""
        stats = {
            "total_nodes": len(self.nodes),
            "nodes_by_kind": {},
            "nodes_by_op": {},
            "max_depth": 0,
            "total_reasons": 0,
            "rejected_chunks": 0,
            "accepted_chunks": 0,
        }
        
        for node in self.nodes.values():
            # Count by kind
            stats["nodes_by_kind"][node.kind] = (
                stats["nodes_by_kind"].get(node.kind, 0) + 1
            )
            
            # Count by op
            stats["nodes_by_op"][node.op] = (
                stats["nodes_by_op"].get(node.op, 0) + 1
            )
            
            # Count reasons
            stats["total_reasons"] += len(node.reasons)
            
            # Count gate decisions
            if node.kind == NodeKind.GATE_DECISION.value:
                if node.payload.get("accepted"):
                    stats["accepted_chunks"] += 1
                else:
                    stats["rejected_chunks"] += 1
        
        # Calculate max depth
        def calc_depth(nid: str, memo: Dict[str, int]) -> int:
            if nid in memo:
                return memo[nid]
            node = self.nodes.get(nid)
            if not node or not node.parents:
                memo[nid] = 0
                return 0
            depth = 1 + max(calc_depth(p, memo) for p in node.parents)
            memo[nid] = depth
            return depth
        
        memo: Dict[str, int] = {}
        for nid in self.nodes:
            depth = calc_depth(nid, memo)
            if depth > stats["max_depth"]:
                stats["max_depth"] = depth
        
        return stats


# =============================================================================
# Convenience: TraceBuilder for pipeline integration
# =============================================================================


class TraceBuilder:
    """
    Helper for building trace graphs during pipeline execution.
    
    Usage:
        builder = TraceBuilder()
        
        # Add input
        turn_id = builder.add_user_turn("What is Apple's 2024 revenue?")
        
        # Add state extraction
        delta_id = builder.add_state_delta({...}, parents=[turn_id])
        
        # ... continue building the graph
        
        graph = builder.graph
    """
    
    def __init__(self, run_id: Optional[str] = None):
        self.graph = TraceGraph(run_id=run_id)
    
    def add_user_turn(
        self,
        text: str,
        turn_number: int = 0,
    ) -> str:
        """Add a user turn node."""
        return self.graph.add(
            op=NodeOp.EXTRACT_STATE.value,
            kind=NodeKind.USER_TURN.value,
            summary=text[:50] + ("..." if len(text) > 50 else ""),
            payload={"text": text, "turn": turn_number},
        )
    
    def add_state_delta(
        self,
        delta_dict: Dict[str, Any],
        parents: List[str],
    ) -> str:
        """Add a state delta extraction node."""
        changes = []
        if delta_dict.get("entities_add"):
            changes.append(f"{len(delta_dict['entities_add'])} entities")
        if delta_dict.get("time"):
            changes.append("time")
        if delta_dict.get("metric"):
            changes.append(f"metric={delta_dict['metric']}")
        
        summary = ", ".join(changes) if changes else "no changes"
        
        return self.graph.add(
            op=NodeOp.EXTRACT_STATE.value,
            kind=NodeKind.STATE_DELTA.value,
            summary=summary,
            payload=delta_dict,
            parents=parents,
        )
    
    def add_state_merge(
        self,
        state_dict: Dict[str, Any],
        conflicts: List[Dict],
        parents: List[str],
    ) -> str:
        """Add a state merge node."""
        summary = f"{len(state_dict.get('entities', []))} entities"
        if state_dict.get("time", {}).get("year"):
            summary += f", year={state_dict['time']['year']}"
        
        reasons = [c.get("reason", "") for c in conflicts]
        
        return self.graph.add(
            op=NodeOp.MERGE_STATE.value,
            kind=NodeKind.STATE_MERGE.value,
            summary=summary,
            payload={"state": state_dict, "conflicts": conflicts},
            parents=parents,
            reasons=reasons,
        )
    
    def add_claim(
        self,
        claim_text: str,
        claim_id: str,
        parents: List[str],
    ) -> str:
        """Add a claim node."""
        return self.graph.add(
            op=NodeOp.EXTRACT_CLAIMS.value,
            kind=NodeKind.CLAIM.value,
            summary=claim_text[:60] + ("..." if len(claim_text) > 60 else ""),
            payload={"claim_id": claim_id, "text": claim_text},
            parents=parents,
        )

    def add_plan(
        self,
        plan_id: str,
        steps_count: int,
        parents: Optional[List[str]] = None,
    ) -> str:
        """Add a retrieval plan node."""
        summary = f"plan={plan_id}, steps={steps_count}"
        return self.graph.add(
            op=NodeOp.PLAN_RETRIEVAL.value,
            kind=NodeKind.PLAN.value,
            summary=summary,
            payload={"plan_id": plan_id, "steps": steps_count},
            parents=parents or [],
        )

    def add_plan_step(
        self,
        step_id: str,
        query: str,
        query_type: str,
        k: int,
        parents: Optional[List[str]] = None,
    ) -> str:
        """Add a plan step / retrieval query node."""
        summary = f"[{query_type}] {query[:60]}" + ("..." if len(query) > 60 else "")
        return self.graph.add(
            op=NodeOp.GENERATE_QUERY.value,
            kind=NodeKind.PLAN_STEP.value,
            summary=summary,
            payload={"step_id": step_id, "query": query, "type": query_type, "k": k},
            parents=parents or [],
        )
    
    def add_retrieval_query(
        self,
        query: str,
        query_type: str,  # "support" or "counter"
        parents: List[str],
    ) -> str:
        """Add a retrieval query node."""
        return self.graph.add(
            op=NodeOp.GENERATE_QUERY.value,
            kind=NodeKind.RETRIEVAL_QUERY.value,
            summary=f"[{query_type}] {query[:40]}...",
            payload={"query": query, "type": query_type},
            parents=parents,
        )
    
    def add_chunk(
        self,
        chunk_text: str,
        source_id: str,
        score: Optional[float],
        parents: List[str],
    ) -> str:
        """Add a retrieved chunk node."""
        return self.graph.add(
            op=NodeOp.RETRIEVE.value,
            kind=NodeKind.CHUNK.value,
            summary=f"[{source_id}] {chunk_text[:40]}...",
            payload={"text": chunk_text, "source_id": source_id, "score": score},
            parents=parents,
            confidence=score,
        )
    
    def add_gate_decision(
        self,
        accepted: bool,
        reasons: List[str],
        constraint_matches: Dict[str, bool],
        parents: List[str],
    ) -> str:
        """Add a gating decision node."""
        status = "ACCEPTED" if accepted else "REJECTED"
        
        return self.graph.add(
            op=NodeOp.CHECK_ELIGIBILITY.value,
            kind=NodeKind.GATE_DECISION.value,
            summary=status,
            payload={"accepted": accepted, "constraint_matches": constraint_matches},
            parents=parents,
            reasons=reasons,
        )
    
    def add_evidence_assessment(
        self,
        role: str,
        support_score: Optional[float],
        contradict_score: Optional[float],
        rationale: Optional[str],
        parents: Optional[List[str]] = None,
    ) -> str:
        """Add an evidence assessment node."""
        sup_val = support_score if support_score is not None else 0.0
        summary = f"{role} sup={sup_val:.2f}"
        return self.graph.add(
            op=NodeOp.AGGREGATE_EVIDENCE.value,
            kind=NodeKind.EVIDENCE_ASSESSMENT.value,
            summary=summary,
            payload={
                "support_score": support_score,
                "contradict_score": contradict_score,
                "rationale": rationale,
                "role": role,
            },
            parents=parents or [],
            confidence=max(
                [s for s in [support_score, contradict_score] if s is not None] or [0.0]
            ),
        )

    def add_judge_call(
        self,
        support_score: float,
        contradict_score: float,
        parents: List[str],
    ) -> str:
        """Add a judge call node."""
        if support_score > contradict_score:
            summary = f"SUPPORT ({support_score:.2f})"
        elif contradict_score > support_score:
            summary = f"CONTRADICT ({contradict_score:.2f})"
        else:
            summary = f"NEUTRAL ({support_score:.2f}/{contradict_score:.2f})"
        
        return self.graph.add(
            op=NodeOp.JUDGE_SUPPORT.value,
            kind=NodeKind.JUDGE_CALL.value,
            summary=summary,
            payload={
                "support_score": support_score,
                "contradict_score": contradict_score,
            },
            parents=parents,
            confidence=max(support_score, contradict_score),
        )
    
    def add_claim_verdict(
        self,
        claim_id: str,
        label: str,
        confidence: float,
        reasons: List[str],
        parents: List[str],
    ) -> str:
        """Add a claim verdict node."""
        return self.graph.add(
            op=NodeOp.AGGREGATE_EVIDENCE.value,
            kind=NodeKind.CLAIM_VERDICT.value,
            summary=f"{label} (conf={confidence:.2f})",
            payload={"claim_id": claim_id, "label": label},
            parents=parents,
            reasons=reasons,
            confidence=confidence,
        )
    
    def add_verdict_report(
        self,
        overall_label: str,
        overall_confidence: float,
        parents: List[str],
    ) -> str:
        """Add the final verdict report node."""
        return self.graph.add(
            op=NodeOp.BUILD_REPORT.value,
            kind=NodeKind.VERDICT_REPORT.value,
            summary=f"{overall_label} (conf={overall_confidence:.2f})",
            payload={"label": overall_label},
            parents=parents,
            confidence=overall_confidence,
        )

    def add_context_pack(
        self,
        facts_count: int,
        token_estimate: int,
        parents: Optional[List[str]] = None,
    ) -> str:
        """Add a context pack node."""
        summary = f"context pack facts={facts_count}, tokens~{token_estimate}"
        return self.graph.add(
            op=NodeOp.BUILD_CONTEXT_PACK.value,
            kind=NodeKind.CONTEXT_PACK.value,
            summary=summary,
            payload={"facts": facts_count, "tokens": token_estimate},
            parents=parents or [],
        )
