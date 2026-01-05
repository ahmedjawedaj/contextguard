#!/usr/bin/env python3
"""
ContextGuard Hero Demo: Micrograd-Style Trace Visualization

This example demonstrates the "killer feature" of ContextGuard:
the ability to visualize the entire verification pipeline as a DAG,
just like Andrej Karpathy's micrograd shows computation graphs.

The trace shows:
1. How user input becomes state constraints
2. How claims are extracted and planned
3. How evidence is retrieved and gated
4. How verdicts are determined
5. Why each decision was made

Run this example to generate:
- trace.dot: Graphviz DOT file
- trace.png: Rendered graph (if graphviz installed)
- report.md: Verdict report in markdown

Usage:
    python examples/05_trace_graphviz.py
"""

import sys
from pathlib import Path

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from contextguard.core.specs import (
    StateDelta,
    EntityRef,
    TimeConstraint,
    SourcePolicy,
    SourceType,
    Claim,
)
from contextguard.core.merge import merge_state, create_initial_state
from contextguard.core.trace import TraceBuilder
from contextguard.retrieve.protocols import MockRetriever
from contextguard.retrieve.planner import plan_retrieval
from contextguard.retrieve.gating import gate_chunks, summarize_gating
from contextguard.verify.claim_splitter import RuleBasedClaimSplitter
from contextguard.verify.judges import RuleBasedJudge
from contextguard.verify.aggregate import ClaimAggregator, OverallAggregator
from contextguard.verify.report import build_report, render_report


def create_mock_corpus():
    """Create a mock corpus with finance-related chunks."""
    retriever = MockRetriever()
    
    # Apple chunks
    retriever.add_chunk(
        text="Apple Inc. reported revenue of $383 billion for fiscal year 2023, "
             "representing a 3% decline from the previous year. The company's "
             "iPhone segment contributed $200 billion to the total revenue.",
        source_id="apple_10k_2023",
        source_type=SourceType.PRIMARY,
        entity_ids=["AAPL"],
        year=2023,
    )
    
    retriever.add_chunk(
        text="Apple's 2024 revenue projections indicate growth returning to the company, "
             "with analysts expecting approximately $400 billion in total revenue. "
             "The projection is based on strong iPhone 15 sales and services growth.",
        source_id="apple_analyst_2024",
        source_type=SourceType.SECONDARY,
        entity_ids=["AAPL"],
        year=2024,
    )
    
    retriever.add_chunk(
        text="Apple announced that 2024 projections were revised downward due to "
             "China market weakness. The new guidance suggests $380-390 billion range.",
        source_id="apple_earnings_call_q1_2024",
        source_type=SourceType.PRIMARY,
        entity_ids=["AAPL"],
        year=2024,
    )
    
    # Microsoft chunks
    retriever.add_chunk(
        text="Microsoft Corporation achieved record revenue of $211 billion in FY2023, "
             "a 7% increase year-over-year. Azure cloud services grew by 29%.",
        source_id="msft_10k_2023",
        source_type=SourceType.PRIMARY,
        entity_ids=["MSFT"],
        year=2023,
    )
    
    retriever.add_chunk(
        text="Microsoft's 2024 guidance projects revenue between $230-240 billion, "
             "driven by continued AI integration and cloud growth.",
        source_id="msft_guidance_2024",
        source_type=SourceType.PRIMARY,
        entity_ids=["MSFT"],
        year=2024,
    )
    
    # Some noise/irrelevant chunks
    retriever.add_chunk(
        text="The technology sector has seen significant volatility in 2023, "
             "with many companies adjusting their growth expectations.",
        source_id="tech_news_general",
        source_type=SourceType.TERTIARY,
        entity_ids=[],
        year=2023,
    )
    
    return retriever


def run_multi_turn_verification():
    """
    Simulate a multi-turn verification conversation.
    
    Turn 1: "Compare Apple and Microsoft revenue"
    Turn 2: "Now do 2024 projections"
    Turn 3: "Only use primary sources"
    """
    
    print("=" * 70)
    print("ContextGuard Hero Demo: Multi-Turn Verification with Trace")
    print("=" * 70)
    
    # Initialize components
    retriever = create_mock_corpus()
    splitter = RuleBasedClaimSplitter()
    judge = RuleBasedJudge()
    claim_aggregator = ClaimAggregator()
    overall_aggregator = OverallAggregator()
    
    # Initialize trace builder
    trace = TraceBuilder()
    
    # Initialize state
    thread_id = "demo_thread_1"
    state = create_initial_state(thread_id)
    
    # =========================================================================
    # TURN 1: "Compare Apple and Microsoft revenue"
    # =========================================================================
    
    print("\n--- Turn 1: 'Compare Apple and Microsoft revenue' ---\n")
    
    turn1_text = "Compare Apple and Microsoft revenue"
    turn1_id = trace.add_user_turn(turn1_text, turn_number=1)
    
    # Extract state delta (simulated - normally LLM)
    delta1 = StateDelta(
        entities_add=[
            EntityRef(entity_id="AAPL", display_name="Apple Inc."),
            EntityRef(entity_id="MSFT", display_name="Microsoft Corporation"),
        ],
        metric="revenue",
    )
    
    delta1_id = trace.add_state_delta(
        delta_dict={"entities_add": [{"entity_id": "AAPL"}, {"entity_id": "MSFT"}], "metric": "revenue"},
        parents=[turn1_id],
    )
    
    # Merge state
    merge_result = merge_state(state, delta1, turn_id=1)
    state = merge_result.state
    
    state_id = trace.add_state_merge(
        state_dict={"entities": ["AAPL", "MSFT"], "metric": "revenue"},
        conflicts=[],
        parents=[delta1_id],
    )
    
    print("State after Turn 1:")
    print(f"  Entities: {[e.entity_id for e in state.entities]}")
    print(f"  Metric: {state.metric}")
    
    # =========================================================================
    # TURN 2: "Now do 2024 projections"
    # =========================================================================
    
    print("\n--- Turn 2: 'Now do 2024 projections' ---\n")
    
    turn2_text = "Now do 2024 projections"
    turn2_id = trace.add_user_turn(turn2_text, turn_number=2)
    
    # Extract state delta
    delta2 = StateDelta(
        time=TimeConstraint(year=2024),
        metric="projections",
    )
    
    delta2_id = trace.add_state_delta(
        delta_dict={"time": {"year": 2024}, "metric": "projections"},
        parents=[turn2_id, state_id],
    )
    
    # Merge state (entities persist!)
    merge_result = merge_state(state, delta2, turn_id=2)
    state = merge_result.state
    
    state_id = trace.add_state_merge(
        state_dict={"entities": ["AAPL", "MSFT"], "time": {"year": 2024}, "metric": "projections"},
        conflicts=[c.model_dump() for c in merge_result.conflicts],
        parents=[delta2_id],
    )
    
    print("State after Turn 2 (entities carried forward!):")
    print(f"  Entities: {[e.entity_id for e in state.entities]}")
    print(f"  Year: {state.time.year}")
    print(f"  Metric: {state.metric}")
    
    # =========================================================================
    # TURN 3: "Only use primary sources"
    # =========================================================================
    
    print("\n--- Turn 3: 'Only use primary sources' ---\n")
    
    turn3_text = "Only use primary sources"
    turn3_id = trace.add_user_turn(turn3_text, turn_number=3)
    
    # Extract state delta
    delta3 = StateDelta(
        source_policy=SourcePolicy(
            allowed_source_types=[SourceType.PRIMARY],
        ),
    )
    
    delta3_id = trace.add_state_delta(
        delta_dict={"source_policy": {"allowed_source_types": ["PRIMARY"]}},
        parents=[turn3_id, state_id],
    )
    
    # Merge state
    merge_result = merge_state(state, delta3, turn_id=3)
    state = merge_result.state
    
    state_id = trace.add_state_merge(
        state_dict={
            "entities": ["AAPL", "MSFT"],
            "time": {"year": 2024},
            "metric": "projections",
            "source_policy": {"allowed_source_types": ["PRIMARY"]},
        },
        conflicts=[],
        parents=[delta3_id],
    )
    
    print("State after Turn 3 (all constraints accumulated!):")
    print(f"  Entities: {[e.entity_id for e in state.entities]}")
    print(f"  Year: {state.time.year}")
    print(f"  Metric: {state.metric}")
    print(f"  Allowed Sources: {[s.value for s in state.source_policy.allowed_source_types]}")
    
    # =========================================================================
    # VERIFICATION PIPELINE
    # =========================================================================
    
    print("\n--- Running Verification Pipeline ---\n")
    
    # Create verification claim
    claim_text = "Apple and Microsoft 2024 revenue projections comparison"
    claims = [
        Claim(
            claim_id="claim_1",
            text="Apple's 2024 revenue projection is approximately $400 billion",
            entities=["AAPL"],
            metric="projections",
            time=TimeConstraint(year=2024),
        ),
        Claim(
            claim_id="claim_2",
            text="Microsoft's 2024 revenue projection is between $230-240 billion",
            entities=["MSFT"],
            metric="projections",
            time=TimeConstraint(year=2024),
        ),
    ]
    
    # Add claim nodes to trace
    claim_node_ids = []
    for claim in claims:
        cid = trace.add_claim(claim.text, claim.claim_id, parents=[state_id])
        claim_node_ids.append(cid)
    
    # Plan retrieval
    plan = plan_retrieval(claims, state, total_k=20)
    
    print("Retrieval Plan:")
    print(f"  Total steps: {len(plan.steps)}")
    print(f"  Support queries: {len(plan.get_support_steps())}")
    print(f"  Counter queries: {len(plan.get_counter_steps())}")
    
    # Execute retrieval
    all_chunks = []
    retrieval_node_ids = []
    
    for step in plan.steps:
        query_id = trace.add_retrieval_query(
            step.query,
            step.query_type.value,
            parents=claim_node_ids,
        )
        
        chunks = retriever.search(step.query, k=step.k)
        
        for chunk in chunks:
            chunk_id = trace.add_chunk(
                chunk.text[:100],
                chunk.provenance.source_id,
                chunk.score,
                parents=[query_id],
            )
            all_chunks.append((chunk, chunk_id))
            retrieval_node_ids.append(chunk_id)
    
    print(f"  Retrieved {len(all_chunks)} chunks total")
    
    # Gate chunks
    gated = gate_chunks([c for c, _ in all_chunks], state)
    gating_summary = summarize_gating(gated)
    
    print("\nGating Results:")
    print(f"  Accepted: {gating_summary['accepted']}")
    print(f"  Rejected: {gating_summary['rejected']}")
    print(f"  Rejection reasons: {gating_summary['rejection_reasons']}")
    
    # Add gate decisions to trace
    gate_node_ids = []
    for i, g in enumerate(gated):
        _, chunk_node_id = all_chunks[i]
        gate_id = trace.add_gate_decision(
            accepted=g.accepted,
            reasons=[r.value for r in g.decision.reasons],
            constraint_matches=g.decision.constraint_matches,
            parents=[chunk_node_id],
        )
        gate_node_ids.append(gate_id)
    
    # Judge claims
    accepted_chunks = [g.chunk for g in gated if g.accepted]
    claim_verdicts = []
    verdict_node_ids = []
    
    for claim in claims:
        # Get relevant accepted chunks
        relevant = [c for c in accepted_chunks if any(e in c.entity_ids for e in claim.entities)]
        
        if not relevant:
            # No evidence
            verdict = claim_aggregator.aggregate(claim, [], 0, len(gated))
        else:
            # Judge and aggregate
            judge_results = judge.score_batch(claim, relevant, state)
            
            # Add judge nodes
            for jr in judge_results:
                trace.add_judge_call(
                    jr.support_score,
                    jr.contradict_score,
                    parents=gate_node_ids[:len(relevant)],
                )
            
            verdict = claim_aggregator.aggregate(claim, judge_results, len(relevant), 0)
        
        claim_verdicts.append(verdict)
        
        vid = trace.add_claim_verdict(
            claim.claim_id,
            verdict.label.value,
            verdict.confidence,
            [r.value for r in verdict.reasons],
            parents=gate_node_ids[:3],  # Link to some gates
        )
        verdict_node_ids.append(vid)
        
        print(f"\nClaim: {claim.text[:50]}...")
        print(f"  Verdict: {verdict.label.value} (confidence: {verdict.confidence:.0%})")
    
    # Aggregate overall verdict
    overall_label, overall_confidence, warnings = overall_aggregator.aggregate(claim_verdicts)
    
    # Add final verdict node
    final_id = trace.add_verdict_report(
        overall_label.value,
        overall_confidence,
        parents=verdict_node_ids,
    )
    
    print(f"\n{'=' * 70}")
    print(f"OVERALL VERDICT: {overall_label.value} (confidence: {overall_confidence:.0%})")
    print(f"{'=' * 70}")
    
    # =========================================================================
    # BUILD REPORT
    # =========================================================================
    
    report = build_report(
        thread_id=thread_id,
        state=state,
        claim_verdicts=claim_verdicts,
        overall_label=overall_label,
        overall_confidence=overall_confidence,
        warnings=warnings,
        retrieval_stats={
            "total": len(all_chunks),
            "accepted": gating_summary["accepted"],
            "rejected": gating_summary["rejected"],
        },
    )
    
    # =========================================================================
    # OUTPUT FILES
    # =========================================================================
    
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Save trace as DOT
    dot_path = output_dir / "trace.dot"
    trace.graph.save_dot(
        str(dot_path),
        title="ContextGuard Multi-Turn Verification Trace",
        rankdir="TB",
    )
    print(f"\nTrace saved to: {dot_path}")
    
    # Try to render as PNG
    png_path = output_dir / "trace.png"
    svg_result = trace.graph.render_svg(str(output_dir / "trace.svg"))
    if svg_result:
        print(f"Trace SVG rendered to: {svg_result}")
    else:
        print("(Install graphviz to render PNG: brew install graphviz)")
    
    # Save report
    report_path = output_dir / "report.md"
    with open(report_path, 'w') as f:
        f.write(render_report(report, format="markdown"))
    print(f"Report saved to: {report_path}")
    
    # Print explain output
    print("\n" + "=" * 70)
    print("TRACE EXPLANATION (micrograd-style)")
    print("=" * 70)
    print(trace.graph.explain_verdict())
    
    # Print stats
    stats = trace.graph.get_stats()
    print("\n" + "=" * 70)
    print("TRACE STATISTICS")
    print("=" * 70)
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Max depth: {stats['max_depth']}")
    print(f"Accepted chunks: {stats['accepted_chunks']}")
    print(f"Rejected chunks: {stats['rejected_chunks']}")
    
    return report, trace.graph


if __name__ == "__main__":
    run_multi_turn_verification()
