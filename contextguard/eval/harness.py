"""
ContextGuard Eval Harness

Runs a tiny JSONL fixture through the pipeline with optional ablations:
- Gating on/off
- Counter-evidence on/off

Example:
  python -m contextguard.eval.harness --data tests/fixtures/eval.jsonl --k 5
  python -m contextguard.eval.harness --data tests/fixtures/eval.jsonl --k 5 --disable-gating --disable-counter

Outputs a metrics JSON to stdout (and optionally to a file).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from contextguard import (
    StateSpec,
    EntityRef,
    TimeConstraint,
    Claim,
    plan_retrieval,
    gate_chunks,
    RuleBasedJudge,
    aggregate_claim,
    VerdictLabel,
    MockRetriever,
)
from .metrics import compute_verdict_accuracy, compute_evidence_pr, fever_score


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EvalExample:
    id: str
    claim: str
    label: VerdictLabel
    entities: List[str]
    year: Optional[int]
    evidence_chunks: List[Dict[str, Any]]
    gold_evidence_ids: List[str]


# ---------------------------------------------------------------------------
# Loading fixtures
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def parse_examples(items: List[Dict[str, Any]]) -> List[EvalExample]:
    examples: List[EvalExample] = []
    for it in items:
        examples.append(
            EvalExample(
                id=it["id"],
                claim=it["claim"],
                label=VerdictLabel(it["label"]),
                entities=it.get("entities", []),
                year=it.get("year"),
                evidence_chunks=it.get("evidence", []),
                gold_evidence_ids=it.get("gold_evidence_ids", []),
            )
        )
    return examples


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def run_eval(
    examples: List[EvalExample],
    k: int,
    disable_gating: bool,
    disable_counter: bool,
) -> Dict[str, Any]:
    preds: List[VerdictLabel] = []
    golds: List[VerdictLabel] = []
    retrieved_ids: List[List[str]] = []
    gold_ids: List[List[str]] = []
    support_hits: List[bool] = []

    judge = RuleBasedJudge()

    for ex in examples:
        # Build state
        state = StateSpec(
            thread_id=f"t-{ex.id}",
            entities=[EntityRef(entity_id=e) for e in ex.entities],
            time=TimeConstraint(year=ex.year) if ex.year else TimeConstraint(),
        )

        # Build retriever with provided chunks
        retriever = MockRetriever()
        for chunk in ex.evidence_chunks:
            retriever.add_chunk(
                text=chunk["text"],
                source_id=chunk["id"],
                entity_ids=chunk.get("entities", ex.entities),
                year=chunk.get("year", ex.year),
            )

        # Make a claim
        claim = Claim(
            claim_id=ex.id,
            text=ex.claim,
            entities=ex.entities,
            time=TimeConstraint(year=ex.year) if ex.year else None,
        )

        # Plan
        plan = plan_retrieval(
            [claim],
            state,
            total_k=k,
            enable_counter=not disable_counter,
        )

        # Execute retrieval
        all_chunks = []
        for step in plan.steps:
            chunks = retriever.search(step.query, filters=step.filters, k=step.k)
            all_chunks.extend(chunks)

        # Gate
        if disable_gating:
            accepted_chunks = all_chunks
        else:
            gated = gate_chunks(all_chunks, state)
            accepted_chunks = [g.chunk for g in gated if g.accepted]

        # Judge + aggregate
        judge_results = judge.score_batch(claim, accepted_chunks, state)
        claim_verdict = aggregate_claim(claim, judge_results)

        preds.append(claim_verdict.label)
        golds.append(ex.label)
        retrieved_ids.append([c.provenance.source_id for c in accepted_chunks])
        gold_ids.append(ex.gold_evidence_ids)
        support_hits.append(bool(set(gold_ids[-1]) & set(retrieved_ids[-1])) or not gold_ids[-1])

    verdict_metrics = compute_verdict_accuracy(preds, golds)
    evidence_pr = compute_evidence_pr(retrieved_ids, gold_ids)
    fever = fever_score(preds, golds, support_hits)

    return {
        "examples": len(examples),
        "verdict_accuracy": verdict_metrics.accuracy,
        "confusion": verdict_metrics.confusion,
        "evidence_precision": evidence_pr.precision,
        "evidence_recall": evidence_pr.recall,
        "fever_score": fever,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="ContextGuard eval harness")
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL eval file")
    parser.add_argument("--k", type=int, default=5, help="Top-k per claim")
    parser.add_argument("--disable-gating", action="store_true", help="Disable gating (for ablation)")
    parser.add_argument("--disable-counter", action="store_true", help="Disable counter-evidence (for ablation)")
    parser.add_argument("--output", type=str, default=None, help="Optional path to write metrics JSON")
    args = parser.parse_args()

    items = load_jsonl(Path(args.data))
    examples = parse_examples(items)
    metrics = run_eval(
        examples=examples,
        k=args.k,
        disable_gating=args.disable_gating,
        disable_counter=args.disable_counter,
    )
    print(json.dumps(metrics, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
