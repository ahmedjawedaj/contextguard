"""
Evaluation metrics for verification and evidence retrieval.

This module provides lightweight, CI-friendly scoring:
- Verdict accuracy and confusion counts
- Evidence precision/recall (set-based)
- FEVER-style score (label correct AND evidence meets threshold)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

from ..core.specs import VerdictLabel


@dataclass
class VerdictMetrics:
    accuracy: float
    confusion: Dict[str, Dict[str, int]]


@dataclass
class EvidencePR:
    precision: float
    recall: float


def compute_verdict_accuracy(
    preds: Sequence[VerdictLabel],
    golds: Sequence[VerdictLabel],
) -> VerdictMetrics:
    """Compute simple accuracy and confusion counts."""
    assert len(preds) == len(golds), "preds and golds must align"
    correct = 0
    confusion: Dict[str, Dict[str, int]] = {}
    for p, g in zip(preds, golds):
        if p == g:
            correct += 1
        confusion.setdefault(g.value, {})
        confusion[g.value][p.value] = confusion[g.value].get(p.value, 0) + 1
    acc = correct / len(preds) if preds else 0.0
    return VerdictMetrics(accuracy=acc, confusion=confusion)


def compute_evidence_pr(
    retrieved: Iterable[Iterable[str]],
    gold: Iterable[Iterable[str]],
) -> EvidencePR:
    """
    Set-based precision/recall over evidence IDs per example.
    Each element of retrieved/gold is a list/set of IDs for that example.
    """
    total_prec = 0.0
    total_rec = 0.0
    n = 0
    for r, g in zip(retrieved, gold):
        r_set = set(r)
        g_set = set(g)
        n += 1
        if r_set:
            total_prec += len(r_set & g_set) / len(r_set)
        else:
            total_prec += 0.0
        if g_set:
            total_rec += len(r_set & g_set) / len(g_set)
        else:
            total_rec += 0.0
    if n == 0:
        return EvidencePR(precision=0.0, recall=0.0)
    return EvidencePR(precision=total_prec / n, recall=total_rec / n)


def fever_score(
    preds: Sequence[VerdictLabel],
    golds: Sequence[VerdictLabel],
    support_hits: Sequence[bool],
) -> float:
    """
    FEVER-style score: label correct AND evidence condition met (support_hits).
    In our simplified variant, support_hits[i] indicates if required evidence
    was retrieved for example i.
    """
    assert len(preds) == len(golds) == len(support_hits)
    good = 0
    for p, g, hit in zip(preds, golds, support_hits):
        if p == g and hit:
            good += 1
    return good / len(preds) if preds else 0.0
