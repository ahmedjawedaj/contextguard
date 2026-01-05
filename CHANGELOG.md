# Changelog

## 0.1.0 (Unreleased)

- Core contracts: `StateSpec`, `Claim`, `Chunk`, `Verdict`, `ReasonCode`.
- Merge engine: carryover + reset semantics with conflict detection.
- Retrieval planner: support + counter-evidence queries, coverage-first.
- Evidence gating: hard constraints (entity/time/source policy), diversity, reason codes.
- Judges: rule-based, LLM-based interface, NLI-ready hook.
- Aggregation: per-claim + overall verdicts with confidence and coverage signals.
- Reports: JSON/Markdown/HTML, plus facts-first context pack for safe RAG.
- Trace DAG: micrograd-style explainability; Graphviz DOT/SVG export.
- Storage: SQLite-backed state/fact/run store.
- Hero demo: `examples/05_trace_graphviz.py` end-to-end multi-turn verification.
- Packaging: `pyproject.toml` with hatchling; optional extras (`demo`, `nli`, `dev`); MIT `LICENSE`.

Release checklist
-----------------
- Bump version in `contextguard/__init__.py`.
- Update this changelog with the new version and date.
- `pip install hatch && hatch build` (wheel + sdist).
- Tag release: `git tag -a vX.Y.Z -m "vX.Y.Z"` and push tags.
- Publish (CI): ensure `PYPI_TOKEN` is set; GitHub Actions publish job runs on tagged releases.


