# Examples Guide

Key scripts
- `examples/05_trace_graphviz.py`: hero demo; generates report + DOT/SVG trace.
- `examples/06_minimal_proof.py`: minimal pipeline; writes `examples/output/minimal_trace.dot`.
- `examples/07_integrations.py`: wiring for retrying provider, S3 store, async runner.
- `examples/01-04_*`: article, conversation, enterprise corpus, web+corpus scenarios.

Render a trace
```bash
python examples/05_trace_graphviz.py
dot -Tpng examples/output/trace.dot -o examples/output/trace.png
```

Minimal proof trace
```bash
python examples/06_minimal_proof.py
dot -Tpng minimal_trace.dot -o minimal_trace.png
```

