# Contributing to ContextGuard

## Setup
1. Fork and clone the repo.
2. Create a virtualenv and install deps:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e .[dev]
   ```

## Development
- Lint: `ruff check .`
- Type-check: `mypy contextguard`
- Tests: `pytest`

## Pull requests
- Include repro steps and, when applicable, screenshots of trace outputs.
- Ensure lint, mypy, and pytest pass before pushing.

## Releases
- Bump version in `contextguard/__init__.py` and update `CHANGELOG.md`.
- Build: `hatch build`.
- Tag and publish via CI (requires `PYPI_TOKEN`).

