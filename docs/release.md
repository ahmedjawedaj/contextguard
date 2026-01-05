# Release Checklist

Versioning
- Follow SemVer (MAJOR.MINOR.PATCH). Tag format: `vX.Y.Z`.
- Update `CHANGELOG.md` with the release entry.

Pre-flight
- Tests: `pytest`, `ruff`, `mypy`.
- Build: `hatch build`.
- Twine check: `twine check dist/*`.
- Smoke install: `pip install dist/*.whl`.

Publish
- TestPyPI (optional but recommended): publish with test token; install and smoke test.
- PyPI: tag `vX.Y.Z` -> GitHub Actions publish job runs (requires `PYPI_TOKEN` secret).
- Docs: `mkdocs build`; deploy to GitHub Pages (configure Pages to serve `site/` or use CI action).

Post-release
- Verify PyPI listing and README rendering.
- Add release notes to GitHub Release referencing CHANGELOG.
- Increment dev version if continuing development.

