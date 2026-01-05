# Contributing to ContextGuard

Thank you for your interest in contributing to ContextGuard!

## Development Setup

1. Fork and clone the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/contextguard.git
   cd contextguard
   ```

2. Create a virtualenv and install dev dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. Run the test suite:
   ```bash
   ruff check .        # Lint
   mypy contextguard   # Type-check
   pytest              # Tests
   ```

## Git Flow

We follow a simplified Git Flow branching model:

### Branches

| Branch | Purpose |
|--------|---------|
| `main` | Production-ready code. Protected, requires PR. |
| `develop` | Integration branch for features. Protected, requires PR. |
| `feature/*` | New features (branch from `develop`) |
| `bugfix/*` | Bug fixes (branch from `develop`) |
| `hotfix/*` | Urgent production fixes (branch from `main`) |
| `release/*` | Release preparation (branch from `develop`) |

### Workflow

1. **Create a feature branch** from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**, commit with conventional commits:
   ```bash
   git commit -m "feat: add new verification mode"
   git commit -m "fix: handle edge case in gating"
   git commit -m "docs: update API reference"
   ```

3. **Push and create a PR** targeting `develop`:
   ```bash
   git push origin feature/your-feature-name
   # Open PR on GitHub targeting 'develop' branch
   ```

4. **After review**, your PR will be merged to `develop`.

5. **Releases** are cut from `develop` → `main` with version tags.

### Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

## Pull Request Guidelines

- Target the `develop` branch (unless it's a hotfix)
- Ensure all CI checks pass (ruff, mypy, pytest)
- Include tests for new features
- Update documentation if needed
- Link related issues using `Fixes #123` or `Closes #123`

## Release Process

1. Create a release branch: `release/v0.x.0`
2. Bump version in `contextguard/__init__.py`
3. Update `CHANGELOG.md`
4. Merge to `main` and tag: `git tag v0.x.0`
5. Push tag to trigger PyPI publish: `git push origin v0.x.0`
6. Merge `main` back to `develop`

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.
