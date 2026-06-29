# Contributing to BYO-SLM

Thanks for your interest in improving BYO-SLM! This guide covers the workflow,
standards, and expectations for contributions.

## Code of Conduct

Be respectful and constructive. We follow the spirit of the
[Contributor Covenant](https://www.contributor-covenant.org/).

## Getting started

```bash
git clone https://github.com/your-org/build-your-own-slm.git
cd build-your-own-slm
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
make install   # editable install with dev extras
make hooks     # pre-commit hooks (ruff, mypy, hygiene checks)
```

See [docs/developer-guide.md](docs/developer-guide.md) for a codebase tour.

## Development workflow

1. **Branch** off `main`: `git checkout -b feat/short-description`.
2. **Make focused changes** with tests and docs alongside the code.
3. **Run the local gate** — it mirrors CI:
   ```bash
   make check        # ruff lint + ruff format check + mypy + pytest (≥95% cov)
   ```
   Or individually: `make lint`, `make format`, `make typecheck`, `make test`.
4. **Commit** using [Conventional Commits](https://www.conventionalcommits.org/):
   `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `chore:`, `ci:`.
5. **Open a PR** against `main` with a clear description and rationale. Link any
   related issues. CI must be green.

## Standards

| Area | Requirement |
|------|-------------|
| Language | Python 3.10+, `from __future__ import annotations` |
| Formatting | `ruff format` (100 cols) — enforced |
| Linting | `ruff check` clean — enforced |
| Typing | `mypy src` clean (strict) — enforced |
| Tests | New/changed behaviour covered; suite ≥ 95% coverage — enforced |
| Docs | Update inline docstrings and relevant `docs/*.md` |
| Logging | `structlog` events as `noun.verb` with structured kwargs |

PRs that lower coverage below 95% or break `make check` will not be merged until
addressed.

## Testing expectations

- Unit tests for pure logic (`tests/unit/`), integration tests for the
  train/serve path, API, and CLI (`tests/integration/`).
- Tests must be deterministic — seed RNGs and pass explicit generators.
- Keep the suite fast; reuse the session-scoped `trained_model_dir` fixture
  rather than training new models per test.

## Documentation changes

Update the relevant document(s) under `docs/` and cross-references in the README.
If you change the API surface, regenerate the committed spec:

```bash
python - <<'PY'
import json
from slm.api.app import create_app
from slm.config import Settings
app = create_app(Settings(api_keys="", log_level="WARNING"), engine=None, load_model=False)
json.dump(app.openapi(), open("docs/openapi.json", "w"), indent=2)
PY
```

## Reporting bugs & requesting features

Open a GitHub issue with: expected vs. actual behaviour, a minimal reproduction,
your environment (OS, Python, torch, device), and logs/stack traces. For security
issues, **do not** open a public issue — see
[docs/security.md](docs/security.md#reporting-a-vulnerability).

## Release process

Maintainers tag `vX.Y.Z` (semver); the [release workflow](.github/workflows/release.yml)
builds the sdist/wheel and publishes a versioned container image. Update
[CHANGELOG.md](CHANGELOG.md) before tagging.

## License

By contributing, you agree your contributions are licensed under the project's
[Apache 2.0 License](LICENSE).
