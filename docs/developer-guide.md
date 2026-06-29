# Developer Guide

A guided tour of the codebase and the conventions for extending it.

- [Getting set up](#getting-set-up)
- [Repository tour](#repository-tour)
- [Conventions](#conventions)
- [Testing](#testing)
- [How to extend](#how-to-extend)
- [Reading order](#reading-order)

## Getting set up

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
make install      # pip install -e ".[dev]"
make hooks        # install pre-commit hooks
make check        # lint + typecheck + tests (CI parity)
```

`make help` lists every task.

## Repository tour

The dependency arrow points *inward* — delivery code (CLI/API) depends on
application code (engine/trainer), which depends on domain code (model,
tokenizer, sampling), which depends on foundation code (config, logging, utils).
Nothing in `model/`, `tokenizer.py`, or `generation/` imports `api/`.

| Path | What lives here | Start at |
|------|-----------------|----------|
| `config.py` | `Settings` (env) + `ExperimentConfig` (YAML), all validated | `Settings`, `ExperimentConfig` |
| `tokenizer.py` | Byte-level BPE: incremental-index training, encode/decode, JSON I/O | `BPETokenizer.train`, `.encode` |
| `data.py` | `prepare_dataset` (text→tokenizer→uint16 bins), memmap `get_batch` | `prepare_dataset` |
| `model/layers.py` | `LayerNorm`, `CausalSelfAttention`, `MLP`, `Block` | `CausalSelfAttention.forward` |
| `model/gpt.py` | `GPT` — embeddings, blocks, tied head, optimizer groups | `GPT.forward` |
| `generation/sampler.py` | `GenerationConfig`, logit filters, `generate_tokens` | `generate_tokens` |
| `training/schedule.py` | `cosine_lr` warmup+decay | `cosine_lr` |
| `training/checkpoint.py` | Atomic save/load, `build_model` | `save_checkpoint` |
| `training/trainer.py` | The optimization loop, AMP, eval, resume | `Trainer.train` |
| `inference/engine.py` | `InferenceEngine` — load, generate, stream, limits | `InferenceEngine.generate` |
| `api/app.py` | `create_app` factory: middleware, routers, error handlers, lifespan | `create_app` |
| `api/routes.py` | Health, metrics, `/v1/models`, `/v1/completions` (+SSE) | `create_completion` |
| `api/security.py` | API-key auth + `RateLimiter` | `api_key_dependency`, `RateLimiter` |
| `api/middleware.py` | Request id, metrics, access logs, secure headers | `RequestContextMiddleware` |
| `cli.py` | Typer commands wiring everything together | `app` |

## Conventions

- **Python 3.10+**, `from __future__ import annotations` everywhere; modern
  typing (`X | Y`, `list[...]`).
- **Typed & validated**: pydantic v2 models for config and API; `mypy` is strict
  on `src/` (`make typecheck`).
- **Formatting/lint**: `ruff` (format + lint), 100-col lines (`make format`).
- **Docstrings**: module, class, and non-trivial function docstrings explain the
  *why*; comments flag non-obvious decisions.
- **Logging**: `structlog` via `get_logger(__name__)`; emit events as
  `log.info("noun.verb", key=value)` (machine-friendly), never f-strings.
- **Errors**: raise `ValueError` for bad input in the domain/engine — the API
  maps it to a `400` envelope. Don't leak internals in production.
- **Determinism in tests**: seed everything; pass explicit `torch.Generator`s.

## Testing

```bash
make test                    # full suite + coverage (gate ≥95%)
pytest tests/unit -q         # fast unit tests
pytest -k tokenizer -q       # by keyword
pytest -m "not slow"         # skip slow markers
```

Layout: `tests/unit/` (pure logic) and `tests/integration/` (training, engine,
API, CLI). A session-scoped fixture (`trained_model_dir`) trains one tiny model
and the API/engine tests reuse it, so the whole suite runs in seconds while still
covering the real train→serve path. Add tests beside the behaviour they cover and
keep coverage ≥95% (CI enforces `--cov-fail-under=95`).

## How to extend

**Add a generation parameter** (e.g. `min_p`):
1. Add the field to `GenerationConfig` (validated) and apply it in
   `sample_next_token`.
2. Surface it in `api/schemas.py::CompletionRequest.to_generation_config` and the
   CLI `generate` command.
3. Add unit tests in `tests/unit/test_sampler.py`.

**Add an API endpoint:**
1. Add request/response models to `api/schemas.py`.
2. Add the handler to `api/routes.py` (depend on `rate_limit_dependency` for
   authn/z + limiting; access the engine via `_engine(request)`).
3. Test it in `tests/integration/test_api.py`.

**Swap the positional scheme (RoPE/ALiBi):** implement it inside
`model/layers.py::CausalSelfAttention` and remove the learned
`position_embedding` in `GPT.__init__`; the rest of the stack is unaffected.

**Add a KV cache:** extend `generate_tokens` (the single decode loop) to thread a
cache between steps; nothing else changes.

**New model size:** copy `configs/small.yaml`, adjust dims, run
`prepare-data` + `train`. See [training.md](training.md#scaling-the-model-up).

## Reading order

For a first pass, read in dependency order: `config.py` → `tokenizer.py` →
`model/layers.py` → `model/gpt.py` → `generation/sampler.py` →
`training/trainer.py` → `inference/engine.py` → `api/app.py` → `api/routes.py`.
The architecture and diagrams are in [architecture.md](architecture.md).
