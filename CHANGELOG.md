# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-29

The first release: a complete, production-grade Small Language Model — tokenizer,
model, training, inference, and serving — built from first principles.

### Added

**Tokenizer**
- From-scratch byte-level BPE (`BPETokenizer`) with GPT-2-style pre-tokenization
  and byte↔unicode mapping; lossless round-trip on arbitrary UTF-8.
- Fast incremental pair-count training index; trainable, JSON-serialisable,
  special-token aware, with streaming-friendly `id_to_bytes`.

**Model**
- Configurable GPT (`GPT`): pre-norm transformer blocks, multi-head causal
  self-attention via fused `scaled_dot_product_attention` (with a manual
  fallback), 4× GELU MLP, weight tying, GPT-2 scaled init, optional gradient
  checkpointing.
- AdamW optimizer grouping (decay/no-decay) with fused-kernel support on CUDA.

**Data & training**
- `prepare_dataset`: corpus → tokenizer → `uint16` memory-mapped train/val
  binaries (with optional URL download).
- `Trainer`: mixed precision (bf16/fp16 + `GradScaler`), gradient accumulation,
  cosine LR schedule with warmup, gradient clipping, periodic evaluation,
  best-checkpoint saving, and resumable state.
- Atomic checkpoint save/load carrying weights, config, optimizer, and recipe.

**Inference & generation**
- Sampling: temperature, top-k, top-p (nucleus), repetition penalty, seedable
  reproducibility (`GenerationConfig`, `generate_tokens`).
- `InferenceEngine`: load a model directory, enforce prompt/token limits,
  buffered `generate` and streaming `stream` with correct incremental UTF-8
  decoding and stop sequences.

**HTTP API**
- FastAPI app factory with OpenAI-style `/v1/completions` (buffered + SSE),
  `/v1/models`, `/healthz`, `/readyz`, `/metrics`.
- API-key authentication (constant-time), per-identity token-bucket rate
  limiting, strict request validation, uniform error envelope, request-id
  propagation, security headers, CORS allowlist, and Prometheus metrics.

**Tooling & ops**
- `slm` CLI (`prepare-data`, `train`, `generate`, `serve`, `info`).
- Structured logging (console/JSON) via structlog.
- Multi-stage non-root Dockerfile, docker-compose stack with Prometheus,
  GitHub Actions CI (lint, mypy, tests on 3.10–3.12, image build) and release.
- Experiment recipes (`configs/tiny.yaml`, `configs/small.yaml`).

**Quality**
- ~98% test coverage across unit and integration suites (CI gate ≥ 95%).
- Comprehensive documentation: architecture, API, training, deployment,
  security, operations runbook, and developer guide.

[Unreleased]: https://github.com/your-org/build-your-own-slm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/build-your-own-slm/releases/tag/v0.1.0
