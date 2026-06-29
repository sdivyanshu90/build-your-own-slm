# Deployment Guide

How to run the BYO-SLM inference API in development, with Docker, with Compose,
and on Kubernetes — plus CI/CD, scaling, and rollback.

- [Prerequisites](#prerequisites)
- [Local (bare metal)](#local-bare-metal)
- [Docker](#docker)
- [Docker Compose](#docker-compose)
- [Kubernetes](#kubernetes)
- [Configuration reference](#configuration-reference)
- [CI/CD](#cicd)
- [Scaling & autoscaling](#scaling--autoscaling)
- [Rollback strategy](#rollback-strategy)
- [Hardening checklist](#hardening-checklist)

## Prerequisites

A trained model directory containing `model.pt` + `tokenizer.json` (produce one
with `slm prepare-data` + `slm train`, see [training.md](training.md)). The
serving process needs only that directory and the Python runtime.

## Local (bare metal)

```bash
export SLM_MODEL_DIR=checkpoints/tiny
export SLM_API_KEYS=$(python -c "import secrets;print(secrets.token_urlsafe(32))")
slm serve --host 0.0.0.0 --port 8000
# production-style (multiple workers, JSON logs):
SLM_LOG_JSON=true uvicorn slm.api.app:create_app --factory --host 0.0.0.0 --port 8000 --workers 4
```

> On a single GPU prefer **replicas over workers** (each worker loads its own
> copy of the model into VRAM).

## Docker

```bash
docker build -t byo-slm:latest .
docker run --rm -p 8000:8000 \
  -e SLM_API_KEYS="$SLM_API_KEYS" \
  -e SLM_MODEL_DIR=/app/checkpoints/tiny \
  -v "$PWD/checkpoints:/app/checkpoints:ro" \
  byo-slm:latest
```

The image is multi-stage and runs as a non-root user (`slm`, uid 1001); the
model is **mounted read-only**, never baked in. A built-in `HEALTHCHECK` polls
`/healthz`. For GPU, base the image on an NVIDIA CUDA image and install the CUDA
build of PyTorch (swap the index URL in the `Dockerfile`), then run with
`--gpus all` and `-e SLM_DEVICE=cuda`.

## Docker Compose

```bash
# API only
docker compose up --build api
# API + Prometheus
docker compose --profile monitoring up --build
```

Set secrets via the environment or an `.env` file (compose reads `SLM_API_KEYS`,
`SLM_RATE_LIMIT_PER_MINUTE`, …). Prometheus scrapes `api:8000/metrics` using
[`docker/prometheus.yml`](../docker/prometheus.yml).

## Kubernetes

A minimal, production-shaped manifest (probes, resources, non-root, read-only
model via a PVC or initContainer):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: byo-slm }
spec:
  replicas: 3
  selector: { matchLabels: { app: byo-slm } }
  template:
    metadata: { labels: { app: byo-slm } }
    spec:
      securityContext: { runAsNonRoot: true, runAsUser: 1001 }
      containers:
        - name: api
          image: ghcr.io/your-org/build-your-own-slm:0.1.0
          ports: [{ containerPort: 8000 }]
          env:
            - { name: SLM_ENV, value: production }
            - { name: SLM_LOG_JSON, value: "true" }
            - { name: SLM_MODEL_DIR, value: /models/tiny }
            - name: SLM_API_KEYS
              valueFrom: { secretKeyRef: { name: byo-slm-keys, key: api-keys } }
          volumeMounts:
            - { name: models, mountPath: /models, readOnly: true }
          readinessProbe:
            httpGet: { path: /readyz, port: 8000 }
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet: { path: /healthz, port: 8000 }
            periodSeconds: 15
          resources:
            requests: { cpu: "1", memory: 1Gi }
            limits: { cpu: "2", memory: 4Gi }
      volumes:
        - name: models
          persistentVolumeClaim: { claimName: byo-slm-models }
---
apiVersion: v1
kind: Service
metadata: { name: byo-slm }
spec:
  selector: { app: byo-slm }
  ports: [{ port: 80, targetPort: 8000 }]
```

`readyz` gates traffic until the model is loaded; `healthz` drives restarts.
Terminate TLS at the Ingress/load balancer.

## Configuration reference

All knobs are `SLM_`-prefixed environment variables (see `.env.example` and
`slm.config.Settings`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `SLM_ENV` | development | development / staging / production (redacts errors) |
| `SLM_LOG_LEVEL` / `SLM_LOG_JSON` | INFO / false | Verbosity / structured JSON logs |
| `SLM_MODEL_DIR` | ./checkpoints/tiny | Directory with `model.pt` + `tokenizer.json` |
| `SLM_DEVICE` | auto | auto / cpu / cuda / mps |
| `SLM_MAX_NEW_TOKENS` | 256 | Hard ceiling per request |
| `SLM_MAX_PROMPT_TOKENS` | 1024 | Reject longer prompts |
| `SLM_API_HOST` / `SLM_API_PORT` | 0.0.0.0 / 8000 | Bind address |
| `SLM_CORS_ORIGINS` | `*` | Comma-separated allowlist |
| `SLM_API_KEYS` | _(empty)_ | Comma-separated keys; empty disables auth |
| `SLM_RATE_LIMIT_PER_MINUTE` / `SLM_RATE_LIMIT_BURST` | 60 / 10 | Token bucket |

## CI/CD

[`/.github/workflows/ci.yml`](../.github/workflows/ci.yml) on every push/PR:

1. **lint** — `ruff check`, `ruff format --check`, `mypy`.
2. **test** — matrix on Python 3.10/3.11/3.12, `pytest` with `--cov-fail-under=95`.
3. **docker** — build the image (with GitHub Actions layer cache).

[`release.yml`](../.github/workflows/release.yml) on a `v*` tag builds the sdist
+ wheel and pushes a versioned image to GHCR.

## Scaling & autoscaling

- **Horizontal**: increase `replicas`; add an HPA on CPU or a custom metric
  (e.g. `slm_http_in_flight_requests`).
- **Vertical**: give replicas more CPU/GPU; raise resource limits.
- The service is **stateless**, so scaling is linear and safe. The only caveat
  is the in-process rate limiter (per-replica) — use a Redis-backed limiter for
  globally consistent limits (see [security.md](security.md#rate-limiting)).

## Rollback strategy

1. **Image rollback** — redeploy the previous tag (`kubectl rollout undo
   deployment/byo-slm`, or repoint the Compose image). The app is stateless.
2. **Model rollback** — point `SLM_MODEL_DIR` at a previous checkpoint directory
   and restart; checkpoints are immutable and versioned in object storage.
3. **Config rollback** — revert the env/Secret change and restart.

Roll out with a readiness gate (`/readyz`) and a small surge so a bad model
version never receives traffic.

## Hardening checklist

- [ ] `SLM_ENV=production`, `SLM_LOG_JSON=true`
- [ ] `SLM_API_KEYS` set (auth enabled), keys from a Secret manager
- [ ] `SLM_CORS_ORIGINS` restricted to known origins
- [ ] TLS terminated at the edge; HSTS sent on HTTPS
- [ ] Resource requests/limits set; readiness + liveness probes wired
- [ ] Model volume mounted read-only; container runs as non-root
- [ ] Metrics scraped; alerts configured (see [operations-runbook.md](operations-runbook.md))
