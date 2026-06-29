# Operations Runbook

For on-call engineers operating the BYO-SLM inference service.

- [Service overview](#service-overview)
- [Dashboards & golden signals](#dashboards--golden-signals)
- [Alerts](#alerts)
- [Health checks](#health-checks)
- [Common incidents](#common-incidents)
- [Routine operations](#routine-operations)
- [Backup & restore](#backup--restore)
- [Capacity planning](#capacity-planning)
- [Log queries](#log-queries)

## Service overview

| Property | Value |
|----------|-------|
| Process | `uvicorn slm.api.app:create_app --factory` |
| Port | 8000 (HTTP) |
| State | Stateless; loads read-only `model.pt` + `tokenizer.json` at startup |
| Liveness | `GET /healthz` |
| Readiness | `GET /readyz` (503 until model loaded) |
| Metrics | `GET /metrics` (Prometheus) |
| Logs | structured JSON on stderr (`SLM_LOG_JSON=true`) |

## Dashboards & golden signals

Build a dashboard from the exported metrics:

- **Latency** — `histogram_quantile(0.95, sum(rate(slm_http_request_duration_seconds_bucket[5m])) by (le, path))`
- **Traffic** — `sum(rate(slm_http_requests_total[5m])) by (path)`
- **Errors** — `sum(rate(slm_http_requests_total{status=~"5.."}[5m])) / sum(rate(slm_http_requests_total[5m]))`
- **Saturation** — `slm_http_in_flight_requests`
- **Throughput** — `rate(slm_tokens_generated_total[5m])` (tokens/sec)

## Alerts

| Alert | Condition (5m) | Severity | First action |
|-------|----------------|----------|--------------|
| HighErrorRate | 5xx ratio > 2% | page | Check logs by `request_id`; recent deploy? |
| HighLatencyP95 | p95 > 2s (CPU) | warn | Check `in_flight`/CPU; scale out |
| NotReady | `up{job=byo-slm}`==1 but `/readyz` 503 > 5m | page | Model failed to load — check `engine.load_failed` logs |
| Saturation | `in_flight` ≥ replicas × N | warn | Scale replicas (HPA or manual) |
| RateLimitSpike | sharp rise in `status="429"` | info | Possible abuse or under-provisioned limits |
| PodCrashLoop | restarts > 3 / 10m | page | Inspect liveness failures, OOM |

## Health checks

```bash
curl -fsS localhost:8000/healthz   # liveness — process up
curl -isS localhost:8000/readyz    # readiness — 200 only when model loaded
```

Orchestrators use `/readyz` to gate traffic and `/healthz` to decide restarts.

## Common incidents

### 503 on /v1/* (model not loaded)
- **Diagnose:** `/readyz` returns 503; logs show `engine.not_found` or
  `engine.load_failed`.
- **Fix:** verify `SLM_MODEL_DIR` points at a directory containing both
  `model.pt` and `tokenizer.json`; check the volume mount; confirm the
  checkpoint matches this code version (`Unsupported checkpoint version` ⇒
  rebuild/migrate). Restart after correcting.

### Elevated latency
- **Diagnose:** p95 up, `in_flight` high. Autoregressive decoding is one forward
  pass per token, so latency scales with `max_tokens`.
- **Fix:** scale out replicas; consider a GPU (`SLM_DEVICE=cuda`); lower
  `SLM_MAX_NEW_TOKENS`; ensure clients use streaming for perceived latency.

### 5xx spike after deploy
- **Fix:** roll back the image/model (`kubectl rollout undo deployment/byo-slm`).
  The service is stateless; rollback is immediate. Correlate by `request_id`.

### OOM / pod kills
- **Fix:** raise memory limits, or reduce concurrency. Model + tokenizer memory
  is fixed per replica; spikes come from concurrency × context length.

### 429 storms
- **Fix:** confirm whether legitimate (raise `SLM_RATE_LIMIT_PER_MINUTE`/replicas)
  or abuse (tighten limits, block the key/IP at the edge).

## Routine operations

- **Deploy a new model:** publish the new checkpoint dir to the model store,
  point `SLM_MODEL_DIR` at it (or update the PVC), and do a rolling restart.
  `/readyz` ensures no traffic hits a replica until its model is loaded.
- **Rotate API keys:** add the new key to `SLM_API_KEYS`, deploy, then remove the
  old key after clients migrate (both valid during overlap).
- **Scale:** adjust `replicas`/HPA. Prefer replicas over uvicorn workers on GPU.

## Backup & restore

**What to back up:** the checkpoint directory (`model.pt`, `tokenizer.json`) and
the experiment recipe (`configs/*.yaml`) + source corpus. The recipe + corpus
deterministically reproduce a model; the checkpoint is the fast path.

**Restore:** pull the checkpoint from object storage into the model volume and
restart, or re-run `slm prepare-data` + `slm train` from the recipe.

- **RPO:** last saved checkpoint (controlled by `train.eval_interval`).
- **RTO:** image pull + model load (seconds–minutes for small models).

## Capacity planning

Throughput per replica ≈ `tokens_per_second / average_tokens_per_request`.
Measure `rate(slm_tokens_generated_total[5m])` per replica at saturation, then
provision `replicas = peak_tokens_per_sec / per_replica_tokens_per_sec × headroom`.

## Log queries

Logs are JSON with stable keys (`event`, `request_id`, `status`, `duration_ms`,
`path`). Examples (jq / log backend):

```bash
# All lines for one request
... | jq 'select(.request_id=="<id>")'
# Slow requests
... | jq 'select(.event=="request.completed" and .duration_ms>1000)'
# Model load failures
... | jq 'select(.event=="engine.load_failed")'
```
