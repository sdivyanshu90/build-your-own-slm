# API Reference

Base URL: `http://<host>:<port>` (default `http://localhost:8000`). The machine
readable contract is served at `/openapi.json` and committed at
[`docs/openapi.json`](openapi.json); interactive explorers live at `/docs`
(Swagger UI) and `/redoc`.

- [Authentication](#authentication)
- [Versioning](#versioning)
- [Rate limiting](#rate-limiting)
- [Errors](#errors)
- [Endpoints](#endpoints)
  - [GET /healthz](#get-healthz)
  - [GET /readyz](#get-readyz)
  - [GET /metrics](#get-metrics)
  - [GET /v1/models](#get-v1models)
  - [POST /v1/completions](#post-v1completions)
- [Streaming](#streaming)
- [Status codes](#status-codes)

## Authentication

When the deployment sets `SLM_API_KEYS` (a comma-separated list), all `/v1/*`
endpoints require a valid key. Supply it either way:

```
X-API-Key: <key>
# or
Authorization: Bearer <key>
```

Keys are compared in constant time and never logged (a salted, truncated id is
logged instead). If `SLM_API_KEYS` is empty, authentication is **disabled** —
intended only for local development.

Generate a key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Versioning

Inference endpoints are namespaced under `/v1`. Backwards-incompatible changes
introduce `/v2` while `/v1` continues to function. Health/metrics endpoints are
unversioned infrastructure.

## Rate limiting

Each identity (API key, or client IP when unauthenticated) gets a token bucket:
`SLM_RATE_LIMIT_BURST` capacity refilling at `SLM_RATE_LIMIT_PER_MINUTE / 60`
per second. Exceeding it yields `429 Too Many Requests` with a `Retry-After`
header (seconds).

## Errors

Every error shares one envelope:

```json
{
  "error": {
    "type": "invalid_request",
    "message": "Prompt has 2048 tokens, exceeding the limit of 1024.",
    "request_id": "0f1a2b3c4d5e6f7081920a1b2c3d4e5f"
  }
}
```

`type` values: `http_error`, `validation_error`, `invalid_request`,
`internal_error`. The `request_id` matches the `X-Request-ID` response header
for log correlation. In production (`SLM_ENV=production`) `internal_error`
messages are redacted to `"Internal server error."`.

## Endpoints

### GET /healthz

Liveness. Always `200` if the process is reachable.

```json
{ "status": "ok", "version": "0.1.0", "model_loaded": true }
```

### GET /readyz

Readiness. Returns `503` until a model is loaded (so orchestrators do not route
traffic to a replica that cannot serve). Body shape matches `/healthz`.

### GET /metrics

Prometheus exposition (text format). Series include:

| Metric | Type | Labels | Meaning |
|--------|------|--------|---------|
| `slm_http_requests_total` | counter | method, path, status | Request counts |
| `slm_http_request_duration_seconds` | histogram | method, path | Latency |
| `slm_http_in_flight_requests` | gauge | – | Concurrent requests |
| `slm_tokens_generated_total` | counter | – | Completion tokens produced |

### GET /v1/models

Lists the served model and its architecture.

```bash
curl -s localhost:8000/v1/models -H "x-api-key: $KEY" | jq
```

```json
{
  "object": "list",
  "data": [{
    "id": "byo-slm", "object": "model",
    "parameters": 3221128, "parameters_human": "3.22M",
    "context_length": 256, "n_layer": 4, "n_head": 4, "n_embd": 128,
    "device": "cpu"
  }]
}
```

### POST /v1/completions

Generate a completion. Request body (`application/json`):

| Field | Type | Default | Constraints |
|-------|------|---------|-------------|
| `prompt` | string | — | required, ≤ 100 000 chars |
| `max_tokens` | int | 128 | 1 ≤ x ≤ 8192 (also clamped by `SLM_MAX_NEW_TOKENS`) |
| `temperature` | float | 0.8 | 0 ≤ x ≤ 5 (0 = greedy) |
| `top_k` | int / null | 40 | ≥ 1 |
| `top_p` | float / null | 0.95 | 0 < x ≤ 1 |
| `repetition_penalty` | float | 1.0 | 1 ≤ x ≤ 2 |
| `seed` | int / null | null | set for reproducible sampling |
| `stop` | string[] / null | null | ≤ 8 strings that halt generation |
| `stream` | bool | false | SSE stream when true |

Unknown fields are rejected (`422`). Example:

```bash
curl -s localhost:8000/v1/completions \
  -H "content-type: application/json" -H "x-api-key: $KEY" \
  -d '{"prompt":"To be, or not to be","max_tokens":40,"temperature":0.7,"seed":1}' | jq
```

```json
{
  "id": "cmpl-7f3c…",
  "object": "text_completion",
  "model": "byo-slm",
  "choices": [{ "index": 0, "text": ", that is the question…", "finish_reason": "length" }],
  "usage": { "prompt_tokens": 7, "completion_tokens": 40, "total_tokens": 47 }
}
```

`finish_reason` is `length` (hit `max_tokens`) or `stop` (end-of-text token or a
`stop` string matched).

## Streaming

Set `"stream": true` to receive Server-Sent Events. Each event is a JSON chunk;
the stream terminates with a finish event and then the literal `[DONE]`:

```
data: {"id":"cmpl-…","object":"text_completion.chunk","choices":[{"index":0,"text":"To","finish_reason":null}]}
data: {"id":"cmpl-…","object":"text_completion.chunk","choices":[{"index":0,"text":" be","finish_reason":null}]}
data: {"id":"cmpl-…","object":"text_completion.chunk","choices":[{"index":0,"text":"","finish_reason":"stop"}]}
data: [DONE]
```

Multi-byte UTF-8 characters are split across tokens safely — the engine only
emits complete characters. If the client disconnects, generation is cancelled.

```python
import httpx, json
with httpx.stream("POST", "http://localhost:8000/v1/completions",
                  headers={"x-api-key": KEY},
                  json={"prompt": "Hello", "max_tokens": 50, "stream": True}) as r:
    for line in r.iter_lines():
        if line.startswith("data:") and "[DONE]" not in line:
            print(json.loads(line[5:])["choices"][0]["text"], end="", flush=True)
```

## Status codes

| Code | When |
|------|------|
| 200 | Success (JSON or SSE stream) |
| 400 | `invalid_request` — input violates an engine limit (e.g. prompt too long) |
| 401 | Missing/invalid API key |
| 422 | Request body failed schema validation |
| 429 | Rate limit exceeded (`Retry-After` provided) |
| 500 | Unhandled server error (redacted in production) |
| 503 | Model not loaded / not ready |
