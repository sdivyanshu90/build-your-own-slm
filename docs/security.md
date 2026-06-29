# Security

Threat model, controls, and the OWASP mapping for BYO-SLM. Security is layered;
no single control is load-bearing.

- [Assets & threat model](#assets--threat-model)
- [Authentication](#authentication)
- [Authorization](#authorization)
- [Input validation](#input-validation)
- [Rate limiting & abuse](#rate-limiting--abuse)
- [Transport & headers](#transport--headers)
- [Secrets management](#secrets-management)
- [Logging & auditing](#logging--auditing)
- [Supply chain](#supply-chain)
- [OWASP API Top 10 mapping](#owasp-api-top-10-mapping)
- [Reporting a vulnerability](#reporting-a-vulnerability)

## Assets & threat model

| Asset | Threat | Primary control |
|-------|--------|-----------------|
| Model weights / tokenizer | Exfiltration, tampering | Read-only mount, image provenance |
| API availability | DoS, runaway generation | Rate limiting, input limits, resource caps |
| API keys / secrets | Leak via logs/errors | Constant-time compare, no key logging, redaction |
| Inference compute | Resource exhaustion (huge prompts/outputs) | `max_prompt_tokens`, `max_new_tokens`, context crop |
| Host | RCE / privilege escalation | Non-root container, minimal image, no shell input eval |

Out of scope: model *content* safety (toxicity/jailbreak filtering) — add a
moderation layer in front for that use case.

## Authentication

Opaque API keys via `X-API-Key` or `Authorization: Bearer`. Keys are compared
with `hmac.compare_digest` (constant time) against `SLM_API_KEYS`. Auth is
enforced whenever at least one key is configured; an empty list disables it
(development only). Rotate keys by adding the new key, deploying, then removing
the old one (both are accepted during overlap).

## Authorization

The current model is **all-or-nothing** per valid key (any key may call any
`/v1` endpoint). The single extension point is `api_key_dependency`, which
returns a caller identity — attach scopes/quotas there to implement per-key
authorization without touching route code.

## Input validation

- **Strict schemas**: every request model sets `extra="forbid"`; unknown fields
  are rejected (`422`). All numeric parameters are bounded (`temperature ≤ 5`,
  `max_tokens ≤ 8192`, `top_p ≤ 1`, ≤ 8 stop strings, prompt ≤ 100 000 chars).
- **Engine limits**: prompts over `SLM_MAX_PROMPT_TOKENS` or the model context
  are rejected (`400`); `max_tokens` is clamped to `SLM_MAX_NEW_TOKENS`.
- **No injection surface**: there is no SQL, no shell, no template rendering, and
  no `eval` on request data — the only "interpreter" is the model itself, which
  receives integer token ids, not executable text.

## Rate limiting & abuse

A per-identity token bucket (`SLM_RATE_LIMIT_BURST` capacity,
`SLM_RATE_LIMIT_PER_MINUTE/60` refill) smooths bursts and bounds spend; `429`
includes `Retry-After`. The default limiter is **in-process** (per replica). For
globally consistent limits across replicas, replace `RateLimiter` with a
Redis-backed implementation exposing the same `allow(identity) -> (bool, float)`
contract — only that class changes.

## Transport & headers

Every response carries hardening headers (`X-Content-Type-Options: nosniff`,
`X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, a restrictive CSP,
`Cross-Origin-Opener-Policy`). `Strict-Transport-Security` is emitted on HTTPS.
**Terminate TLS at the edge** (Ingress/LB); the app speaks HTTP inside the
trust boundary. CORS is an explicit allowlist (`SLM_CORS_ORIGINS`); credentials
are not allowed and methods are limited to `GET`/`POST`.

## Secrets management

- Provide secrets via environment/secret manager (K8s Secret, Vault, SSM) —
  never commit them. `.env` is git-ignored; only `.env.example` is tracked.
- API keys are never written to logs; structured logs include only a salted,
  truncated identifier (`key_<12 hex>`).
- `detect-private-key` runs as a pre-commit hook.

## Logging & auditing

Structured (JSON in production) logs include a `request_id` (also returned as
`X-Request-ID`) bound to every line for a request, the method/path, status, and
latency — enough for audit and correlation without logging request bodies or
secrets. Ship logs to a central, access-controlled, retention-bounded store.

## Supply chain

- Dependencies are pinned with floors/ceilings in `pyproject.toml`; the lockable
  surface is small and well-known (PyTorch, FastAPI, pydantic).
- CI runs on every PR; add `pip-audit`/Dependabot to alert on CVEs.
- The release image is built in CI and published to GHCR with semver + SHA tags
  for provenance; sign with cosign for full supply-chain attestation.
- Multi-stage build discards the compiler toolchain from the runtime image,
  shrinking the attack surface.

## OWASP API Security Top 10 (2023) mapping

| Risk | Mitigation here |
|------|-----------------|
| API1 Broken Object Level Auth | No object ids/multi-tenant data; single model resource |
| API2 Broken Authentication | Constant-time key auth, no key logging, rotation supported |
| API3 Broken Object Property Auth | `extra="forbid"` strict schemas, explicit response models |
| API4 Unrestricted Resource Consumption | Prompt/token limits, rate limiting, container resource caps |
| API5 Broken Function Level Auth | Single function level; extension point documented |
| API6 Unrestricted Business Flows | Token bucket + per-key identity |
| API7 SSRF | No outbound fetch from request data (only operator-set `source_url` at prep time) |
| API8 Security Misconfiguration | Secure-by-default headers, prod error redaction, non-root image |
| API9 Improper Inventory Mgmt | Versioned `/v1`, committed OpenAPI spec |
| API10 Unsafe Consumption of APIs | No third-party API calls in the serving path |

## Reporting a vulnerability

Please report security issues privately to the maintainers (see
`CONTRIBUTING.md`) rather than opening a public issue. We aim to acknowledge
within 72 hours.
