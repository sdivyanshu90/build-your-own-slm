"""Load / stress test for the BYO-SLM API using Locust.

This file is intentionally NOT collected by pytest (its name does not match the
``test_*`` pattern). It is an operational tool, run separately.

Install and run:

    pip install locust
    # against a running server (set a key if auth is enabled):
    SLM_LOAD_API_KEY=<key> locust -f tests/load/locustfile.py --host http://localhost:8000

Then open http://localhost:8089 and configure users / spawn-rate, or run headless:

    locust -f tests/load/locustfile.py --host http://localhost:8000 \
        --headless -u 50 -r 10 -t 2m --csv results

Use it to find the saturation point (where p95 latency degrades) and to validate
rate-limit behaviour (429s under burst) before sizing replicas — see
docs/operations-runbook.md#capacity-planning.
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, between, task

_API_KEY = os.getenv("SLM_LOAD_API_KEY", "")
_HEADERS = {"x-api-key": _API_KEY} if _API_KEY else {}

_PROMPTS = [
    "Once upon a time",
    "To be, or not to be",
    "The meaning of life is",
    "In the beginning",
    "ROMEO:",
]


class CompletionUser(HttpUser):
    """A virtual user that hits the completion + metadata endpoints."""

    wait_time = between(0.5, 2.0)

    @task(10)
    def completion(self) -> None:
        payload = {
            "prompt": random.choice(_PROMPTS),
            "max_tokens": random.choice([16, 32, 64]),
            "temperature": 0.8,
            "top_p": 0.95,
        }
        with self.client.post(
            "/v1/completions", json=payload, headers=_HEADERS, catch_response=True
        ) as resp:
            # 429s are an expected, valid outcome under load — don't count them
            # as failures.
            if resp.status_code in (200, 429):
                resp.success()
            else:
                resp.failure(f"unexpected status {resp.status_code}")

    @task(1)
    def models(self) -> None:
        self.client.get("/v1/models", headers=_HEADERS, name="/v1/models")

    @task(1)
    def health(self) -> None:
        self.client.get("/healthz", name="/healthz")
