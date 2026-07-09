"""Phase 4e — load test for the FastAPI serving layer.

This measures the HTTP + guardrails + Prometheus overhead of this service,
NOT real LLM inference latency — that depends entirely on whatever backend
VLLM_BASE_URL points at, which this repo has no GPU to run. Run against a
live server with no backend configured, the realistic worst case (backend
totally down) still has to be handled gracefully under load — that's
exactly what src.serve.guardrails.safe_generate exists for.

Run against a live `uvicorn src.serve.app:app` process:

    uvicorn src.serve.app:app &
    locust -f tests/load/locustfile.py --host http://127.0.0.1:8000 --headless \\
        -u 20 -r 5 -t 30s --csv reports/load_test --html reports/load_test.html
"""

import random

from locust import HttpUser, between, task

_TRANSCRIPTS = [
    "Hello, this is Microsoft support calling. We detected a virus on your computer.",
    "Hi, just confirming your dentist appointment for Tuesday at 3pm.",
    "Congratulations, you've won a free vacation! Just provide your card for a deposit.",
    "Hi, this is the delivery driver, I'm outside and can't find the entrance.",
    "This is the IRS. You owe back taxes and must pay immediately with gift cards.",
]


class TriageUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(5)
    def triage_text(self):
        self.client.post("/triage/text", json={"transcript": random.choice(_TRANSCRIPTS)})

    @task(1)
    def health(self):
        self.client.get("/health")

    @task(1)
    def ready(self):
        self.client.get("/ready")

    @task(1)
    def metrics(self):
        self.client.get("/metrics")
