"""
Integration tests for the FastAPI endpoints.
The LLM (Ollama) is mocked via unittest.mock.patch — no external service needed.
"""
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import Scenario
from app.generator.factory import generate_logs

client = TestClient(app)

# ---------------------------------------------------------------------------
# Mock LLM helpers
# ---------------------------------------------------------------------------

_RCA_RESPONSE = "The Spark executor exhausted heap memory due to an unbounded partition."

_REPORT_JSON = {
    "hypotheses": ["No JVM memory cap", "Dataset grew unexpectedly"],
    "impact": "ETL pipeline failed for 6 hours.",
    "recommended_actions": ["Set spark.executor.memory", "Add monitoring"],
}


def _patch_llms():
    """Patch call_llm and call_llm_report independently (different module namespaces)."""
    p1 = patch("app.agents.rca_agent.call_llm", return_value=_RCA_RESPONSE)
    p2 = patch("app.agents.report_agent.call_llm_report", return_value=_REPORT_JSON)
    return p1, p2


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /generate/
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", [s.value for s in Scenario])
def test_generate_returns_logs(scenario: str) -> None:
    response = client.post("/generate/", json={"scenario": scenario, "n_logs": 10})
    assert response.status_code == 200
    body = response.json()
    assert body["scenario"] == scenario
    assert len(body["logs"]) == 10
    assert all(isinstance(l, str) for l in body["logs"])


def test_generate_default_count() -> None:
    response = client.post("/generate/", json={"scenario": "oom_crash"})
    assert response.status_code == 200
    assert len(response.json()["logs"]) == 50


def test_generate_zero_logs() -> None:
    response = client.post("/generate/", json={"scenario": "db_timeout", "n_logs": 0})
    assert response.status_code == 200
    assert response.json()["logs"] == []


def test_generate_invalid_scenario() -> None:
    response = client.post("/generate/", json={"scenario": "not_a_scenario"})
    assert response.status_code == 422


def test_generate_missing_scenario() -> None:
    response = client.post("/generate/", json={"n_logs": 5})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /analyze/
# ---------------------------------------------------------------------------

def test_analyze_empty_logs_returns_422() -> None:
    response = client.post("/analyze/", json={"logs": []})
    assert response.status_code == 422


def test_analyze_missing_logs_field_returns_422() -> None:
    response = client.post("/analyze/", json={})
    assert response.status_code == 422


def test_analyze_garbage_logs_returns_valid_response() -> None:
    """Unparseable lines → no events → no clusters → rca skips LLM → report uses fallback root_cause."""
    p1, p2 = _patch_llms()
    with p1, p2:
        response = client.post("/analyze/", json={"logs": ["not a log", "also garbage"]})
    assert response.status_code == 200
    body = response.json()
    assert body["events"] == []
    assert body["clusters"] == []
    assert "report" in body
    assert body["report"]["root_cause"]


@pytest.mark.parametrize("scenario", [s.value for s in Scenario])
def test_analyze_full_pipeline(scenario: str) -> None:
    logs = generate_logs(Scenario(scenario), n_logs=30)
    p1, p2 = _patch_llms()
    with p1, p2:
        response = client.post("/analyze/", json={"logs": logs})
    assert response.status_code == 200
    body = response.json()
    assert len(body["events"]) == 30
    assert "clusters" in body
    assert body["report"]["root_cause"] == _RCA_RESPONSE
    assert isinstance(body["report"]["hypotheses"], list)
    assert isinstance(body["report"]["recommended_actions"], list)


def test_analyze_response_schema() -> None:
    """Verify the response has all required top-level keys."""
    logs = generate_logs(Scenario.OOM_CRASH, n_logs=20)
    p1, p2 = _patch_llms()
    with p1, p2:
        response = client.post("/analyze/", json={"logs": logs})
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"events", "clusters", "report"}
    report = body["report"]
    assert set(report.keys()) >= {"root_cause", "hypotheses", "impact", "recommended_actions"}
