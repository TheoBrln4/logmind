"""
Tests for exception handlers and startup behavior in app/main.py.
"""
from unittest.mock import patch, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

def test_ollama_connect_error_returns_503() -> None:
    """ConnectError from Ollama should produce a 503, not a raw 500."""
    with patch("app.agents.rca_agent.call_llm", side_effect=httpx.ConnectError("refused")):
        from app.generator.factory import generate_logs
        from app.models.schemas import Scenario
        logs = generate_logs(Scenario.OOM_CRASH, n_logs=20)
        response = client.post("/analyze/", json={"logs": logs})
    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]


def test_ollama_connect_timeout_returns_503() -> None:
    with patch("app.agents.rca_agent.call_llm", side_effect=httpx.ConnectTimeout("timeout")):
        from app.generator.factory import generate_logs
        from app.models.schemas import Scenario
        logs = generate_logs(Scenario.DB_TIMEOUT, n_logs=20)
        response = client.post("/analyze/", json={"logs": logs})
    assert response.status_code == 503


def test_generic_exception_returns_500() -> None:
    with patch("app.agents.rca_agent.call_llm", side_effect=RuntimeError("unexpected")):
        from app.generator.factory import generate_logs
        from app.models.schemas import Scenario
        logs = generate_logs(Scenario.OOM_CRASH, n_logs=20)
        response = client.post("/analyze/", json={"logs": logs})
    assert response.status_code == 500
    assert "RuntimeError" in response.json()["detail"]


def test_503_response_is_json() -> None:
    with patch("app.agents.rca_agent.call_llm", side_effect=httpx.ConnectError("refused")):
        from app.generator.factory import generate_logs
        from app.models.schemas import Scenario
        logs = generate_logs(Scenario.OOM_CRASH, n_logs=20)
        response = client.post("/analyze/", json={"logs": logs})
    assert response.headers["content-type"].startswith("application/json")
    assert "detail" in response.json()
