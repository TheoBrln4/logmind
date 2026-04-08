from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from app.agents.rca_agent import build_prompt, build_rag_context, call_llm, rca_agent
from app.models.schemas import Cluster, LogEvent, LogLevel, Scenario
from app.generator.factory import generate_logs
from app.agents.parser_agent import parser_agent
from app.agents.pattern_agent import pattern_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_event(message: str, level: LogLevel = LogLevel.ERROR) -> LogEvent:
    return LogEvent(
        timestamp=datetime(2026, 1, 1),
        level=level,
        service="svc",
        message=message,
        raw=f"2026-01-01 00:00:00.000 {level} svc {message}",
    )


def make_cluster(cluster_id: int, representative: str, size: int = 3) -> Cluster:
    return Cluster(cluster_id=cluster_id, size=size, representative=representative)


_MOCK_ROOT_CAUSE = "The Spark executor ran out of heap memory due to a large partition scan with no memory limit configured."


def _mock_llm_response(*args, **kwargs) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {"response": _MOCK_ROOT_CAUSE}
    return mock


# ---------------------------------------------------------------------------
# build_rag_context — unit tests (no LLM needed)
# ---------------------------------------------------------------------------

def test_build_rag_context_returns_string() -> None:
    events = [make_event("heap used 2048 MB"), make_event("GC overhead exceeded")]
    clusters = [make_cluster(0, "OutOfMemoryError heap space")]
    result = build_rag_context(clusters, events)
    assert isinstance(result, str)


def test_build_rag_context_empty_events() -> None:
    clusters = [make_cluster(0, "OutOfMemoryError")]
    assert build_rag_context(clusters, []) == ""


def test_build_rag_context_empty_clusters() -> None:
    events = [make_event("some error")]
    assert build_rag_context([], events) == ""


def test_build_rag_context_no_duplicates() -> None:
    events = [make_event("heap used 2048 MB")] * 10
    clusters = [make_cluster(0, "heap used 2048 MB")]
    context = build_rag_context(clusters, events)
    lines = [l for l in context.splitlines() if l]
    assert len(lines) == len(set(lines))


def test_build_rag_context_contains_relevant_message() -> None:
    events = [
        make_event("OutOfMemoryError Java heap space"),
        make_event("unrelated info message", level=LogLevel.INFO),
    ]
    clusters = [make_cluster(0, "OutOfMemoryError heap")]
    context = build_rag_context(clusters, events)
    assert "OutOfMemoryError" in context


# ---------------------------------------------------------------------------
# build_prompt — unit tests
# ---------------------------------------------------------------------------

def test_build_prompt_contains_cluster_info() -> None:
    clusters = [make_cluster(0, "SQLTimeout retry exceeded", size=5)]
    prompt = build_prompt(clusters, "some context")
    assert "Cluster 0" in prompt
    assert "SQLTimeout retry exceeded" in prompt
    assert "5 events" in prompt


def test_build_prompt_contains_rag_context() -> None:
    clusters = [make_cluster(0, "error")]
    context = "specific log context line"
    prompt = build_prompt(clusters, context)
    assert context in prompt


def test_build_prompt_ends_with_root_cause_cue() -> None:
    clusters = [make_cluster(0, "error")]
    prompt = build_prompt(clusters, "ctx")
    assert "Root cause:" in prompt


def test_build_prompt_multiple_clusters() -> None:
    clusters = [make_cluster(i, f"error type {i}") for i in range(3)]
    prompt = build_prompt(clusters, "ctx")
    assert "Cluster 0" in prompt
    assert "Cluster 1" in prompt
    assert "Cluster 2" in prompt


# ---------------------------------------------------------------------------
# call_llm — mocked HTTP
# ---------------------------------------------------------------------------

@patch("app.agents.rca_agent.httpx.post", side_effect=_mock_llm_response)
def test_call_llm_returns_string(mock_post) -> None:
    result = call_llm("some prompt")
    assert isinstance(result, str)
    assert len(result) > 0


@patch("app.agents.rca_agent.httpx.post", side_effect=_mock_llm_response)
def test_call_llm_strips_whitespace(mock_post) -> None:
    result = call_llm("some prompt")
    assert result == result.strip()


@patch("app.agents.rca_agent.httpx.post", side_effect=_mock_llm_response)
def test_call_llm_sends_correct_model(mock_post) -> None:
    from config import settings
    call_llm("prompt")
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["model"] == settings.ollama_model


# ---------------------------------------------------------------------------
# rca_agent node — mocked LLM
# ---------------------------------------------------------------------------

def _full_state(scenario: Scenario, n_logs: int = 50) -> dict:
    raw = generate_logs(scenario, n_logs=n_logs)
    s = {"raw_logs": raw, "events": [], "clusters": [], "root_cause": "", "report": None}
    s = parser_agent(s)
    s = pattern_agent(s)
    return s


@patch("app.agents.rca_agent.call_llm", return_value=_MOCK_ROOT_CAUSE)
def test_rca_agent_sets_root_cause(mock_llm) -> None:
    state = _full_state(Scenario.OOM_CRASH)
    result = rca_agent(state)
    assert result["root_cause"] == _MOCK_ROOT_CAUSE


@patch("app.agents.rca_agent.call_llm", return_value=_MOCK_ROOT_CAUSE)
@pytest.mark.parametrize("scenario", list(Scenario))
def test_rca_agent_runs_on_all_scenarios(mock_llm, scenario: Scenario) -> None:
    state = _full_state(scenario)
    result = rca_agent(state)
    assert isinstance(result["root_cause"], str)
    assert len(result["root_cause"]) > 0


@patch("app.agents.rca_agent.call_llm", return_value=_MOCK_ROOT_CAUSE)
def test_rca_agent_preserves_events_and_clusters(mock_llm) -> None:
    state = _full_state(Scenario.DB_TIMEOUT)
    result = rca_agent(state)
    assert result["events"] == state["events"]
    assert result["clusters"] == state["clusters"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@patch("app.agents.rca_agent.call_llm", return_value="")
def test_rca_agent_empty_llm_response(mock_llm) -> None:
    """An empty LLM response should still populate root_cause (as empty string)."""
    state = _full_state(Scenario.OOM_CRASH)
    result = rca_agent(state)
    assert result["root_cause"] == ""


def test_build_rag_context_single_event_single_cluster() -> None:
    events = [make_event("OutOfMemoryError heap")]
    clusters = [make_cluster(0, "OutOfMemoryError")]
    context = build_rag_context(clusters, events)
    assert isinstance(context, str)
    assert "OutOfMemoryError" in context


def test_build_prompt_empty_rag_context() -> None:
    clusters = [make_cluster(0, "error")]
    prompt = build_prompt(clusters, "")
    assert "Root cause:" in prompt


def test_rca_agent_no_clusters_skips_llm() -> None:
    state = {
        "raw_logs": [],
        "events": [make_event("some info", level=LogLevel.INFO)],
        "clusters": [],
        "root_cause": "",
        "report": None,
    }
    result = rca_agent(state)
    assert "No error clusters" in result["root_cause"]
