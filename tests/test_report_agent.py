from unittest.mock import MagicMock, patch
import json

import pytest

from app.agents.report_agent import (
    build_report_prompt,
    call_llm_report,
    parse_llm_output,
    report_agent,
)
from app.models.schemas import Cluster, RCAReport, Scenario
from app.generator.factory import generate_logs
from app.agents.parser_agent import parser_agent
from app.agents.pattern_agent import pattern_agent
from app.agents.rca_agent import rca_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT_CAUSE = "The Spark executor ran out of heap memory due to unbounded partition size."

VALID_LLM_JSON = {
    "hypotheses": [
        "No memory limit configured on Spark executor JVM",
        "Input dataset grew significantly since last run",
    ],
    "impact": "Daily ETL pipeline failed; downstream dashboards received no data for 6 hours.",
    "recommended_actions": [
        "Set spark.executor.memory to an appropriate value",
        "Add partition size checks before the Spark job",
        "Configure alerting on heap usage > 80%",
    ],
}

VALID_LLM_JSON_STR = json.dumps(VALID_LLM_JSON)


def make_cluster(cluster_id: int = 0, representative: str = "OOM error", size: int = 4) -> Cluster:
    return Cluster(cluster_id=cluster_id, size=size, representative=representative)


def _mock_llm(json_body: dict):
    def _side_effect(*args, **kwargs):
        mock = MagicMock()
        mock.raise_for_status.return_value = None
        mock.json.return_value = {"response": json.dumps(json_body)}
        return mock
    return _side_effect


def _mock_llm_with_fences(json_body: dict):
    """Simulate a model that wraps JSON in markdown code fences."""
    def _side_effect(*args, **kwargs):
        mock = MagicMock()
        mock.raise_for_status.return_value = None
        mock.json.return_value = {"response": f"```json\n{json.dumps(json_body)}\n```"}
        return mock
    return _side_effect


# ---------------------------------------------------------------------------
# build_report_prompt — unit tests
# ---------------------------------------------------------------------------

def test_prompt_contains_root_cause() -> None:
    prompt = build_report_prompt(ROOT_CAUSE, [make_cluster()])
    assert ROOT_CAUSE in prompt


def test_prompt_contains_cluster_info() -> None:
    cluster = make_cluster(0, "OutOfMemoryError heap space", size=7)
    prompt = build_report_prompt(ROOT_CAUSE, [cluster])
    assert "Cluster 0" in prompt
    assert "OutOfMemoryError heap space" in prompt
    assert "7 events" in prompt


def test_prompt_requests_json_keys() -> None:
    prompt = build_report_prompt(ROOT_CAUSE, [make_cluster()])
    assert "hypotheses" in prompt
    assert "impact" in prompt
    assert "recommended_actions" in prompt


def test_prompt_ends_with_json_cue() -> None:
    prompt = build_report_prompt(ROOT_CAUSE, [make_cluster()])
    assert prompt.strip().endswith("JSON:")


# ---------------------------------------------------------------------------
# call_llm_report — mocked HTTP
# ---------------------------------------------------------------------------

@patch("app.agents.report_agent.httpx.post", side_effect=_mock_llm(VALID_LLM_JSON))
def test_call_llm_report_returns_dict(mock_post) -> None:
    result = call_llm_report("some prompt")
    assert isinstance(result, dict)


@patch("app.agents.report_agent.httpx.post", side_effect=_mock_llm(VALID_LLM_JSON))
def test_call_llm_report_has_expected_keys(mock_post) -> None:
    result = call_llm_report("some prompt")
    assert "hypotheses" in result
    assert "impact" in result
    assert "recommended_actions" in result


@patch("app.agents.report_agent.httpx.post", side_effect=_mock_llm_with_fences(VALID_LLM_JSON))
def test_call_llm_report_strips_markdown_fences(mock_post) -> None:
    result = call_llm_report("some prompt")
    assert isinstance(result, dict)
    assert "hypotheses" in result


# ---------------------------------------------------------------------------
# parse_llm_output — unit tests (no HTTP)
# ---------------------------------------------------------------------------

def test_parse_llm_output_returns_rca_report() -> None:
    report = parse_llm_output(VALID_LLM_JSON, ROOT_CAUSE)
    assert isinstance(report, RCAReport)


def test_parse_llm_output_root_cause_preserved() -> None:
    report = parse_llm_output(VALID_LLM_JSON, ROOT_CAUSE)
    assert report.root_cause == ROOT_CAUSE


def test_parse_llm_output_hypotheses() -> None:
    report = parse_llm_output(VALID_LLM_JSON, ROOT_CAUSE)
    assert len(report.hypotheses) >= 1
    assert all(isinstance(h, str) for h in report.hypotheses)


def test_parse_llm_output_impact() -> None:
    report = parse_llm_output(VALID_LLM_JSON, ROOT_CAUSE)
    assert isinstance(report.impact, str)
    assert len(report.impact) > 0


def test_parse_llm_output_recommended_actions() -> None:
    report = parse_llm_output(VALID_LLM_JSON, ROOT_CAUSE)
    assert len(report.recommended_actions) >= 1


def test_parse_llm_output_fallback_on_missing_keys() -> None:
    report = parse_llm_output({}, ROOT_CAUSE)
    assert report.hypotheses  # fallback list
    assert report.impact
    assert report.recommended_actions


def test_parse_llm_output_fallback_on_bad_types() -> None:
    bad_data = {"hypotheses": "not a list", "impact": 42, "recommended_actions": None}
    report = parse_llm_output(bad_data, ROOT_CAUSE)
    assert isinstance(report.hypotheses, list)
    assert isinstance(report.impact, str)
    assert isinstance(report.recommended_actions, list)


def test_parse_llm_output_stores_raw_in_metadata() -> None:
    report = parse_llm_output(VALID_LLM_JSON, ROOT_CAUSE)
    assert "raw_llm_output" in report.metadata


# ---------------------------------------------------------------------------
# report_agent node — mocked LLM
# ---------------------------------------------------------------------------

def _pipeline_state(scenario: Scenario, n_logs: int = 50) -> dict:
    """Run the full pipeline up to rca_agent with a mocked LLM."""
    raw = generate_logs(scenario, n_logs=n_logs)
    state = {"raw_logs": raw, "events": [], "clusters": [], "root_cause": "", "report": None}
    state = parser_agent(state)
    state = pattern_agent(state)

    with patch("app.agents.rca_agent.call_llm", return_value=ROOT_CAUSE):
        state = rca_agent(state)

    return state


@patch("app.agents.report_agent.call_llm_report", return_value=VALID_LLM_JSON)
def test_report_agent_sets_report(mock_llm) -> None:
    state = _pipeline_state(Scenario.OOM_CRASH)
    result = report_agent(state)
    assert isinstance(result["report"], RCAReport)


@patch("app.agents.report_agent.call_llm_report", return_value=VALID_LLM_JSON)
@pytest.mark.parametrize("scenario", list(Scenario))
def test_report_agent_runs_on_all_scenarios(mock_llm, scenario: Scenario) -> None:
    state = _pipeline_state(scenario)
    result = report_agent(state)
    assert result["report"] is not None
    assert result["report"].root_cause


@patch("app.agents.report_agent.call_llm_report", return_value=VALID_LLM_JSON)
def test_report_agent_preserves_events_clusters(mock_llm) -> None:
    state = _pipeline_state(Scenario.DB_TIMEOUT)
    result = report_agent(state)
    assert result["events"] == state["events"]
    assert result["clusters"] == state["clusters"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_parse_llm_output_empty_lists_use_fallback() -> None:
    """_safe_list treats empty lists as invalid and returns the fallback."""
    data = {"hypotheses": [], "impact": "some impact", "recommended_actions": []}
    report = parse_llm_output(data, ROOT_CAUSE)
    assert report.hypotheses  # fallback, not empty
    assert report.recommended_actions  # fallback, not empty


def test_call_llm_report_invalid_json_returns_empty_dict() -> None:
    """LLM returning non-JSON prose should not raise — returns {}."""
    def _bad_json(*args, **kwargs):
        m = MagicMock()
        m.raise_for_status.return_value = None
        m.json.return_value = {"response": "Sorry, I cannot help with that."}
        return m

    with patch("app.agents.report_agent.httpx.post", side_effect=_bad_json):
        result = call_llm_report("some prompt")
    assert result == {}


@patch("app.agents.report_agent.call_llm_report", return_value={})
def test_report_agent_json_parse_failure_uses_fallbacks(mock_llm) -> None:
    """If LLM returns {} (parse failure), all fields should use fallback values."""
    state = {
        "raw_logs": [],
        "events": [],
        "clusters": [],
        "root_cause": ROOT_CAUSE,
        "report": None,
    }
    result = report_agent(state)
    report = result["report"]
    assert report.root_cause == ROOT_CAUSE
    assert report.hypotheses
    assert report.impact
    assert report.recommended_actions


@patch("app.agents.report_agent.httpx.post")
def test_call_llm_report_raises_on_http_error(mock_post) -> None:
    """HTTP errors from Ollama should propagate."""
    import httpx as _httpx
    mock_post.side_effect = _httpx.HTTPStatusError(
        "500", request=MagicMock(), response=MagicMock()
    )
    with pytest.raises(_httpx.HTTPStatusError):
        call_llm_report("some prompt")


def test_report_agent_no_root_cause_returns_fallback() -> None:
    state = {
        "raw_logs": [],
        "events": [],
        "clusters": [],
        "root_cause": "",
        "report": None,
    }
    result = report_agent(state)
    assert isinstance(result["report"], RCAReport)
    assert "No root cause" in result["report"].root_cause
