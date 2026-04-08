from datetime import datetime

import pytest

from app.agents.pattern_agent import cluster_events, pattern_agent
from app.models.schemas import Cluster, LogEvent, LogLevel, Scenario
from app.generator.factory import generate_logs
from app.agents.parser_agent import parser_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_event(message: str, level: LogLevel = LogLevel.ERROR) -> LogEvent:
    return LogEvent(
        timestamp=datetime(2026, 1, 1, 0, 0, 0),
        level=level,
        service="test-svc",
        message=message,
        raw=f"2026-01-01 00:00:00.000 {level} test-svc {message}",
    )


# ---------------------------------------------------------------------------
# cluster_events — unit tests
# ---------------------------------------------------------------------------

def test_cluster_events_returns_list_of_clusters() -> None:
    events = [make_event("OutOfMemoryError: heap space") for _ in range(5)]
    clusters = cluster_events(events)
    assert isinstance(clusters, list)
    assert all(isinstance(c, Cluster) for c in clusters)


def test_cluster_events_empty_input() -> None:
    assert cluster_events([]) == []


def test_cluster_events_ignores_info_logs() -> None:
    events = [make_event("all good", level=LogLevel.INFO) for _ in range(10)]
    assert cluster_events(events) == []


def test_cluster_events_groups_similar_messages() -> None:
    """Identical error messages should land in a single cluster."""
    events = [make_event("SQLTimeout: query timed out after 5000 ms") for _ in range(6)]
    clusters = cluster_events(events)
    assert len(clusters) == 1
    assert clusters[0].size == 6


def test_cluster_events_separates_distinct_messages() -> None:
    """Semantically different errors should produce multiple clusters."""
    oom = [make_event("OutOfMemoryError Java heap space executor killed") for _ in range(4)]
    timeout = [make_event("SQLTimeout database connection timed out retry") for _ in range(4)]
    clusters = cluster_events(oom + timeout)
    assert len(clusters) >= 2


def test_cluster_ids_are_unique() -> None:
    events = (
        [make_event("OutOfMemoryError heap") for _ in range(3)]
        + [make_event("SQLTimeout retry") for _ in range(3)]
    )
    clusters = cluster_events(events)
    ids = [c.cluster_id for c in clusters]
    assert len(ids) == len(set(ids))


def test_cluster_representative_is_non_empty() -> None:
    events = [make_event("disk full error") for _ in range(3)]
    clusters = cluster_events(events)
    assert all(c.representative for c in clusters)


def test_cluster_total_size_equals_input() -> None:
    events = [make_event(f"error variant {i % 3}") for i in range(9)]
    clusters = cluster_events(events)
    assert sum(c.size for c in clusters) == 9


def test_cluster_warning_level_included() -> None:
    events = [make_event("high memory pressure GC overhead", level=LogLevel.WARNING) for _ in range(4)]
    clusters = cluster_events(events)
    assert len(clusters) >= 1


def test_cluster_singleton_noise_handled() -> None:
    """A fully unique set of messages should not raise, just return singletons."""
    events = [make_event(f"unique error message number {i}") for i in range(5)]
    clusters = cluster_events(events)
    assert len(clusters) >= 1
    assert sum(c.size for c in clusters) == 5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_cluster_events_single_event() -> None:
    """A single error event should produce exactly one singleton cluster."""
    events = [make_event("OutOfMemoryError heap space")]
    clusters = cluster_events(events)
    assert len(clusters) == 1
    assert clusters[0].size == 1


def test_cluster_events_debug_only_ignored() -> None:
    events = [make_event("debug trace info", level=LogLevel.DEBUG) for _ in range(5)]
    assert cluster_events(events) == []


def test_cluster_events_mixed_levels_only_clusters_relevant() -> None:
    """INFO and DEBUG events should not appear in clusters."""
    info_events = [make_event("all fine", level=LogLevel.INFO) for _ in range(5)]
    error_events = [make_event("disk full error") for _ in range(3)]
    clusters = cluster_events(info_events + error_events)
    total = sum(c.size for c in clusters)
    assert total == 3  # only the 3 ERROR events


def test_cluster_events_custom_eps_min_samples() -> None:
    """Looser eps should merge more aggressively."""
    events = [make_event(f"similar error variant {i}") for i in range(4)]
    clusters_tight = cluster_events(events, eps=0.1, min_samples=2)
    clusters_loose = cluster_events(events, eps=0.99, min_samples=2)
    assert len(clusters_loose) <= len(clusters_tight)


# ---------------------------------------------------------------------------
# pattern_agent — integration with generator + parser
# ---------------------------------------------------------------------------

def _build_state(scenario: Scenario, n_logs: int = 50) -> dict:
    raw_logs = generate_logs(scenario, n_logs=n_logs)
    state = {"raw_logs": raw_logs, "events": [], "clusters": [], "report": None}
    return parser_agent(state)


@pytest.mark.parametrize("scenario", list(Scenario))
def test_pattern_agent_runs_on_all_scenarios(scenario: Scenario) -> None:
    state = _build_state(scenario)
    result = pattern_agent(state)
    assert "clusters" in result
    assert isinstance(result["clusters"], list)
    assert all(isinstance(c, Cluster) for c in result["clusters"])


def test_pattern_agent_oom_produces_clusters() -> None:
    state = _build_state(Scenario.OOM_CRASH, n_logs=60)
    result = pattern_agent(state)
    assert len(result["clusters"]) >= 1


def test_pattern_agent_preserves_events() -> None:
    state = _build_state(Scenario.DB_TIMEOUT, n_logs=30)
    result = pattern_agent(state)
    assert result["events"] == state["events"]


def test_pattern_agent_preserves_report_key() -> None:
    state = _build_state(Scenario.SILENT_FAIL)
    result = pattern_agent(state)
    assert result["report"] is None
