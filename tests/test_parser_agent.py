from datetime import datetime

import pytest

from app.agents.parser_agent import parse_line, parser_agent
from app.models.schemas import LogEvent, LogLevel, Scenario
from app.generator.factory import generate_logs


# ---------------------------------------------------------------------------
# parse_line — unit tests
# ---------------------------------------------------------------------------

VALID_LINE = "2026-03-21 03:03:09.000 INFO     spark-executor-4 Processing partition 0 — heap used 512 MB"


def test_parse_line_returns_log_event() -> None:
    event = parse_line(VALID_LINE)
    assert isinstance(event, LogEvent)


def test_parse_line_timestamp() -> None:
    event = parse_line(VALID_LINE)
    assert event is not None
    assert event.timestamp == datetime(2026, 3, 21, 3, 3, 9)


def test_parse_line_level() -> None:
    event = parse_line(VALID_LINE)
    assert event is not None
    assert event.level == LogLevel.INFO


def test_parse_line_service() -> None:
    event = parse_line(VALID_LINE)
    assert event is not None
    assert event.service == "spark-executor-4"


def test_parse_line_message() -> None:
    event = parse_line(VALID_LINE)
    assert event is not None
    assert "Processing partition 0" in event.message


def test_parse_line_raw_preserved() -> None:
    event = parse_line(VALID_LINE)
    assert event is not None
    assert event.raw == VALID_LINE


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
def test_parse_line_all_levels(level: str) -> None:
    line = f"2026-01-01 00:00:00.000 {level:<8} my-service some message here"
    event = parse_line(line)
    assert event is not None
    assert event.level == LogLevel(level)


def test_parse_line_returns_none_on_garbage() -> None:
    assert parse_line("not a log line at all") is None


def test_parse_line_returns_none_on_empty() -> None:
    assert parse_line("") is None


def test_parse_line_returns_none_on_unknown_level() -> None:
    line = "2026-01-01 00:00:00.000 VERBOSE  svc msg"
    assert parse_line(line) is None


# ---------------------------------------------------------------------------
# parser_agent — integration with generator output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario", list(Scenario))
def test_parser_agent_parses_all_generated_logs(scenario: Scenario) -> None:
    raw_logs = generate_logs(scenario, n_logs=30)
    state = {"raw_logs": raw_logs, "events": [], "clusters": [], "report": None}

    result = parser_agent(state)

    assert len(result["events"]) == 30
    assert all(isinstance(e, LogEvent) for e in result["events"])


def test_parser_agent_preserves_other_state_keys() -> None:
    state = {"raw_logs": [VALID_LINE], "events": [], "clusters": [], "report": None}
    result = parser_agent(state)
    assert result["clusters"] == []
    assert result["report"] is None


def test_parser_agent_skips_unparseable_lines() -> None:
    raw_logs = [VALID_LINE, "garbage line", VALID_LINE]
    state = {"raw_logs": raw_logs, "events": [], "clusters": [], "report": None}
    result = parser_agent(state)
    assert len(result["events"]) == 2


def test_parser_agent_empty_input() -> None:
    state = {"raw_logs": [], "events": [], "clusters": [], "report": None}
    result = parser_agent(state)
    assert result["events"] == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_parse_line_whitespace_only() -> None:
    assert parse_line("   ") is None


def test_parse_line_newline_only() -> None:
    assert parse_line("\n") is None


def test_parse_line_message_with_special_chars() -> None:
    line = "2026-01-01 00:00:00.000 ERROR  svc OutOfMemoryError: Java heap space — executor #3 killed"
    event = parse_line(line)
    assert event is not None
    assert "OutOfMemoryError" in event.message


def test_parse_line_service_with_hyphen_and_digits() -> None:
    line = "2026-01-01 00:00:00.000 WARNING spark-executor-42 something happened"
    event = parse_line(line)
    assert event is not None
    assert event.service == "spark-executor-42"


def test_parser_agent_all_garbage_produces_empty_events() -> None:
    raw_logs = ["garbage", "   ", "", "also not a log"]
    state = {"raw_logs": raw_logs, "events": [], "clusters": [], "report": None}
    result = parser_agent(state)
    assert result["events"] == []
