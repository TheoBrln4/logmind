from __future__ import annotations

import re
from datetime import datetime

import structlog

from app.agents.state import AnalysisState
from app.models.schemas import LogEvent, LogLevel

logger = structlog.get_logger()

# 2026-03-21 03:03:09.000 INFO     spark-executor-4 Processing partition 0 ...
_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})"
    r"\s+(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)"
    r"\s+(?P<service>\S+)"
    r"\s+(?P<message>.+)$"
)

_LEVEL_MAP: dict[str, LogLevel] = {lvl.value: lvl for lvl in LogLevel}


def parse_line(raw: str) -> LogEvent | None:
    """Parse a single raw log line into a LogEvent. Returns None if unparseable."""
    m = _LINE_RE.match(raw.strip())
    if not m:
        logger.warning("parser.skip", raw=raw[:120])
        return None

    return LogEvent(
        timestamp=datetime.strptime(m["ts"], "%Y-%m-%d %H:%M:%S.%f"),
        level=_LEVEL_MAP[m["level"]],
        service=m["service"],
        message=m["message"].strip(),
        raw=raw,
    )


def parser_agent(state: AnalysisState) -> AnalysisState:
    """Normalise raw log strings into typed LogEvent objects."""
    raw_logs = state["raw_logs"]
    events: list[LogEvent] = []
    skipped = 0

    for line in raw_logs:
        event = parse_line(line)
        if event is not None:
            events.append(event)
        else:
            skipped += 1

    logger.info("parser.done", total=len(raw_logs), parsed=len(events), skipped=skipped)
    return {**state, "events": events}
