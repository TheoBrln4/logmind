from __future__ import annotations

from datetime import timedelta

import structlog

from app.agents.state import AnalysisState
from app.models.schemas import LogEvent, LogLevel

logger = structlog.get_logger()

_WINDOW_SIZE = timedelta(seconds=10)
_SPIKE_THRESHOLD = 3
_ALERT_LEVELS = {LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL}


def _compute_peaks(alert_events: list[LogEvent]) -> list[dict]:
    n = len(alert_events)
    if n < _SPIKE_THRESHOLD:
        return []

    # Sliding window: for each left, find max right within _WINDOW_SIZE
    windows: list[tuple[int, int, int]] = []
    right = 0
    for left in range(n):
        if right < left:
            right = left
        while (
            right < n - 1
            and alert_events[right + 1].timestamp - alert_events[left].timestamp <= _WINDOW_SIZE
        ):
            right += 1
        count = right - left + 1
        if count >= _SPIKE_THRESHOLD:
            windows.append((left, right, count))

    if not windows:
        return []

    # Greedily select non-overlapping peaks, highest count first
    windows.sort(key=lambda w: -w[2])
    covered: set[int] = set()
    peaks: list[dict] = []
    for left, right, count in windows:
        if not covered.intersection(range(left, right + 1)):
            window_events = alert_events[left : right + 1]
            peaks.append({
                "start": window_events[0].timestamp,
                "end": window_events[-1].timestamp,
                "count": count,
                "services": sorted({e.service for e in window_events}),
            })
            covered.update(range(left, right + 1))

    peaks.sort(key=lambda p: p["start"])
    return peaks


def _build_summary(peaks: list[dict]) -> str:
    lines = []
    for peak in peaks:
        duration = int((peak["end"] - peak["start"]).total_seconds())
        duration_str = f"{duration}s" if duration > 0 else "<1s"
        services_str = ", ".join(peak["services"])
        start_str = peak["start"].strftime("%H:%M:%S")
        end_str = peak["end"].strftime("%H:%M:%S")
        lines.append(
            f"Pic détecté entre {start_str} et {end_str} : "
            f"{peak['count']} erreurs en {duration_str} sur {services_str}"
        )
    return "\n".join(lines)


def temporal_agent(state: AnalysisState) -> AnalysisState:
    """Detect error spikes using a 10-second sliding window."""
    alert_events = sorted(
        [e for e in state["events"] if e.level in _ALERT_LEVELS],
        key=lambda e: e.timestamp,
    )

    peaks = _compute_peaks(alert_events)
    summary = _build_summary(peaks)

    peak_window = (
        f"{peaks[0]['start'].strftime('%H:%M:%S')}-{peaks[0]['end'].strftime('%H:%M:%S')}"
        if peaks
        else "none"
    )
    logger.info("temporal.done", n_peaks=len(peaks), peak_window=peak_window)

    return {**state, "temporal_summary": summary}
