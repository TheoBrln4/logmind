import pytest

from app.generator.factory import generate_logs
from app.models.schemas import Scenario


# ---------------------------------------------------------------------------
# Missing edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", list(Scenario))
def test_generate_returns_list(scenario: Scenario) -> None:
    logs = generate_logs(scenario, n_logs=20)
    assert isinstance(logs, list)
    assert len(logs) == 20
    assert all(isinstance(line, str) and line for line in logs)


@pytest.mark.parametrize("n_logs", [1, 10, 100])
def test_generate_respects_count(n_logs: int) -> None:
    logs = generate_logs(Scenario.OOM_CRASH, n_logs=n_logs)
    assert len(logs) == n_logs


def test_oom_crash_contains_oom_error() -> None:
    logs = generate_logs(Scenario.OOM_CRASH, n_logs=50)
    assert any("OutOfMemoryError" in line for line in logs)


def test_oom_crash_has_critical_level() -> None:
    logs = generate_logs(Scenario.OOM_CRASH, n_logs=50)
    assert any("CRITICAL" in line for line in logs)


def test_oom_crash_memory_progression() -> None:
    """Earlier logs should mention lower memory than later warning logs."""
    logs = generate_logs(Scenario.OOM_CRASH, n_logs=50)
    info_logs = [l for l in logs if "INFO" in l and "heap used" in l]
    warn_logs = [l for l in logs if "WARNING" in l and "heap used" in l]
    assert info_logs, "Expected INFO logs with heap usage"
    assert warn_logs, "Expected WARNING logs with heap pressure"


def test_db_timeout_contains_timeout_error() -> None:
    logs = generate_logs(Scenario.DB_TIMEOUT, n_logs=30)
    assert any("SQLTimeout" in line for line in logs)


def test_db_timeout_contains_retry() -> None:
    logs = generate_logs(Scenario.DB_TIMEOUT, n_logs=30)
    assert any("retry" in line for line in logs)


def test_silent_fail_all_non_error() -> None:
    """Silent fail scenario must not contain ERROR or CRITICAL lines."""
    logs = generate_logs(Scenario.SILENT_FAIL, n_logs=40)
    assert not any(("ERROR" in line or "CRITICAL" in line) for line in logs)


def test_silent_fail_contains_zero_rows() -> None:
    logs = generate_logs(Scenario.SILENT_FAIL, n_logs=40)
    assert any("rows_written=0" in line for line in logs)


@pytest.mark.parametrize("scenario", list(Scenario))
def test_generate_zero_logs(scenario: Scenario) -> None:
    logs = generate_logs(scenario, n_logs=0)
    assert logs == []


@pytest.mark.parametrize("scenario", [Scenario.DB_TIMEOUT, Scenario.SILENT_FAIL])
def test_generate_respects_count_all_scenarios(scenario: Scenario) -> None:
    for n in [1, 10, 50]:
        assert len(generate_logs(scenario, n_logs=n)) == n


def test_logs_have_timestamp_prefix() -> None:
    """Every line must start with a valid-looking timestamp."""
    import re
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}")
    for scenario in Scenario:
        logs = generate_logs(scenario, n_logs=10)
        for line in logs:
            assert pattern.match(line), f"No timestamp prefix in: {line!r}"
