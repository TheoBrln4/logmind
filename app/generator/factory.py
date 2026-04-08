from __future__ import annotations

import random
from datetime import datetime, timedelta

from faker import Faker

from app.models.schemas import Scenario

fake = Faker()

_LOG_FMT = "{ts} {level:<8} {service} {message}"


def _ts(base: datetime, offset_s: float) -> str:
    return (base + timedelta(seconds=offset_s)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def generate_logs(scenario: Scenario, n_logs: int = 50) -> list[str]:
    """Dispatch to the appropriate scenario generator."""
    generators = {
        Scenario.OOM_CRASH: _oom_crash,
        Scenario.DB_TIMEOUT: _db_timeout,
        Scenario.SILENT_FAIL: _silent_fail,
    }
    return generators[scenario](n_logs)


# ---------------------------------------------------------------------------
# Scenario 1 — OOM Crash
# Spark pipeline: memory rises progressively, ends with OutOfMemoryError
# ---------------------------------------------------------------------------

def _oom_crash(n_logs: int) -> list[str]:
    base = fake.date_time_this_month()
    logs: list[str] = []

    # Allocate phases proportionally; crash gets the remainder (always ≥ 1)
    normal_count = max(0, int(n_logs * 0.6))
    warn_count = max(0, int(n_logs * 0.25))
    crash_count = n_logs - normal_count - warn_count

    executors = [f"spark-executor-{i}" for i in range(1, 5)]
    t = 0.0

    # Phase 1 — normal operation, memory climbing
    base_mem = 512
    for i in range(normal_count):
        mem = base_mem + int((i / normal_count) * 1024)
        svc = random.choice(executors)
        logs.append(_LOG_FMT.format(
            ts=_ts(base, t),
            level="INFO",
            service=svc,
            message=f"Processing partition {i} — heap used {mem} MB / 2048 MB",
        ))
        t += random.uniform(0.5, 2.0)

    # Phase 2 — warnings
    for j in range(warn_count):
        mem = 1800 + j * 20
        svc = random.choice(executors)
        logs.append(_LOG_FMT.format(
            ts=_ts(base, t),
            level="WARNING",
            service=svc,
            message=f"High memory pressure detected — heap used {mem} MB / 2048 MB, GC overhead rising",
        ))
        t += random.uniform(0.2, 1.0)

    # Phase 3 — crash
    for k in range(crash_count):
        svc = random.choice(executors)
        if k == crash_count - 1:
            level, msg = "CRITICAL", "OutOfMemoryError: Java heap space — executor killed"
        else:
            level, msg = "ERROR", f"GC overhead limit exceeded on partition {normal_count + warn_count + k}"
        logs.append(_LOG_FMT.format(ts=_ts(base, t), level=level, service=svc, message=msg))
        t += random.uniform(0.1, 0.5)

    return logs


# ---------------------------------------------------------------------------
# Scenario 2 — DB Timeout
# Repeated SQL timeouts with exponential-backoff retries
# ---------------------------------------------------------------------------

def _db_timeout(n_logs: int) -> list[str]:
    base = fake.date_time_this_month()
    logs: list[str] = []
    service = "etl-worker"
    queries = [
        "SELECT * FROM orders WHERE created_at > ?",
        "INSERT INTO staging_table SELECT * FROM raw_events",
        "UPDATE metrics SET value = ? WHERE date = ?",
    ]
    t = 0.0
    attempt = 0

    for i in range(n_logs):
        query = random.choice(queries)
        timeout_ms = random.randint(5000, 30000)

        if attempt == 0:
            logs.append(_LOG_FMT.format(
                ts=_ts(base, t),
                level="INFO",
                service=service,
                message=f"Executing query: {query}",
            ))
            t += random.uniform(0.1, 0.3)
        elif attempt <= 3:
            backoff = 2 ** attempt
            logs.append(_LOG_FMT.format(
                ts=_ts(base, t),
                level="ERROR",
                service=service,
                message=(
                    f"SQLTimeout: query timed out after {timeout_ms} ms — "
                    f"retry {attempt}/3 in {backoff}s [{query}]"
                ),
            ))
            t += backoff + random.uniform(0, 0.5)
        else:
            logs.append(_LOG_FMT.format(
                ts=_ts(base, t),
                level="CRITICAL",
                service=service,
                message=f"Max retries exceeded for query [{query}] — task aborted",
            ))
            t += random.uniform(1.0, 3.0)
            attempt = -1  # reset for next query group

        attempt += 1

    return logs


# ---------------------------------------------------------------------------
# Scenario 3 — Silent Fail
# Tasks complete with SUCCESS status but produce 0 rows / empty output
# ---------------------------------------------------------------------------

def _silent_fail(n_logs: int) -> list[str]:
    base = fake.date_time_this_month()
    logs: list[str] = []
    services = ["data-loader", "transformer", "validator", "sink-writer"]
    tables = ["fact_sales", "dim_customer", "staging_events", "agg_daily"]
    t = 0.0

    for i in range(n_logs):
        svc = random.choice(services)
        table = random.choice(tables)

        kind = i % 4
        if kind == 0:
            msg = f"Task started — reading from source table '{table}'"
            level = "INFO"
        elif kind == 1:
            msg = f"Task completed — status=SUCCESS duration={random.randint(100, 900)}ms rows_written=0"
            level = "INFO"
        elif kind == 2:
            msg = f"Output validation passed — schema OK, row_count=0 (expected >0)"
            level = "WARNING"
        else:
            msg = f"Downstream task skipped — no input rows from '{table}'"
            level = "WARNING"

        logs.append(_LOG_FMT.format(ts=_ts(base, t), level=level, service=svc, message=msg))
        t += random.uniform(0.3, 1.5)

    return logs
