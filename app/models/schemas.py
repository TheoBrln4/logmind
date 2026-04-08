from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class Scenario(StrEnum):
    OOM_CRASH = "oom_crash"
    DB_TIMEOUT = "db_timeout"
    SILENT_FAIL = "silent_fail"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# --- Log model ---

class LogEvent(BaseModel):
    timestamp: datetime
    level: LogLevel
    service: str
    message: str
    raw: str


# --- Generate endpoint ---

class GenerateRequest(BaseModel):
    scenario: Scenario
    n_logs: int = 50


class GenerateResponse(BaseModel):
    scenario: Scenario
    logs: list[str]


# --- Analyze endpoint ---

class AnalyzeRequest(BaseModel):
    logs: list[str]


class Cluster(BaseModel):
    cluster_id: int
    size: int
    representative: str


class RCAReport(BaseModel):
    root_cause: str
    hypotheses: list[str]
    impact: str
    recommended_actions: list[str]
    metadata: dict[str, Any] = {}


class AnalyzeResponse(BaseModel):
    events: list[LogEvent]
    clusters: list[Cluster]
    report: RCAReport
