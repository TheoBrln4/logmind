from typing import Any, TypedDict

from app.models.schemas import Cluster, LogEvent, RCAReport


class AnalysisState(TypedDict):
    raw_logs: list[str]
    events: list[LogEvent]
    clusters: list[Cluster]
    root_cause: str          # filled by rca_agent, consumed by report_agent
    report: RCAReport | None
    chroma_collection: Any   # chromadb.Collection populated by embed_agent
