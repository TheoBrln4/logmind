from __future__ import annotations

from datetime import datetime

import structlog

from app.agents.embed_agent import _HISTORY_COLLECTION, get_chroma_client, get_embeddings
from app.agents.state import AnalysisState
from app.models.schemas import Cluster, LogEvent, LogLevel

logger = structlog.get_logger()

_ERROR_LEVELS = {LogLevel.ERROR, LogLevel.CRITICAL}
# cosine distance: distance = 1 - similarity; 0.85 similarity → 0.15 distance
_DEDUP_DISTANCE_THRESHOLD = 0.15


def is_worth_persisting(cluster: Cluster, events: list[LogEvent]) -> bool:
    if cluster.size < 3:
        return False
    error_messages = {e.message for e in events if e.level in _ERROR_LEVELS}
    return cluster.representative in error_messages


def _is_duplicate(embedding: list[float], collection) -> bool:
    if collection.count() == 0:
        return False
    results = collection.query(
        query_embeddings=[embedding],
        n_results=1,
    )
    min_distance = results["distances"][0][0]
    return min_distance <= _DEDUP_DISTANCE_THRESHOLD


def persist_agent(state: AnalysisState) -> AnalysisState:
    clusters: list[Cluster] = state["clusters"]
    events: list[LogEvent] = state["events"]
    root_cause: str = state.get("root_cause", "unknown")  # type: ignore[assignment]

    client = get_chroma_client()
    history = client.get_or_create_collection(
        _HISTORY_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    persisted = 0
    now = datetime.utcnow().isoformat()

    for cluster in clusters:
        if not is_worth_persisting(cluster, events):
            continue

        embedding = get_embeddings([cluster.representative])[0]

        if _is_duplicate(embedding, history):
            logger.debug("persist.skip_duplicate", representative=cluster.representative[:80])
            continue

        doc_id = f"realtime_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{cluster.cluster_id}"
        history.add(
            documents=[cluster.representative],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[{
                "scenario": "unknown",
                "root_cause_type": "realtime",
                "source": "realtime",
                "created_at": now,
                "cluster_size": cluster.size,
                "root_cause": root_cause[:200],
            }],
        )
        persisted += 1

    logger.info("persist.done", persisted=persisted, total=len(clusters))
    return state
