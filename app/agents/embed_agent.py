from __future__ import annotations

from datetime import datetime

import chromadb
import httpx
import structlog

from app.agents.state import AnalysisState
from app.models.schemas import LogEvent
from config import settings

logger = structlog.get_logger()

_EMBED_MODEL = "nomic-embed-text"
_COLLECTION_NAME = "log_events"
_HISTORY_COLLECTION = "cluster_history"
_CHROMA_PATH = "/chroma_data"


def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = httpx.post(
        f"{settings.ollama_base_url}/api/embed",
        json={"model": _EMBED_MODEL, "input": texts},
        timeout=130.0,
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=_CHROMA_PATH)


def embed_agent(state: AnalysisState) -> AnalysisState:
    events: list[LogEvent] = state["events"]

    client = get_chroma_client()
    collection = client.get_or_create_collection(_COLLECTION_NAME)

    if not events:
        logger.warning("embed.no_events")
        return {**state, "chroma_collection": collection}

    req_prefix = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    documents = [e.message for e in events]
    embeddings = get_embeddings(documents)
    ids = [f"{req_prefix}_{i}" for i in range(len(events))]
    metadatas = [
        {
            "level": e.level.value,
            "service": e.service,
            "timestamp": e.timestamp.isoformat(),
        }
        for e in events
    ]

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    logger.info("embed.done", n_events=len(events))
    return {**state, "chroma_collection": collection}
