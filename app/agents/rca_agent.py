from __future__ import annotations

import httpx
import structlog

from app.agents.embed_agent import get_chroma_client, get_embeddings, _HISTORY_COLLECTION
from app.agents.state import AnalysisState
from app.models.schemas import Cluster, LogEvent, LogLevel
from config import settings

logger = structlog.get_logger()

_TOP_K = 6
_HISTORY_DISTANCE_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Context building from clusters — semantic search via ChromaDB
# ---------------------------------------------------------------------------

_ERROR_LEVELS = {LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL}
_ERROR_LEVEL_VALUES = [lvl.value for lvl in _ERROR_LEVELS]


def build_rag_context(clusters: list[Cluster], events: list[LogEvent], chroma_collection=None) -> str:
    if not clusters or not events:
        return ""

    queries = [c.representative for c in clusters[:3]]

    # --- Current request context ---
    current: list[str] = []
    if chroma_collection is None:
        seen: set[str] = set()
        for event in events:
            if event.level in _ERROR_LEVELS and event.message not in seen:
                current.append(f"· [courant] {event.message}")
                seen.add(event.message)
        current = current[: _TOP_K * 2]
    else:
        seen: set[str] = set()
        for query in queries:
            query_embedding = get_embeddings([query])[0]
            results = chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=_TOP_K,
                where={"level": {"$in": _ERROR_LEVEL_VALUES}},
            )
            for doc in results["documents"][0]:
                if doc not in seen:
                    current.append(f"· [courant] {doc}")
                    seen.add(doc)

    # --- Historical context from cluster_history ---
    historical: list[str] = []
    try:
        hist_client = get_chroma_client()
        hist_collection = hist_client.get_collection(_HISTORY_COLLECTION)
        seen_hist: set[str] = set()
        for query in queries:
            query_embedding = get_embeddings([query])[0]
            results = hist_collection.query(
                query_embeddings=[query_embedding],
                n_results=_TOP_K,
            )
            for doc, distance, meta in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0],
            ):
                if distance < _HISTORY_DISTANCE_THRESHOLD and doc not in seen_hist:
                    date = meta.get("created_at", "")[:10]
                    root_cause_type = meta.get("root_cause_type", "")
                    suffix = f" — type: {root_cause_type}" if root_cause_type else ""
                    historical.append(f"· [historique - {date}] {doc}{suffix}")
                    seen_hist.add(doc)
    except Exception:
        pass  # cluster_history not yet populated

    all_context = current[:_TOP_K] + historical[:_TOP_K]
    logger.info("rca.rag_context", n_current=len(current), n_historical=len(historical))
    return "\n".join(all_context)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(clusters: list[Cluster], rag_context: str) -> str:
    cluster_summary = "\n".join(
        f"  - Cluster {c.cluster_id} ({c.size} events): {c.representative}"
        for c in clusters
    )
    return (
        "You are an expert SRE analysing a data pipeline incident.\n\n"
        "## Error clusters detected\n"
        f"{cluster_summary}\n\n"
        "## Relevant log context (retrieved)\n"
        f"{rag_context}\n\n"
        "Identify the root cause of this incident in 2-3 sentences. "
        "Be specific and concise.\n"
        "Root cause:"
    )


# ---------------------------------------------------------------------------
# LLM call — Ollama /api/generate
# ---------------------------------------------------------------------------

def call_llm(prompt: str) -> str:
    """POST to Ollama and return the generated text."""
    response = httpx.post(
        f"{settings.ollama_base_url}/api/generate",
        json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
        timeout=300.0,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def rca_agent(state: AnalysisState) -> AnalysisState:
    """Identify the root cause from clusters using LLM."""
    clusters = state["clusters"]

    if not clusters:
        logger.warning("rca.no_clusters")
        return {**state, "root_cause": "No error clusters detected."}

    rag_context = build_rag_context(clusters, state["events"], state.get("chroma_collection"))
    prompt = build_prompt(clusters, rag_context)

    logger.info("rca.calling_llm", model=settings.ollama_model, n_clusters=len(clusters))
    root_cause = call_llm(prompt)
    logger.info("rca.done", root_cause_preview=root_cause[:120])

    return {**state, "root_cause": root_cause}
