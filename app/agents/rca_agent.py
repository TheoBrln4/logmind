from __future__ import annotations

import structlog
import httpx

from app.agents.state import AnalysisState
from app.models.schemas import Cluster, LogEvent, LogLevel
from config import settings

logger = structlog.get_logger()

_TOP_K = 6


# ---------------------------------------------------------------------------
# Context building from clusters
# ---------------------------------------------------------------------------

_ERROR_LEVELS = {LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL}


def build_rag_context(clusters: list[Cluster], events: list[LogEvent]) -> str:
    if not clusters or not events:
        return ""
    
    context = []
    seen = set()
    for event in events:
        if event.level in _ERROR_LEVELS and event.message not in seen:
            context.append(event.message)
            seen.add(event.message)
    return "\n".join(context[:_TOP_K * 2])


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

    rag_context = build_rag_context(clusters, state["events"])
    prompt = build_prompt(clusters, rag_context)

    logger.info("rca.calling_llm", model=settings.ollama_model, n_clusters=len(clusters))
    root_cause = call_llm(prompt)
    logger.info("rca.done", root_cause_preview=root_cause[:120])

    return {**state, "root_cause": root_cause}
