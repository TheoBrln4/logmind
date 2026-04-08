from __future__ import annotations

import json
import structlog
import httpx

from app.agents.state import AnalysisState
from app.models.schemas import Cluster, RCAReport
from config import settings

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_report_prompt(root_cause: str, clusters: list[Cluster]) -> str:
    cluster_summary = "\n".join(
        f"  - Cluster {c.cluster_id} ({c.size} events): {c.representative}"
        for c in clusters
    )
    return (
        "You are an expert SRE writing a post-incident report.\n\n"
        "## Root cause identified\n"
        f"{root_cause}\n\n"
        "## Error clusters\n"
        f"{cluster_summary}\n\n"
        "Write a structured JSON report with exactly these keys:\n"
        '  "hypotheses"        : list of 2-3 strings, plausible contributing factors\n'
        '  "impact"            : one string describing the business/system impact\n'
        '  "recommended_actions": list of 3-5 concrete remediation steps\n\n'
        "Return only valid JSON, no prose before or after.\n"
        "JSON:"
    )


# ---------------------------------------------------------------------------
# LLM call + JSON parsing
# ---------------------------------------------------------------------------

def call_llm_report(prompt: str) -> dict:
    """Call Ollama and parse the JSON response. Returns {} on parse failure."""
    response = httpx.post(
        f"{settings.ollama_base_url}/api/generate",
        json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
        timeout=300.0,
    )
    response.raise_for_status()
    raw = response.json()["response"].strip()

    # Strip markdown code fences if the model wraps the JSON
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        parsed = json.loads(raw)
        # Si le LLM renvoie une liste, prendre le premier élément
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed[0]
        return parsed
    except json.JSONDecodeError:
        logger.warning("report.json_parse_error", raw_preview=raw[:200])
        return {}


def _safe_list(value: object, fallback: list[str]) -> list[str]:
    """Return value if it's a non-empty list of strings, else fallback."""
    if isinstance(value, list) and value and all(isinstance(i, str) for i in value):
        return value
    return fallback


def parse_llm_output(data: dict, root_cause: str) -> RCAReport:
    """Build an RCAReport from the LLM JSON dict, with safe fallbacks."""
    hypotheses = (
        data.get("hypotheses")
        or data.get("hypoetheses")
        or data.get("hypothesis")
    )
    actions = (
        data.get("recommended_actions")
        or data.get("remedied_actions")
        or data.get("actions")
    )
    return RCAReport(
        root_cause=root_cause,
        hypotheses=_safe_list(
            hypotheses,
            ["Unable to determine hypotheses from LLM output."],
        ),
        impact=data.get("impact") if isinstance(data.get("impact"), str) and data.get("impact") else "Impact could not be determined.",
        recommended_actions=_safe_list(
            actions,
            ["Investigate the root cause manually."],
        ),
        metadata={"raw_llm_output": data},
    )


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def report_agent(state: AnalysisState) -> AnalysisState:
    """Write the structured RCA report (hypotheses, impact, actions)."""
    root_cause = state.get("root_cause", "")
    clusters = state["clusters"]

    if not root_cause:
        logger.warning("report.no_root_cause")
        report = RCAReport(
            root_cause="No root cause available.",
            hypotheses=["Root cause analysis was not performed."],
            impact="Unknown.",
            recommended_actions=["Run the full analysis pipeline."],
        )
        return {**state, "report": report}

    prompt = build_report_prompt(root_cause, clusters)
    logger.info("report.calling_llm", model=settings.ollama_model)

    llm_data = call_llm_report(prompt)
    report = parse_llm_output(llm_data, root_cause)

    logger.info("report.done", n_actions=len(report.recommended_actions))
    return {**state, "report": report}
