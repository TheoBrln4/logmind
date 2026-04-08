from fastapi import APIRouter, HTTPException

from app.agents.graph import build_graph
from app.models.schemas import AnalyzeRequest, AnalyzeResponse

router = APIRouter()

# Compile once at import time — avoids rebuilding the graph on every request
_graph = build_graph()


@router.post("/", response_model=AnalyzeResponse)
async def analyze_logs(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run the multi-agent RCA pipeline on the provided logs."""
    if not request.logs:
        raise HTTPException(status_code=422, detail="logs list must not be empty")

    initial_state = {
        "raw_logs": request.logs,
        "events": [],
        "clusters": [],
        "root_cause": "",
        "report": None,
    }

    final_state = await _graph.ainvoke(initial_state)

    if final_state["report"] is None:
        raise HTTPException(status_code=500, detail="Pipeline produced no report")

    return AnalyzeResponse(
        events=final_state["events"],
        clusters=final_state["clusters"],
        report=final_state["report"],
    )
