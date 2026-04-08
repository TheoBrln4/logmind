from fastapi import APIRouter, HTTPException

from app.generator.factory import generate_logs
from app.models.schemas import GenerateRequest, GenerateResponse

router = APIRouter()


@router.post("/", response_model=GenerateResponse)
async def generate_logs_endpoint(request: GenerateRequest) -> GenerateResponse:
    """Generate synthetic logs for a given scenario."""
    if request.n_logs < 0:
        raise HTTPException(status_code=422, detail="n_logs must be >= 0")

    logs = generate_logs(request.scenario, n_logs=request.n_logs)
    return GenerateResponse(scenario=request.scenario, logs=logs)
