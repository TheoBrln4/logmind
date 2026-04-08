import structlog
import structlog.stdlib
import logging
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.routes import analyze, generate

# ---------------------------------------------------------------------------
# Structlog configuration
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    logging.basicConfig(level=logging.INFO)


configure_logging()
logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Log Analyzer",
    version="0.1.0",
    description="Multi-agent RCA system for data pipeline logs.",
)

app.include_router(generate.router, prefix="/generate", tags=["generate"])
app.include_router(analyze.router, prefix="/analyze", tags=["analyze"])


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(httpx.ConnectError)
@app.exception_handler(httpx.ConnectTimeout)
async def ollama_unavailable_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("ollama.unreachable", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=503,
        content={"detail": "LLM service (Ollama) is unreachable. Is it running?"},
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_exception", path=request.url.path, error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}"},
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
