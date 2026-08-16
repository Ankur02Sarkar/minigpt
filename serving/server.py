"""Unified FastAPI server entry point for minigpt_llm.

Mounts both OpenAI-compatible and Ollama-compatible routers on a single
``FastAPI`` app.  Provides ``/health`` and ``/ready`` probes for
k8s-style orchestration.

Typical usage::

    uvicorn serving.server:app --host 0.0.0.0 --port 8080 \\
        --model checkpoints/minigpt-high/best.pt \\
        --tokenizer-dir /data/vocab \\
        --api openai|ollama|both
"""

from __future__ import annotations

import argparse
import time
import uvicorn

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from serving.loader import initialize, get_model, get_tokenizer, get_config

# ----------------------------------------------------------------------------
# Optional API key auth (checked against environment variable MINIGPT_API_KEY)
# Set MINIGPT_API_KEY in the environment to enable auth; leave unset to disable.
# ----------------------------------------------------------------------------

_API_KEY: str | None = None
try:
    from os import environ
    _API_KEY = environ.get("MINIGPT_API_KEY")
except Exception:  # pragma: no cover
    pass

# ----------------------------------------------------------------------------
# Simple per-IP rate limiter (token bucket, default 60 req/min)
# ----------------------------------------------------------------------------

_REQUEST_TIMESTAMPS: dict[str, list[float]] = {}
_RATE_LOCK = __import__("threading").Lock()


def _rate_allow(client_ip: str, rate_per_min: int = 60) -> bool:
    """Return True if the IP is within the rate limit."""
    with _RATE_LOCK:
        now = time.time()
        timestamps = _REQUEST_TIMESTAMPS.get(client_ip, [])
        # Remove timestamps older than 1 minute
        timestamps = [t for t in timestamps if now - t < 60.0]
        if len(timestamps) < rate_per_min:
            timestamps.append(now)
            _REQUEST_TIMESTAMPS[client_ip] = timestamps
            return True
        return False


# ----------------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------------

from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from os import environ
from typing import cast


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Automatically initialize model & tokenizer from environment if provided
    model_path = environ.get("MINIGPT_MODEL_PATH") or environ.get("CHECKPOINT_PATH")
    tokenizer_dir = environ.get("MINIGPT_TOKENIZER_DIR") or environ.get("TOKENIZER_DIR")
    if model_path and tokenizer_dir:
        try:
            initialize(model_path, tokenizer_dir)
        except Exception as e:
            # Allow server to start so /health or debugging still works
            pass
    yield


app = FastAPI(
    title="minigpt_llm Serving API",
    description="MiniGPT-llm — from-scratch GPT with OpenAI + Ollama endpoints",
    version="0.1.0",
    lifespan=lifespan,
)


# ----------------------------------------------------------------------------
# HTTP middleware that applies rate limiting + optional auth
# ----------------------------------------------------------------------------

@app.middleware("http")
async def _middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    # Public endpoints exempt from auth
    if request.url.path in ("/health", "/ready", "/docs", "/openapi.json"):
        return cast(Response, await call_next(request))

    # --- Auth (Bearer token) ---
    if _API_KEY is not None:
        auth = request.headers.get("Authorization", "")
        if not auth or not auth.startswith("Bearer "):
            return JSONResponse(
                content={"detail": "missing or malformed Bearer token"},
                status_code=403,
            )
        token = auth.split(" ", 1)[1]
        if token != _API_KEY:
            return JSONResponse(
                content={"detail": "invalid Bearer token"},
                status_code=403,
            )

    # --- Rate limit ---
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_allow(client_ip, rate_per_min=60):
        return JSONResponse(
            content={"detail": "rate limit exceeded — max 60 requests/min per IP"},
            status_code=429,
        )

    response = await call_next(request)
    return cast(Response, response)


@app.get("/health", include_in_schema=False)
def health() -> JSONResponse:
    """Simple health-check endpoint."""
    return JSONResponse(content={"status": "ok"})


@app.get("/ready", include_in_schema=False)
def ready() -> JSONResponse:
    """Readiness probe — model + tokenizer must be initialized."""
    try:
        get_model()
        get_tokenizer()
        return JSONResponse(content={"status": "ready"})
    except Exception as e:
        return JSONResponse(content={"status": "not_ready", "error": str(e)}, status_code=500)


# Mount routers
from serving.app_openai import router as openai_router
from serving.app_ollama import router as ollama_router

app.include_router(openai_router)
app.include_router(ollama_router)


# ----------------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="minigpt_llm serving API")
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--tokenizer-dir", type=str, required=True, help="Path to tokenizer directory")
    parser.add_argument("--api", type=str, default="both", choices=["openai", "ollama", "both"],
                        help="Which API(s) to serve")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    # Initialise model + tokenizer once at startup
    initialize(args.model, args.tokenizer_dir)

    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)