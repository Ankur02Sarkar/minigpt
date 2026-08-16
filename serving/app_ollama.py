"""Ollama-compatible routes for the serving infrastructure.

Mirrors Ollama's `/api/*` endpoint schema.

- ``POST /api/generate`` — text generation (NDJSON streaming)
- ``POST /api/chat`` — chat (NDJSON streaming)
- ``GET /api/tags`` — list local models (placeholder)
- ``GET /api/version`` — return server version
- ``POST /api/show`` — return model info
"""

from __future__ import annotations

import json
import threading
from typing import Any, AsyncGenerator, cast

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import torch

from serving.loader import get_model, get_tokenizer, get_config

router = APIRouter(prefix="/api", tags=["ollama"])

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _sse_frame(data: dict) -> bytes:
    """Encode a dict as NDJSON (one JSON object + newline) in bytes."""
    return (json.dumps(data, ensure_ascii=False) + "\n").encode("utf-8")


# ----------------------------------------------------------------------------
# Generation payloads
# ----------------------------------------------------------------------------

class GenerateRequest:
    def __init__(
        self,
        model: str,
        prompt: str,
        stream: bool = True,
        options: dict | None = None,
        format: dict | None = None,
        system: str | None = None,
        context: list[str] | None = None,
        raw: bool = False,
    ):
        self.model = model
        self.prompt = prompt
        self.stream = stream
        self.options = options or {}
        self.format = format
        self.system = system
        self.context = context or []
        self.raw = raw


class ChatMessageOllama:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class ChatRequest:
    def __init__(
        self,
        model: str,
        messages: list[ChatMessageOllama],
        stream: bool = True,
        options: dict | None = None,
        keep_alive: str | dict | None = None,
    ):
        self.model = model
        self.messages = messages
        self.stream = stream
        self.options = options or {}
        self.keep_alive = keep_alive


# ----------------------------------------------------------------------------
# Route: POST /api/generate
# ----------------------------------------------------------------------------

@router.post("/generate", response_model=None)
async def ollama_generate(request: Request) -> Response:
    """``POST /api/generate`` — Ollama text generation."""
    payload = await request.json()
    prompt = payload.get("prompt", "")
    stream = payload.get("stream", True)
    system = payload.get("system", "")

    ctx_lines: list[str] = []
    if system:
        ctx_lines.append(f"System: {system}")
    ctx_lines.extend([f"User: {prompt}", "Assistant:"])
    context_str = "\n".join(ctx_lines)

    from serving.loader import get_model as _get_model, get_tokenizer as _get_tokenizer
    from inference.generate import generate as _generate

    model = _get_model()
    tokenizer = _get_tokenizer()

    if stream:
        async def token_stream() -> AsyncGenerator[bytes, None]:
            for piece in _generate(
                model=model,
                tokenizer=tokenizer,
                prompt=context_str,
                max_new_tokens=128,
                stream=True,
            ):
                yield _sse_frame({
                    "response": piece,
                    "done": False,
                    "model": payload.get("model", "minigpt"),
                })
            yield _sse_frame({
                "response": "",
                "done": True,
                "done_reason": "stop",
                "model": payload.get("model", "minigpt"),
            })

        return StreamingResponse(token_stream(), media_type="application/x-ndjson")
    else:
        text = _generate(
            model=model,
            tokenizer=tokenizer,
            prompt=context_str,
            max_new_tokens=128,
            stream=False,
        )
        return JSONResponse(
            content={
                "response": str(text),
                "done": True,
                "done_reason": "stop",
                "model": payload.get("model", "minigpt"),
            }
        )


# ----------------------------------------------------------------------------
# Route: POST /api/chat
# ----------------------------------------------------------------------------

@router.post("/chat", response_model=None)
async def ollama_chat(request: Request) -> Response:
    """``POST /api/chat`` — Ollama chat (multi-turn, NDJSON streaming)."""
    payload = await request.json()
    messages = payload.get("messages", [])
    stream = payload.get("stream", True)

    system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
    user_texts: list[str] = [m.get("content", "") for m in messages if m.get("role") == "user"]
    assistant_texts: list[str] = [m.get("content", "") for m in messages if m.get("role") == "assistant"]

    context_lines: list[str] = [f"System: {system}"]
    for u, a in zip(user_texts, assistant_texts):
        context_lines.append(f"User: {u}")
        context_lines.append(f"Assistant: {a}")
    if len(user_texts) > len(assistant_texts):
        context_lines.append(f"User: {user_texts[-1]}")
    context_lines.append("Assistant:")
    context_str = "\n".join(context_lines)

    from serving.loader import get_model as _get_model, get_tokenizer as _get_tokenizer
    from inference.generate import generate as _generate

    model = _get_model()
    tokenizer = _get_tokenizer()

    if stream:
        async def chat_stream() -> AsyncGenerator[bytes, None]:
            for piece in _generate(
                model=model,
                tokenizer=tokenizer,
                prompt=context_str,
                max_new_tokens=128,
                stream=True,
            ):
                yield _sse_frame({
                    "message": {"role": "assistant", "content": piece},
                    "done": False,
                })
            yield _sse_frame({
                "message": {"role": "assistant", "content": ""},
                "done": True,
            })

        return StreamingResponse(chat_stream(), media_type="application/x-ndjson")
    else:
        text = _generate(
            model=model,
            tokenizer=tokenizer,
            prompt=context_str,
            max_new_tokens=128,
            stream=False,
        )
        return JSONResponse(
            content={
                "message": {"role": "assistant", "content": str(text)},
                "done": True,
            }
        )


# ----------------------------------------------------------------------------
# Route: GET /api/tags
# ----------------------------------------------------------------------------

@router.get("/tags")
async def ollama_tags() -> JSONResponse:
    """``GET /api/tags`` — List local models (placeholder)."""
    return JSONResponse(content={"models": [{"name": "minigpt", "size": 0}]})


# ----------------------------------------------------------------------------
# Route: GET /api/version
# ----------------------------------------------------------------------------

@router.get("/version")
async def ollama_version() -> JSONResponse:
    """``GET /api/version`` — Return server version."""
    import platform
    return JSONResponse(content={"version": "0.1.0", "status": "ok"})


# ----------------------------------------------------------------------------
# Route: POST /api/show
# ----------------------------------------------------------------------------

@router.post("/show")
async def ollama_show(request: Request) -> JSONResponse:
    """``POST /api/show`` — Return model info (minimal)."""
    payload = await request.json()
    model_name = payload.get("model", "minigpt")
    return JSONResponse(content={"model": model_name, "modified": None})