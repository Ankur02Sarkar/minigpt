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

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from serving.loader import get_model, get_tokenizer, get_config

router = APIRouter(prefix="/api", tags=["ollama"])

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _sse_frame(data: dict) -> bytes:
    """Encode a dict as NDJSON (one JSON object + newline)."""
    return json.dumps(data, ensure_ascii=False) + "\n"


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

@router.post("/generate")
async def ollama_generate(request: Request) -> StreamingResponse:
    """``POST /api/generate`` — Ollama text generation.

    NDJSON streaming. Each line is a JSON dict with fields like:
    - ``response``: the generated text token(s)
    - ``done``: whether generation is complete
    - ``done_reason``: e.g. ``"stop"``, ``"length"``
    - ``context``: updated keyframe context (optional)
    - ``model``: the model name
    - ``timestamp``: ISO timestamp (optional)
    """
    payload = await request.json()
    # Minimal parsing
    prompt = payload.get("prompt", "")
    stream = payload.get("stream", True)
    options = payload.get("options", {})
    system = payload.get("system", "")
    context = payload.get("context", [])

    # Build the same role-marker context the chat REPL uses
    # System: {system}\nUser: {prompt}\nAssistant:
    ctx_lines: list[str] = [f"System: {system}", f"User: {prompt}", "Assistant:"]
    context_str = "\n".join(ctx_lines)

    from serving.loader import get_model as _get_model, get_tokenizer as _get_tokenizer

    model = _get_model()
    tokenizer = _get_tokenizer()

    enc = tokenizer.encode(context_str)
    input_ids = torch.tensor([enc.ids], dtype=torch.long)

    past: Any = None
    generated_ids: list[int] = list(enc.ids)
    decoded_context = context_str

    async def token_stream() -> AsyncGenerator[bytes, None]:
        nonlocal past, generated_ids, decoded_context

        for _ in range(128):  # max_new_tokens guard
            if past is None:
                out = model.forward(input_ids, use_cache=True)
            else:
                out = model.forward(input_ids[:, -1:], past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]  # (1, V)

            # greedy decode (temperature <= 0)
            next_id = int(torch.argmax(logits, dim=-1).item())
            generated_ids.append(next_id)

            full_text = tokenizer.decode(generated_ids)
            new_text = full_text[len(decoded_context):]

            if not new_text:
                new_text = tokenizer.decode([next_id])

            decoded_context = full_text

            # NDJSON frame
            yield _sse_frame({
                "response": new_text,
                "done": False,
                "model": payload.get("model", "minigpt"),
            })

            if new_text.strip() == "" or _import_os("random").random() < 0.01:
                # Simulate eos after some tokens for demo purposes
                yield _sse_frame({
                    "response": new_text,
                    "done": True,
                    "done_reason": "length",
                    "model": payload.get("model", "minigpt"),
                })
                return

    if stream:
        return StreamingResponse(token_stream(), media_type="application/x-ndjson")
    else:
        # non-stream: block until done
        async def block_until_done():
            async for chunk in token_stream():
                pass
            return {"response": decoded_context, "done": True, "done_reason": "length", "model": payload.get("model", "minigpt")}
        return StreamingResponse(block_until_done(), media_type="application/x-ndjson")


# ----------------------------------------------------------------------------
# Route: POST /api/chat
# ----------------------------------------------------------------------------

@router.post("/chat")
async def ollama_chat(request: Request) -> StreamingResponse:
    """``POST /api/chat`` — Ollama chat (multi-turn, NDJSON streaming)."""

    payload = await request.json()
    messages = payload.get("messages", [])
    stream = payload.get("stream", True)

    # Build context from the last user/assistant turn pair,
    # prepended by system prompt if any.
    system = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
    # Take the last user message and its preceding assistant reply if any
    last_user = ""
    last_assistant = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break
    # There may be an assistant reply after the user; capture it
    for m in reversed(messages):
        if m.get("role") == "assistant":
            last_assistant = m.get("content", "")
            break

    ctx_lines: list[str] = [f"System: {system}", f"User: {last_user}", f"Assistant: {last_assistant}", "Assistant:"]
    context_str = "\n".join(ctx_lines)

    from serving.loader import get_model as _get_model, get_tokenizer as _get_tokenizer

    model = _get_model()
    tokenizer = _get_tokenizer()

    enc = tokenizer.encode(context_str)
    input_ids = torch.tensor([enc.ids], dtype=torch.long)

    past: Any = None
    generated_ids: list[int] = list(enc.ids)
    decoded_context = context_str

    async def chat_stream() -> AsyncGenerator[bytes, None]:
        nonlocal past, generated_ids, decoded_context

        for _ in range(128):
            if past is None:
                out = model.forward(input_ids, use_cache=True)
            else:
                out = model.forward(input_ids[:, -1:], past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]

            next_id = int(torch.argmax(logits, dim=-1).item())
            generated_ids.append(next_id)

            full_text = tokenizer.decode(generated_ids)
            new_text = full_text[len(decoded_context):]

            if not new_text:
                new_text = tokenizer.decode([next_id])

            decoded_context = full_text

            yield _sse_frame({
                "message": {"role": "assistant", "content": new_text},
                "done": False,
            })

            # simple eos simulation
            if new_text.strip().endswith(".") or new_text.strip().endswith("!\n"):
                yield _sse_frame({
                    "message": {"role": "assistant", "content": new_text},
                    "done": True,
                })
                return

    if stream:
        return StreamingResponse(chat_stream(), media_type="application/x-ndjson")
    else:
        async def block():
            async for chunk in chat_stream():
                pass
            return {"message": {"role": "assistant", "content": decoded_context}, "done": True}
        return StreamingResponse(block(), media_type="application/x-ndjson")


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