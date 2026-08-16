"""OpenAI-compatible routes for the serving infrastructure.

Mirrors OpenAI's `/v1/*` endpoint schema byte-for-byte.
Supports ``stream=true`` (SSE) and ``stream=false``.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, AsyncGenerator, cast

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
import torch

from serving.loader import get_model, get_tokenizer, get_config

router = APIRouter(prefix="/v1", tags=["openai"])

# In-memory rate limiter (per-IP). Disabled when no auth is configured.
_rate_limit_per_ip: dict[str, int] = {}
_rate_limit_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_rate_limit(client_ip: str, max_req: int = 60) -> bool:
    """Simple per-IP token-bucket rate limiter."""
    with _rate_limit_lock:
        current = _rate_limit_per_ip.get(client_ip, 0)
        if current >= max_req:
            return False
        _rate_limit_per_ip[client_ip] = current + 1
        return True


# ---------------------------------------------------------------------------
# Request / Response models (OpenAI schema)
# ---------------------------------------------------------------------------

class ChatMessage:
    """Minimal chat message — OpenAI API uses ``role`` + ``content``."""
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def model_dump(self) -> dict:
        return {"role": self.role, "content": self.content}


class ChatCompletionRequest:
    def __init__(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stream: bool = False,
        stop: list[str] | None = None,
        max_completion_tokens: int | None = None,
        seed: int | None = None,
    ):
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.stream = stream
        self.stop = stop or []
        self.max_completion_tokens = max_completion_tokens
        self.seed = seed

    @staticmethod
    async def from_fastapi_request(req: Request) -> "ChatCompletionRequest":
        payload = await req.json()
        # Parse into our minimal types
        msgs = [ChatMessage(**m) for m in payload.get("messages", [])]
        return ChatCompletionRequest(
            model=payload.get("model", ""),
            messages=msgs,
            temperature=payload.get("temperature"),
            top_p=payload.get("top_p"),
            top_k=payload.get("top_k"),
            stream=payload.get("stream", False),
            stop=payload.get("stop"),
            max_completion_tokens=payload.get("max_completion_tokens"),
            seed=payload.get("seed"),
        )


class CompletionChoice:
    def __init__(self, index: int, text: str, finish_reason: str | None = None):
        self.index = index
        self.message = {"role": "assistant", "content": text}
        self.text = text
        self.finish_reason = finish_reason

    def model_dump(self) -> dict:
        return {
            "index": self.index,
            "message": self.message,
            "text": self.text,
            "finish_reason": self.finish_reason or "stop",
        }


class ChatCompletionResponse:
    def __init__(self, choices: list[CompletionChoice], usage: dict | None = None):
        self.choices = choices
        self.usage = usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def model_dump(self) -> dict:
        return {
            "id": "chatcmpl-" + str(id(self)),
            "object": "chat.completion",
            "created": int(__import__("time").time()),
            "model": "minigpt",
            "choices": [c.model_dump() if hasattr(c, "model_dump") else c for c in self.choices],
            "usage": self.usage,
        }


# ---------------------------------------------------------------------------
# Streaming helper (SSE format for OpenAI)
# ---------------------------------------------------------------------------

def _sse_frame(data: dict) -> bytes:
    """Encode a dict as an SSE ``data:`` frame followed by CRLF."""
    return f"data: {json.dumps(data)}\n\n".encode()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/chat/completions", response_model=None)
async def chat_completions(request: Request) -> Response:
    """``POST /v1/chat/completions`` — OpenAI-compatible chat endpoint.

    - ``stream=true``: returns ``StreamingResponse`` with SSE frames.
    - ``stream=false``: returns a single ``JSONResponse`` with the final choice.
    """
    # ---- rate limit (optional) ----
    # client_ip = request.client.host  # uncomment if auth/rate-limiting enabled
    # if not _check_rate_limit(client_ip, max_req=60):
    #     raise HTTPException(status_code=429, detail="rate limit exceeded")

    body = await request.json()
    # Validate minimal fields
    if "messages" not in body or not body["messages"]:
        raise HTTPException(status_code=400, detail="`messages` is required")

    messages = body["messages"]
    temperature = body.get("temperature", 0.8)
    top_p = body.get("top_p", None)
    top_k = body.get("top_k", None)
    stream = body.get("stream", False)
    stop = body.get("stop", None) or []
    max_new_tokens = body.get("max_completion_tokens", 128)
    seed = body.get("seed", None)

    # ---- build prompt from chat history ----
    # The base LM expects plain text with role markers.
    # We concatenate: System: {system}\nUser: {u1}\nAssistant: {a1}\n...
    # For simplicity, we join all user/assistant turns.
    system_prompt = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
    user_texts: list[str] = []
    assistant_texts: list[str] = []
    for m in messages:
        if m.get("role") == "system":
            system_prompt = m.get("content", "")
        elif m.get("role") == "user":
            user_texts.append(m.get("content", ""))
        elif m.get("role") == "assistant":
            assistant_texts.append(m.get("content", ""))

    # Build context the same way the chat REPL does
    context_lines: list[str] = [f"System: {system_prompt}"]
    for u, a in zip(user_texts, assistant_texts):
        context_lines.append(f"User: {u}")
        context_lines.append(f"Assistant: {a}")
    # Trailing Assistant: cue (no text) — matches the chat REPL format
    context_lines.append("Assistant:")
    context = "\n".join(context_lines)

    # ---- generate ----
    from serving.loader import get_model as _get_model, get_tokenizer as _get_tokenizer

    model = _get_model()
    tokenizer = _get_tokenizer()

    enc = tokenizer.encode(context)
    input_ids = torch.tensor([enc.ids], dtype=torch.long)

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(int(seed))

    past: Any = None
    generated_ids: list[int] = list(enc.ids)
    decoded_context = tokenizer.decode(generated_ids)

    from inference.generate import generate as _generate
    import asyncio

    if stream:
        async def token_stream() -> AsyncGenerator[bytes, None]:
            gen = _generate(
                model=model,
                tokenizer=tokenizer,
                prompt=context,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_strings=stop,
                stream=True,
                seed=seed,
            )
            for piece in gen:
                yield _sse_frame({
                    "id": "chatcmpl-" + str(int(time.time() * 1000)),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "minigpt",
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": piece}, "finish_reason": None}],
                })
                await asyncio.sleep(0)

            yield _sse_frame({
                "id": "chatcmpl-" + str(int(time.time() * 1000)),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "minigpt",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            })
            yield b"data: [DONE]\n\n"

        return StreamingResponse(token_stream(), media_type="text/event-stream")
    else:
        generated_text = _generate(
            model=model,
            tokenizer=tokenizer,
            prompt=context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_strings=stop,
            stream=False,
            seed=seed,
        )
        choice = CompletionChoice(index=0, text=str(generated_text), finish_reason="stop")
        response = ChatCompletionResponse(choices=[choice])
        return JSONResponse(content=response.model_dump())


@router.get("/models")
async def list_models() -> JSONResponse:
    """``GET /v1/models`` — List available OpenAI-compatible models."""
    now = int(time.time())
    return JSONResponse(
        content={
            "object": "list",
            "data": [
                {
                    "id": "minigpt-high",
                    "object": "model",
                    "created": now,
                    "owned_by": "minigpt",
                },
                {
                    "id": "minigpt-low",
                    "object": "model",
                    "created": now,
                    "owned_by": "minigpt",
                },
                {
                    "id": "minigpt",
                    "object": "model",
                    "created": now,
                    "owned_by": "minigpt",
                },
            ],
        }
    )


@router.post("/completions", response_model=None)
async def legacy_completions(request: Request) -> Response:
    """``POST /v1/completions`` — Legacy single-prompt completion endpoint."""
    body = await request.json()
    prompt = body.get("prompt", "")
    temperature = body.get("temperature", 0.8)
    top_p = body.get("top_p", None)
    top_k = body.get("top_k", None)
    stream = body.get("stream", False)
    stop = body.get("stop", [])
    max_new_tokens = body.get("max_tokens", 128)
    seed = body.get("seed", None)

    from serving.loader import get_model as _get_model, get_tokenizer as _get_tokenizer
    from inference.generate import generate as _generate
    import asyncio

    model = _get_model()
    tokenizer = _get_tokenizer()

    if stream:
        async def stream_legacy() -> AsyncGenerator[bytes, None]:
            for piece in _generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_strings=stop,
                stream=True,
                seed=seed,
            ):
                yield _sse_frame({
                    "id": "cmpl-" + str(int(time.time() * 1000)),
                    "object": "text_completion",
                    "created": int(time.time()),
                    "choices": [{"text": piece, "index": 0, "finish_reason": None}],
                    "model": "minigpt",
                })
                await asyncio.sleep(0)

            yield _sse_frame({
                "id": "cmpl-" + str(int(time.time() * 1000)),
                "object": "text_completion",
                "created": int(time.time()),
                "choices": [{"text": "", "index": 0, "finish_reason": "stop"}],
                "model": "minigpt",
            })
            yield b"data: [DONE]\n\n"

        return StreamingResponse(stream_legacy(), media_type="text/event-stream")
    else:
        text = _generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_strings=stop,
            stream=False,
            seed=seed,
        )
        return JSONResponse(
            content={
                "id": "cmpl-" + str(int(time.time() * 1000)),
                "object": "text_completion",
                "created": int(time.time()),
                "model": "minigpt",
                "choices": [{"text": str(text), "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        )