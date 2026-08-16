"""OpenAI-compatible routes for the serving infrastructure.

Mirrors OpenAI's `/v1/*` endpoint schema byte-for-byte.
Supports ``stream=true`` (SSE) and ``stream=false``.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, AsyncGenerator, cast

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse

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
    def from_fastapi_request(req: Request) -> "ChatCompletionRequest":
        payload = payload = req.json()
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
        self.text = text
        self.finish_reason = finish_reason


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
            "choices": self.choices,
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

@router.post("/chat/completions")
async def chat_completions(request: Request) -> StreamingResponse | JSONResponse:
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

    stopped = False
    final_text = ""

    async def token_stream() -> AsyncGenerator[bytes, None]:
        nonlocal past, generated_ids, stopped, final_text

        for _ in range(max_new_tokens):
            # feed last token (or full sequence on first step)
            if past is None:
                out = model.forward(input_ids, use_cache=True)
            else:
                out = model.forward(input_ids[:, -1:], past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]  # (1, V)

            # greedy or sample
            if temperature <= 0 or top_k is not None:
                next_id = int(torch.argmax(logits, dim=-1).item())
            else:
                probs = torch.softmax(logits, dim=-1)
                # nucleus (top-p)
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    probs_s = torch.softmax(sorted_logits, dim=-1)
                    cum = torch.cumsum(probs_s, dim=-1)
                    mask = cum > top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                    logits = torch.full_like(logits, float("-inf"))
                    logits.scatter_(1, sorted_idx, sorted_logits)
                    probs = torch.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1, generator=generator).item())

            generated_ids.append(next_id)

            # decode only the *new* tokens (post-prompt)
            full_text = tokenizer.decode(generated_ids)
            new_text = full_text[len(decoded_context):]

            if new_text:
                piece = new_text
            else:
                piece = tokenizer.decode([next_id])

            # stop-string check (against generated text only)
            for s in stop:
                if s and s in piece:
                    # truncate at the stop string boundary
                    piece = piece[: piece.rfind(s)]
                    stopped = True
                    break
            if stopped:
                # yield final piece then return
                yield _sse_frame({"choices": [{"index": 0, "delta": {"role": "assistant", "content": piece}, "finish_reason": "stop"}]})
                yield _sse_frame({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                return

            # yield SSE token
            yield _sse_frame({
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": piece}, "finish_reason": None}]
            })

            if stopped:
                return

            # update context for next iteration (append the piece)
            decoded_context = full_text

    if stream:
        return StreamingResponse(token_stream(), media_type="text/event-stream")
    else:
        # non-stream: run to completion then return JSON
        async def run_and_return() -> JSONResponse:
            async for chunk in token_stream():
                pass  # consume stream
            # final choice
            choice = CompletionChoice(index=0, text=final_text, finish_reason="stop")
            response = ChatCompletionResponse(choices=[choice])
            return JSONResponse(content=response.model_dump())

        return await run_and_return()