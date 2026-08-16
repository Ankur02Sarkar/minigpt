# MiniGPT API Documentation & Reference Manual

Complete guide to running and interacting with the `minigpt_llm` serving container and its OpenAI-compatible and Ollama-compatible HTTP REST endpoints.

---

## 1. Quick Start: How to Run

### Method A: Running with Docker (Recommended)

To start the serving container with the API key configured:

```bash
docker run -d \
  --name minigpt_serve \
  -p 8080:8080 \
  -e MINIGPT_API_KEY=<MINIGPT_API_KEY> \
  minigpt_llm:0.1.0
```

To run with mounted model checkpoints and tokenizer from disk (e.g. on Azure GPU VM or custom training output):

```bash
docker run -d \
  --name minigpt_serve \
  -p 8080:8080 \
  -e MINIGPT_API_KEY=<MINIGPT_API_KEY> \
  -v /path/to/checkpoints:/opt/minigpt_llm/checkpoints:ro \
  -v /path/to/tokenizer:/opt/minigpt_llm/tokenizer:ro \
  minigpt_llm:0.1.0
```

> **Note on Local CPU Execution**: When running locally without mounted Azure GPU checkpoints, the server automatically boots an in-memory test model to verify all API interfaces end-to-end.

---

### Method B: Running Directly with Python

```bash
# 1. Activate your virtual environment
source .venv/bin/activate

# 2. Export environment variables
export MINIGPT_API_KEY=<MINIGPT_API_KEY>

# 3. Start the Uvicorn server
uvicorn serving.server:app --host 0.0.0.0 --port 8080
```

---

## 2. Authentication & Headers

| Header | Format | Required | Description |
| --- | --- | --- | --- |
| `Authorization` | `Bearer <MINIGPT_API_KEY>` | Yes (for model/gen APIs) | Bearer token authorization |
| `Content-Type` | `application/json` | Yes (for POST requests) | Request body serialization format |

*Public Endpoints (No Auth Required)*:

- `GET /health` — Liveness health check
- `GET /ready` — Readiness probe

---

## 3. Verified Endpoints & Examples

All requests below use the configured API key: `<MINIGPT_API_KEY>`.

### 3.1 Health & Diagnostics

#### `GET /health`

Returns `{"status": "ok"}` when the HTTP server process is running.

```bash
curl -s http://localhost:8080/health
```

#### `GET /ready`

Returns `{"status": "ready"}` when the model and tokenizer are initialized in memory and ready to handle inference.

```bash
curl -s http://localhost:8080/ready
```

---

### 3.2 OpenAI-Compatible API (`/v1/*`)

#### 1. List Available Models (`GET /v1/models`)

```bash
curl -s http://localhost:8080/v1/models \
  -H "Authorization: Bearer <MINIGPT_API_KEY>"
```

**Response**:

```json
{
  "object": "list",
  "data": [
    { "id": "minigpt-high", "object": "model", "created": 1786889616, "owned_by": "minigpt" },
    { "id": "minigpt-low", "object": "model", "created": 1786889616, "owned_by": "minigpt" },
    { "id": "minigpt", "object": "model", "created": 1786889616, "owned_by": "minigpt" }
  ]
}
```

---

#### 2. Chat Completions (Non-Streaming) (`POST /v1/chat/completions`)

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <MINIGPT_API_KEY>" \
  -d '{
    "model": "minigpt-high",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Once upon a time in a faraway kingdom,"}
    ],
    "temperature": 0.8,
    "max_completion_tokens": 64,
    "stream": false
  }'
```

**Response**:

```json
{
  "id": "chatcmpl-139777295009888",
  "object": "chat.completion",
  "created": 1786889630,
  "model": "minigpt",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "there lived a kind young prince who loved reading books."
      },
      "text": "there lived a kind young prince who loved reading books.",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

---

#### 3. Chat Completions (Streaming SSE) (`POST /v1/chat/completions`)

```bash
curl -N -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <MINIGPT_API_KEY>" \
  -d '{
    "model": "minigpt-high",
    "messages": [
      {"role": "user", "content": "Tell me a short story."}
    ],
    "temperature": 0.7,
    "max_completion_tokens": 32,
    "stream": true
  }'
```

**Stream Chunk Format (SSE)**:

```
data: {"id": "chatcmpl-1786890280568", "object": "chat.completion.chunk", "created": 1786890280, "model": "minigpt", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Once"}, "finish_reason": null}]}

data: {"id": "chatcmpl-1786890280589", "object": "chat.completion.chunk", "created": 1786890280, "model": "minigpt", "choices": [{"index": 0, "delta": {"role": "assistant", "content": " upon"}, "finish_reason": null}]}

data: {"id": "chatcmpl-1786890280697", "object": "chat.completion.chunk", "created": 1786890280, "model": "minigpt", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]
```

---

#### 4. Legacy Prompt Completions (`POST /v1/completions`)

Supports traditional single-prompt OpenAI completions (both streaming and non-streaming):

```bash
curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <MINIGPT_API_KEY>" \
  -d '{
    "prompt": "The little cat sat on the",
    "temperature": 0.8,
    "max_tokens": 20,
    "stream": false
  }'
```

---

### 3.3 Ollama-Compatible API (`/api/*`)

#### 1. Version (`GET /api/version`)

```bash
curl -s http://localhost:8080/api/version \
  -H "Authorization: Bearer <MINIGPT_API_KEY>"
```

**Response**:

```json
{
  "version": "0.1.0",
  "status": "ok"
}
```

---

#### 2. Model Tags (`GET /api/tags`)

```bash
curl -s http://localhost:8080/api/tags \
  -H "Authorization: Bearer <MINIGPT_API_KEY>"
```

**Response**:

```json
{
  "models": [
    { "name": "minigpt", "size": 0 }
  ]
}
```

---

#### 3. Text Generation (`POST /api/generate`)

Supports both standard single JSON responses (`"stream": false`) and NDJSON streaming (`"stream": true`):

**Non-Streaming Example**:

```bash
curl -s http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <MINIGPT_API_KEY>" \
  -d '{
    "model": "minigpt",
    "prompt": "Once upon a time",
    "stream": false
  }'
```

**Streaming Example (NDJSON)**:

```bash
curl -s http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <MINIGPT_API_KEY>" \
  -d '{
    "model": "minigpt",
    "prompt": "Once upon a time",
    "stream": true
  }'
```

**Stream Chunk Format (NDJSON)**:

```json
{"response": "there", "done": false, "model": "minigpt"}
{"response": " was", "done": false, "model": "minigpt"}
{"response": " a", "done": false, "model": "minigpt"}
{"response": "", "done": true, "done_reason": "stop", "model": "minigpt"}
```

---

#### 4. Chat (`POST /api/chat`)

Supports multi-turn chat messages with NDJSON streaming (`"stream": true`) or final message JSON (`"stream": false`):

```bash
curl -s http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <MINIGPT_API_KEY>" \
  -d '{
    "model": "minigpt",
    "messages": [
      {"role": "user", "content": "Hello, who are you?"}
    ],
    "stream": false
  }'
```

---

## 4. Stopping the Server

```bash
docker stop minigpt_serve
```
