"""Generation core — encode prompt, autoregressive decode with KV-cache, decode to text.

Re-implements the decode loop here (Option A) instead of extending ``GPT.generate()``
so the model layer stays tokenizer-agnostic and ``stop_strings`` + streaming live in
the inference layer. Mirrors ``model._sample_next`` semantics (greedy / top_k / top_p)
so a non-stream ``generate(...)`` call matches ``GPT.generate()`` output for the same
seed and sampling params.

Usage::

    from inference.generate import generate
    text = generate(model, tok, "Once upon", max_new_tokens=50, stream=False)
    # or stream token-by-token:
    for piece in generate(model, tok, "Once upon", max_new_tokens=50, stream=True):
        print(piece, end="", flush=True)
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from minigpt_llm.model import GPT, ModelConfig
from minigpt_llm.tokenizer.load import eos_token_id, load_tokenizer

__all__ = ["generate", "load_model", "main"]

_DTYPE = torch.long


def _sample_next(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample or greedy-select next token ids ``(B, 1)``.

    Mirrors ``minigpt_llm.model.model._sample_next`` so a non-stream call here
    matches ``GPT.generate()`` output for the same seed + sampling params. The
    ``generator`` arg makes sampling reproducible from the caller's seed.
    """
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        cutoff = values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < cutoff, torch.finfo(logits.dtype).min)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        mask = cum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, torch.finfo(sorted_logits.dtype).min)
        logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
        logits.scatter_(1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    if generator is not None:
        next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    else:
        next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def generate(
    model: GPT,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    stop_strings: list[str] | None = None,
    stream: bool = False,
    device: torch.device | str = "cpu",
    seed: int | None = None,
    on_token: Callable[[int, str, float, list[tuple[int, float]]], None] | None = None,
) -> str | Iterator[str]:
    """Generate text from ``prompt`` with KV-cache.

    Args:
        model: a trained ``GPT`` (`.eval()` is called here).
        tokenizer: a ``tokenizers`` ``ByteLevelBPETokenizer`` (``.encode(s).ids``,
            ``.decode(ids)``).
        prompt: text prompt.
        max_new_tokens: hard cap on new tokens.
        temperature: ``<= 0`` selects greedy (argmax).
        top_k / top_p: optional nucleus filtering (disabled if None).
        stop_strings: optional list of substrings — generation halts as soon as
            any one appears in the decoded text *including* the token that
            completed it. The stop string is excluded from the final output.
        stream: if ``True``, return a generator yielding incremental text
            pieces (one per new token); if ``False`` return the full string.
        device: device to place tensors on.
        seed: if set, sampling is reproducible via a local ``torch.Generator``
            (does not touch the global RNG state).

    Returns:
        Generated text only (excludes the prompt) — matching OpenAI API
        semantics (``choices[0].text`` is the model's continuation, not
        prompt+continuation). ``stream=False`` returns the full generated
        string; ``stream=True`` returns an iterator of incremental pieces.
    """
    if max_new_tokens < 1:
        return "" if not stream else iter("")

    model.eval()
    dev = torch.device(device)
    stops = stop_strings or []

    enc = tokenizer.encode(prompt)
    ids = enc.ids
    eos = eos_token_id(tokenizer)

    # Respect model context window if available
    max_ctx = getattr(getattr(model, "config", None), "max_position_embeddings", 1024)
    if len(ids) >= max_ctx:
        ids = ids[-(max_ctx - 1):]
    allowed_new_tokens = min(max_new_tokens, max_ctx - len(ids))
    if allowed_new_tokens < 1:
        return "" if not stream else iter("")

    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device=dev)
        generator.manual_seed(int(seed))

    input_ids = torch.tensor([ids], dtype=_DTYPE, device=dev)
    prompt_text = tokenizer.decode(ids)
    generated_ids = list(ids)
    past: list[Any] | None = None
    # decoded_new tracks ONLY the post-prompt text (for stop-string checks + return).
    decoded_new = ""

    def _step() -> tuple[str | None, bool]:
        """Run one decode step; return ``(piece, stopped)``.

        ``piece`` is the incremental text yielded to a streaming caller (or None
        if the stop string consumed it); ``stopped`` is True on eos / stop-string.

        Stop strings are matched against the *generated* text only (not the prompt),
        matching OpenAI API semantics.
        """
        nonlocal past, decoded_new, input_ids

        if past is None:
            out = model.forward(input_ids, use_cache=True)
        else:
            out = model.forward(input_ids[:, -1:], past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]  # (B, V)

        next_id = _sample_next(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )  # (B, 1)
        next_id_int = int(next_id.item())
        generated_ids.append(next_id_int)
        input_ids = next_id

        # Full decode (prompt + new); ByteLevel BPE needs surrounding context.
        full_text = tokenizer.decode(generated_ids)
        new_all = full_text[len(prompt_text) :]
        piece = new_all[len(decoded_new) :]
        decoded_new = new_all

        stopped_by_eos = eos is not None and next_id_int == eos
        stopped_by_str = False
        if stops:
            for s in stops:
                if s and s in decoded_new:
                    idx = decoded_new.find(s)
                    piece = piece[: max(0, idx - len(decoded_new) + len(piece))]
                    decoded_new = decoded_new[:idx]
                    stopped_by_str = True
                    break
        stopped = stopped_by_eos or stopped_by_str
        # --- on-token observation callback ---
        if on_token is not None:
            probs = F.softmax(logits, dim=-1)
            entropy = float(-(probs * torch.log(probs + 1e-10)).sum())
            top5_vals, top5_idx = torch.topk(probs, 5)
            top5: list[tuple[int, float]] = [
                (int(idx.item()), float(val.item()))
                for val, idx in zip(top5_vals, top5_idx, strict=False)
            ]
            on_token(next_id_int, piece, entropy, top5)
        return (piece, stopped)

    def _run() -> str:
        for _ in range(allowed_new_tokens):
            _, stopped = _step()
            if stopped:
                break
        return decoded_new

    def _gen() -> Iterator[str]:
        for _ in range(allowed_new_tokens):
            piece, stopped = _step()
            if piece:
                yield piece
            if stopped:
                return

    if stream:
        return _gen()
    return _run()


def load_model(checkpoint: Path, device: torch.device) -> tuple[GPT, int]:
    """Load a ``GPT`` + its trained step from a training checkpoint."""
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ModelConfig(**payload["config"]["model"])
    model = GPT(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, int(payload.get("step", -1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--stop", nargs="*", default=None, help="stop strings (substring match)")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--greedy", action="store_true", help="override with temperature=0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, step = load_model(args.checkpoint, device)
    tok = load_tokenizer(args.tokenizer_dir)
    temperature = 0.0 if args.greedy else args.temperature
    top_k = None if args.greedy else args.top_k

    print(
        f"# checkpoint={args.checkpoint.name} step={step} "
        f"mode={'greedy' if args.greedy else f'temp={temperature} top_k={top_k}'}\n"
    )

    if args.stream:
        for piece in generate(
            model,
            tok,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=args.top_p,
            stop_strings=args.stop,
            stream=True,
            device=device,
            seed=args.seed,
        ):
            print(piece, end="", flush=True)
        print()
    else:
        text = generate(
            model,
            tok,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=args.top_p,
            stop_strings=args.stop,
            stream=False,
            device=device,
            seed=args.seed,
        )
        print(text)


if __name__ == "__main__":
    main()
