"""Generate text samples from a trained checkpoint for docs/SAMPLES.md."""

from __future__ import annotations

import argparse
from pathlib import Path

import structlog
import torch

from minigpt_llm.model import GPT, ModelConfig
from minigpt_llm.tokenizer.load import eos_token_id, load_tokenizer

__all__ = ["main"]

log = structlog.get_logger(__name__)


def load_model(checkpoint: Path, device: torch.device) -> tuple[GPT, int]:
    """Load a GPT model + trained step from a training checkpoint."""
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ModelConfig(**payload["config"])
    model = GPT(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    log.info(
        "model_loaded",
        checkpoint=str(checkpoint),
        step=payload.get("step"),
        best_val_loss=payload.get("best_val_loss"),
        params=sum(p.numel() for p in model.parameters()),
    )
    return model, int(payload.get("step", -1))


def generate_one(
    model: GPT,
    token_ids: list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    eos: int | None,
    device: torch.device,
) -> list[int]:
    """Run one generation pass and return the full token sequence."""
    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    out = model.generate(
        x,
        max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=eos,
    )
    return out[0].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--greedy", action="store_true", help="override with temperature=0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, step = load_model(args.checkpoint, device)
    tok = load_tokenizer(args.tokenizer_dir)
    eos = eos_token_id(tok)
    temperature = 0.0 if args.greedy else args.temperature
    top_k = None if args.greedy else args.top_k
    mode = "greedy" if args.greedy else f"temp={temperature} top_k={top_k}"

    print(f"# samples — checkpoint={args.checkpoint.name} step={step} mode={mode}\n")
    for prompt in args.prompts:
        ids = tok.encode(prompt).ids
        generated = generate_one(
            model,
            ids,
            max_new_tokens=args.max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos=eos,
            device=device,
        )
        text = tok.decode(generated)
        n_new = len(generated) - len(ids)
        print(f"## prompt: {prompt!r} (+{n_new} tokens)\n")
        print(text)
        print()


if __name__ == "__main__":
    main()
