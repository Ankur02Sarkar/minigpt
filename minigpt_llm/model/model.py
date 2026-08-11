"""Full GPT causal LM: embed → blocks → norm → tied LM head + generate."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from minigpt_llm.model.attention import KVCache
from minigpt_llm.model.block import DecoderBlock, RMSNorm
from minigpt_llm.model.config import ModelConfig
from minigpt_llm.model.init import apply_weight_init, assert_not_bfloat16

__all__ = ["CausalLMOutputWithPast", "GPT"]


@dataclass
class CausalLMOutputWithPast:
    """Minimal causal LM output (no transformers dependency)."""

    loss: torch.Tensor | None
    logits: torch.Tensor
    past_key_values: list[KVCache] | None = None


class GPT(nn.Module):
    """From-scratch GPT with RoPE, SwiGLU, pre-norm RMSNorm, optional weight tying."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(DecoderBlock(config) for _ in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Init before weight tying so shared embed is not double-initialized.
        apply_weight_init(self)
        if config.tie_weights:
            self.lm_head.weight = self.embed_tokens.weight

    @classmethod
    def from_config(cls, config: ModelConfig) -> GPT:
        return cls(config)

    def num_parameters(self, *, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        past_key_values: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutputWithPast:
        """Forward pass.

        Args:
            input_ids: ``(B, T)`` token ids.
            labels: optional ``(B, T)``; if set, CE loss on next-token prediction
                (``logits[..., :-1]`` vs ``labels[..., 1:]`` when same length).
            past_key_values: optional per-layer KV cache for incremental decode.
            use_cache: if True, return updated past_key_values.
        """
        b, t = input_ids.shape
        if t > self.config.max_position_embeddings and past_key_values is None:
            raise ValueError(
                f"sequence length {t} exceeds max_position_embeddings "
                f"{self.config.max_position_embeddings}"
            )

        x = self.embed_dropout(self.embed_tokens(input_ids))
        presents: list[KVCache] | None = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values is not None else None
            x, present = layer(x, past_key_value=past, use_cache=use_cache)
            if use_cache and present is not None and presents is not None:
                presents.append(present)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss: torch.Tensor | None = None
        if labels is not None:
            # Shift for causal LM: predict token t+1 from position t
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation with KV-cache.

        ``temperature <= 0`` selects greedy (argmax). Never uses bfloat16.
        """
        assert_not_bfloat16(next(self.parameters()).dtype)
        self.eval()
        if max_new_tokens < 1:
            return input_ids

        generated = input_ids
        past: list[KVCache] | None = None

        for _ in range(max_new_tokens):
            if past is None:
                # Prefill full prompt
                out = self.forward(generated, use_cache=True)
            else:
                # Decode one new token
                out = self.forward(
                    generated[:, -1:],
                    past_key_values=past,
                    use_cache=True,
                )
            past = out.past_key_values
            logits = out.logits[:, -1, :]  # (B, V)

            next_id = _sample_next(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            generated = torch.cat([generated, next_id], dim=1)
            if eos_token_id is not None and (next_id == eos_token_id).all():
                break

        return generated


def _sample_next(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    """Sample or greedy-select next token ids ``(B, 1)``."""
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
        # Remove tokens with cumulative prob above top_p (keep at least 1)
        mask = cum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, torch.finfo(sorted_logits.dtype).min)
        # Scatter back
        logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
        logits.scatter_(1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
