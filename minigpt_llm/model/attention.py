"""Causal multi-head self-attention with RoPE and optional KV-cache."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minigpt_llm.model.config import ModelConfig
from minigpt_llm.model.rope import RotaryEmbedding, apply_rotary_emb

__all__ = ["CausalSelfAttention"]

# past_key_value: (key, value) each (B, n_heads, T, head_dim)
KVCache = tuple[torch.Tensor, torch.Tensor]


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention (no bias on projections — LLaMA style)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.rope = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )

    def _shape(self, x: torch.Tensor, b: int, t: int) -> torch.Tensor:
        # (B, T, H) -> (B, n_heads, T, head_dim)
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        b, t, _ = x.shape
        q = self._shape(self.q_proj(x), b, t)
        k = self._shape(self.k_proj(x), b, t)
        v = self._shape(self.v_proj(x), b, t)

        offset = 0
        if past_key_value is not None:
            offset = past_key_value[0].shape[2]

        cos, sin = self.rope(t, offset=offset)
        # cos/sin: (T, head_dim) → broadcast over batch/heads
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        present: KVCache | None = (k, v) if use_cache else None

        # Attention
        if self.training:
            # Explicit causal mask for train path (unit-testable)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # q length may be t, k length may be longer with cache
            q_len, k_len = scores.shape[-2], scores.shape[-1]
            # Causal: query position i attends to keys j <= i + offset
            # positions: queries at [offset, offset+q_len), keys at [0, k_len)
            q_pos = torch.arange(offset, offset + q_len, device=x.device).view(-1, 1)
            k_pos = torch.arange(k_len, device=x.device).view(1, -1)
            mask = k_pos > q_pos  # True where blocked
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
            attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            if self.dropout > 0:
                attn = F.dropout(attn, p=self.dropout, training=True)
            y = torch.matmul(attn, v)
        else:
            # Fast path: SDPA with is_causal only when no cache / full sequence
            if past_key_value is None and offset == 0:
                y = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True,
                )
            else:
                # With cache, build additive mask
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                q_len, k_len = scores.shape[-2], scores.shape[-1]
                q_pos = torch.arange(offset, offset + q_len, device=x.device).view(-1, 1)
                k_pos = torch.arange(k_len, device=x.device).view(1, -1)
                mask = k_pos > q_pos
                scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
                attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
                y = torch.matmul(attn, v)

        y = y.transpose(1, 2).contiguous().view(b, t, self.hidden_size)
        y = self.resid_dropout(self.o_proj(y))
        return y, present
