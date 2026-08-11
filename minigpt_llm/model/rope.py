"""Rotary positional embeddings (RoPE)."""

from __future__ import annotations

import torch

__all__ = ["RotaryEmbedding", "apply_rotary_emb"]


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to ``x`` of shape ``(..., T, D)`` with even ``D``.

    ``cos``/``sin`` have shape ``(T, D)`` or broadcastable to ``x``.
    """
    # Split last dim into pairs
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    # cos/sin stored as interleaved pairs matching even/odd
    cos_e = cos[..., ::2]
    sin_e = sin[..., ::2]
    # Rotate: (x1, x2) -> (x1 cos - x2 sin, x1 sin + x2 cos)
    o1 = x1 * cos_e - x2 * sin_e
    o2 = x1 * sin_e + x2 * cos_e
    out = torch.empty_like(x)
    out[..., ::2] = o1
    out[..., 1::2] = o2
    return out


class RotaryEmbedding(torch.nn.Module):
    """Caches cos/sin tables for RoPE up to ``max_seq_len``."""

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
        # Interleave so apply_rotary_emb can index even/odd
        emb = torch.stack((freqs, freqs), dim=-1).flatten(-2)  # (T, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin for positions ``[offset, offset+seq_len)``."""
        end = offset + seq_len
        if end > self.max_seq_len:
            raise ValueError(f"sequence end {end} exceeds max_seq_len {self.max_seq_len}")
        cos = self.cos_cached[offset:end]
        sin = self.sin_cached[offset:end]
        return cos, sin
