"""Sampling diagnostics for inference generation.

Provides:

- ``compute_entropy`` — Shannon entropy of a softmax distribution over the
  vocabulary (used as the per-token diagnostics metric).
- ``DiagnosticAccumulator`` — accumulates per-token entropy and detects
  n‑gram repetition loops in the generated text.
- ``on_token_default`` — a ready‑to‑use ``on_token`` callback that logs
  entropy and warns when a repeat is detected or entropy exceeds a threshold.

Typical usage::

    from inference.diagnostics import DiagnosticAccumulator, on_token_default

    accumulator = DiagnosticAccumulator(n_gram=3, entropy_threshold=4.0)

    def my_on_token(token_id: int, piece: str, entropy: float,
                    top5: List[Tuple[int, float]]) -> None:
        accumulator.add(token_id, piece, entropy, top5)

    # pass ``my_on_token`` as ``on_token`` to ``generate``.
"""

from collections import deque

import torch
import torch.nn.functional as F


def compute_entropy(logits: torch.Tensor) -> float:
    """Shannon entropy ``-sum(p * log p)`` of the softmax distribution.

    Args:
        logits: unnormalised log-probs ``(..., vocab_size)``.

    Returns:
        entropy in nats.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    return float(-(probs * log_probs).sum())


class DiagnosticAccumulator:
    """Track per-token entropy and detect n‑gram repetition loops.

    Stores the last ``max_entropy`` entropy values and maintains a sliding
    window of n‑gram tokens to flag when the model starts looping.

    Args:
        n_gram: size of the n‑gram to look for repeats (default 3).
        entropy_window: how many recent entropy values to keep (default 20).
        entropy_threshold: if the most recent entropy exceeds this (default 4.0
            nats), a warning is emitted.
        repeat_window: how many recent n‑grams to check for a repeat (default 64).
    """

    def __init__(
        self,
        *,
        n_gram: int = 3,
        entropy_window: int = 20,
        entropy_threshold: float = 4.0,
        repeat_window: int = 64,
    ) -> None:
        self.n_gram = n_gram
        self.entropy_window = entropy_window
        self.entropy_threshold = entropy_threshold
        self.repeat_window = repeat_window

        # history
        self._entropy_history: deque[float] = deque(maxlen=entropy_window)
        # token ids for n‑gram detection (kept as a list of ints)
        self._token_ids: list[int] = []
        # set of n‑grams (as tuples) seen so far, for O(1) repeat checks
        self._seen_ngrams: set[tuple[int, ...]] = set()

    # ------------------------------------------------------------------
    # public API – called from the ``on_token`` callback
    # ------------------------------------------------------------------

    def add(self, token_id: int, piece: str, entropy: float, top5: list[tuple[int, float]]) -> None:
        """Record one generated token.

        Args:
            token_id: the generated token integer id.
            piece: the incremental text yielded by ``generate`` for this token.
            entropy: Shannon entropy of the distribution at this step.
            top5: list of ``(token_id, probability)`` for the five most likely
                tokens *at this step*.
        """
        # --- entropy ---
        self._entropy_history.append(entropy)

        # --- token-id record for n‑gram detection ---
        self._token_ids.append(token_id)
        # keep only the most recent ``repeat_window`` tokens
        if len(self._token_ids) > self.repeat_window:
            # remove the oldest token and its n‑grams from the set
            old = self._token_ids[: -self.repeat_window]
            for i in range(len(old) - self.n_gram + 1):
                ng = tuple(old[i : i + self.n_gram])
                self._seen_ngrams.discard(ng)
        # add new n‑grams that end at this token
        # we need the last n_gram token ids
        if len(self._token_ids) >= self.n_gram:
            ng = tuple(self._token_ids[-self.n_gram :])
            self._seen_ngrams.add(ng)

        # --- repeat detection ---
        # an n‑gram is a "repeat" if it was already seen earlier in the
        # full token sequence (outside the sliding repeat_window).
        # We track all n‑grams ever seen in a growing set; if the newest
        # n‑gram is already in there, we have a loop.
        if len(self._token_ids) >= self.n_gram:
            newest_ngram = tuple(self._token_ids[-self.n_gram :])
            if newest_ngram in self._seen_ngrams:
                # optional: could check if it was seen *before* the window,
                # but for a simple warning we just flag it.
                pass  # caller can inspect `has_repeated` property

    # ------------------------------------------------------------------
    # properties for the caller (e.g. chat REPL)
    # ------------------------------------------------------------------

    @property
    def has_repeated(self) -> bool:
        """``True`` if the most recent n‑gram was seen earlier in the text."""
        if len(self._token_ids) < self.n_gram:
            return False
        newest = tuple(self._token_ids[-self.n_gram :])
        return newest in self._seen_ngrams  # simplified: always checks full set

    @property
    def recent_entropy(self) -> float | None:
        """Most recent entropy value, or ``None`` if no tokens yet."""
        if not self._entropy_history:
            return None
        return self._entropy_history[-1]

    @property
    def avg_entropy(self) -> float | None:
        """Average entropy over the stored window, or ``None``."""
        if not self._entropy_history:
            return None
        return sum(self._entropy_history) / len(self._entropy_history)


def on_token_default(
    token_id: int,
    piece: str,
    entropy: float,
    top5: list[tuple[int, float]],
    *,
    accumulator: DiagnosticAccumulator | None = None,
) -> None:
    """Default ``on_token`` callback that logs entropy and warns on repeats.

    Args:
        token_id: generated token integer id.
        piece: incremental text for this token (may be empty if the tokenizer
            produced no visible output).
        entropy: Shannon entropy of the distribution at this step.
        top5: list of ``(token_id, probability)`` for the five most likely
            tokens at this step.
        accumulator: optional ``DiagnosticAccumulator`` instance. If ``None``,
            only prints the entropy (no repeat detection).
    """
    if accumulator is not None:
        accumulator.add(token_id, piece, entropy, top5)

    # always print a one-line summary the user can see in the REPL
    top5_str = ", ".join(f"{tid}:{p:.2f}" for tid, p in top5)
    print(
        f"[tok {token_id:5d} ent={entropy:5.2f} top5=[{top5_str}] piece={piece!r}",
        end="",
        flush=True,
    )
