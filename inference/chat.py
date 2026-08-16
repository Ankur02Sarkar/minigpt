"""Interactive chat REPL — multi-turn prompts with KV-cache streaming generation.

The model is a base LM (no chat template, not instruction-tuned), so multi-turn
history is formatted via plain-text concatenation with role markers
(``System:``/``User:``/``Assistant:``) that give the base LM a fighting chance
to imitate dialogue. History is kept in memory per session; ``generate()`` is
called with ``stream=True`` so produced tokens flush to stdout incrementally.

Usage::

    python -m inference.chat --checkpoint best.pt --tokenizer-dir vocab/ \
        --system "You are a helpful assistant." --temperature 0.8

Slash commands (typed at the ``> `` prompt): ``/reset`` ``/system <text>``
``/temp <f>`` ``/tokens <n>`` ``/help`` ``/quit`` ``/exit``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from inference.generate import generate, load_model
from minigpt_llm.tokenizer.load import load_tokenizer

__all__ = ["ChatSession", "main"]

_SYSTEM_DEFAULT = "You are a helpful assistant."
_SYSTEM_MARKER = "System:"
_USER_MARKER = "User:"
_ASSISTANT_MARKER = "Assistant:"
# Separator turns use a single space + trailing marker so the model continues
# the assistant turn in place; a final trailing marker avoids a leading space
# bleed in the generated text.
_TURN_SEP = "\n"


@dataclass
class ChatParams:
    """Mutable live parameters a user can change mid-session via slash commands."""

    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int | None = 50
    top_p: float | None = None
    seed: int | None = 42


@dataclass
class ChatSession:
    """In-memory multi-turn chat state for a base LM (no chat template).

    History is a list of ``(role, text)`` tuples; ``build_context()`` formats
    it into a single prompt string with role markers, truncated from the left
    if it exceeds the model's ``max_position_embeddings`` (so the most recent
    turns fit).
    """

    model: torch.nn.Module
    tokenizer: Any
    system: str = _SYSTEM_DEFAULT
    max_position_embeddings: int = 1024
    history: list[tuple[str, str]] = field(default_factory=list)
    params: ChatParams = field(default_factory=ChatParams)

    def reset(self) -> None:
        """Clear the history (the system prompt is preserved)."""
        self.history.clear()

    def build_context(self) -> str:
        """Format system prompt + history into a single prompt for the base LM.

        Layout::

            System: {system}
            User: {u1}
            Assistant: {a1}
            User: {u2}
            Assistant:

        The trailing ``Assistant:`` (no text) cues the model to continue the
        assistant turn in place. If the full string exceeds the model context
        window, oldest user/assistant turns are dropped from the front until
        it fits (the system prompt always stays).
        """
        lines: list[str] = [f"{_SYSTEM_MARKER} {self.system}"]
        for role, text in self.history:
            lines.append(f"{_role_marker(role)} {text}")
        lines.append(f"{_ASSISTANT_MARKER}:")  # trailing cue, no leading space
        full = _TURN_SEP.join(lines)
        # Truncate from the left (drop oldest turns) if too long. We always keep
        # the system line + the trailing marker line.
        while self._token_len(full) > self.max_position_embeddings and len(self.history) > 0:
            self.history.pop(0)  # drop oldest (role, text) pair
            lines = [f"{_SYSTEM_MARKER} {self.system}"]
            for role, text in self.history:
                lines.append(f"{_role_marker(role)} {text}")
            lines.append(f"{_ASSISTANT_MARKER}:")
            full = _TURN_SEP.join(lines)
        return full

    def _token_len(self, text: str) -> int:
        return len(self.tokenizer.encode(text).ids)

    def user_turn(self, text: str, *, on_token: Any = None) -> str:
        """Append a user message, stream the assistant reply, persist it.

        Returns the generated assistant text (prompt excluded). The streamed
        pieces are written to stdout as they arrive so the user sees live
        output. ``on_token`` is an optional per-step callback forwarded to
        ``generate()`` (used by the Phase 5.3 diagnostics hook).
        """
        self.history.append(("user", text))
        context = self.build_context()
        pieces: list[str] = []
        gen = generate(
            self.model,
            self.tokenizer,
            context,
            max_new_tokens=self.params.max_new_tokens,
            temperature=self.params.temperature,
            top_k=self.params.top_k,
            top_p=self.params.top_p,
            stop_strings=[_TURN_SEP + _USER_MARKER],
            stream=True,
            device=next(self.model.parameters()).device,
            seed=self.params.seed,
        )
        for piece in gen:
            pieces.append(piece)
            sys.stdout.write(piece)
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
        assistant_text = "".join(pieces)
        self.history.append(("assistant", assistant_text))
        return assistant_text


def _role_marker(role: str) -> str:
    return {"user": _USER_MARKER, "assistant": _ASSISTANT_MARKER}[role]


_SLASH_HELP = """\
Slash commands:
  /reset            clear history (keep system prompt)
  /system <text>    set / replace the system prompt
  /temp <float>     set temperature for subsequent turns
  /tokens <int>     set max_new_tokens for subsequent turns
  /help             show this help
  /quit | /exit     quit
"""


def _handle_slash(line: str, session: ChatSession) -> bool:
    """Handle a slash command. Returns True if the REPL should continue, False to quit."""
    parts = line.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""
    if cmd == "/reset":
        session.reset()
        print("(history cleared)")
    elif cmd == "/system":
        session.system = arg.strip() or _SYSTEM_DEFAULT
        print("(system prompt set)")
    elif cmd == "/temp":
        try:
            session.params.temperature = float(arg)
            print(f"(temperature = {session.params.temperature})")
        except ValueError:
            print("(usage: /temp <float>)")
    elif cmd == "/tokens":
        try:
            session.params.max_new_tokens = int(arg)
            print(f"(max_new_tokens = {session.params.max_new_tokens})")
        except ValueError:
            print("(usage: /tokens <int>)")
    elif cmd == "/help":
        print(_SLASH_HELP)
    elif cmd in ("/quit", "/exit"):
        return False
    else:
        print(f"(unknown command: {cmd}; try /help)")
    return True


def _run_repl(session: ChatSession, *, input_fn=input, out=sys.stdout) -> None:
    """Read-eval-print loop. ``input_fn`` is injectable for headless tests."""
    print(_SLASH_HELP, file=out)
    while True:
        try:
            line = input_fn("> ")
        except (EOFError, KeyboardInterrupt):
            print("", file=out)
            return
        if not line.strip():
            continue
        if line.startswith("/"):
            if not _handle_slash(line, session):
                return
            continue
        session.user_turn(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--system", type=str, default=_SYSTEM_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, step = load_model(args.checkpoint, device)
    tok = load_tokenizer(args.tokenizer_dir)
    max_pos = getattr(model.config, "max_position_embeddings", 1024)
    session = ChatSession(
        model=model,
        tokenizer=tok,
        system=args.system,
        max_position_embeddings=max_pos,
        params=ChatParams(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k and args.top_k > 0 else None,
            top_p=args.top_p,
            seed=args.seed,
        ),
    )
    print(
        f"# checkpoint={args.checkpoint.name} step={step} "
        f"ctx={max_pos} temp={args.temperature} top_k={args.top_k}\n",
        file=sys.stderr,
    )
    _run_repl(session)


if __name__ == "__main__":
    main()
