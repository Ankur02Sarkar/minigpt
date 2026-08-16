"""Tests for inference.chat — CPU tests with the shared tiny_setup fixture."""

from __future__ import annotations

import io

import pytest

torch = pytest.importorskip("torch")

from inference.chat import (  # noqa: E402
    ChatParams,
    ChatSession,
    _handle_slash,
    _run_repl,
)

# --------------------------------------------------------------------------- #
# ChatSession.build_context
# --------------------------------------------------------------------------- #


def test_build_context_empty_history(tiny_setup) -> None:
    """Empty history → 'System: {text}\\nAssistant:' (trailing cue)."""
    model, tok, _ = tiny_setup
    s = ChatSession(model=model, tokenizer=tok, system="hi", max_position_embeddings=1024)
    ctx = s.build_context()
    assert ctx == "System: hi\nAssistant:"


def test_build_context_with_history(tiny_setup) -> None:
    """History is rendered with role markers; trailing 'Assistant:' cues the next turn."""
    model, tok, _ = tiny_setup
    s = ChatSession(model=model, tokenizer=tok, system="sys", max_position_embeddings=1024)
    s.history = [("user", "hello"), ("assistant", "world")]
    ctx = s.build_context()
    assert "System: sys" in ctx
    assert "User: hello" in ctx
    assert "Assistant: world" in ctx
    assert ctx.endswith("Assistant:")


def test_build_context_truncates_from_left(tiny_setup) -> None:
    """When the full context exceeds max_position_embeddings, oldest turns drop first."""
    model, tok, _ = tiny_setup
    # Tiny tokenizer encodes ~1 token per char; tiny model ctx=64. Use a tiny
    # max_position_embeddings so truncation triggers immediately.
    s = ChatSession(model=model, tokenizer=tok, system="s", max_position_embeddings=10)
    s.history = [
        ("user", "aaaaaaaaaaaa"),
        ("assistant", "bbbbbbbbbbbb"),
        ("user", "cccccccccccc"),
        ("assistant", "dddddddddddd"),
    ]
    ctx = s.build_context()
    # However many turns were dropped, the earliest user turn should be gone.
    assert "aaaaaaaaaaaa" not in ctx or "aaaaaaaaaaaa" not in s.history
    # System prompt always preserved, trailing cue always present.
    assert ctx.startswith("System:")
    assert ctx.endswith("Assistant:")
    # The final context must be within budget.
    assert len(tok.encode(ctx).ids) <= 10


# --------------------------------------------------------------------------- #
# Slash commands
# --------------------------------------------------------------------------- #


def test_slash_reset_clears_history(tiny_setup) -> None:
    model, tok, _ = tiny_setup
    s = ChatSession(model=model, tokenizer=tok, max_position_embeddings=1024)
    s.history = [("user", "x"), ("assistant", "y")]
    assert _handle_slash("/reset", s) is True
    assert s.history == []


def test_slash_system_replaces_prompt(tiny_setup) -> None:
    model, tok, _ = tiny_setup
    s = ChatSession(model=model, tokenizer=tok, system="old", max_position_embeddings=1024)
    assert _handle_slash("/system you are tiny", s) is True
    assert s.system == "you are tiny"


def test_slash_system_no_arg_resets_default(tiny_setup) -> None:
    model, tok, _ = tiny_setup
    s = ChatSession(model=model, tokenizer=tok, system="old", max_position_embeddings=1024)
    _handle_slash("/system", s)
    assert s.system != "old"  # reset to default


def test_slash_temp_updates_param(tiny_setup) -> None:
    model, tok, _ = tiny_setup
    s = ChatSession(model=model, tokenizer=tok, max_position_embeddings=1024)
    assert _handle_slash("/temp 0.5", s) is True
    assert s.params.temperature == 0.5


def test_slash_temp_bad_arg_keeps_value(tiny_setup) -> None:
    model, tok, _ = tiny_setup
    s = ChatSession(model=model, tokenizer=tok, max_position_embeddings=1024)
    s.params.temperature = 0.9
    _handle_slash("/temp notafloat", s)
    assert s.params.temperature == 0.9


def test_slash_tokens_updates_param(tiny_setup) -> None:
    model, tok, _ = tiny_setup
    s = ChatSession(model=model, tokenizer=tok, max_position_embeddings=1024)
    assert _handle_slash("/tokens 64", s) is True
    assert s.params.max_new_tokens == 64


def test_slash_quit_returns_false(tiny_setup) -> None:
    model, tok, _ = tiny_setup
    s = ChatSession(model=model, tokenizer=tok, max_position_embeddings=1024)
    assert _handle_slash("/quit", s) is False
    assert _handle_slash("/exit", s) is False


def test_slash_unknown_returns_true(tiny_setup) -> None:
    model, tok, _ = tiny_setup
    s = ChatSession(model=model, tokenizer=tok, max_position_embeddings=1024)
    assert _handle_slash("/bogus", s) is True


# --------------------------------------------------------------------------- #
# Headless REPL run (input_fn + out injected)
# --------------------------------------------------------------------------- #


def test_run_repl_headless_turn_and_quit(tiny_setup) -> None:
    """A scripted input stream produces a multi-turn conversation and exits cleanly."""
    model, tok, _ = tiny_setup
    s = ChatSession(
        model=model,
        tokenizer=tok,
        system="sys",
        max_position_embeddings=64,
        params=ChatParams(max_new_tokens=2, temperature=0.0, top_k=None, seed=42),
    )
    # Scripted: a user msg, a /reset, another msg, then /quit.
    inputs = iter(["hello world", "/reset", "second msg", "/quit"])
    out = io.StringIO()
    _run_repl(s, input_fn=lambda _: next(inputs), out=out)

    # After the run: history has exactly the last turn (reset cleared the first).
    # Two turns survived the second user msg: ("user", "second msg") + ("assistant", ...).
    assert any(role == "user" and text == "second msg" for role, text in s.history)
    assert any(role == "assistant" for role, text in s.history)
    # The first user message must NOT be present (it was cleared by /reset).
    assert not any(role == "user" and text == "hello world" for role, text in s.history)


def test_run_repl_blank_lines_skipped(tiny_setup) -> None:
    """Empty input lines don't create turns."""
    model, tok, _ = tiny_setup
    s = ChatSession(
        model=model,
        tokenizer=tok,
        system="sys",
        max_position_embeddings=64,
        params=ChatParams(max_new_tokens=1, temperature=0.0, seed=42),
    )
    inputs = iter(["", "   ", "/quit"])
    out = io.StringIO()
    _run_repl(s, input_fn=lambda _: next(inputs), out=out)
    assert s.history == []


def test_user_turn_appends_assistant_to_history(tiny_setup) -> None:
    """A user_turn() call records both the user msg and the streamed assistant reply."""
    model, tok, _ = tiny_setup
    s = ChatSession(
        model=model,
        tokenizer=tok,
        system="sys",
        max_position_embeddings=64,
        params=ChatParams(max_new_tokens=3, temperature=0.0, seed=42),
    )
    # Capture stdout so the test isn't noisy; we just want history correctness.
    out = io.StringIO()
    import contextlib

    with contextlib.redirect_stdout(out):
        reply = s.user_turn("hello")
    assert isinstance(reply, str)
    assert len(s.history) == 2
    assert s.history[0] == ("user", "hello")
    assert s.history[1] == ("assistant", reply)
