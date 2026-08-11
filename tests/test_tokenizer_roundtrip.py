"""Train a tiny BPE on a fixture and verify round-trip."""

from __future__ import annotations

from pathlib import Path

from minigpt_llm.data.paths import DataPaths
from minigpt_llm.tokenizer.load import load_tokenizer, make_encode_batch
from minigpt_llm.tokenizer.train_bpe import train_bpe


def test_train_bpe_roundtrip(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path)
    paths.ensure_dirs()
    # Enough varied text for a tiny BPE
    lines = [f"The quick brown fox jumps over the lazy dog number {i}." for i in range(200)] + [
        f"Once upon a time there was a little story about item {i}." for i in range(200)
    ]
    paths.all_text_cleaned.write_text("\n".join(lines) + "\n", encoding="utf-8")

    stats = train_bpe(paths, vocab_size=500, min_frequency=2, force=True)
    assert stats["vocab_size"] >= 100
    assert (paths.vocab / "vocab.json").is_file()
    assert (paths.vocab / "merges.txt").is_file()
    assert stats["roundtrip_rate"] >= 0.95

    tok = load_tokenizer(paths)
    encode_fn = make_encode_batch(tok)
    batch_ids = encode_fn(["The quick brown fox jumps over the lazy dog number 1."])
    assert len(batch_ids) == 1
    assert len(batch_ids[0]) > 0
