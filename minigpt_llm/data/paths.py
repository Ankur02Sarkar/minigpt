"""Canonical on-disk layout under the portable Azure data disk."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["DataPaths"]


@dataclass(frozen=True)
class DataPaths:
    """Resolved paths for raw / cleaned / tokenized / vocab artifacts.

    All durable data lives under ``root`` (typically ``/data`` on Azure).
    Never use ``/mnt`` for anything that must survive VM deallocate.
    """

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def cleaned(self) -> Path:
        return self.root / "cleaned"

    @property
    def tokenized(self) -> Path:
        return self.root / "tokenized"

    @property
    def vocab(self) -> Path:
        return self.root / "vocab"

    @property
    def tinystories_raw(self) -> Path:
        return self.raw / "tinystories.txt"

    @property
    def wikitext_raw(self) -> Path:
        return self.raw / "wikitext103.txt"

    @property
    def all_text_cleaned(self) -> Path:
        return self.cleaned / "all_text.txt"

    @property
    def tinystories_bin(self) -> Path:
        return self.tokenized / "tinystories.bin"

    @property
    def wikitext_bin(self) -> Path:
        return self.tokenized / "wikitext.bin"

    @property
    def fineweb_bin(self) -> Path:
        return self.tokenized / "fineweb.bin"

    @property
    def val_bin(self) -> Path:
        return self.tokenized / "val.bin"

    @property
    def meta_pkl(self) -> Path:
        return self.tokenized / "meta.pkl"

    def ensure_dirs(self) -> None:
        """Create the standard subdirectory layout."""
        for d in (self.raw, self.cleaned, self.tokenized, self.vocab):
            d.mkdir(parents=True, exist_ok=True)
