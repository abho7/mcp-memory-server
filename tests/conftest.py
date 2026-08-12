"""Shared fixtures.

Most tests run against a deterministic stand-in embedder rather than the
real MiniLM model: it keeps the suite fast and offline, and it makes
similarity assertions exact instead of dependent on what a neural net
happens to think. The real model is exercised separately in
test_embedder.py, which is opt-in.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_memory.store import MemoryStore  # noqa: E402

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Bag-of-words hashed into a small vector space, then L2-normalized.

    Shares the two properties the store actually depends on: a fixed
    dimension, and unit-length vectors whose cosine distance tracks word
    overlap -- so texts about the same thing really do land near each other
    and ranking assertions mean something.
    """

    dim = 64
    model_name = "test-hashing-embedder"

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.call_count = 0

    def encode(self, texts):
        texts = list(texts)
        self.call_count += len(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                # hashlib, not hash(): str hashing is salted per process,
                # so built-in hash() would embed the same text differently
                # in a later session and break every persistence test.
                digest = hashlib.sha1(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dim
                out[row, bucket] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-12, None)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "memories"


@pytest.fixture
def store(data_dir: Path, embedder: HashingEmbedder) -> MemoryStore:
    return MemoryStore(data_dir=data_dir, embedder=embedder)


@pytest.fixture
def reopen(data_dir: Path):
    """Open a fresh MemoryStore over the same directory.

    This is the whole point of the project -- a brand new process finding
    what a previous one wrote -- so every persistence test goes through it.
    """

    def _reopen(**kwargs) -> MemoryStore:
        kwargs.setdefault("embedder", HashingEmbedder())
        return MemoryStore(data_dir=data_dir, **kwargs)

    return _reopen
