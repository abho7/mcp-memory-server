"""Local sentence embeddings via ONNX Runtime.

Uses the all-MiniLM-L6-v2 weights (the sentence-transformers model) exported
to ONNX, so embedding runs fully offline with no API calls and without
pulling in PyTorch -- onnxruntime + tokenizers is roughly 90MB against
~2.5GB for a torch install, and it starts cold in well under a second.

The output is byte-for-byte the same computation sentence-transformers
performs for this model: take the token embeddings from the final layer,
mean-pool them under the attention mask, then L2-normalize. Normalizing
matters downstream: with unit-length vectors, the HNSW index's cosine
distance is `1 - cosine_similarity`, so similarity is just `1 - distance`.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_SEQ_LENGTH = 256

_HF_BASE = f"https://huggingface.co/{MODEL_REPO}/resolve/main"
_FILES = {
    "model.onnx": f"{_HF_BASE}/onnx/model.onnx",
    "tokenizer.json": f"{_HF_BASE}/tokenizer.json",
}


def default_cache_dir() -> Path:
    """Where model weights get cached. Override with MCP_MEMORY_MODEL_DIR."""
    override = os.environ.get("MCP_MEMORY_MODEL_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "mcp-memory-hnsw" / MODEL_NAME


class Embedder(Protocol):
    """The only thing MemoryStore needs from an embedding backend.

    Kept as a Protocol so tests can substitute a cheap deterministic stub
    instead of loading a real model for every case.
    """

    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (len(texts), dim) float32 array of unit-length vectors."""
        ...


class MiniLMEmbedder:
    """all-MiniLM-L6-v2 running under ONNX Runtime.

    Model files are downloaded once on first use and cached on disk; every
    later session loads from cache and never touches the network.
    """

    dim = EMBEDDING_DIM
    model_name = MODEL_NAME

    def __init__(self, cache_dir: Path | str | None = None, *, allow_download: bool = True):
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
        self._allow_download = allow_download
        self._session = None
        self._tokenizer = None
        self._input_names: set[str] = set()

    # -- model files ----------------------------------------------------

    def _ensure_files(self) -> dict[str, Path]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for filename, url in _FILES.items():
            target = self.cache_dir / filename
            if not target.exists() or target.stat().st_size == 0:
                if not self._allow_download:
                    raise FileNotFoundError(
                        f"{target} is missing and downloads are disabled. "
                        f"Fetch it manually from {url}"
                    )
                self._download(url, target)
            paths[filename] = target
        return paths

    @staticmethod
    def _download(url: str, target: Path) -> None:
        # Download to a temp file and rename, so an interrupted download can
        # never leave a truncated file that looks valid on the next run.
        tmp = target.with_suffix(target.suffix + ".partial")
        req = urllib.request.Request(url, headers={"User-Agent": "mcp-memory-hnsw/0.1"})
        with urllib.request.urlopen(req) as response, open(tmp, "wb") as fh:
            while chunk := response.read(1 << 20):
                fh.write(chunk)
        os.replace(tmp, target)

    # -- lazy init ------------------------------------------------------

    def _load(self) -> None:
        if self._session is not None:
            return

        import onnxruntime as ort
        from tokenizers import Tokenizer

        paths = self._ensure_files()

        tokenizer = Tokenizer.from_file(str(paths["tokenizer.json"]))
        tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
        tokenizer.enable_padding()
        self._tokenizer = tokenizer

        options = ort.SessionOptions()
        # An MCP server is not the right place to saturate every core; one
        # intra-op thread keeps latency predictable for the short, single
        # sentences this store actually embeds.
        options.intra_op_num_threads = 1
        options.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(paths["model.onnx"]), options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}

    # -- encoding -------------------------------------------------------

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        self._load()
        assert self._tokenizer is not None and self._session is not None

        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        # Some exports of this model drop token_type_ids; only feed what the
        # graph actually declares.
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}

        token_embeddings = self._session.run(None, feeds)[0]
        return _mean_pool_and_normalize(token_embeddings, attention_mask)


def _mean_pool_and_normalize(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean-pool (B, T, H) token embeddings under the mask, then L2-normalize.

    Padding tokens carry real activations, so averaging without the mask
    would let sequence length leak into the embedding -- two identical
    sentences batched with different neighbours would embed differently.
    """
    mask = attention_mask[..., None].astype(np.float32)
    summed = (token_embeddings.astype(np.float32) * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    pooled = summed / counts

    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return (pooled / np.clip(norms, 1e-12, None)).astype(np.float32)
