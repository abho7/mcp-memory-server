"""Embedding backend: the pooling maths, and (opt-in) the real model.

The pooling tests run everywhere and need no model. The tests that load
real MiniLM weights are skipped unless MCP_MEMORY_TEST_REAL_MODEL=1, since
they download ~90MB on a cold cache.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from mcp_memory.embedder import (
    EMBEDDING_DIM,
    MiniLMEmbedder,
    _mean_pool_and_normalize,
    default_cache_dir,
)

requires_real_model = pytest.mark.skipif(
    os.environ.get("MCP_MEMORY_TEST_REAL_MODEL") != "1",
    reason="set MCP_MEMORY_TEST_REAL_MODEL=1 to run against real MiniLM weights",
)


# -- pooling maths ----------------------------------------------------------


def test_mean_pool_averages_unmasked_tokens_only():
    # Two tokens real, one padding. The padding row is deliberately huge:
    # if it leaked into the average the result would be nowhere near [1,0].
    token_embeddings = np.array([[[2.0, 0.0], [0.0, 0.0], [99.0, 99.0]]])
    attention_mask = np.array([[1, 1, 0]])

    pooled = _mean_pool_and_normalize(token_embeddings, attention_mask)

    assert pooled.shape == (1, 2)
    assert np.allclose(pooled[0], [1.0, 0.0])


def test_mean_pool_output_is_unit_length():
    rng = np.random.default_rng(0)
    token_embeddings = rng.normal(size=(4, 7, 16))
    attention_mask = np.ones((4, 7), dtype=np.int64)

    pooled = _mean_pool_and_normalize(token_embeddings, attention_mask)

    assert np.allclose(np.linalg.norm(pooled, axis=1), 1.0, atol=1e-6)


def test_padding_does_not_change_a_sequences_embedding():
    """Same sentence, batched with a longer one, must embed identically --
    otherwise recall would depend on batch composition."""
    tokens = np.array([[1.0, 2.0], [3.0, 4.0]])

    alone = _mean_pool_and_normalize(tokens[None, ...], np.array([[1, 1]]))
    padded_tokens = np.array([[[1.0, 2.0], [3.0, 4.0], [50.0, -50.0]]])
    padded = _mean_pool_and_normalize(padded_tokens, np.array([[1, 1, 0]]))

    assert np.allclose(alone, padded, atol=1e-6)


def test_mean_pool_survives_an_all_padding_row():
    """A fully masked row must not divide by zero."""
    pooled = _mean_pool_and_normalize(np.ones((1, 3, 4)), np.zeros((1, 3), dtype=np.int64))
    assert np.isfinite(pooled).all()


# -- wiring -----------------------------------------------------------------


def test_encode_of_empty_list_returns_empty_array_without_loading_model():
    embedder = MiniLMEmbedder(allow_download=False)
    out = embedder.encode([])
    assert out.shape == (0, EMBEDDING_DIM)


def test_missing_files_with_downloads_disabled_raises_actionable_error(tmp_path):
    embedder = MiniLMEmbedder(cache_dir=tmp_path, allow_download=False)
    with pytest.raises(FileNotFoundError, match="huggingface.co"):
        embedder.encode(["anything"])


def test_cache_dir_honours_environment_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MCP_MEMORY_MODEL_DIR", str(tmp_path / "models"))
    assert default_cache_dir() == tmp_path / "models"


def test_declared_dim_matches_minilm():
    assert MiniLMEmbedder.dim == EMBEDDING_DIM == 384


# -- the real model ---------------------------------------------------------


@requires_real_model
def test_real_model_produces_normalized_384_dim_vectors():
    embedder = MiniLMEmbedder()
    out = embedder.encode(["hello world", "a second sentence"])

    assert out.shape == (2, 384)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


@requires_real_model
def test_real_model_places_related_sentences_closer_than_unrelated():
    embedder = MiniLMEmbedder()
    vectors = embedder.encode(
        [
            "The cat sat on the mat",
            "A kitten rested on the rug",
            "Quarterly revenue exceeded analyst forecasts",
        ]
    )
    related = float(vectors[0] @ vectors[1])
    unrelated = float(vectors[0] @ vectors[2])

    assert related > unrelated
    assert related > 0.5


@requires_real_model
def test_real_model_is_deterministic_across_calls():
    embedder = MiniLMEmbedder()
    first = embedder.encode(["stability check"])
    second = embedder.encode(["stability check"])
    assert np.allclose(first, second, atol=1e-6)
