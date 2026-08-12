"""Locates the hnsw-engine checkout and imports VectorDB from it.

The engine is vendored as a plain git clone rather than a pip package, and
its own modules import each other as top-level names (`from hnsw.index
import HNSWIndex`), so its `src/` directory has to be on sys.path before
anything else can import it. Keeping that in one place means the rest of
the package can just `from mcp_memory.engine import VectorDB`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def engine_src_dir() -> Path:
    """Path to hnsw-engine/src. Override with MCP_MEMORY_ENGINE_DIR."""
    override = os.environ.get("MCP_MEMORY_ENGINE_DIR")
    root = Path(override) if override else _PROJECT_ROOT / "hnsw-engine"
    return root / "src"


def _install_path() -> None:
    src = str(engine_src_dir())
    if src not in sys.path:
        sys.path.insert(0, src)


_install_path()

try:
    from hnsw.index import HNSWIndex  # noqa: E402
    from vectordb.store import VectorDB  # noqa: E402
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    raise ImportError(
        f"Could not import the HNSW engine from {engine_src_dir()}.\n"
        "Clone it first:\n"
        "  git clone https://github.com/abho7/vectordb-hnsw.git ./hnsw-engine\n"
        "or point MCP_MEMORY_ENGINE_DIR at an existing checkout."
    ) from exc

__all__ = ["HNSWIndex", "VectorDB", "engine_src_dir"]
