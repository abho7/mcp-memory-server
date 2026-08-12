"""Persistent semantic memory for Claude, backed by a from-scratch HNSW index.

The vector index itself lives in the sibling `hnsw-engine/` checkout and is
treated as a read-only dependency; everything in this package is the layer
that turns it into a durable, tag-aware memory store with an MCP interface.
"""

__version__ = "0.1.0"
