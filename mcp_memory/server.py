"""MCP server exposing the HNSW-backed memory store as four tools.

Run it directly (`python -m mcp_memory.server`) and it speaks MCP over
stdio, which is what Claude Code and Claude Desktop launch it as.

The store is created lazily on the first tool call rather than at import.
A first run has to download ~90MB of model weights, and doing that during
module import would stall the MCP handshake long enough for the client to
give up on the server before it ever registers its tools.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp_memory import __version__
from mcp_memory.store import MemoryStore, default_data_dir

try:  # mcp >= 2.0
    from mcp.server import MCPServer as _ServerClass
except ImportError:  # pragma: no cover - mcp 1.x fallback
    from mcp.server.fastmcp import FastMCP as _ServerClass

INSTRUCTIONS = """\
Persistent semantic memory across sessions, backed by a local HNSW vector \
index and a local embedding model (no external API calls).

Use store_memory to save a durable fact worth recalling in a later session \
(preferences, project decisions, context that is not in the code). Use \
search_memory to recall by meaning rather than exact wording. Memories \
survive restarts."""

server = _ServerClass(
    name="memory-hnsw",
    version=__version__,
    instructions=INSTRUCTIONS,
)

_store: MemoryStore | None = None


def get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore(data_dir=default_data_dir())
    return _store


def set_store(store: MemoryStore | None) -> None:
    """Inject a store (used by the tests to avoid loading a real model)."""
    global _store
    _store = store


def _format_memory_line(index: int, memory_dict: dict) -> str:
    tags = memory_dict.get("tags") or []
    tag_str = f" [{', '.join(tags)}]" if tags else ""
    similarity = memory_dict.get("similarity")
    score = f" (similarity {similarity:.3f})" if similarity is not None else ""
    return (
        f"{index}. {memory_dict['text']}{tag_str}{score}\n"
        f"   id={memory_dict['id']}  stored={memory_dict['created_at']}"
    )


@server.tool(
    name="store_memory",
    description=(
        "Save a durable memory. The text is embedded locally and added to the "
        "HNSW index so it can be recalled semantically in any later session. "
        "Optional tags allow filtered listing. Returns the new memory's id."
    ),
)
def store_memory(text: str, tags: list[str] | None = None) -> str:
    store = get_store()
    try:
        memory = store.store(text, tags)
    except ValueError as exc:
        return f"Could not store memory: {exc}"

    tag_str = f" with tags [{', '.join(memory.tags)}]" if memory.tags else ""
    return (
        f"Stored memory {memory.id}{tag_str}.\n"
        f"{len(store)} memories now in the index."
    )


@server.tool(
    name="search_memory",
    description=(
        "Recall the k memories closest in meaning to the query, using vector "
        "similarity rather than keyword matching. Returns them ranked nearest "
        "first with a cosine similarity score in [-1, 1] (1.0 = identical)."
    ),
)
def search_memory(query: str, k: int = 5) -> str:
    store = get_store()
    try:
        hits = store.search(query, k)
    except ValueError as exc:
        return f"Could not search: {exc}"

    if not hits:
        if len(store) == 0:
            return "No memories stored yet."
        return f"No matches for {query!r}."

    lines = [f"{len(hits)} memory match(es) for {query!r}, nearest first:", ""]
    lines += [
        _format_memory_line(i, hit.to_public_dict()) for i, hit in enumerate(hits, 1)
    ]
    return "\n".join(lines)


@server.tool(
    name="list_memories",
    description=(
        "List stored memories, newest first. Pass a tag to return only "
        "memories carrying it (matched case-insensitively)."
    ),
)
def list_memories(tag: str | None = None) -> str:
    store = get_store()
    memories = store.list(tag)

    if not memories:
        if tag:
            known = store.all_tags()
            hint = f" Known tags: {', '.join(known)}." if known else ""
            return f"No memories tagged {tag!r}.{hint}"
        return "No memories stored yet."

    scope = f" tagged {tag!r}" if tag else ""
    lines = [f"{len(memories)} memory(ies){scope}, newest first:", ""]
    lines += [
        _format_memory_line(i, m.to_public_dict()) for i, m in enumerate(memories, 1)
    ]
    return "\n".join(lines)


@server.tool(
    name="delete_memory",
    description=(
        "Delete a memory by its id (as shown by search_memory or "
        "list_memories). The deletion persists across sessions."
    ),
)
def delete_memory(id: str) -> str:
    store = get_store()
    if store.delete(id):
        return f"Deleted memory {id}. {len(store)} memories remain."
    return f"No memory with id {id!r}. Use list_memories to see valid ids."


def main() -> None:
    # Touch the data dir early so a bad MCP_MEMORY_DIR fails loudly at
    # startup rather than on the first store_memory call.
    data_dir = default_data_dir()
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    if os.environ.get("MCP_MEMORY_EAGER_LOAD") == "1":
        get_store()

    server.run("stdio")


if __name__ == "__main__":
    main()
