# mcp-memory-hnsw

[![tests](https://github.com/abho7/mcp-memory-server/actions/workflows/tests.yml/badge.svg)](https://github.com/abho7/mcp-memory-server/actions/workflows/tests.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Persistent semantic memory for Claude Code and Claude Desktop, backed by a
from-scratch HNSW vector index and a local embedding model. Nothing leaves
the machine: no embedding API, no hosted vector database, no network calls
after the model is cached once.

Claude gets four tools — `store_memory`, `search_memory`, `list_memories`,
`delete_memory` — and memories written in one session are recallable in
every later one.

## How it fits together

```
  Claude Code / Desktop
          │  MCP over stdio
  ┌───────▼─────────────────────────────────────┐
  │ mcp_memory/server.py    four MCP tools      │
  │ mcp_memory/store.py     persistence, tags   │
  │ mcp_memory/embedder.py  MiniLM via ONNX     │
  └───────┬─────────────────────────────────────┘
          │  imports, never modifies
  ┌───────▼─────────────────────────────────────┐
  │ hnsw-engine/   VectorDB + HNSWIndex         │
  │                (github.com/abho7/vectordb-hnsw) │
  └─────────────────────────────────────────────┘
          │
     ~/.claude/memory-hnsw/
       memories.json   source of truth: text, tags, ids
       index.npz       graph snapshot + vectors
```

`hnsw-engine/` is a plain checkout, treated as a read-only dependency. Two
things it does not provide — persistence and enumeration — are supplied by
`store.py` from the outside, so the engine can be `git pull`ed without
merge conflicts.

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/abho7/vectordb-hnsw.git ./hnsw-engine
pip install -r requirements.txt
pytest                     # 63 tests
```

Then register the server with Claude Code (`-s user` makes it available in
every project, not just this one):

```bash
claude mcp add memory-hnsw -s user -- /absolute/path/to/python /absolute/path/to/mcp-memory-server/run_server.py
```

Point it at `run_server.py` rather than `-m mcp_memory.server`: the CLI's
own option parser claims the `-m` and rejects the command. The launcher
also puts the package on `sys.path` itself, so the server does not depend
on the client's working directory. Check it came up with `claude mcp list`.

For Claude Desktop, add it to `claude_desktop_config.json` (on Windows,
`%APPDATA%\Claude\`; on macOS, `~/Library/Application Support/Claude/`):

```json
{
  "mcpServers": {
    "memory-hnsw": {
      "command": "/absolute/path/to/python",
      "args": ["/absolute/path/to/mcp-memory-server/run_server.py"]
    }
  }
}
```

Use an absolute path to the Python executable in both cases. Desktop and
Claude Code do not inherit your shell's `PATH`, and a bare `python` that
resolves in your terminal is the most common reason a server shows up
dead. On Windows, escape backslashes in the JSON (`C:\\Users\\...`).

Newly added servers are picked up at startup, so restart Claude Code or
Claude Desktop before the tools appear.

Copy `SKILL.md` into `.claude/skills/memory-hnsw/SKILL.md` (project) or
`~/.claude/skills/memory-hnsw/SKILL.md` (global) so Claude knows when to
reach for the tools rather than only what they do.

The first `store_memory` call downloads ~90MB of MiniLM weights to
`~/.cache/mcp-memory-hnsw/`. Every run after that is fully offline.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `MCP_MEMORY_DIR` | `~/.claude/memory-hnsw` | where memories are stored |
| `MCP_MEMORY_MODEL_DIR` | `~/.cache/mcp-memory-hnsw/all-MiniLM-L6-v2` | model weight cache |
| `MCP_MEMORY_ENGINE_DIR` | `./hnsw-engine` | path to the engine checkout |
| `MCP_MEMORY_EAGER_LOAD` | unset | `1` loads the model at startup instead of on first use |

## Demo: recall across two sessions

Two separate OS processes sharing nothing but a directory on disk — which
is what two Claude sessions are. This is real captured output.

**Session 1** writes and exits:

```console
$ python scripts/demo_two_sessions.py write --reset
=== SESSION 1 (process A) ===

Opened store at C:\Users\satya\mcp-memory-server\.demo-memory in 0.00s (0 existing)

  stored 685814f899be  [decision, db]  We picked Postgres over MySQL because we need JSONB indexing
  stored eb48e4499d68  [ops]  The staging deploy key rotates on the first of every month
  stored 6f1dbc3c81bf  [preference]  Ali prefers code review comments to be specific and short
  stored 4ba2f6b59477  [testing]  Integration tests live in tests/integration and need Docker running

4 memories written to disk.
Session 1 now exits. Nothing stays in RAM.
```

**Session 2** is a brand new process:

```console
$ python scripts/demo_two_sessions.py read
=== SESSION 2 (process B, started fresh) ===

Loaded 4 memories in 0.13s (restored from snapshot)

  query: "which database did we choose and why"
    -> [0.393] We picked Postgres over MySQL because we need JSONB indexing

  query: "how should I write review feedback"
    -> [0.375] Ali prefers code review comments to be specific and short

  query: "what do I need installed to run the slower tests"
    -> [0.378] Integration tests live in tests/integration and need Docker running

Tag filter (list_memories tag='ops'):
  eb48e4499d68  The staging deploy key rotates on the first of every month
```

None of those queries share meaningful wording with the text they matched —
"which database did we choose" against "We picked Postgres over MySQL", "run
the slower tests" against "Integration tests ... need Docker". That is the
point of embedding rather than grepping.

In Claude, the same exchange looks like:

> **Session 1** — "Remember that we went with Postgres over MySQL for JSONB indexing."
> → *`store_memory` → Stored memory 685814f899be with tags [decision, db].*
>
> **Session 2, next day** — "Remind me which database we settled on?"
> → *`search_memory("which database did we settle on")` → We picked Postgres over MySQL because we need JSONB indexing*

## Design notes

**Persistence, given an engine that has none.** `hnsw-engine`'s
ARCHITECTURE.md lists in-memory-only as a known limitation: every restart
rebuilds from scratch. Rebuilding is not viable at MCP startup — the engine's
published build cost is 5.1s for 500 vectors at dim=32, and MiniLM's 384
dimensions make each distance computation over ten times more expensive.

So the store snapshots the built graph (vectors, per-layer adjacency in CSR
form, entry point) to `index.npz` and restores it directly. Measured on this
machine with 200 memories:

| | time |
|---|---|
| load from snapshot | 0.08s |
| load by rebuilding | 4.38s |
| warm query | 5ms |
| first query (includes model load) | 0.48s |

Restoring means writing into the engine's private attributes, which would
normally be fragile. It is safe here because it is guarded: `memories.json`
is the source of truth, the snapshot carries a fingerprint of exactly which
entries it should contain, and every structural assumption is checked before
use. Any mismatch — stale snapshot, corrupt file, changed engine layout —
falls back to rebuilding from the stored text. The worst case is a slow
startup, never a wrong or lost memory. Both paths are tested, including a
test that they return identical search results.

**Rebuilding is also the compaction pass.** The engine soft-deletes: a
deleted node stays in the graph and is filtered out of results, so the index
only ever grows. Its docs list compaction as future work. A rebuild here
drops tombstones, so deleting a lot and forcing one rebuild reclaims the
space.

**IDs are never recycled.** Re-inserting an id the engine has soft-deleted
leaves the old vector live in the graph under that same id, which surfaces
as a duplicate hit carrying stale text. (The engine's own
`test_reinsert_after_delete_succeeds` passes because it only asks for `k=1`.)
Every `store_memory` mints a fresh id, so the situation never arises.

**ONNX instead of PyTorch.** Same all-MiniLM-L6-v2 weights and the same
mean-pool-then-normalize computation sentence-transformers performs, run
under onnxruntime: ~90MB installed against ~2.5GB, and a cold start well
under a second. Embeddings are unit length, so the engine's cosine distance
is exactly `1 - similarity`.

**Similarity scores read low.** MiniLM puts genuine paraphrase matches
around 0.3–0.5, not 0.9. Rank matters; the absolute number does not.
`SKILL.md` tells Claude to read them relatively.

## Tests

```bash
pytest                                        # 63 tests, no model needed
MCP_MEMORY_TEST_REAL_MODEL=1 pytest           # + 3 against real MiniLM weights
```

Most tests use a deterministic hashing embedder rather than the neural
model: fast, offline, and it makes ranking assertions exact instead of
dependent on what a neural net happens to think.

| File | Covers |
|---|---|
| `test_store.py` | CRUD, tags, snapshot round-trip, all three rebuild-fallback triggers, compaction, id recycling, graph-structure fidelity, both load paths agreeing |
| `test_tools.py` | each of the four tools, MCP registration and schemas, dispatch through `server.call_tool`, cross-session recall through the tools |
| `test_embedder.py` | mean-pooling under the attention mask, padding invariance, unit length; real-model tests are opt-in |

The engine's own 28 tests still run from `hnsw-engine/` (`cd hnsw-engine &&
pytest`) and are unaffected — nothing in that directory was modified.

## Limitations

- **No update tool.** Superseding a memory is store-then-delete.
- **Tag filtering happens after search**, not inside it, so `list_memories`
  filters by tag but `search_memory` ranks on text alone.
- **Single writer.** Two servers pointed at one data directory will clobber
  each other's writes; each write rewrites both files atomically, but there
  is no cross-process lock.
- **Plaintext on disk.** `memories.json` is human-readable and unencrypted.
  Don't store secrets in it.
- **Everything in RAM while running.** Fine for thousands of memories at
  384 dimensions; this is not a billion-scale store.
