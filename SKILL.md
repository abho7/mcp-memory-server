---
name: memory-hnsw
description: Persistent semantic memory across sessions, stored in a local HNSW vector index with local embeddings and no external API calls. Use when the user asks you to remember something for later, refers to something decided or discussed in an earlier session, asks what you already know about a topic or project, or when you learn a durable fact (a preference, a project decision, a convention) that would be worth having in a future session. Also use when the user asks to review, list, or forget stored memories.
---

# Persistent memory (HNSW)

Four MCP tools backed by a local vector index. Memories survive restarts;
recall is by meaning, not keyword match.

| Tool | Signature | Returns |
|---|---|---|
| `store_memory` | `(text: str, tags: list[str] = [])` | the new memory's id |
| `search_memory` | `(query: str, k: int = 5)` | k nearest memories, similarity-ranked |
| `list_memories` | `(tag: str = None)` | all memories, newest first, optional tag filter |
| `delete_memory` | `(id: str)` | confirmation |

## When to store

Store facts whose value outlives the current session:

- Decisions and their reasoning — *"We picked Postgres over MySQL because we need JSONB indexing"*
- Stable preferences — *"Prefers short, specific review comments"*
- Project conventions not visible in the code — *"Integration tests need Docker running"*
- Environment and process facts — *"The staging deploy key rotates monthly"*

Do **not** store: anything already in the repo (code structure, git history,
CLAUDE.md), transient state (*"the build is currently failing"*), or secrets
and credentials. The store is plaintext JSON on disk.

Write each memory as a self-contained sentence. `"Use pnpm, not npm"` is
recallable in six months; `"use that instead"` is not.

## When to search

Search before assuming you lack context — at the start of work on a
familiar project, when the user references a past decision, or when a
question sounds like it has an established answer. It is cheap; a miss
returns nothing and costs one call.

Phrase the query as the question you actually have. Embeddings match on
meaning, so `"which database did we choose and why"` finds a memory worded
`"We picked Postgres over MySQL..."` despite sharing almost no words.

## Tags

Tags are free-form, matched case-insensitively, and used only by
`list_memories`. A small stable vocabulary works best: `decision`,
`preference`, `ops`, `testing`, `convention`. `search_memory` ignores tags
entirely — it ranks on the text.

## Reading results

`search_memory` returns a cosine similarity in `[-1, 1]`. Judge hits
relatively, not against a fixed threshold: the top result is the best
available match, and MiniLM scores genuine paraphrase matches around
0.3–0.5 rather than near 1.0. Treat a top score below roughly 0.15 as
probably unrelated, and say so rather than forcing a connection.

Every result carries its `id`, which is what `delete_memory` takes.

## Deleting

Delete when the user says to forget something, or when a stored fact is
superseded. Superseding is two calls — `store_memory` the new fact, then
`delete_memory` the stale one — since there is no update tool. Confirm with
the user before deleting anything they did not explicitly point at.
