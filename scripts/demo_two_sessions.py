#!/usr/bin/env python3
"""Demo: a memory stored in one process, recalled by a different one.

Two separate OS processes with no shared state but the data directory --
which is exactly what "session 1" and "session 2" of Claude Code are.

    python scripts/demo_two_sessions.py write    # session 1, then exits
    python scripts/demo_two_sessions.py read     # session 2, fresh process

Add --reset to the write step to start from an empty store.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_memory.store import MemoryStore  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parent.parent / ".demo-memory"

MEMORIES = [
    ("We picked Postgres over MySQL because we need JSONB indexing", ["decision", "db"]),
    ("The staging deploy key rotates on the first of every month", ["ops"]),
    ("Ali prefers code review comments to be specific and short", ["preference"]),
    ("Integration tests live in tests/integration and need Docker running", ["testing"]),
]

QUERIES = [
    "which database did we choose and why",
    "how should I write review feedback",
    "what do I need installed to run the slower tests",
]


def _open_store() -> tuple[MemoryStore, float]:
    started = time.perf_counter()
    store = MemoryStore(data_dir=DEMO_DIR)
    return store, time.perf_counter() - started


def write_session(reset: bool) -> None:
    if reset and DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)

    print("=== SESSION 1 (process A) ===\n")
    store, elapsed = _open_store()
    print(f"Opened store at {DEMO_DIR} in {elapsed:.2f}s ({len(store)} existing)\n")

    for text, tags in MEMORIES:
        memory = store.store(text, tags)
        print(f"  stored {memory.id}  [{', '.join(memory.tags)}]  {memory.text}")

    print(f"\n{len(store)} memories written to disk.")
    print("Session 1 now exits. Nothing stays in RAM.\n")
    print("Next:  python scripts/demo_two_sessions.py read")


def read_session() -> None:
    print("=== SESSION 2 (process B, started fresh) ===\n")
    if not DEMO_DIR.exists():
        print("No demo store yet. Run the write step first.")
        raise SystemExit(1)

    store, elapsed = _open_store()
    load_path = "rebuilt from text" if store.rebuilt_on_load else "restored from snapshot"
    print(f"Loaded {len(store)} memories in {elapsed:.2f}s ({load_path})\n")

    for query in QUERIES:
        print(f'  query: "{query}"')
        for hit in store.search(query, k=1):
            print(f'    -> [{hit.similarity:.3f}] {hit.memory.text}')
        print()

    print("Note none of those queries share their wording with the stored text --")
    print("the match is semantic, from the local MiniLM embeddings.\n")

    print("Tag filter (list_memories tag='ops'):")
    for memory in store.list("ops"):
        print(f"  {memory.id}  {memory.text}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["write", "read"])
    parser.add_argument("--reset", action="store_true", help="clear the demo store first")
    args = parser.parse_args()

    if args.step == "write":
        write_session(args.reset)
    else:
        read_session()


if __name__ == "__main__":
    main()
