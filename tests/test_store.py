"""MemoryStore: CRUD behaviour, and the persistence layer the engine lacks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mcp_memory.store import INDEX_FILE, MEMORIES_FILE, MemoryStore, normalize_tags


# -- storing and searching --------------------------------------------------


def test_store_returns_memory_with_id_and_timestamp(store):
    memory = store.store("The deploy script lives in ops/deploy.sh", ["ops"])
    assert memory.id
    assert memory.text == "The deploy script lives in ops/deploy.sh"
    assert memory.tags == ["ops"]
    assert memory.created_at
    assert len(store) == 1


def test_store_rejects_empty_text(store):
    for blank in ("", "   ", "\n\t"):
        with pytest.raises(ValueError, match="empty memory"):
            store.store(blank)
    assert len(store) == 0


def test_store_strips_surrounding_whitespace(store):
    memory = store.store("  padded text  ")
    assert memory.text == "padded text"


def test_search_ranks_semantically_closest_first(store):
    store.store("Postgres connection pooling is handled by pgbouncer")
    store.store("The office coffee machine is on the third floor")
    store.store("Database connection limits are configured in postgres config")

    hits = store.search("postgres database connection", k=3)

    assert len(hits) == 3
    top_two = {hit.memory.text for hit in hits[:2]}
    assert "The office coffee machine is on the third floor" not in top_two
    # Ranked nearest-first, so scores must be non-increasing.
    assert [h.similarity for h in hits] == sorted(
        (h.similarity for h in hits), reverse=True
    )


def test_search_similarity_is_one_for_exact_text(store):
    store.store("exact match target")
    hit = store.search("exact match target", k=1)[0]
    assert hit.similarity == pytest.approx(1.0, abs=1e-6)


def test_search_on_empty_store_returns_nothing(store):
    assert store.search("anything", k=5) == []


def test_search_k_larger_than_store_returns_all(store):
    store.store("one")
    store.store("two")
    assert len(store.search("one", k=50)) == 2


def test_search_rejects_bad_input(store):
    store.store("something")
    with pytest.raises(ValueError, match="empty query"):
        store.search("  ")
    with pytest.raises(ValueError, match="k must be at least 1"):
        store.search("something", k=0)


# -- listing ---------------------------------------------------------------


def test_list_returns_newest_first(store):
    first = store.store("first memory")
    second = store.store("second memory")
    third = store.store("third memory")

    listed = [m.id for m in store.list()]
    assert listed == [third.id, second.id, first.id]


def test_list_filters_by_tag(store):
    store.store("uses tabs not spaces", ["style"])
    store.store("deploys on fridays are banned", ["ops", "policy"])
    store.store("untagged note")

    assert [m.text for m in store.list("style")] == ["uses tabs not spaces"]
    assert [m.text for m in store.list("ops")] == ["deploys on fridays are banned"]
    assert len(store.list()) == 3


def test_list_tag_match_is_case_insensitive(store):
    store.store("tagged memory", ["Ops"])
    assert len(store.list("ops")) == 1
    assert len(store.list("OPS")) == 1


def test_list_unknown_tag_returns_empty(store):
    store.store("tagged memory", ["ops"])
    assert store.list("nonexistent") == []


def test_tags_are_normalized(store):
    memory = store.store("note", ["  ops  ", "", "OPS", "policy"])
    # Trimmed, blanks dropped, case-insensitive duplicate removed, order kept.
    assert memory.tags == ["ops", "policy"]


def test_normalize_tags_handles_none():
    assert normalize_tags(None) == []
    assert normalize_tags([]) == []


def test_all_tags_lists_tags_in_use(store):
    store.store("a", ["ops", "style"])
    store.store("b", ["policy"])
    assert store.all_tags() == ["ops", "policy", "style"]


# -- deleting ---------------------------------------------------------------


def test_delete_removes_from_list_and_search(store):
    keep = store.store("keep this memory about kubernetes")
    drop = store.store("drop this memory about kubernetes")

    assert store.delete(drop.id) is True

    assert [m.id for m in store.list()] == [keep.id]
    assert len(store) == 1
    hit_ids = {h.memory.id for h in store.search("kubernetes memory", k=10)}
    assert drop.id not in hit_ids
    assert keep.id in hit_ids


def test_delete_unknown_id_returns_false(store):
    assert store.delete("does-not-exist") is False


def test_delete_is_not_repeatable(store):
    memory = store.store("transient")
    assert store.delete(memory.id) is True
    assert store.delete(memory.id) is False


def test_get_returns_none_after_delete(store):
    memory = store.store("transient")
    store.delete(memory.id)
    assert store.get(memory.id) is None


# -- persistence: the cross-session guarantee -------------------------------


def test_memory_survives_reopen(store, reopen):
    stored = store.store("The staging database password rotates every 90 days", ["ops"])

    revived = reopen()

    assert len(revived) == 1
    recalled = revived.search("how often does the staging db password change", k=1)
    assert recalled[0].memory.id == stored.id
    assert recalled[0].memory.text == stored.text
    assert recalled[0].memory.tags == ["ops"]


def test_reopen_uses_snapshot_not_rebuild(store, reopen):
    store.store("a memory")
    store.store("another memory")

    revived = reopen()

    assert revived.rebuilt_on_load is False
    assert len(revived) == 2


def test_deletes_survive_reopen(store, reopen):
    keep = store.store("keep me")
    drop = store.store("drop me")
    store.delete(drop.id)

    revived = reopen()

    assert [m.id for m in revived.list()] == [keep.id]
    assert revived.get(drop.id) is None
    assert drop.id not in {h.memory.id for h in revived.search("drop me", k=10)}


def test_writes_are_visible_without_explicit_save(data_dir, embedder):
    store = MemoryStore(data_dir=data_dir, embedder=embedder)
    store.store("autosaved")
    assert (data_dir / MEMORIES_FILE).exists()
    assert (data_dir / INDEX_FILE).exists()


def test_empty_store_reopens_cleanly(data_dir, embedder, reopen):
    MemoryStore(data_dir=data_dir, embedder=embedder)
    revived = reopen()
    assert len(revived) == 0
    assert revived.list() == []


def test_many_memories_round_trip(store, reopen):
    texts = [f"memory number {i} about topic {i % 7}" for i in range(60)]
    for text in texts:
        store.store(text)

    revived = reopen()

    assert len(revived) == 60
    assert {m.text for m in revived.list()} == set(texts)
    assert revived.rebuilt_on_load is False


# -- persistence: the rebuild fallback --------------------------------------


def test_missing_snapshot_falls_back_to_rebuild(store, reopen, data_dir):
    stored = store.store("rebuild me from the source of truth", ["tag"])

    (data_dir / INDEX_FILE).unlink()
    revived = reopen()

    assert revived.rebuilt_on_load is True
    assert len(revived) == 1
    hit = revived.search("rebuild me from the source of truth", k=1)[0]
    assert hit.memory.id == stored.id
    assert hit.memory.tags == ["tag"]


def test_corrupt_snapshot_falls_back_to_rebuild(store, reopen, data_dir):
    store.store("survives a corrupt index file")

    (data_dir / INDEX_FILE).write_bytes(b"this is not a valid npz archive")
    revived = reopen()

    assert revived.rebuilt_on_load is True
    assert len(revived) == 1
    assert revived.search("survives a corrupt index file", k=1)[0].similarity > 0.99


def test_snapshot_disagreeing_with_manifest_forces_rebuild(store, reopen, data_dir):
    """A stale snapshot must never win over memories.json."""
    store.store("first")
    snapshot_after_first = (data_dir / INDEX_FILE).read_bytes()

    store.store("second")
    # Put the one-entry snapshot back alongside the two-entry manifest.
    (data_dir / INDEX_FILE).write_bytes(snapshot_after_first)

    revived = reopen()

    assert revived.rebuilt_on_load is True
    assert len(revived) == 2
    assert {m.text for m in revived.list()} == {"first", "second"}


def test_rebuild_compacts_tombstones(store, reopen, data_dir):
    """The engine soft-deletes and never reclaims; a rebuild is the
    compaction pass that does."""
    keep = store.store("keep")
    for i in range(5):
        doomed = store.store(f"delete {i}")
        store.delete(doomed.id)

    manifest = json.loads((data_dir / MEMORIES_FILE).read_text(encoding="utf-8"))
    assert len(manifest["tombstones"]) == 5

    (data_dir / INDEX_FILE).unlink()  # force the rebuild path
    revived = reopen()

    assert revived.rebuilt_on_load is True
    compacted = json.loads((data_dir / MEMORIES_FILE).read_text(encoding="utf-8"))
    assert compacted["tombstones"] == []
    assert [m.id for m in revived.list()] == [keep.id]
    assert len(revived._db._index.vectors) == 1


def test_unreadable_manifest_raises_rather_than_silently_resetting(data_dir, embedder):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / MEMORIES_FILE).write_text("{not valid json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="source of truth"):
        MemoryStore(data_dir=data_dir, embedder=embedder)


def test_dimension_change_is_refused(store, data_dir):
    """Vectors from two different embedding models are not comparable, so
    reopening with a different model must fail loudly, not mix them."""
    store.store("stored with a 64-dim embedder")

    from tests.conftest import HashingEmbedder

    with pytest.raises(ValueError, match="not comparable"):
        MemoryStore(data_dir=data_dir, embedder=HashingEmbedder(dim=32))


# -- persistence: durability details ----------------------------------------


def test_saving_leaves_no_temp_files_behind(store, data_dir):
    store.store("one")
    store.store("two")
    store.delete(store.list()[0].id)

    leftovers = [p.name for p in data_dir.iterdir() if ".tmp" in p.name]
    assert leftovers == []
    assert sorted(p.name for p in data_dir.iterdir()) == [INDEX_FILE, MEMORIES_FILE]


def test_manifest_is_human_readable_json(store, data_dir):
    store.store("readable", ["tag"])
    manifest = json.loads((data_dir / MEMORIES_FILE).read_text(encoding="utf-8"))

    assert manifest["version"] == 1
    assert manifest["dim"] == 64
    assert manifest["metric"] == "cosine"
    assert len(manifest["memories"]) == 1
    assert manifest["memories"][0]["text"] == "readable"
    assert manifest["memories"][0]["tags"] == ["tag"]


def test_internal_ids_are_never_recycled(store, reopen):
    """Reusing an id the engine has soft-deleted leaves the old vector live
    in the graph under that id -- so ids must always be fresh."""
    first = store.store("first")
    store.delete(first.id)
    second = store.store("second")

    assert second.internal_id != first.internal_id
    assert second.id != first.id

    revived = reopen()
    hits = revived.search("first", k=10)
    assert all(h.memory.text == "second" for h in hits)


def test_snapshot_preserves_graph_structure_exactly(store, reopen):
    for i in range(25):
        store.store(f"structural fidelity check number {i}")

    original = store._db._index
    revived = reopen()
    restored = revived._db._index

    assert restored.entry_point == original.entry_point
    assert restored.max_layer == original.max_layer
    assert len(restored.layers) == len(original.layers)
    for before, after in zip(original.layers, restored.layers):
        assert before == after
    for node_id, vector in original.vectors.items():
        assert np.allclose(restored.vectors[node_id], vector, atol=1e-6)


def test_search_quality_matches_across_snapshot_and_rebuild(data_dir, embedder, reopen):
    """The two load paths must agree, or recall would quietly depend on
    which one happened to run."""
    store = MemoryStore(data_dir=data_dir, embedder=embedder)
    for i in range(30):
        store.store(f"note about subject {i % 5} variant {i}")
    query = "note about subject 3"

    from_snapshot = [h.memory.id for h in reopen().search(query, k=5)]

    (data_dir / INDEX_FILE).unlink()
    rebuilt_store = reopen()
    from_rebuild = [h.memory.id for h in rebuilt_store.search(query, k=5)]

    assert rebuilt_store.rebuilt_on_load is True
    assert from_snapshot == from_rebuild
