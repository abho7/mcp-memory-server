"""The four MCP tools, plus the wiring that exposes them to a client."""

from __future__ import annotations

import asyncio

import pytest

from mcp_memory import server as server_module
from mcp_memory.server import (
    delete_memory,
    list_memories,
    search_memory,
    server,
    store_memory,
)
from mcp_memory.store import MemoryStore


@pytest.fixture(autouse=True)
def wired_store(data_dir, embedder):
    """Point the tool module at a throwaway store instead of the real
    on-disk one, so tests never touch the user's actual memories."""
    store = MemoryStore(data_dir=data_dir, embedder=embedder)
    server_module.set_store(store)
    yield store
    server_module.set_store(None)


# -- registration -----------------------------------------------------------


def test_all_four_tools_are_registered_with_the_server():
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == {
        "store_memory",
        "search_memory",
        "list_memories",
        "delete_memory",
    }


def _input_schema(tool):
    # mcp 2.x renamed Tool.inputSchema to input_schema.
    return getattr(tool, "input_schema", None) or tool.inputSchema


def test_registered_tools_have_descriptions_and_schemas():
    tools = {t.name: t for t in asyncio.run(server.list_tools())}

    assert "embedded locally" in tools["store_memory"].description
    assert set(_input_schema(tools["store_memory"])["properties"]) == {"text", "tags"}
    assert _input_schema(tools["store_memory"])["required"] == ["text"]

    assert set(_input_schema(tools["search_memory"])["properties"]) == {"query", "k"}
    assert set(_input_schema(tools["list_memories"])["properties"]) == {"tag"}
    assert set(_input_schema(tools["delete_memory"])["properties"]) == {"id"}


def test_tools_are_callable_through_the_server(wired_store):
    """Exercise the real MCP dispatch path, not just the Python function."""
    asyncio.run(
        server.call_tool("store_memory", {"text": "routed through MCP", "tags": ["x"]})
    )
    assert len(wired_store) == 1

    result = asyncio.run(server.call_tool("search_memory", {"query": "routed", "k": 1}))
    rendered = str(result)
    assert "routed through MCP" in rendered


# -- store_memory -----------------------------------------------------------


def test_store_memory_reports_id_and_count(wired_store):
    out = store_memory("Prefers pytest over unittest", ["preference"])

    assert "Stored memory" in out
    assert "tags [preference]" in out
    assert "1 memories now in the index" in out

    memory = wired_store.list()[0]
    assert memory.id in out
    assert memory.text == "Prefers pytest over unittest"


def test_store_memory_without_tags(wired_store):
    out = store_memory("A memory with no tags")
    assert "tags" not in out
    assert wired_store.list()[0].tags == []


def test_store_memory_rejects_empty_text_without_raising(wired_store):
    out = store_memory("   ")
    assert out.startswith("Could not store memory:")
    assert len(wired_store) == 0


# -- search_memory ----------------------------------------------------------


def test_search_memory_returns_ranked_results(wired_store):
    store_memory("The CI pipeline runs on GitHub Actions", ["ci"])
    store_memory("Lunch options near the office are limited")
    store_memory("GitHub Actions workflows live in .github/workflows")

    out = search_memory("github actions ci", k=2)

    assert "2 memory match(es)" in out
    assert "similarity" in out
    assert "Lunch options" not in out


def test_search_memory_respects_k(wired_store):
    for i in range(5):
        store_memory(f"memory {i}")
    out = search_memory("memory", k=2)
    assert "2 memory match(es)" in out


def test_search_memory_on_empty_store(wired_store):
    assert search_memory("anything") == "No memories stored yet."


def test_search_memory_rejects_bad_k(wired_store):
    store_memory("something")
    assert search_memory("something", k=0).startswith("Could not search:")


def test_search_memory_output_includes_ids_for_followup_deletion(wired_store):
    memory = wired_store.store("find me by id")
    out = search_memory("find me by id", k=1)
    assert f"id={memory.id}" in out


# -- list_memories ----------------------------------------------------------


def test_list_memories_shows_all_newest_first(wired_store):
    store_memory("oldest")
    store_memory("newest")

    out = list_memories()

    assert "2 memory(ies)" in out
    assert out.index("newest") < out.index("oldest")


def test_list_memories_filters_by_tag(wired_store):
    store_memory("tagged one", ["ops"])
    store_memory("tagged two", ["style"])

    out = list_memories("ops")

    assert "tagged one" in out
    assert "tagged two" not in out
    assert "tagged 'ops'" in out


def test_list_memories_unknown_tag_suggests_known_ones(wired_store):
    store_memory("tagged", ["ops"])
    out = list_memories("nope")
    assert "No memories tagged 'nope'" in out
    assert "Known tags: ops" in out


def test_list_memories_on_empty_store(wired_store):
    assert list_memories() == "No memories stored yet."


# -- delete_memory ----------------------------------------------------------


def test_delete_memory_removes_it(wired_store):
    memory = wired_store.store("delete me")

    out = delete_memory(memory.id)

    assert f"Deleted memory {memory.id}" in out
    assert "0 memories remain" in out
    assert wired_store.list() == []


def test_delete_memory_unknown_id_is_a_clear_message_not_an_error(wired_store):
    out = delete_memory("nope123")
    assert "No memory with id 'nope123'" in out
    assert "list_memories" in out


def test_delete_memory_then_search_does_not_return_it(wired_store):
    keep = wired_store.store("keep this one about caching")
    drop = wired_store.store("drop this one about caching")

    delete_memory(drop.id)
    out = search_memory("caching", k=10)

    assert keep.id in out
    assert drop.id not in out


# -- cross-session behaviour through the tools ------------------------------


def test_tools_recall_memories_stored_in_a_previous_session(data_dir, embedder):
    """The README's demo, as a test: two independent stores over one
    directory, standing in for two separate Claude sessions."""
    from tests.conftest import HashingEmbedder

    session_one = MemoryStore(data_dir=data_dir, embedder=embedder)
    server_module.set_store(session_one)
    store_memory("We chose Postgres over MySQL for JSONB support", ["decision"])

    # Nothing carried over but the directory on disk.
    session_two = MemoryStore(data_dir=data_dir, embedder=HashingEmbedder())
    server_module.set_store(session_two)

    recalled = search_memory("which database did we pick and why", k=1)
    assert "Postgres over MySQL" in recalled

    listed = list_memories("decision")
    assert "Postgres over MySQL" in listed
