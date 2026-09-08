"""Durable, tag-aware memory store on top of the HNSW vector index.

The engine (`hnsw-engine/`) is deliberately in-memory only -- its own
ARCHITECTURE.md lists persistence as future work -- and it has no way to
enumerate what it holds. This module supplies both, without modifying the
engine, by keeping two files side by side:

    memories.json   source of truth: id, text, tags, timestamps
    index.npz       snapshot of the built graph + embedding vectors

Loading prefers the snapshot, which restores in milliseconds. If the
snapshot is missing, corrupt, or disagrees with memories.json, the store
silently rebuilds the graph from the text instead. That fallback is what
makes it safe to snapshot the engine's internals at all: if a future
version of the engine changes its internal layout, the restore fails its
checks and the rebuild path takes over, so the worst case is a slow
startup rather than lost or corrupted memories.

Rebuilding is also how space from deleted entries is reclaimed. The engine
soft-deletes (marks and filters, never removing the node from the graph),
so tombstones accumulate in the snapshot; a rebuild drops them, which
makes it the compaction pass the engine's docs describe as missing.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from mcp_memory.embedder import EMBEDDING_DIM, MODEL_NAME, Embedder, MiniLMEmbedder
from mcp_memory.engine import HNSWIndex, VectorDB

SCHEMA_VERSION = 1
MEMORIES_FILE = "memories.json"
INDEX_FILE = "index.npz"

DEFAULT_METRIC = "cosine"
DEFAULT_M = 16
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_SEED = 17


def default_data_dir() -> Path:
    """Where memories live. Override with MCP_MEMORY_DIR."""
    override = os.environ.get("MCP_MEMORY_DIR")
    if override:
        return Path(override)
    return Path.home() / ".claude" / "memory-hnsw"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_tags(tags: Iterable[str] | None) -> list[str]:
    """Strip, drop blanks, de-duplicate case-insensitively, keep input order."""
    if not tags:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for raw in tags:
        tag = str(raw).strip()
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
    return out


@dataclass
class Memory:
    id: str
    text: str
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now)
    internal_id: int = -1

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "tags": list(self.tags),
            "created_at": self.created_at,
        }


@dataclass
class SearchHit:
    memory: Memory
    distance: float

    @property
    def similarity(self) -> float:
        """Cosine similarity. Embeddings are unit-length and the engine's
        cosine metric returns `1 - cos_sim`, so this inverts exactly."""
        return 1.0 - self.distance

    def to_public_dict(self) -> dict[str, Any]:
        out = self.memory.to_public_dict()
        out["similarity"] = round(self.similarity, 4)
        return out


class MemoryStore:
    """CRUD + semantic search over persisted memories."""

    def __init__(
        self,
        data_dir: Path | str | None = None,
        *,
        embedder: Embedder | None = None,
        metric: str = DEFAULT_METRIC,
        M: int = DEFAULT_M,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        seed: int = DEFAULT_SEED,
        autosave: bool = True,
    ):
        self.data_dir = Path(data_dir) if data_dir else default_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._embedder: Embedder = embedder if embedder is not None else MiniLMEmbedder()
        self.dim = getattr(self._embedder, "dim", EMBEDDING_DIM)
        self.model_name = getattr(self._embedder, "model_name", MODEL_NAME)

        self.metric = metric
        self.M = M
        self.ef_construction = ef_construction
        self.seed = seed
        self.autosave = autosave

        self._memories: dict[str, Memory] = {}
        self._tombstones: dict[str, int] = {}  # external id -> internal id
        self._next_internal_id = 0
        self._vectors: dict[str, np.ndarray] = {}  # external id -> embedding
        self._db: VectorDB = self._new_db()
        self._rebuilt_on_load = False

        self.load()

    # -- construction helpers -------------------------------------------

    def _new_db(self) -> VectorDB:
        return VectorDB(
            dim=self.dim,
            metric=self.metric,
            M=self.M,
            ef_construction=self.ef_construction,
            seed=self.seed,
        )

    @property
    def memories_path(self) -> Path:
        return self.data_dir / MEMORIES_FILE

    @property
    def index_path(self) -> Path:
        return self.data_dir / INDEX_FILE

    @property
    def rebuilt_on_load(self) -> bool:
        """True if the last load fell back to rebuilding from text."""
        return self._rebuilt_on_load

    def __len__(self) -> int:
        return len(self._memories)

    # -- public API ------------------------------------------------------

    def store(self, text: str, tags: Sequence[str] | None = None) -> Memory:
        text = (text or "").strip()
        if not text:
            raise ValueError("cannot store an empty memory")

        memory = Memory(
            id=uuid.uuid4().hex[:12],
            text=text,
            tags=normalize_tags(tags),
            created_at=_utc_now(),
            internal_id=self._next_internal_id,
        )
        vector = self._embed_one(text)

        # Always a fresh id, never a recycled one. This began as a workaround:
        # re-inserting an id the engine had soft-deleted used to leave the old
        # vector live in the graph under that same id, surfacing as a duplicate
        # hit carrying stale text. The engine fixed that in vectordb-hnsw
        # 4fd7eca (tombstones are keyed by internal id now, so a reused id
        # retires its predecessor for good), and recycling ids would be safe.
        # Fresh ids stay because they earn their place on their own: internal_id
        # is a monotonic counter, which is what breaks created_at ties in list()
        # and what the snapshot's tombstone map is keyed on.
        self._db.insert(memory.id, vector, metadata=self._metadata_for(memory))
        self._next_internal_id += 1

        self._memories[memory.id] = memory
        self._vectors[memory.id] = vector

        if self.autosave:
            self.save()
        return memory

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        query = (query or "").strip()
        if not query:
            raise ValueError("cannot search with an empty query")
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if not self._memories:
            return []

        query_vector = self._embed_one(query)
        raw = self._db.search(query_vector, k=min(k, len(self._memories)))

        hits: list[SearchHit] = []
        for result in raw:
            memory = self._memories.get(result["id"])
            if memory is None:
                # Defensive: an id in the graph with no record means the two
                # files drifted. Skip rather than surface a half-record.
                continue
            hits.append(SearchHit(memory=memory, distance=float(result["distance"])))
        return hits

    def list(self, tag: str | None = None) -> list[Memory]:
        """Newest first, optionally filtered to one tag (case-insensitive).

        Timestamps are only second-resolution, so several memories stored in
        the same second would tie; internal_id breaks the tie because it is
        a monotonic insertion counter, and a rebuild reassigns it in the
        existing order, so insertion order survives both load paths.
        """
        memories = list(self._memories.values())
        if tag:
            needle = tag.strip().lower()
            memories = [m for m in memories if any(t.lower() == needle for t in m.tags)]
        return sorted(memories, key=lambda m: (m.created_at, m.internal_id), reverse=True)

    def delete(self, memory_id: str) -> bool:
        """Returns False if the id is unknown or already deleted."""
        memory = self._memories.pop(memory_id, None)
        if memory is None:
            return False

        self._db.delete(memory_id)
        self._vectors.pop(memory_id, None)
        self._tombstones[memory_id] = memory.internal_id

        if self.autosave:
            self.save()
        return True

    def get(self, memory_id: str) -> Memory | None:
        return self._memories.get(memory_id)

    def all_tags(self) -> list[str]:
        counts: dict[str, str] = {}
        for memory in self._memories.values():
            for tag in memory.tags:
                counts.setdefault(tag.lower(), tag)
        return sorted(counts.values(), key=str.lower)

    # -- embedding --------------------------------------------------------

    def _embed_one(self, text: str) -> np.ndarray:
        vectors = self._embedder.encode([text])
        vector = np.asarray(vectors, dtype=np.float64).reshape(-1)
        if vector.shape != (self.dim,):
            raise ValueError(
                f"embedder returned dim {vector.shape[0]}, index expects {self.dim}"
            )
        return vector

    def _metadata_for(self, memory: Memory) -> dict[str, Any]:
        return {"text": memory.text, "tags": list(memory.tags), "created_at": memory.created_at}

    # -- persistence: writing ---------------------------------------------

    def save(self) -> None:
        self._write_memories()
        self._write_snapshot()

    def _manifest(self) -> dict[str, Any]:
        return {
            "version": SCHEMA_VERSION,
            "embedding_model": self.model_name,
            "dim": self.dim,
            "metric": self.metric,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "seed": self.seed,
            "next_internal_id": self._next_internal_id,
            "memories": [
                {
                    "id": m.id,
                    "internal_id": m.internal_id,
                    "text": m.text,
                    "tags": m.tags,
                    "created_at": m.created_at,
                }
                for m in self._memories.values()
            ],
            "tombstones": [
                {"id": mid, "internal_id": iid} for mid, iid in self._tombstones.items()
            ],
        }

    def _write_memories(self) -> None:
        _atomic_write_text(
            self.memories_path,
            json.dumps(self._manifest(), indent=2, ensure_ascii=False),
        )

    def _fingerprint(self) -> str:
        """Identifies exactly which entries the graph should contain.

        The snapshot is only trusted if this matches, so any edit to
        memories.json that the snapshot does not reflect forces a rebuild
        rather than a silently stale index.
        """
        payload = {
            "version": SCHEMA_VERSION,
            "dim": self.dim,
            "metric": self.metric,
            "model": self.model_name,
            "next_internal_id": self._next_internal_id,
            "live": sorted((m.id, m.internal_id) for m in self._memories.values()),
            "dead": sorted(self._tombstones.items()),
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _write_snapshot(self) -> None:
        index: HNSWIndex = self._db._index

        internal_ids = sorted(index.vectors.keys())
        vectors = (
            np.stack([index.vectors[i] for i in internal_ids]).astype(np.float32)
            if internal_ids
            else np.zeros((0, self.dim), dtype=np.float32)
        )

        arrays: dict[str, np.ndarray] = {
            "fingerprint": np.array(self._fingerprint()),
            "internal_ids": np.array(internal_ids, dtype=np.int64),
            "vectors": vectors,
            "entry_point": np.array(
                -1 if index.entry_point is None else index.entry_point, dtype=np.int64
            ),
            "max_layer": np.array(index.max_layer, dtype=np.int64),
            "num_layers": np.array(len(index.layers), dtype=np.int64),
        }

        # Adjacency goes out in CSR form (nodes + offsets + flat neighbours)
        # so an empty neighbour list stays distinguishable from an absent
        # node -- the engine relies on that distinction during insertion.
        for layer_num, adjacency in enumerate(index.layers):
            nodes = sorted(adjacency.keys())
            offsets = [0]
            flat: list[int] = []
            for node in nodes:
                flat.extend(adjacency[node])
                offsets.append(len(flat))
            arrays[f"L{layer_num}_nodes"] = np.array(nodes, dtype=np.int64)
            arrays[f"L{layer_num}_offsets"] = np.array(offsets, dtype=np.int64)
            arrays[f"L{layer_num}_neighbors"] = np.array(flat, dtype=np.int64)

        _atomic_write_npz(self.index_path, arrays)

    # -- persistence: reading ----------------------------------------------

    def load(self) -> None:
        self._rebuilt_on_load = False
        manifest = self._read_manifest()
        if manifest is None:
            self._memories, self._tombstones = {}, {}
            self._next_internal_id = 0
            self._vectors = {}
            self._db = self._new_db()
            return

        self._apply_manifest(manifest)

        if self._restore_snapshot():
            return

        self._rebuild()
        self._rebuilt_on_load = True

    def _read_manifest(self) -> dict[str, Any] | None:
        if not self.memories_path.exists():
            return None
        try:
            manifest = json.loads(self.memories_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(
                f"{self.memories_path} is unreadable ({exc}). It is the source of "
                "truth for stored memories; move it aside to start fresh."
            ) from exc

        stored_dim = manifest.get("dim")
        if stored_dim is not None and stored_dim != self.dim:
            raise ValueError(
                f"{self.memories_path} was written with dim {stored_dim} "
                f"(model {manifest.get('embedding_model')!r}) but this store uses "
                f"dim {self.dim} (model {self.model_name!r}). Vectors from two "
                "different embedding models are not comparable -- use a separate "
                "data dir, or delete this one to re-embed from scratch."
            )
        return manifest

    def _apply_manifest(self, manifest: dict[str, Any]) -> None:
        self.metric = manifest.get("metric", self.metric)
        self.M = manifest.get("M", self.M)
        self.ef_construction = manifest.get("ef_construction", self.ef_construction)
        self.seed = manifest.get("seed", self.seed)

        self._memories = {}
        for record in manifest.get("memories", []):
            memory = Memory(
                id=record["id"],
                text=record["text"],
                tags=list(record.get("tags", [])),
                created_at=record.get("created_at", _utc_now()),
                internal_id=int(record.get("internal_id", -1)),
            )
            self._memories[memory.id] = memory

        self._tombstones = {
            record["id"]: int(record["internal_id"])
            for record in manifest.get("tombstones", [])
        }

        known = [m.internal_id for m in self._memories.values()] + list(
            self._tombstones.values()
        )
        self._next_internal_id = max(
            int(manifest.get("next_internal_id", 0)),
            (max(known) + 1) if known else 0,
        )
        self._vectors = {}

    def _restore_snapshot(self) -> bool:
        """Rehydrate the graph from index.npz. False means 'rebuild instead'.

        This is the one place that touches the engine's private attributes.
        Every assumption it makes is checked first, and any failure returns
        False rather than raising, so a layout change upstream costs a slow
        startup and nothing more.
        """
        if not self.index_path.exists():
            return False

        try:
            with np.load(self.index_path, allow_pickle=False) as data:
                if str(data["fingerprint"]) != self._fingerprint():
                    return False

                internal_ids = data["internal_ids"].tolist()
                vectors = data["vectors"]
                if vectors.shape[1:] != (self.dim,) or len(internal_ids) != len(vectors):
                    return False

                index = HNSWIndex(
                    dim=self.dim,
                    metric=self.metric,
                    M=self.M,
                    ef_construction=self.ef_construction,
                    seed=self.seed,
                )
                index.vectors = {
                    int(i): np.asarray(v, dtype=np.float64)
                    for i, v in zip(internal_ids, vectors)
                }

                layers: list[dict[int, list[int]]] = []
                for layer_num in range(int(data["num_layers"])):
                    nodes = data[f"L{layer_num}_nodes"].tolist()
                    offsets = data[f"L{layer_num}_offsets"].tolist()
                    flat = data[f"L{layer_num}_neighbors"].tolist()
                    adjacency = {
                        int(node): [int(n) for n in flat[offsets[i] : offsets[i + 1]]]
                        for i, node in enumerate(nodes)
                    }
                    layers.append(adjacency)
                index.layers = layers

                entry_point = int(data["entry_point"])
                index.entry_point = None if entry_point < 0 else entry_point
                index.max_layer = int(data["max_layer"])
        except (KeyError, ValueError, OSError, IndexError):
            return False

        # Every node the graph can walk to must resolve to a record, or
        # search would raise on an unmapped id mid-query.
        mapped = {m.internal_id for m in self._memories.values()} | set(
            self._tombstones.values()
        )
        if not set(index.vectors).issubset(mapped):
            return False

        self._db = self._new_db()
        self._db._index = index
        self._db._id_map = {m.id: m.internal_id for m in self._memories.values()}
        self._db._id_map.update(self._tombstones)
        self._db._reverse_id_map = {v: k for k, v in self._db._id_map.items()}
        self._db._metadata = {
            m.id: self._metadata_for(m) for m in self._memories.values()
        }
        self._db._metadata.update({mid: {} for mid in self._tombstones})
        # .values(), not the dict: the engine tombstones by internal id, so
        # handing it external ids leaves every tombstone inert -- deleted
        # entries then get ranked, eat the result budget, and crowd live
        # memories out of search. _tombstones is external id -> internal id.
        self._db._deleted = set(self._tombstones.values())
        self._db._next_internal_id = self._next_internal_id

        self._vectors = {
            m.id: index.vectors[m.internal_id]
            for m in self._memories.values()
            if m.internal_id in index.vectors
        }
        return True

    def _rebuild(self) -> None:
        """Re-embed and re-insert every live memory into a fresh graph.

        Tombstones are dropped here rather than carried over, so a rebuild
        doubles as the compaction pass the engine does not implement.
        """
        self._db = self._new_db()
        self._vectors = {}
        self._tombstones = {}

        memories = sorted(self._memories.values(), key=lambda m: m.internal_id)
        if not memories:
            self._next_internal_id = 0
            return

        vectors = self._embedder.encode([m.text for m in memories])
        for offset, memory in enumerate(memories):
            vector = np.asarray(vectors[offset], dtype=np.float64).reshape(-1)
            memory.internal_id = offset
            self._db.insert(memory.id, vector, metadata=self._metadata_for(memory))
            self._vectors[memory.id] = vector

        self._next_internal_id = len(memories)
        self.save()


def _atomic_write_text(path: Path, content: str) -> None:
    """Write via temp file + rename so a crash mid-write cannot truncate
    the existing file -- these two files are the whole memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp.npz")
    os.close(fd)
    try:
        with open(tmp_name, "wb") as fh:
            np.savez(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
