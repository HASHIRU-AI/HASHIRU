"""Saved user memories: keyed notes the agents can add, search and delete.

``SqliteMemory`` persists in one file shared by every session and ranks search results with SQLite FTS5 (bm25), so
no embedding model is needed. ``SessionMemory`` keeps a private copy of data/memory.json per session and never
writes it back, which keeps benchmark runs reproducible.

Each memory keeps the labels its writer held (for example ``untrusted`` after reading a web page), so text that
crosses sessions through memory keeps its labels when it is read back.
"""
import contextlib
import itertools
import json
import re
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

__all__ = ["MAX_KEY_CHARS", "MAX_LIST", "MemoryStore", "SessionMemory", "SqliteMemory", "open_memory", "words"]

MAX_KEY_CHARS = 200
MAX_TEXT_CHARS = 4_000
MAX_LIST = 200


class MemoryStore(Protocol):
    def add(self, key: str, text: str, labels: list[str] | None = None) -> None: ...
    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]: ...
    def delete(self, key: str) -> bool: ...
    def keys(self, limit: int = MAX_LIST) -> tuple[list[str], int]: ...


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class SessionMemory:
    def __init__(self, seed: dict[str, str] | None = None) -> None:
        self._order = itertools.count()  # "newest" by insertion: clock readings can tie
        self._items: dict[str, tuple[str, list[str], int]] = {
            k: (t, [], next(self._order)) for k, t in (seed or {}).items()}

    def add(self, key: str, text: str, labels: list[str] | None = None) -> None:
        self._items.pop(key, None)
        self._items[key] = (text[:MAX_TEXT_CHARS], sorted(set(labels or [])), next(self._order))

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Rank by how many query words each memory (key and text) contains; an empty query gives the newest."""
        wanted = set(words(query))
        scored = [(len(wanted & set(words(f"{k} {t}"))), when, k, t, labels)
                  for k, (t, labels, when) in self._items.items()]
        hits = sorted((h for h in scored if h[0] or not wanted), key=lambda h: (-h[0], -h[1]))
        return [{"key": k, "text": t, "labels": labels} for _, _, k, t, labels in hits[:limit]]

    def delete(self, key: str) -> bool:
        return self._items.pop(key, None) is not None

    def keys(self, limit: int = MAX_LIST) -> tuple[list[str], int]:
        newest = sorted(self._items, key=lambda k: -self._items[k][2])
        return newest[:limit], len(newest)


class SqliteMemory:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            # key is searchable too (as in SessionMemory): "units" should find a memory saved under that key
            db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS memories "
                       "USING fts5(key, text, labels UNINDEXED, updated UNINDEXED)")

    @contextlib.contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        with contextlib.closing(sqlite3.connect(self.path, timeout=10)) as db, db:  # commit, then close
            yield db

    def add(self, key: str, text: str, labels: list[str] | None = None) -> None:
        with self._db() as db:
            db.execute("DELETE FROM memories WHERE key = ?", (key,))
            db.execute("INSERT INTO memories (key, text, labels, updated) VALUES (?, ?, ?, ?)",
                       (key, text[:MAX_TEXT_CHARS], json.dumps(sorted(set(labels or []))), time.time()))

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Best matches by bm25 for any query word; an empty query returns the most recent memories."""
        terms = words(query)
        with self._db() as db:
            if terms:  # quote each word so user text can never be FTS5 query syntax
                rows = db.execute("SELECT key, text, labels FROM memories WHERE memories MATCH ? "
                                  "ORDER BY bm25(memories) LIMIT ?",
                                  (" OR ".join(f'"{t}"' for t in terms), limit)).fetchall()
            else:
                rows = db.execute("SELECT key, text, labels FROM memories ORDER BY updated DESC, rowid DESC "
                                  "LIMIT ?", (limit,)).fetchall()
        return [{"key": k, "text": t, "labels": json.loads(labels or "[]")} for k, t, labels in rows]

    def delete(self, key: str) -> bool:
        with self._db() as db:
            return db.execute("DELETE FROM memories WHERE key = ?", (key,)).rowcount > 0

    def keys(self, limit: int = MAX_LIST) -> tuple[list[str], int]:
        with self._db() as db:
            total = db.execute("SELECT count(*) FROM memories").fetchone()[0]
            newest = [k for (k,) in db.execute("SELECT key FROM memories ORDER BY updated DESC, rowid DESC LIMIT ?",
                                               (limit,))]
        return newest, int(total)


def seed_of(data_dir: Path) -> dict[str, str]:
    path = data_dir / "memory.json"
    return {str(k): str(v) for k, v in json.loads(path.read_text()).items()} if path.exists() else {}


def open_memory(memory_db: str, data_dir: Path) -> MemoryStore:
    """The shared store if memory_db is set (a new one starts from data_dir/memory.json); otherwise a fresh
    per-session copy of memory.json."""
    if not memory_db:
        return SessionMemory(seed_of(data_dir))
    new = not Path(memory_db).exists()
    store = SqliteMemory(Path(memory_db))
    if new:
        for key, text in seed_of(data_dir).items():
            store.add(key, text)
    return store
