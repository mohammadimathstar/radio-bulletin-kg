"""Bounded, self-managing Wikidata response cache.

Replaces the previous single ``wd_cache.json``, which had two practical
problems: it was rewritten IN FULL on every save (O(cache) work every 25
clusters, so late in a 15k-cluster run most time went into rewriting a huge
file), and it never expired or shrank (the reason it had to be deleted by hand).

Design
------
* Storage: one SQLite database, one row per entry — key, zlib-compressed JSON
  value, kind, creation time, expiry time.  Writes are per-key upserts, so
  saving costs O(new entries), not O(cache).
* Kinds with separate TTLs (``config.cache_ttl_days``): ``search`` results are
  the bulk of the entries and go stale, so they expire first; entities, label
  lookups, per-item facts and class closures are stable and live long.
* :meth:`prune` deletes expired rows (indexed), and if the file is still above
  ``cache_max_mb`` deletes the oldest entries of the least valuable kinds
  (search first, summaries next).  Called on open and periodically during the
  run, so the file stays bounded without manual deletes.
* Thread-safe: a single connection guarded by a lock, WAL journal mode.

Key scheme (kept compatible with the previous cache so scripts that share it —
e.g. candidate_profiles — keep working; the prefix selects the kind):

    s|lang|limit|term   wbsearchentities results          (kind search)
    c|limit|query       Cirrus full-text results          (kind search)
    eL|QID              entity, light props               (kind entity)
    e|QID               entity incl. claims               (kind entity)
    f|QID               P31 + gate dates (SPARQL)         (kind facts)
    C|classes           subclass closure of class roots   (kind closure)
    l|QID               English/Dutch label of an item    (kind label)
    w|lang|title        Wikipedia summary                 (kind summary)
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import zlib
from pathlib import Path

# prefix -> cache kind (drives the TTL and the eviction order of prune()).
# Facts use the versioned "f2|": the "f|" entries written before the SPARQL
# time-format fix carry no dates — left unkinded, so they age out as "misc".
_KIND_BY_PREFIX = {"s": "search", "c": "search", "eL": "entity", "e": "entity",
                   "f2": "facts", "C": "closure", "l": "label", "w": "summary"}

# Least valuable first: prune() evicts whole kinds, oldest entries first,
# when the size cap is reached (search results dominate the size and are the
# cheapest to refetch).
_EVICT_ORDER = ["search", "summary", "label", "misc", "facts", "entity", "closure"]


def kind_of(key: str) -> str:
    """Cache kind of a key from its ``prefix|...`` scheme ('search' if unknown)."""
    prefix = key.split("|", 1)[0]
    return _KIND_BY_PREFIX.get(prefix, "misc")


class SQLiteCache:
    """Tiny key/value API (``get`` / ``set`` / ``save``) over a bounded SQLite store.

    ``save()`` exists only for drop-in compatibility with the old JSON cache —
    SQLite commits every write itself, so it is a no-op here.
    """

    def __init__(self, path: str, ttl_days: dict, max_mb: int = 512):
        self.path = Path(path)
        self.ttl_days, self.max_mb = ttl_days, max_mb
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._con = sqlite3.connect(str(self.path), check_same_thread=False)
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("""CREATE TABLE IF NOT EXISTS kv (
            key TEXT PRIMARY KEY, value BLOB NOT NULL, kind TEXT NOT NULL,
            created REAL NOT NULL, expires REAL NOT NULL)""")
        self._con.execute("CREATE INDEX IF NOT EXISTS kv_kind_created ON kv(kind, created)")
        self._con.execute("CREATE INDEX IF NOT EXISTS kv_expires ON kv(expires)")
        self._con.commit()
        self.prune()

    # ------------------------------------------------------------------ core
    def get(self, key: str):
        """Cached value, or None when absent/expired."""
        now = time.time()
        with self._lock:
            row = self._con.execute("SELECT value, expires FROM kv WHERE key = ?",
                                    (key,)).fetchone()
            if row is None or row[1] <= now:
                return None
            return json.loads(zlib.decompress(row[0]).decode("utf-8"))

    def set(self, key: str, value, kind: str | None = None) -> None:
        """Insert/overwrite one entry with the TTL of its kind."""
        kind = kind or kind_of(key)
        ttl = float(self.ttl_days.get(kind, 30)) * 86400
        blob = zlib.compress(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"), 6)
        now = time.time()
        with self._lock:
            self._con.execute(
                "INSERT INTO kv(key, value, kind, created, expires) VALUES (?,?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, kind=excluded.kind, "
                "created=excluded.created, expires=excluded.expires",
                (key, blob, kind, now, now + ttl))
            self._con.commit()

    def save(self) -> None:      # compatibility no-op; every set() already committed
        pass

    def close(self) -> None:
        """Prune once more, VACUUM if the file is still oversized, then close.

        VACUUM rewrites the database file, dropping the pages that deleted rows
        left behind — done once at the end of a run, not on every prune.
        """
        try:
            self.prune()
            if self._file_mb() > self.max_mb:
                self._con.execute("VACUUM")
        except sqlite3.Error:
            pass
        with self._lock:
            self._con.close()

    # ------------------------------------------------------------------ maintenance
    def prune(self, extra_kinds_first: list[str] | None = None) -> dict:
        """Delete expired entries; if still above the cap, evict oldest-first.

        The cap decision uses the LOGICAL size (bytes of live values) — deleted
        pages keep the FILE big until VACUUM, so a file-size test here would
        evict far too much.  Eviction walks the kinds in _EVICT_ORDER (anything
        in ``extra_kinds_first`` goes first — used to make room for a known
        burst), deleting the oldest rows of each kind until the logical size is
        under ``cache_max_mb``.  The FILE is compacted by :meth:`close` (VACUUM)
        when it is still oversized.  Returns counters for the progress log.
        """
        now = time.time()
        with self._lock:
            stats = {"expired": self._con.execute(
                "DELETE FROM kv WHERE expires <= ?", (now,)).rowcount}
            self._con.commit()
            evicted = 0
            order = list(extra_kinds_first or [])
            order += [k for k in _EVICT_ORDER if k not in order]
            for kind in order:
                while self._logical_mb() > self.max_mb:
                    n = self._con.execute(
                        "DELETE FROM kv WHERE rowid IN "
                        "(SELECT rowid FROM kv WHERE kind = ? ORDER BY created LIMIT 500)",
                        (kind,)).rowcount
                    if n == 0:
                        break
                    evicted += n
                if self._logical_mb() <= self.max_mb:
                    break
            self._con.commit()
            stats["evicted"] = evicted
            stats["logical_mb"] = round(self._logical_mb(), 1)
            return stats

    def stats(self) -> dict:
        with self._lock:
            rows = self._con.execute(
                "SELECT kind, COUNT(*), SUM(LENGTH(value)) FROM kv GROUP BY kind").fetchall()
        return {"file_mb": round(self._file_mb(), 1),
                "logical_mb": round(self._logical_mb(), 1),
                "entries": {k: {"n": n, "mb": round((s or 0) / 1e6, 1)} for k, n, s in rows}}

    # ------------------------------------------------------------------ internals
    def _logical_mb(self) -> float:
        """Size of the LIVE (non-expired) entries — the basis for the cap."""
        now = time.time()
        row = self._con.execute(
            "SELECT SUM(LENGTH(value)) FROM kv WHERE expires > ?", (now,)).fetchone()
        return (row[0] or 0) / 1e6

    def _file_mb(self) -> float:
        main = self.path.stat().st_size if self.path.exists() else 0
        wal = Path(str(self.path) + "-wal").stat().st_size if Path(str(self.path) + "-wal").exists() else 0
        return (main + wal) / 1e6