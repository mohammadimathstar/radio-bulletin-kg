"""All access to Wikidata: search, entity fetch, type/date facts, class closures.

Speed model (why this replaces the old serial calls — 15k clusters used to take
days):

* ONE shared :class:`requests` session per thread (keep-alive; the old code did
  a fresh TCP+TLS handshake per call).
* NO artificial ``time.sleep`` — a token-bucket :class:`RateLimiter` with
  separate budgets for the MediaWiki API, SPARQL and REST endpoints.
* COALESCED BATCHING: worker threads ask for single items (one candidate's
  facts/entity); a :class:`Batcher` collects them across clusters and fires one
  request per batch.  wbgetentities is capped at 50 ids/request by the API, the
  SPARQL facts query at ``sparql_batch`` ids — but requests now come from ALL
  clusters together, deduplicated (the same QID is never fetched twice, which
  matters: place candidates like countries recur in hundreds of clusters).
* LIGHT ENTITY FETCH: clusters fetch labels/descriptions/aliases/sitelinks
  only.  P31 and the gate dates come from ONE compact SPARQL query per batch
  (:meth:`facts`) instead of shipping every claim of every candidate in the
  entity payloads — the old ``claims`` prop made each response 10-60x larger
  and was the main driver of run time and cache size.  Full claims are fetched
  only where they are really used (pass-2 coherence; candidate_profiles).
* LOCAL TYPE CHECK: the P279* subclass closure of the allowed root classes is
  fetched ONCE (cached for weeks), so "is this candidate a Person/Place/..."
  becomes a local set intersection instead of a SPARQL query per 100 candidates.
"""
from __future__ import annotations

import re
import threading
import time
from concurrent.futures import Future

import requests

QID_RE = re.compile(r"Q\d+$")
# Date properties used by the temporal gate (see features.temporal_status).
START_PROPS = ("P569", "P571", "P729", "P580")
END_PROPS = ("P570", "P576", "P730", "P582")
FACT_PROPS = ("P31",) + START_PROPS + END_PROPS


# ------------------------------------------------------------------ plumbing
class RateLimiter:
    """Thread-safe token spacing: at most ``rate`` requests/s per kind."""

    def __init__(self, rates: dict[str, float]):
        self._rates = dict(rates)
        self._next: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, kind: str) -> None:
        """Reserve ONE slot and sleep until it arrives — no loop.

        (A "reserve again after waking" loop live-locks under load: every
        waking thread finds the slot pushed forward by the other threads'
        re-reservations and sleeps again, forever.)
        """
        rate = self._rates.get(kind, 8.0)
        with self._lock:
            now = time.monotonic()
            t = max(now, self._next.get(kind, 0.0))
            self._next[kind] = t + 1.0 / rate
        if t > now:
            time.sleep(t - now)


class Batcher:
    """Coalesce single-item lookups from many threads into batched network calls.

    ``fetch_many(chunk)`` must return ``{item: value}`` for every item of the
    chunk (a missing item maps to None).  Items are accumulated and fetched
    when the batch is full or the oldest pending item is ``max_wait`` seconds
    old.  Exceptions are re-raised in every waiting thread, so a network error
    fails that cluster (which is then retried on the next run) instead of
    silently producing wrong data.
    """

    def __init__(self, fetch_many, batch_size: int = 50, max_wait: float = 0.2):
        self._fetch, self._size, self._wait = fetch_many, batch_size, max_wait
        self._pending: dict[str, list] = {}
        self._oldest = 0.0
        self._lock = threading.Lock()
        self._wake = threading.Event()
        threading.Thread(target=self._drain_loop, daemon=True).start()

    def get(self, item: str):
        f = Future()
        with self._lock:
            self._pending.setdefault(item, []).append(f)
            if not self._oldest:
                self._oldest = time.monotonic()
            full = len(self._pending) >= self._size
        self._wake.set()
        if full:
            self._drain()
        return f.result()

    def _drain_loop(self) -> None:
        while True:
            self._wake.wait(timeout=0.05)
            self._wake.clear()
            if self._pending and time.monotonic() - self._oldest >= self._wait:
                self._drain()

    def _drain(self) -> None:
        with self._lock:
            if not self._pending:
                return
            chunk, self._pending, self._oldest = self._pending, {}, 0.0
        try:
            values = self._fetch(list(chunk))
            for item, futs in chunk.items():
                v = values.get(item)          # None = item missing/not found
                for f in futs:
                    f.set_result(v)
        except BaseException as ex:
            for futs in chunk.values():
                for f in futs:
                    f.set_exception(ex)


class WDClient:
    """Wikidata access with caching, batching, rate limiting and retries."""

    def __init__(self, cfg, cache):
        self.cfg, self.cache = cfg, cache
        self.limiter = RateLimiter({"api": cfg.rate_api, "sparql": cfg.rate_sparql,
                                    "rest": cfg.rate_sparql * 2})
        self._local = threading.local()
        self._closures: dict[str, set] = {}          # class-root key -> P279* closure
        self._closure_lock = threading.Lock()        # one fetch per root, not one per cluster
        self._batches: dict[str, Batcher] = {
            "light": Batcher(self._fetch_entities_light, 50, 0.2),
            "full": Batcher(self._fetch_entities_full, 50, 0.2),
            "facts": Batcher(self._fetch_facts, cfg.sparql_batch, 0.3),
        }

    # ------------------------------------------------------------------ http
    @property
    def session(self) -> requests.Session:
        if not hasattr(self._local, "s"):
            s = requests.Session()
            s.headers.update({"User-Agent": self.cfg.user_agent})
            self._local.s = s
        return self._local.s

    def _request(self, url: str, params: dict, kind: str, timeout: float,
                 retries: int | None = None) -> dict:
        """GET with rate limit + retries; raises requests.RequestException on failure."""
        last = None
        for attempt in range(retries if retries is not None else self.cfg.retries):
            self.limiter.wait(kind)
            try:
                r = self.session.get(url, params=params, timeout=timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    last = requests.RequestException(f"HTTP {r.status_code}")
                    retry_after = float(r.headers.get("Retry-After") or 0)
                    time.sleep(retry_after or 2 ** attempt)
                    continue
                r.raise_for_status()
                data = r.json()
                if isinstance(data, dict) and "error" in data:
                    raise requests.RequestException(f"API error: {data['error']}")
                return data
            except (requests.ConnectionError, requests.Timeout, ValueError) as ex:
                last, data = ex, None
                time.sleep(2 ** attempt)
                continue
            except requests.RequestException as ex:
                if "API error" in str(ex):        # a definitive API error: do not retry
                    raise
                last = ex
                time.sleep(2 ** attempt)
                continue
        raise last or requests.RequestException("request failed")

    def sparql(self, query: str, retries: int | None = None) -> list[dict]:
        """Run a SPARQL query; returns the result bindings."""
        data = self._request(self.cfg.sparql, dict(query=query, format="json"),
                             "sparql", self.cfg.timeout_sparql, retries=retries)
        return data.get("results", {}).get("bindings", [])

    # ------------------------------------------------------------------ search
    def search(self, term: str, lang: str, limit: int | None = None) -> list[str]:
        """wbsearchentities: prefix match on labels/aliases in ONE language. Cached."""
        limit = min(limit or self.cfg.search_limit, 50)
        k = f"s|{lang}|{limit}|{term}"
        if (v := self.cache.get(k)) is not None:
            return v
        data = self._request(self.cfg.wd_api, dict(
            action="wbsearchentities", search=term, language=lang, uselang=lang,
            type="item", limit=limit, format="json"), "api", self.cfg.timeout_api)
        ids = [x["id"] for x in data.get("search", [])]
        self.cache.set(k, ids)
        return ids

    def cirrus(self, term: str, filt: str = "", limit: int | None = None) -> list[str]:
        """Full-text search (CirrusSearch) over labels+aliases+descriptions of ALL languages.

        ``filt`` is a Cirrus operator string such as "haswbstatement:P31=Q5";
        callers append "~" to terms for fuzzy matching.
        """
        limit = min(limit or self.cfg.cirrus_limit, 50)
        q = f"{term} {filt}".strip()
        k = f"c|{limit}|{q}"
        if (v := self.cache.get(k)) is not None:
            return v
        data = self._request(self.cfg.wd_api, dict(
            action="query", list="search", srsearch=q, srnamespace=0,
            srlimit=limit, srprop="", format="json"), "api", self.cfg.timeout_api)
        ids = [x["title"] for x in data.get("query", {}).get("search", [])
               if QID_RE.fullmatch(x.get("title", ""))]
        self.cache.set(k, ids)
        return ids

    # ------------------------------------------------------------------ entities
    def entity(self, qid: str, claims: bool = False):
        """One entity (light props, plus claims when asked); None if missing."""
        return self._batches["full" if claims else "light"].get(qid)

    def entities(self, qids: list[str], claims: bool = False) -> dict[str, dict]:
        """Several entities through the same batcher (deduplicated across threads)."""
        b = self._batches["full" if claims else "light"]
        return {q: v for q in dict.fromkeys(qids) if (v := b.get(q))}

    def _fetch_entities(self, qids: list[str], claims: bool) -> dict:
        props = "labels|descriptions|aliases|sitelinks" + ("|claims" if claims else "")
        out = {}
        for i in range(0, len(qids), 50):
            data = self._request(self.cfg.wd_api, dict(
                action="wbgetentities", ids="|".join(qids[i:i + 50]),
                props=props, languages="|".join(self.cfg.langs), format="json"),
                "api", 60.0)
            for q, e in data.get("entities", {}).items():
                if "missing" not in e:
                    out[q] = e
        return out

    def _fetch_entities_light(self, qids: list[str]) -> dict:
        out = {}
        for q in qids:
            if (v := self.cache.get(f"eL|{q}")) is not None:
                out[q] = v
        need = [q for q in qids if q not in out]
        for q, e in self._fetch_entities(need, claims=False).items():
            self.cache.set(f"eL|{q}", e)
            out[q] = e
        return out

    def _fetch_entities_full(self, qids: list[str]) -> dict:
        out = {}
        for q in qids:
            if (v := self.cache.get(f"e|{q}")) is not None:
                out[q] = v
        need = [q for q in qids if q not in out]
        for q, e in self._fetch_entities(need, claims=True).items():
            self.cache.set(f"e|{q}", e)
            out[q] = e
        return out

    # ------------------------------------------------------------------ P31 + gate dates
    def facts(self, qid: str) -> dict:
        """P31 values + gate dates of one item, from one compact SPARQL batch. Cached.

        Returns ``{"p31": [QID...], "dates": {pid: ["YYYY-MM-DD", ...]}}`` —
        exactly what the type gate and the temporal gate need.
        """
        return self._batches["facts"].get(qid) or {"p31": [], "dates": {}}

    def _fetch_facts(self, qids: list[str]) -> dict:
        # f2| : versioned key — the f| entries written before the SPARQL time
        # format fix carry NO dates; they must be ignored, not reused.
        out = {}
        for q in qids:
            if (v := self.cache.get(f"f2|{q}")) is not None:
                out[q] = v
        need = [q for q in qids if q not in out]
        props = " ".join(f"wdt:{p}" for p in FACT_PROPS)
        for i in range(0, len(need), self.cfg.sparql_batch):
            chunk = need[i:i + self.cfg.sparql_batch]
            query = ("SELECT ?q ?p ?t WHERE { VALUES ?q { %s } VALUES ?p { %s } ?q ?p ?t }"
                     % (" ".join(f"wd:{q}" for q in chunk), props))
            per_q = {q: {"p31": [], "dates": {}} for q in chunk}
            for b in self.sparql(query):
                q = b["q"]["value"].rsplit("/", 1)[-1]
                pid = b["p"]["value"].rsplit("/", 1)[-1]
                val = b["t"]["value"]
                if pid == "P31":
                    per_q[q]["p31"].append(val.rsplit("/", 1)[-1])
                elif (d := parse_wd_time(val)) is not None:
                    per_q[q]["dates"].setdefault(pid, []).append(d.isoformat())
            for q, v in per_q.items():
                self.cache.set(f"f2|{q}", v)
                out[q] = v
        return out

    # ------------------------------------------------------------------ type checking
    def closure(self, classes: list[str]) -> set:
        """All classes that are (transitively) subclasses of any of ``classes``.

        ONE SPARQL query PER ROOT CLASS, each cached for weeks; after that the
        type gate is a local set intersection (no per-candidate SPARQL).  Broad
        roots are genuinely big ("geographical feature" subtree: >1e6 classes,
        ~40s once) — which is exactly why the closure is fetched once and not
        queried per candidate.  The fetch runs under a lock: without it the
        first N clusters of a type all race into the same miss and download the
        same roots N times in parallel.
        """
        with self._closure_lock:
            key = "|".join(sorted(classes))
            if key not in self._closures:
                union = set()
                for root in classes:
                    union |= set(self._closure_root(root))
                self._closures[key] = union
            return self._closures[key]

    def _closure_root(self, root: str) -> list:
        """Closure of ONE root class (cached for weeks), fetched with retries."""
        if (v := self.cache.get(f"C|{root}")) is None:
            t0 = time.monotonic()
            query = f"SELECT DISTINCT ?x WHERE {{ ?x wdt:P279* wd:{root} }}"
            v = sorted({b["x"]["value"].rsplit("/", 1)[-1] for b in self.sparql(query)})
            self.cache.set(f"C|{root}", v)
            print(f"      [closure] {root}: {len(v)} classes "
                  f"({time.monotonic() - t0:.0f}s)", flush=True)
        return v

    def closure_roots(self, roots, pool=None) -> None:
        """Warm the closure cache for every uncached root, in PARALLEL.

        Called once at start-up, before pass 1.  Without it the first clusters
        of each type stall on their root closures one after another, and the
        early clusters/s rate (and the ETA built on it) is meaningless.
        """
        todo = [r for r in dict.fromkeys(roots) if self.cache.get(f"C|{r}") is None]
        if not todo:
            return
        if pool is None or len(todo) == 1:
            for r in todo:
                self._closure_root(r)
            return
        list(pool.map(self._closure_root, todo))

    def is_a(self, qid_p31: list[str], classes: list[str]) -> bool:
        """True when any direct P31 of the item reaches one of ``classes`` via P279*."""
        cl = self.closure(classes)
        return any(p in cl for p in qid_p31)


# ------------------------------------------------------------------ claim helpers
def claim_ids(e: dict, pid: str) -> list[str]:
    """QIDs used as values of property ``pid`` in a FULL entity (item-valued claims)."""
    return [c["mainsnak"]["datavalue"]["value"]["id"]
            for c in e.get("claims", {}).get(pid, [])
            if c["mainsnak"].get("datavalue") and c["mainsnak"]["datavalue"]["type"] == "wikibase-entityid"]


def claim_date(e: dict, pid: str):
    """First time-valued claim of ``pid`` as a ``date`` (missing month/day -> 1), or None."""
    for c in e.get("claims", {}).get(pid, []):
        dv = c["mainsnak"].get("datavalue")
        if dv and dv["type"] == "time":
            return parse_wd_time(dv["value"]["time"])
    return None


def parse_wd_time(t: str):
    """Wikidata time value -> date; BC years -> None.

    Handles BOTH wire formats: the API's ``+1947-12-22T00:00:00Z`` (leading
    plus) and the SPARQL endpoint's ``1947-12-22T00:00:00Z`` (no sign).  A
    parser written for only one of the two silently drops EVERY date of the
    other — this happened once, which is why the sign must be parsed, not
    sliced by position.
    """
    from datetime import date

    m = re.match(r"^([+-]?)(\d{4,})-(\d{2})-(\d{2})", t or "")
    if not m or m.group(1) == "-":
        return None
    try:
        return date(int(m.group(2)), int(m.group(3)), int(m.group(4)))
    except ValueError:
        return None


def best_label(e: dict, langs=("en", "nl")) -> str | None:
    """English, else Dutch, else any label of the entity."""
    labels = e.get("labels", {})
    for lang in langs:
        if lang in labels:
            return labels[lang]["value"]
    return next(iter(labels.values()), {}).get("value")