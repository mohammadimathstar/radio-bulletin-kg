"""Run the two-pass Wikidata linking pipeline (CLI entry point).

Repo layout expected (paths added to sys.path automatically):

    repo/
      linking/            <- this package
      entity_resolution/  <- the NEW clustering package (cues tables)
      utils/io.py         <- shared JSONL helpers

Usage
-----
    python -m linking.run_linking                              # full run, resumable
    python -m linking.run_linking --limit 30                   # quick test: 30 biggest clusters
    python -m linking.run_linking --restart                    # ignore an existing pass-1 file
    python -m linking.run_linking --config linking.json        # any config field as JSON
    python -m linking.run_linking --workers 8 --no-embeddings  # throughput knobs
    python -m linking.run_linking --max-candidates 120 --stored-candidates 40 \
               --search-limit 30 --cirrus-limit 40             # recall knobs

Pass 1 (network, resumable, written incrementally):
    for every corpus cluster: retrieval (RRF over channels, concurrent) -> hard
    gates (type via local P279* closure, not-yet-existing) -> features -> decision.
    One line per cluster is appended to ``work_dir/wikidata_pass1.jsonl`` as soon
    as it is finished (tail -f works); a rerun skips finished clusters.

Pass 2 (cached, fast):
    coherence bonus from accepted clusters sharing a document -> re-decide ->
    consistency checks across clusters -> final ``wikidata_links.jsonl``.

Output records keep the exact key set of the previous version, so
candidate_profiles.py and the judge step continue to work (they are adapted
separately).  Types are UPPER-case ("PERSON") as before.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, wait, ThreadPoolExecutor
from pathlib import Path

# make sibling packages importable whether the package lives at repo root or nested
_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from .cache import SQLiteCache  # noqa: E402
from .clusters import expansion_hints, load_clusters  # noqa: E402
from .config import LinkingConfig  # noqa: E402
from .cues import NAT_GROUPS, TITLES  # noqa: E402
from .decide import decide, total  # noqa: E402
from .features import ALLOWED_CLASSES, COUNTRY_CLASSES, RELATION_PROPS, Embedder, Scorer  # noqa: E402
from .retrieve import Retriever  # noqa: E402
from .wikidata import WDClient, claim_ids  # noqa: E402
from .io import append_jsonl, read_jsonl, write_jsonl  # noqa: E402


class Linker:
    """Wires cache + client + retrieval + scoring together; runs both passes."""

    def __init__(self, cfg: LinkingConfig):
        self.cfg = cfg
        self.cache = SQLiteCache(cfg.cache_path, cfg.cache_ttl_days, cfg.cache_max_mb)
        self.wd = WDClient(cfg, self.cache)
        self.queries_pool = ThreadPoolExecutor(cfg.queries_workers)      # search fan-out
        self.clusters_pool = ThreadPoolExecutor(cfg.workers)             # clusters in flight
        self.embedder = Embedder(cfg)
        self.retriever = Retriever(self.wd, cfg, NAT_GROUPS, TITLES, self.queries_pool)
        self.scorer = Scorer(self.wd, cfg, self.embedder)

    def close(self):
        self.queries_pool.shutdown(wait=True)
        self.clusters_pool.shutdown(wait=True)
        self.cache.close()

    # ------------------------------------------------------------------ one cluster
    def link_one(self, c: dict, hint_forms: list[str]) -> dict:
        """Retrieve + score + decide one cluster; raises on network/API errors."""
        cfg = self.cfg
        qids, rrf, found = self.retriever.retrieve(c, extra=hint_forms)
        cands, rejected, n_retrieved = self.scorer.score_cluster(c, qids, rrf, found, hint_forms)
        expansion = None
        if cfg.expand and len(cands) < cfg.min_candidates:
            more, rej = self._expand_cluster(c, hint_forms, {x["qid"] for x in cands})
            expansion = dict(before=len(cands), after=len(cands) + len(more))
            cands, rejected = cands + more, rejected + rej
        cands.sort(key=lambda x: total(x["feats"], cfg.w, cfg.penalty), reverse=True)
        res = decide(c, cands, cfg)
        if not cands:                    # explain the nil
            res["reason"] = (f"no type/time-valid candidate ({n_retrieved} retrieved; "
                             f"rejected: {dict(Counter(r['why'] for r in rejected))})")
        return dict(cluster_id=c["id"], cluster_key=c["key"], type=c["type"],
                    n_mentions=c["n_mentions"], n_docs=c["n_docs"], result=res,
                    candidates=cands[:cfg.max_stored_candidates],
                    rejected=rejected[:10], expansion=expansion)

    def _expand_cluster(self, c: dict, hint_forms: list[str], known: set):
        """Search more candidates for a cluster and score them (same gates/features)."""
        qids, rrf, found = self.retriever.retrieve_more(c, hint_forms, known)
        if not qids:
            return [], []
        more, rej, _ = self.scorer.score_cluster(c, qids, rrf, found, hint_forms)
        return more, rej

    def _warmup_closures(self, clusters: list[dict]) -> None:
        """Fetch every root closure the input needs, in parallel, BEFORE pass 1.

        Without this the first clusters of each type stall on their root
        closures (up to ~40s per broad root) and the early clusters/s rate —
        and the ETA built on it — is meaningless.
        """
        roots = []
        for t in {c["type"] for c in clusters}:
            roots += ALLOWED_CLASSES.get(t, [])
        if any(c["type"] == "PLACE" for c in clusters):
            roots += COUNTRY_CLASSES
        t0 = time.monotonic()
        self.wd.closure_roots(roots, pool=self.queries_pool)
        print(f"[warmup] class closures ready ({time.monotonic() - t0:.0f}s)", flush=True)

    def _link_one_retry(self, c: dict, hints: list):
        """link_one with in-run retries: a transient network error should not
        push a cluster to the next run (the cache makes retries nearly free)."""
        for attempt in range(max(1, self.cfg.cluster_retries)):
            try:
                return self.link_one(c, hints)
            except Exception:
                if attempt == self.cfg.cluster_retries - 1:
                    raise
                time.sleep(self.cfg.retry_backoff * (attempt + 1))

    # ------------------------------------------------------------------ pass 1
    def pass1(self, clusters: list[dict], hints: dict, restart: bool) -> dict[str, dict]:
        """Link every cluster; append each result to the pass-1 file when done.

        Resumable: cluster ids already in the file are skipped.  A cluster whose
        retries are exhausted (persistent network/API error) is printed as
        [skip] and left for the next run — it is not written, so a wrong "nil"
        can never be stored because of an empty or error response.
        """
        cfg = self.cfg
        p1 = Path(cfg.pass1_path)
        p1.parent.mkdir(parents=True, exist_ok=True)
        if restart and p1.exists():
            p1.unlink()
        records = {r["cluster_id"]: r for r in read_jsonl(cfg.pass1_path)}
        todo = [c for c in clusters if c["id"] not in records]
        print(f"pass 1: {len(records)} done, {len(todo)} to do "
              f"({cfg.workers} clusters in flight, {cfg.queries_workers} concurrent searches)",
              flush=True)

        t0, done_n = time.monotonic(), 0
        last_t, last_n = t0, 0                        # window for the recent rate
        futures = {self.clusters_pool.submit(self._link_one_retry, c, hints.get(c["id"], [])): c
                   for c in todo}
        pending = set(futures)
        with open(cfg.pass1_path, "a", encoding="utf-8") as fh:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    c = futures[fut]
                    try:
                        rec = fut.result()
                    except Exception as ex:      # noqa: BLE001 — one cluster failing
                        print(f"[skip] {c['id']}: {ex!r}", flush=True)
                        continue                    # retried on the next run
                    append_jsonl(fh, rec)
                    records[c["id"]] = rec
                    done_n += 1
                    if done_n % 25 == 0:
                        now = time.monotonic()
                        recent = (done_n - last_n) / max(now - last_t, 1e-9)
                        avg = done_n / max(now - t0, 1e-9)
                        eta = (len(todo) - done_n) / max(recent, 1e-9) / 60
                        print(f"  {done_n}/{len(todo)}  (recent {recent:.1f}/s, "
                              f"avg {avg:.1f}/s, ~{eta:.0f} min left)", flush=True)
                        last_t, last_n = now, done_n
                        self.cache.prune()
        return records

    # ------------------------------------------------------------------ pass 2
    def pass2(self, clusters: list[dict], records: dict[str, dict]) -> list[dict]:
        """Coherence bonus + consistency checks; returns the final records.

        Coherence: for each cluster, take QIDs of ACCEPTED clusters sharing a
        document.  A candidate gets min(k/3, 1) where k counts how many of its
        RELATION_PROPS values are those sibling QIDs or the siblings' own relation
        values (same country, part of the same body, ...).  Then decide() is
        re-run.  Clusters without accepted siblings keep coherence = null.
        Consistency: one QID used by several clusters is flagged (type_conflict
        if the types differ -> accepted is demoted to review, else
        possible_duplicate_clusters).
        """
        cfg = self.cfg
        clusters = [c for c in clusters if c["id"] in records]
        by_doc = defaultdict(list)
        for c in clusters:
            for d in c["docs"]:
                by_doc[d].append(c["id"])

        results = {c["id"]: records[c["id"]]["result"] for c in clusters}
        cands = {c["id"]: records[c["id"]]["candidates"] for c in clusters}

        new_results = {}
        for c in clusters:
            cs = cands[c["id"]]
            if not cs:
                continue
            sibs = {results[s]["qid"] for d in c["docs"] for s in by_doc[d]
                    if s != c["id"] and results[s]["status"] == "accepted"}
            if not sibs:
                continue
            sib_ents = self.wd.entities(list(sibs), claims=True)
            sib_rel = set()
            for e in sib_ents.values():
                for p in RELATION_PROPS:
                    sib_rel |= set(claim_ids(e, p))
            ents = self.wd.entities([x["qid"] for x in cs], claims=True)
            for cand in cs:
                e = ents.get(cand["qid"], {})
                rel = {x for p in RELATION_PROPS for x in claim_ids(e, p)}
                cand["feats"]["coherence"] = min((len(rel & sibs) + len(rel & sib_rel)) / 3, 1.0)
            new_results[c["id"]] = decide(c, cs, cfg)
        results.update(new_results)               # after the loop: siblings use pass-1 status

        by_qid = defaultdict(list)
        for c in clusters:
            r = results[c["id"]]
            if r["qid"] and r["status"] != "nil":
                by_qid[r["qid"]].append(c)
        for cs_ in by_qid.values():
            if len(cs_) > 1:
                kind = ("type_conflict" if len({x["type"] for x in cs_}) > 1
                        else "possible_duplicate_clusters")
                for x in cs_:
                    r = results[x["id"]]
                    r["flags"].append(kind)
                    if kind == "type_conflict" and r["status"] == "accepted":
                        r["status"] = "review"
                        r["blockers"].append("type_conflict")

        final = []
        for c in clusters:
            ranked = sorted(cands[c["id"]],
                            key=lambda x: total(x["feats"], cfg.w, cfg.penalty), reverse=True)
            final.append(dict(
                cluster_id=c["id"], cluster_key=c["key"], type=c["type"],
                n_mentions=c["n_mentions"], n_docs=c["n_docs"], **results[c["id"]],
                candidates=[dict(qid=x["qid"], label=x["label"],
                                 score=round(total(x["feats"], cfg.w, cfg.penalty), 3))
                            for x in ranked[:cfg.max_stored_candidates]],
                expanded=bool(records[c["id"]].get("expansion"))))
        return final


# ------------------------------------------------------------------ main
def _skipped_record(c: dict) -> dict:
    """Record for a cluster whose type is not enabled (cfg.entity_types).

    Same key set as :meth:`link_one` so downstream tools see a uniform file;
    the cluster never touches the network.  Skipped records are NOT written to
    the pass-1 file, so enabling the type later re-links the cluster without
    --restart; a real decision from an earlier run with a broader --types wins.
    """
    return dict(cluster_id=c["id"], cluster_key=c["key"], type=c["type"],
                n_mentions=c["n_mentions"], n_docs=c["n_docs"],
                result=dict(status="nil", qid=None, label=None, score=0.0,
                            margin=0.0, evidence={}, flags=[],
                            blockers=["type_not_selected"], alternatives=[],
                            best_candidate=None,
                            reason=f"cluster type '{c['type']}' is not enabled "
                                   "(--types / config entity_types)"),
                candidates=[], rejected=[], expansion=None)


def run(cfg: LinkingConfig, limit: int | None = None, restart: bool = False) -> None:
    """Pass 1 (incremental, resumable) then pass 2 and write the final file."""
    types = set(cfg.entity_types)
    unknown = sorted(types - set(ALLOWED_CLASSES))
    if unknown:
        raise SystemExit(f"unknown entity types {unknown} in entity_types; "
                         f"allowed: {sorted(ALLOWED_CLASSES)}")
    clusters = load_clusters(cfg.in_path, cfg.cue_min_count, cfg.cue_min_share)
    clusters.sort(key=lambda c: -c["n_mentions"])      # biggest clusters first
    if limit:
        clusters = clusters[:limit]
    selected = [c for c in clusters if c["type"] in types]
    skipped = [c for c in clusters if c["type"] not in types]
    if skipped:
        counts = dict(Counter(c["type"] for c in skipped))
        print(f"linking {len(selected)} clusters of {sorted(types)}; "
              f"{len(skipped)} clusters of other types are skipped: {counts}", flush=True)
    hints = expansion_hints(selected)

    linker = Linker(cfg)
    cache_stats = {}
    try:
        linker._warmup_closures(selected)
        records = linker.pass1(selected, hints, restart)

        missing = [c["id"] for c in selected if c["id"] not in records]
        if missing:
            print(f"WARNING: {len(missing)} clusters not linked yet (network/API errors). "
                  f"Rerun the script to process only those. First few: {missing[:5]}", flush=True)

        # Types not enabled: nil records with blocker "type_not_selected" (never
        # linked, not written to the pass-1 file; a real decision from an
        # earlier run with a broader --types wins via setdefault).
        for c in skipped:
            records.setdefault(c["id"], _skipped_record(c))

        final = linker.pass2(clusters, records)
        cache_stats = linker.cache.stats()
    finally:
        linker.close()

    write_jsonl(final, cfg.out_path)
    print(Counter(r["status"] for r in final), f"-> {cfg.out_path}")
    print("most common blockers:",
          Counter(b for r in final for b in r["blockers"]).most_common(6))
    print(f"cache: {cache_stats.get('entries', {})}")
    print(f"cache file: {cfg.cache_path}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Link corpus clusters to Wikidata QIDs")
    ap.add_argument("--in", dest="in_path", default=None, help="input global_clusters.json")
    ap.add_argument("--out", dest="out_path", default=None, help="output wikidata_links.jsonl")
    ap.add_argument("--work-dir", default=None, help="dir for the pass-1 file and the cache")
    ap.add_argument("--config", default=None,
                    help="JSON file overriding any config field (see linking/config.py)")
    ap.add_argument("--limit", type=int, default=None, help="only the N clusters with most mentions")
    ap.add_argument("--restart", action="store_true", help="delete the existing pass-1 file and start over")
    ap.add_argument("--workers", type=int, default=None, help="clusters processed concurrently (default 6)")
    ap.add_argument("--queries-workers", type=int, default=None, help="concurrent search requests (default 12)")
    ap.add_argument("--no-embeddings", action="store_true", help="skip the context feature (faster, feature=null)")
    ap.add_argument("--no-expand", action="store_true", help="do not search more candidates for clusters with few")
    ap.add_argument("--max-candidates", type=int, default=None, help=f"default {LinkingConfig().max_candidates}")
    ap.add_argument("--stored-candidates", type=int, default=None, help="candidates stored per record")
    ap.add_argument("--search-limit", type=int, default=None, help="hits per (form, language), max 50")
    ap.add_argument("--cirrus-limit", type=int, default=None, help="Cirrus hits per form, max 50")
    ap.add_argument("--min-candidates", type=int, default=None, help="search more when a cluster has fewer valid candidates")
    ap.add_argument("--cache-max-mb", type=int, default=None, help="keep the cache below this size (default 512)")
    ap.add_argument("--types", default=None,
                    help="comma-separated entity types to link, e.g. PERSON,PLACE,ORGANIZATION,VEHICLE "
                         "(default PERSON,PLACE,ORGANIZATION,VEHICLE; other clusters are emitted as nil "
                         "with blocker type_not_selected)")
    a = ap.parse_args(argv)

    cfg = LinkingConfig()
    if a.config:
        cfg.apply_json(a.config)
    if a.types:
        cfg.entity_types = [t.strip().upper() for t in a.types.split(",") if t.strip()]
    for name, val in [("in_path", a.in_path), ("out_path", a.out_path),
                      ("work_dir", a.work_dir), ("workers", a.workers),
                      ("queries_workers", a.queries_workers),
                      ("max_candidates", a.max_candidates),
                      ("max_stored_candidates", a.stored_candidates),
                      ("search_limit", a.search_limit), ("cirrus_limit", a.cirrus_limit),
                      ("min_candidates", a.min_candidates), ("cache_max_mb", a.cache_max_mb)]:
        if val is not None:
            setattr(cfg, name, val)
    if a.no_embeddings:
        cfg.use_embeddings = False
    if a.no_expand:
        cfg.expand = False

    run(cfg, limit=a.limit, restart=a.restart)


if __name__ == "__main__":
    main()