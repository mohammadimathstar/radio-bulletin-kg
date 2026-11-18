# Wikidata linking stage (`linking/`)

Links the corpus clusters produced by the NEW `entity_resolution` package
(`global_clusters.json`) to Wikidata QIDs, with an accept / review / nil
decision per cluster.  The output format (`wikidata_links.jsonl`) is unchanged
from the previous version, so `candidate_profiles.py` and the judge step keep
working.

```
linking/
  config.py       ALL tunable parameters (LinkingConfig; JSON file + CLI flags)
  cache.py        bounded SQLite response cache — replaces wd_cache.json
  wikidata.py     WDClient: search / Cirrus / entities / facts / class closures
  clusters.py     loader for the NEW global_clusters.json (+ expansion hints)
  cues.py         person-cue tables, re-exported from entity_resolution.cues
  retrieve.py     query forms, RRF fusion, candidate expansion (concurrent)
  features.py     hard gates (type, temporal) + features (label, context, ...)
  decide.py       weighted scoring + accept/review/nil decision
  run_linking.py  CLI: pass 1 (concurrent, resumable) + pass 2 (coherence)
utils/io.py       shared JSONL helpers
```

## Run

```bash
python -m linking.run_linking                                # full run, resumable
python -m linking.run_linking --limit 30                     # quick test: 30 biggest clusters
python -m linking.run_linking --restart                      # delete the pass-1 file, start over
python -m linking.run_linking --config linking.json          # override any config field
python -m linking.run_linking --workers 8 --no-embeddings    # throughput knobs
python -m linking.run_linking --max-candidates 120 --stored-candidates 40 \
        --search-limit 30 --cirrus-limit 40                  # recall knobs
```

Repo layout assumed (added to `sys.path` automatically):

```
repo/
  linking/              this package
  entity_resolution/    the new clustering package
  utils/io.py
```

Pass 1 appends one JSON line per finished cluster to
`work_dir/wikidata_pass1.jsonl` immediately (`tail -f` works); a rerun skips
finished clusters, and a cluster that hit a network/API error is left out and
retried on the next run — a wrong "nil" is never stored because of an error.
Pass 2 adds the coherence bonus (accepted clusters sharing a document),
re-decides, flags duplicate-QID conflicts and writes the final file.

## What changed, and why it is fast now

The previous version processed 15k clusters in days.  The bottlenecks, and
their fixes:

| Old | New |
| --- | --- |
| serial HTTP, `time.sleep(0.1)` after every call, fresh TLS handshake per call | shared sessions, a token-bucket rate limiter, `workers` clusters x `queries_workers` searches in flight |
| every entity fetched WITH all claims (10-60x larger payloads; P31 + dates hidden inside) | light entity fetch (labels/descriptions/aliases/sitelinks); P31 + gate dates come from ONE compact SPARQL batch per ~300 candidates (`f\|QID`) |
| SPARQL P31/P279* type check per 100 candidates | P279\* closure of each allowed root fetched ONCE per root (cached ~90 days), type gate is a local set intersection |
| per-cluster wbgetentities batches | requests coalesced ACROSS clusters and deduplicated (countries recur in hundreds of clusters) |
| mention sentences re-encoded per candidate | one embedding batch per cluster, embeddings memoized in memory |
| `wd_cache.json` rewritten IN FULL every 25 clusters, growing forever | SQLite cache: per-key upsert, per-kind TTL (search 14d, entities/facts/closures 90d), size cap with eviction of least valuable kinds first, VACUUM on close |

Cache keys keep the old `s|`, `c|`, `e|`, `l|`, `w|` scheme, so scripts that
shared `wd_cache.json` can move to the same SQLite file (see `cache.kind_of`).
Facts are versioned `f2|`: entries under the old `f|` were written while
`parse_wd_time` still expected the API's leading `+` and therefore carry NO
dates (the temporal gate saw nothing); they are ignored and age out as "misc".

Expected speed: roughly 1-2 s wall time per cluster amortised (with the default
6+12 workers) — 15k clusters in the range of hours, not days.  Three things
make the early minutes look much slower than the steady state:

* **Closures are warmed up once, at start-up, in parallel** (`[warmup] class
  closures ready`) — before that change they were fetched lazily, so the first
  clusters of each type stalled on their root closures (up to ~40 s per broad
  root), and several concurrent clusters could even fetch the SAME root in
  parallel (no lock; fixed).
* **The progress line shows two rates** — `recent` (last 25 clusters) and
  `avg` (since start).  The ETA uses the recent rate, so it drops sharply
  after warm-up; the old line averaged everything since t=0 and showed
  e.g. "~8435 min left" right after the closures.
* **Reuse the same `--work-dir`**: the closure/entity/search cache lives there
  (`wd_cache.db`).  A fresh work dir pays the one-time closure cost again; a
  reused one starts with zero `[closure]` lines.

`[skip] id: error` means a cluster raised after `cluster_retries` (default 3,
with backoff — transient hiccups like `ChunkedEncodingError` are retried
in-run).  Skipped clusters are not written, so nothing wrong is stored; rerun
the script (without `--restart`) to process only those.  If skips are
FREQUENT, Wikimedia is throttling you: lower `--queries-workers` (and
`rate_api`/`rate_sparql`) instead of raising them.

## Retrieval notes (CirrusSearch quirks)

Two facts about Wikidata's search shape the retrieval code:

* **wbsearchentities is a prefix match** on labels/aliases per language — it
  cannot find an entity whose name is a translation or a periphrasis.
* **CirrusSearch effectively requires every word of the query to match
  somewhere** ("stock exchange Yorks" -> 165 hits, NYSE rank 1;
  "stock exchange Yorksche" -> 0).  One old-Dutch word that appears in no
  Wikidata text poisons the whole query.

Therefore, besides the plain form queries:

* PERSON clusters fire cue queries ("Name British", "Name general");
* ORGANIZATION clusters fire **wordbook queries** (`ORG_CUE_WORDS` in
  `retrieve.py`): "New-Yorksche Beurs" -> "New Yorks Stock exchange" — the
  Dutch org-type word ("beurs") is replaced by its English phrase
  ("stock exchange") and the adjectival inflection is stripped (the trailing
  "-e"/"-che" is the inflection; the -s stays, giving "New-Yorks", which the
  Cirrus stemmer maps back to "york");
* all Cirrus queries go through `cirrus_term()` — hyphens become spaces (a
  leading hyphen would be an EXCLUSION operator in CirrusSearch);
* expansion forms include modernized and suffix-stripped variants;
* rejected candidates carry `found_by` (channel, form, rank), so a nil can be
  audited from the pass-1 file alone.

A translation-gap case ("New-Yorksche Beurs" -> NYSE, Q13677) still decides as
nil with the right candidate stored — the label feature cannot bridge a
translation, so acceptance is deliberately left to the profiles/JEV step.  If
you want such clusters auto-reviewed instead, relax the `weak_label` gate in
`decide.py` (e.g. allow review when `domain == 1` and retrieval is high).

## Config

`config.py` holds every knob with a documented default.  IMPORTANT: your old
`linking_config.py` was not available for this rewrite — the SCORE weights
(`w`), thresholds (`accept`, `review`, `margin`, `label_min`), context
rescaling (`ctx_floor`/`ctx_span`) and cue-support rules are sensible defaults,
NOT your tuned values.  Re-tune them on gold data, or pass your values via a
JSON file:

```json
{"accept": 0.72, "margin": 0.06, "w": {"label": 0.35, "context": 0.25, "...": 0}}
```

## Entity types

The corpus has six coarse types — `PERSON`, `PLACE`, `ORGANIZATION`, `VEHICLE`,
`EVENT`, `LEGAL_DOCUMENT` — and the mention subtype is NOT used as a type: the
former SHIP/SUBMARINE/AIRCRAFT types gate on the combined `VEHICLE` roots, and
battles/summits/meetings on the combined `EVENT` roots
(`features.ALLOWED_CLASSES`).

Which types get linked is a knob: `cfg.entity_types` (default
`PERSON,PLACE,ORGANIZATION,VEHICLE`) or the CLI flag
`--types PERSON,PLACE,ORGANIZATION,VEHICLE`.  Clusters of types outside the
list are emitted as nil records with blocker `type_not_selected` — no network
calls, same record key set — and are NOT written to the pass-1 file, so
enabling a type later links those clusters without `--restart`.  Unknown type
names abort the run up front.

## Verify before trusting

* the class roots in `features.ALLOWED_CLASSES` / `COUNTRY_CLASSES` (they match
  the corpus's coarse types; vehicles and events gate on combined roots);
* the temporal date properties used by the gate (P569/P570 for persons,
  P571/P729/P580 start and P576/P730/P582 end otherwise); ships' sink/retire
  dates often live in `P793`+`P585` (significant event), which the gate does
  NOT read — such vessels simply have no end date (`ok`, not `ended`);
* the score weights/thresholds (see above).

## Compatibility with the next steps

* final records keep the exact key set of the previous version
  (`cluster_id, cluster_key, type, n_mentions, n_docs, status, qid, label,
  score, margin, evidence, flags, blockers, alternatives, candidates,
  expanded, best_candidate, reason`), types stay UPPER-case;
* `linking.wikidata` exposes the pieces `candidate_profiles.py` needs
  (`wd_entities`-equivalent `entities(qids, claims=True)`, batched label
  lookup to be added there, claim helpers `claim_ids` / `claim_date`);
* person-cue tables come from `entity_resolution.cues` via `linking.cues`, so
  clustering and linking cannot drift apart.
