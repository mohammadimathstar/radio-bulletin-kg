# Candidate profiles (`candidate_profiles/`)

Compact, type-specific facts about a Wikidata item, shown to the judge (an LLM
or JEV) that has to decide which candidate a mention refers to.  The package
makes no decisions itself: it only collects facts (Wikidata claims + Wikipedia
summaries), stores them as one JSONL line per candidate, and renders a
deterministic text block per profile for the judge prompt.

It is **independent of candidate extraction**: the links JSONL file that the
extraction stage wrote is the only input, read with the repo's shared
`utils/io.py` helpers.  There is no import of, or host detection for, the
extraction code — point `final_path` at the file and run.

Where it sits in the pipeline:

```
entity_resolution        candidate_extraction              candidate_profiles              candidate_selection
global_clusters.json  ->  wikidata_links.jsonl          ->  candidate_profiles.jsonl  ->  (future judge)
clusters                 clusters + candidate QIDs         facts per candidate QID         picks the right QID
                         + accept/review/nil               (this package, read-only
                                                           for the judge)
```

Dependencies: `requests` and the repo's `utils/io.py` — nothing else.

## What each file does

```
candidate_profiles/
  __init__.py    public API: build_profile, profile_block, load_profiles,
                 select_records, candidate_pairs, main, normalize_type, spec_for, Config
  types.py       WHICH properties a profile contains, per type (pure data)
  extract.py     HOW values are pulled out of a Wikidata entity (pure, no I/O)
  config.py      ALL tunable parameters in one Config dataclass + JSON/CLI overrides
  api.py         the network layer: entity fetch, label resolution, Wikipedia
                 summaries — all cached in one JSON file
  profile.py     build_profile(): assembles the profile dict of one candidate
  render.py      profile_block(): deterministic text view for the judge prompt
  collect.py     the runner: batch CLI over the links file (resumable)
  __main__.py    makes `python -m candidate_profiles` work
  README.md      this file
```

### types.py — the profile specs

`PROFILE_SPEC` maps each of the six canonical types to the Wikidata
properties its profile contains (see the field tables below).  Every type
literal a profile can be asked for resolves through `TYPE_ALIASES` /
`normalize_type()` onto one of those six, so the canonical names of the
clustering (`"Person"`, `"Vehicle"`) and the historical fine-grained literals
of the old IE (`"SHIP"`, `"ARMED_CONFLICT"`, …) share one spec — and one
stored profile per `(qid, type)`:

| literal | spec key | | literal | spec key |
|---|---|---|---|---|
| PERSON | PERSON | | SHIP, SUBMARINE, AIRCRAFT | VEHICLE |
| ORGANIZATION | ORGANIZATION | | ARMED_CONFLICT | EVENT |
| PLACE | PLACE | | DIPLOMATIC_MEETING | EVENT |
| EVENT | EVENT | | LEGALDOCUMENT | LEGAL_DOCUMENT |
| LEGAL_DOCUMENT | LEGAL_DOCUMENT | | VEHICLE | VEHICLE |

`spec_for(etype)` returns the spec entry (empty for unknown types — a profile
is still built, just with the common fields only).

### extract.py — pure claim extractors

Take the full entity payload (wbgetentities WITH claims) and return plain
values; no I/O, no network, trivially testable:

| function | returns |
|---|---|
| `claim_ids(e, pid)` | QIDs used as values of property `pid` (capped at `max_values`) |
| `time_str(e, pid)` | first time claim, formatted by its precision: `1925-12-05` (day) / `1925-12` (month) / `1925` (year); BC years keep the sign |
| `coordinates(e)` | `lat, lon` (2 decimals) from P625 |
| `quantity(e, pid)` | first quantity claim rounded and thousand-separated (`22,584`) |
| `positions(e)` | positions held (P39) as `(qid, start year, end year)` from qualifiers P580/P582 |

### config.py — every parameter in one place

One `Config` dataclass carries the defaults; the module singleton
`config.CONFIG` is what the code reads (at call time, so overrides apply
without reimporting).  Precedence: **defaults < `--config` JSON file < CLI
flags**.

| field | default | meaning |
|---|---|---|
| `final_path` | `./data/output/links/wikidata_links.jsonl` | links file of candidate extraction (input) |
| `out_path` | `./data/output/links/candidate_profiles.jsonl` | profiles output (appended, resumable) |
| `cache_path` | `./data/output/links/profiles_wd_cache.json` | JSON cache of entities, labels, summaries |
| `topk` | 10 | candidates per cluster (set >= the judge's topk) |
| `statuses` | `["review"]` | pipeline statuses to cover |
| `below` / `min_score` | – | also cover clusters with score below this / skip clusters below this |
| `limit` | – | build at most N profiles in this run |
| `batch` | 50 | profiles per chunk (entities fetched per chunk) |
| `api_batch` | 50 | QIDs per wbgetentities call (MediaWiki max 50) |
| `no_summary` | False | skip Wikipedia summaries (much faster) |
| `summary_chars` | 400 | trim length of the summary |
| `wd_api` / `user_agent` | wikidata.org / `candidate-profiles/1.0 …` | endpoint + User-Agent |
| `timeout` / `summary_timeout` / `request_sleep` | 60 / 20 / 0.1 | per-request timeout (s) / summary timeout / pause between calls |
| `entity_langs` / `label_langs` / `summary_langs` | `en\|nl\|de\|fr` / `en\|nl` / `[en, nl]` | language preferences |
| `max_values` / `max_aliases` | 6 / 8 | values kept per item property / aliases kept |

A `--config profiles.json` file overrides any subset (keys = field names,
unknown keys are ignored); every field is ALSO auto-generated as a CLI flag
(`--topk`, `--statuses review,nil`, `--min-score 0.3`, `--final-path …`,
`--no-summary` …), so there is no separate flag table to keep in sync.  In
code: `config.Config()` for a fresh copy, `config.CONFIG` for the singleton.

### api.py — the network layer

| function | does |
|---|---|
| `wd_entities(qids, claims=True)` | wbgetentities in batches of `api_batch`, cache-first (`ep\|QID`); returns `{qid: entity}` |
| `wd_labels(qids)` | English (else Dutch) labels, batched and cached (`l\|QID`); unknown labels fall back to the QID |
| `wiki_summary(e, max_chars)` | intro of the English (else Dutch) Wikipedia article (`enwiki`/`nlwiki` sitelink), cached (`w\|lang\|title`); a missing article yields `""`, other errors are raised so the batch can skip and retry later |

All three go through one `JsonCache` file (`cache_path`) and `check_api()`
(which turns MediaWiki API errors into `requests.RequestException`).

### profile.py — build_profile

`build_profile(qid, etype, with_summary=True, summary_chars=400)` fetches the
entity and returns the profile dict.  Common fields for every type:

`qid, type, label, label_other_languages` (≤4)`, description, description_nl,
aliases` (nl/en/de/fr, ≤`max_aliases`, main label and other-language labels
excluded)`, instance_of` (labels of P31), `sitelinks` (count),
`wikipedia_summary`.

Type-specific fields come from the spec: item-valued properties are resolved
to labels, dates are formatted with their precision (via `time_str`),
positions get their qualifier years.  **Empty fields are omitted** (None, ""
and [] never appear), which keeps profiles and judge blocks short.  A missing
entity does not crash: the profile degrades to the common fields with the QID
as label.

### render.py — profile_block

`profile_block(prof)` renders one profile deterministically (same profile ->
same text, suitable as an LLM prompt input): first line `label (qid) -
description`, then `aka:` (aliases + other-language labels), then the
type-specific fields in a fixed order with readable names (`born:`,
`operator:`, `positions:` … per `BLOCK_FIELDS`), then `desc (nl)`, sitelinks
and the quoted Wikipedia extract.  Unknown/legacy fields (e.g. profiles stored
by an older version) are appended instead of dropped, so no information is
lost in rendering.

### collect.py — the batch runner

Selection, collection and storage over the links file (JSONL I/O via
`utils/io.py`):

| function | does |
|---|---|
| `select_records(records, statuses, below=None, min_score=None)` | records that have candidates AND (`status` in `statuses` OR `score` < `below`) AND `score` >= `min_score` |
| `candidate_pairs(records, topk)` | unique `(qid, normalized cluster type)` over the first `topk` candidates per record |
| `load_profiles(path)` | stored profiles as `{(qid, type): profile}`; `{}` if the file is missing |
| `main()` / `cli()` | fetch entities per chunk, build profiles, append to `out_path`, save the cache; network errors print `[skip …]` and leave that profile for the next run |

Resumable: profiles already in the file are skipped, so reruns only fill the
gaps.  The judge step only READS the result (`load_profiles`) — it makes no
Wikidata/Wikipedia calls.

### __main__.py

`python -m candidate_profiles …` — forwards to `collect.cli()`.

## Profile contents per type

Common fields first (see build_profile above), then:

| type | item-valued fields (property) | date fields (property) | extras |
|---|---|---|---|
| PERSON | occupations (P106), citizenship (P27), place_of_birth (P19), place_of_death (P20), member_of_party (P102), employer (P108), military_rank (P410) | birth_date (P569), death_date (P570) | positions_held: P39 with P580/P582 qualifier years, `Name (1940-1945)` |
| ORGANIZATION | country (P17), headquarters (P159), industry (P452), chairperson (P488), parent_organization (P749), founded_by (P112) | inception (P571), dissolved (P576) | |
| PLACE | country (P17), located_in (P131) | inception (P571), dissolved (P576) | coordinates (P625), population (P1082) |
| VEHICLE | operator (P137), owner (P127), manufacturer (P176), country (P17), country_of_origin (P495), ship_class (P289) | inception (P571), service_entry (P729), service_retirement (P730), first_flight (P606) | |
| EVENT | location (P276), participants (P710), part_of (P361), country (P17) | point_in_time (P585), start (P580), end (P582) | |
| LEGAL_DOCUMENT | signatories (P1891), jurisdiction (P1001), country (P17) | inception (P571), publication_date (P577), point_in_time (P585) | |

## Output format

One JSONL line per candidate at `out_path`:

```json
{"qid": "Q8016", "type": "PERSON", "profile": {"qid": "Q8016", "type": "PERSON",
  "label": "Winston Churchill", "description": "British statesman, soldier and writer",
  "aliases": ["Winston Spencer Churchill", "Sir Winston Churchill"],
  "instance_of": ["human"], "occupations": ["politician", "journalist", "painter", "historian"],
  "citizenship": ["United Kingdom"], "birth_date": "1874-11-30", "place_of_birth": ["Blenheim Palace"],
  "death_date": "1965-01-24", "place_of_death": ["London"],
  "positions_held": ["Prime Minister of the United Kingdom (1940-1945)",
                     "Prime Minister of the United Kingdom (1951-1955)"],
  "member_of_party": ["Conservative Party"], "sitelinks": 266,
  "wikipedia_summary": "Sir Winston Leonard Spencer Churchill was a British statesman, ..."}}
```

## Run

```bash
python -m candidate_profiles --topk 10 --statuses review              # defaults from config.py
python -m candidate_profiles --config profiles.json --limit 20        # JSON file + flags
python -m candidate_profiles --topk 15 --statuses review,nil --min-score 0.3 --no-summary
python -m candidate_profiles --final-path /other/wikidata_links.jsonl # different links file
```

Run it from a directory that CONTAINS the package (repo root / `src/`),
never from inside the package folder itself: there `types.py` shadows the
stdlib `types` module and even `python -m` fails to start.

Use the same `--statuses`/`--below` and a `--topk` at least as large as the
judge's, so every candidate the judge will see has a stored profile.

## Use in the judge step

```python
from candidate_profiles import load_profiles, profile_block

profiles = load_profiles("./data/output/links/candidate_profiles.jsonl")  # {(qid, type): profile}

for cand in record["candidates"][:topk]:
    block = profile_block(profiles[(cand["qid"], record["type_normalized"])])
    # prompt: "Given the context: <sentence/paragraph>
    #          Which candidate does <mention> refer to?
    #          Candidate A: <block A>  Candidate B: <block B> ..."
```

What a block looks like (live example, summary truncated):

```
Winston Churchill (Q8016) - British statesman, soldier and writer (1874–1965)
aka: Winston Spencer Churchill, Sir Winston Churchill
instance of: human
occupations: politician, journalist, painter, historian
citizenship: United Kingdom
born: 1874-11-30
born in: Blenheim Palace, Woodstock
died: 1965-01-24
died in: London
positions: Prime Minister of the United Kingdom (1940-1945), Prime Minister of the United Kingdom (1951-1955)
party: Conservative Party
sitelinks: 266
Wikipedia: "Sir Winston Leonard Spencer Churchill was a British statesman, ..."
```

Key semantics:

* profiles are keyed by the **cluster's** type: a ship retrieved as a PERSON
  candidate gets a PERSON profile — its `instance_of: ship` then exposes the
  mismatch to the judge instead of hiding it;
* the stored `type` is `normalize_type()` of the cluster type, so canonical
  (`"Person"`) and historical (`"SHIP"`) literals share one stored profile;
* empty fields are omitted; a failed network call leaves the profile unstored
  (rerun resumes);
* entity payloads, labels and summaries are cached, so reruns and the judge
  step cost no API calls.

## Recommended repo layout

The pipeline reads as three stages in three folders:

```
repo/
  candidate_extraction/    clusters -> candidate QIDs + accept/review/nil
  candidate_profiles/      this package (Wikidata/Wikipedia facts per candidate)
  candidate_selection/     future: the LLM judge (profiles + context -> chosen QID)
```

`candidate_profiles` only needs `utils/io.py` as a sibling (same import
bootstrap as `candidate_extraction/run_linking.py`) and `requests`.  All
in-package imports are relative, so the folder name itself is free.

## Tests

`scripts/test_candidate_profiles.py` — 77 offline checks: the Config dataclass
(JSON + CLI overrides, auto-generated flags), fixture entities for all six
types, normalization, omissions, block rendering, resumable batch run — plus
`--live` verification of all six types on real Wikidata/Wikipedia.
