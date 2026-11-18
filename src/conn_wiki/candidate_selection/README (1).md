# candidate_judge — the judge step

The third stage of the pipeline: for every entity of every document, show the
mention context, the candidates of its corpus cluster and their profiles to
**JEV AI (TypeSafe System One)**, and let it decide which candidate the
mention refers to — or `nil`.

```
candidate_extraction  ->  candidate_profiles  ->  candidate_judge
(candidates per corpus     (facts per candidate,     (documents + candidates
 cluster, Wikidata)         keyed (qid, type))        + profiles -> decision)
```

Reads   `resolved_documents.jsonl`   entities per document      (IE/resolution step)
        `wikidata_links.jsonl`       candidates per corpus cluster (candidate_extraction;
                                     final file AND pass-1 file both work)
        `candidate_profiles.jsonl`   facts per candidate           (candidate_profiles)
Writes  `decisions.jsonl`            one line per judged entity (appended; a rerun
                                     skips finished entities)

## Modules

| file | does |
|---|---|
| `config.py` | ALL tunables in one `Config` dataclass (defaults < `--config` JSON < CLI flags); module singleton `config.CONFIG`; `build_parser()` auto-generates one flag per field |
| `texts.py` | `squash` / `snippet` / `snippets`: the `[[marked]]` mention context per entity (port of the old decide_common text logic; the constants are config fields now) |
| `jev.py` | the JEV part: `build()` (payload: instructions + state + options) and `send()` (lazy `typesafe-sdk` import; any SDK failure is wrapped in RuntimeError); `main()` = CLI |
| `collect.py` | `gold_spans` / `select_docs` / `judge()` policy / `main(build, send, name)`: the shared, model-agnostic run loop |

## Install / run

```bash
pip install typesafe-sdk
export TYPESAFE_API_KEY=...

python -m candidate_judge --dry-run 3                   # show what would be sent, no API call
python -m candidate_judge --limit-docs 5 --out-path small.jsonl
python -m candidate_judge --config judge.json           # JSON file + flags
python -m candidate_judge --gold gold.json --doc-ids radio:bulletin:1939:09:01
python -m candidate_judge --entity-types ""             # judge EVERY type that has candidates
python -m candidate_judge                               # everything (rerun skips finished entities)
```

Run it from a directory that CONTAINS the package (repo root / `src/`), never
from inside the package folder itself.  `--dry-run` and the offline tests
need NO SDK and no key: the SDK is imported lazily inside `send()`.

## Inputs

* **resolved documents** — one line per document: `doc_id` (a date is parsed
  from its 3rd–5th `:`-separated parts), `clusters[]` with `cluster_id`,
  `global_cluster` (the corpus cluster id in the links file), `type`,
  `representative`, optional `surface_forms`/`nationality`/`title`/`role`,
  and `mentions[]` with `start`/`end`/`text` and `sentence`/`sentence_start`
  (+ `context`/`context_start` for short sentences).
* **links file** — BOTH record shapes of candidate extraction: the final
  file (decision fields flat: `status`, `qid`, `label`, `candidates`, ...)
  and the `wikidata_pass1.jsonl` intermediate (decision nested under
  `result`).  `collect._field()` reads either.
* **profiles file** — candidate_profiles output, lines
  `{"qid": ..., "type": ..., "profile": {...}}` with the type NORMALIZED
  (`PERSON`, `VEHICLE`, ...).  Lookups resolve
  `normalize_type(cluster type)` by IMPORTING
  `candidate_profiles.types.normalize_type` — not a copied alias table — so
  new entity types/aliases cannot drift between the two packages.

## What is sent to the model

`build(entity)` returns `{"instructions", "state", "options"}`:
`state` = `{mention (entity_type, name_in_text, spellings_in_text,
nationality/titles/roles), bulletin_date, context_sentences,
other_entities_in_this_bulletin}`; `options` = `{qid: value}` plus
`NONE_OF_THE_ABOVE` described as "none of the listed items is the entity
meant in the text".

The option value comes from `option_style`:

| style | value |
|---|---|
| `json` (default) | the FULL stored profile dict, as collected — **whatever new fields the profile builder learns for an entity type are sent automatically**, no code change needed |
| `text` | `candidate_profiles.render.profile_block()`: deterministic text ("label (qid) - description", "aka:", the type-specific facts, sitelinks, Wikipedia extract) — unknown/legacy profile fields are appended automatically |

Candidates without a stored profile fall back to `{"qid", "label"}`.

## Which entities are judged

An entity is judged when ALL of these hold:

1. its corpus cluster has at least one candidate (clusters without candidates
   are skipped — there is nothing to choose from);
2. the cluster status is in `statuses` (default `review,nil` — `accepted`
   clusters are already decided by the pipeline);
3. the cluster type is in `entity_types` — **default `PERSON, ORGANIZATION,
   PLACE, VEHICLE`, the four types your corpus has candidates for**.  Both
   the cluster's type and the config values are normalized before the
   comparison (`candidate_profiles.types.normalize_type`), so casing never
   matters and the historical aliases count as their canonical type:
   `SHIP`/`SUBMARINE`/`AIRCRAFT` clusters are judged (→ VEHICLE), while
   `ARMED_CONFLICT`/`DIPLOMATIC_MEETING` (→ EVENT) and `LEGAL_DOCUMENT`
   clusters are skipped.  `--entity-types ""` lifts the limit; the payload
   still shows the cluster's own type to the model.  Skipped clusters are
   counted as `other_type` in the summary line (only clusters that would
   otherwise be judged reach that counter).  `candidate_profiles` accepts
   the same `--entity-types`, so it never builds a profile the judge will
   not read.

## Policy and output

`judge(probs, min_p, min_ratio)` → `p1 < min_p` ⇒ `uncertain`; winner is
`NONE_OF_THE_ABOVE` ⇒ `nil`; `p1/p2 < min_ratio` ⇒ `uncertain`; else `link`.
The model must answer ONLY with the options given — any other key is a
`[skip]` and retried on the next run, exactly like a network error.

One `decisions.jsonl` line per judged entity, keys: `doc_id, cluster_id,
global_cluster, type, representative, pipeline_status, pipeline_qid,
top_candidate, options, outcome, qid, probabilities, backend, model,
mention_spans`.

Resumable on `(doc_id, cluster_id)`; `--restart` deletes the output first.
The text snippets and the known-entities context are built per DOCUMENT:
accepted clusters of the same bulletin (and entities linked earlier in the
same run) are shown to the model as `other_entities_in_this_bulletin` (at
most `max_known`).

## Small experiments

`--dry-run N` (print the first N requests), `--limit-docs N` (default: ALL
documents — the old script defaulted to 110 for experiments),
`--doc-ids a,b`, `--gold gold.json` (only annotated entities; clusters
without a `qid` count as NIL, or `missing=skip` semantics),
`--max-entities N`, plus the tunables `--min-sentence-chars`, `--max-chars`,
`--max-snippets`, `--max-known`, `--option-style`, `--model`.

## What changed vs the uploaded decide_common.py / decide_jev.py

* **package + config**: argparse constants and `BASE` paths → `config.py`
  dataclass with `--config judge.json` and auto-generated flags (same
  pattern as the other two packages); `--out` → `--out-path`.
* **new pipeline files**: the links records of candidate_extraction have a
  different key set AND a pass-1 variant (decision under `result`) — both
  accepted; profile lookups use the normalized type (the old
  `profiles.get((c["qid"], link["type"]))` would MISS, since the profiles
  file stores `PERSON`-style types while the cluster says `Person`).
* **dynamic profile features** (explicit requirement): full profile dict is
  sent as-is (`option_style=json`); `option_style=text` uses
  `profile_block()`, which renders unknown/legacy fields too; type aliases
  come from `candidate_profiles.types` (imported, single source of truth).
* **entity_types filter** (explicit requirement): decisions are limited to
  the types that have candidates — default PERSON/ORGANIZATION/PLACE/VEHICLE,
  normalized + alias-aware; skipped clusters counted as `other_type`; the
  same filter exists in candidate_profiles.
* **robustness**: SDK imported lazily with a helpful error; SDK exceptions
  wrapped in RuntimeError so the loop stays model-agnostic; list config
  fields split correctly even when their default is `None` (`doc_ids`,
  which the old "current value is a list" check would have missed);
  `--limit-docs` defaults to all documents.
* **unchanged on purpose**: the instructions wording, the state/options
  layout, the `[[...]]` marking and snippet rules, the judge policy, the
  decision line keys, `--restart`, the gold-file experiments.

## Tests

`scripts/test_candidate_judge.py` — 58 offline checks (no SDK, no network):
text preparation, the link/nil/uncertain policy, gold spans + document
selection, config flags and list splitting, and a full end-to-end run on
tempdir fixtures with a stubbed model: exact line keys, resumability,
error/unexpected-option skips retried on rerun, `--no-summary`,
`--restart`, `--max-entities`, `--dry-run`, the pass-1 links shape, both
option styles, and the entity_types filter (default four types, alias
resolution, lift/override).
