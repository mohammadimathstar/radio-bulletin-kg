Person, Organization, Place

Good timing to settle the structure before code accrues. The organizing principle I'd recommend: **one package directory per pipeline stage, because each stage has its own cache and its own I/O contract** — that lets you re-run scoring without re-fetching Wikidata, or re-run JEV linking without re-clustering. The other load-bearing decision is separating **mention** (a surface occurrence, which is what the IE JSONL actually contains) from **entity** (the resolved real-world thing / QID) in your data model — the whole resolution pipeline exists to lift one to the other.

Here's a structure that maps 1:1 onto the steps you described:

```
radio-bulletins-kg/
├── README.md
├── pyproject.toml              # pip install -e . makes the package importable
├── .env.example                # JEV_API_KEY, WIKIDATA_USER_AGENT, embedding model
├── .gitignore                  # ignore data/, cache/, *.log
├── config/
│   ├── pipeline.yaml           # scoring weights, top_k candidates, thresholds
│   ├── type_mapping.yaml       # IE type/subtype -> Wikidata class (P31) + filter (P106)
│   └── namespaces.yaml         # KG prefix prefixes (ex:, wd:, wdt:)
├── radio_bulletins_kg/
│   ├── __init__.py
│   ├── models.py               # dataclasses: Mention, Cluster, Candidate, Entity, Relation, KGNode, KGEdge
│   ├── io/
│   │   └── ie_loader.py        # read radio-bulletins-ie JSONL -> Mention/Relation objects
│   ├── clustering/
│   │   ├── normalize.py        # surface normalization (titles, accents, case) -- mirror IE's normalize_surface
│   │   └── cluster.py          # group mentions by (normalized surface, broad type) across the WHOLE corpus
│   ├── wikidata/
│   │   ├── client.py           # SPARQL + wbsearchentities + wbgetentities, rate-limited
│   │   ├── search.py           # candidate retrieval per cluster, filtered by P31 / subclass
│   │   ├── profile.py          # download + cache candidate profiles (claims dump)
│   │   └── cache.py            # disk cache keyed by query string -> never refetch
│   ├── scoring/
│   │   ├── features.py         # feature funcs: string_sim, type_match, occupation_filter
│   │   ├── embeddings.py       # context embedding model wrapper (multilingual, Dutch)
│   │   └── ranker.py           # weighted combination from config/pipeline.yaml -> ranked candidates
│   ├── linking/
│   │   ├── jev_client.py       # JEV API wrapper (prompt build, retry, parse)
│   │   └── resolver.py         # top-N candidates + context -> JEV -> QID (or NIL)
│   ├── enrich/
│   │   └── merge.py            # fold Wikidata profile properties into resolved entities
│   ├── kg/
│   │   ├── build.py            # assemble nodes + edges from enriched entities + IE relations
│   │   └── serialize.py        # RDF/turtle, JSON-LD, GraphML, Neo4j CSV backends
│   └── pipeline.py             # orchestrator: --stage clustering|candidates|score|link|enrich|kg
├── scripts/                    # thin CLI wrappers + one-off analysis
├── data/
│   ├── input/                  # the IE JSONL (gitignored; keep one tiny sample in repo)
│   ├── interim/                # clusters.jsonl, candidates.jsonl, scores.jsonl, links.jsonl
│   └── output/                 # kg.ttl, kg.jsonld, kg.graphml, neo4j_csv/
├── cache/                      # wikidata API response cache (gitignored)
├── notebooks/
└── tests/
```

How each piece maps to your plan:

- **"clustering using string matching + type filtering"** → `clustering/`. Crucially, cluster over the *entire corpus*, not per-bulletin — the payoff of entity resolution is merging `den Bel HENDRICKX` across the 30 bulletins he appears in. Reuse the same surface normalization the IE repo already has so cluster keys stay consistent.
- **"extract candidates from wikidata per cluster"** → `wikidata/search.py`. Query `wbsearchentities` per cluster label, then filter by `P31`/subclass using `config/type_mapping.yaml`. That mapping table is the bridge between your IE taxonomy and Wikidata — e.g. `Athlete` → `Q5` (human) with `P106` occupation ∈ {`Q2309784` cyclist, `Q15117306` footballer, …}; `GeopoliticalEntity` → `Q6256`/`Q3624078`; `SportsClub` → `Q476028`. Ground it in the exact subtype set you already built.
- **"weighted combination: context embedding + word match + type + occupation filter"** → `scoring/`. Make each signal a separate feature function in `features.py` and combine them in `ranker.py` using weights read from `config/pipeline.yaml`, so tuning is a config edit, not a code change.
- **"download profiles of candidates"** → `wikidata/profile.py` + `cache.py`. This is the most important module to get right: `wbgetentities` is slow and rate-limited, and you'll re-run stages constantly, so cache every response to disk keyed by QID/query and never refetch.
- **"use JEV to connect each mention to a QID"** → `linking/`. Isolate JEV behind `jev_client.py` so prompt construction and parsing live in one place.
- **"enrich entities by wikidata"** → `enrich/merge.py`.
- **"create KG"** → `kg/build.py` + `kg/serialize.py`.

Four design decisions you'll hit early — worth deciding now rather than mid-code:

1. **Mention vs entity in `models.py`.** The IE JSONL is mention-level (each "entity" dict is really one occurrence with its own `context` and offsets). Make `Mention` and `Entity` distinct dataclasses; a `Cluster` holds mentions, an `Entity` holds the resolved QID + Wikidata profile + the set of mentions that map to it. This keeps the resolution pipeline honest.
2. **NIL handling.** Many 1939 local Dutch figures (a Deurne burgemeester, a museum director) have no Wikidata QID. JEV must be able to return NIL, and you need a scheme for minting local URIs (`<your-kg>/entity/<cluster-id>`) for those, so they still become KG nodes. Without this, unresolved mentions silently disappear.
3. **JEV per-cluster vs per-mention.** You said "connect each mention to a QID," but a cluster shares its candidate set. Cheapest correct approach: resolve per-cluster when the cluster is clean (one QID for the whole group), and fall back to per-mention only when JEV finds the cluster ambiguous. This cuts JEV calls dramatically.
4. **KG serialization format.** RDF/turtle is Wikidata-native and interoperable; a property-graph export (Neo4j CSV or GraphML) is better for visualizing and querying the result. I'd have `serialize.py` support both behind one `build.py` so the graph construction itself is format-agnostic.

Two small operational notes: gitignore `data/` and `cache/` (keep one tiny IE sample in the repo so tests and README examples run); and give the orchestrator (`pipeline.py`) a `--stage` flag so each stage is independently runnable and resumes from the previous stage's `data/interim/` output. That resumability is what makes the stage-per-directory layout earn its keep.

Want me to scaffold this — create the repo skeleton with `pyproject.toml`, the package dirs with `__init__.py` stubs, the `config/*.yaml` templates (including a first draft of `type_mapping.yaml` filled in from your actual Person/Place/Organization/Event subtype taxonomy), and a `models.py` with the Mention/Cluster/Candidate/Entity dataclasses? I can do that under `/home/z/my-project/download/radio-bulletins-kg/` so you can copy it into your new repo.