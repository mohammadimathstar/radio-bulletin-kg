"""All tunable parameters of the Wikidata linking stage in one dataclass.

This replaces the previous ``config/linking_config.py``: the dataclass carries
the DEFAULTS, an optional JSON file can override any subset, and CLI flags
override both (see ``run_linking.py``).

NOTE ON DEFAULTS: the previous values lived in your ``linking_config.py``, which
was not available to this rewrite.  The defaults below reproduce the documented
behaviour of v8 ("recognition knobs" from the docstring) but the SCORE weights,
thresholds and cache TTLs are sensible starting points, NOT tuned values —
adjust them on your gold data before trusting the accept/review/nil split.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

WD_API = "https://www.wikidata.org/w/api.php"
SPARQL = "https://query.wikidata.org/sparql"


@dataclass
class LinkingConfig:
    # ---- paths ----
    in_path: str = "./data/interim/global_clusters.json"       # new clustering output
    out_path: str = "./data/output/links/wikidata_links.jsonl"
    work_dir: str = "./data/output/links"                     # pass-1 file + cache live here
    pass1_name: str = "wikidata_pass1.jsonl"                  # resumable intermediate
    cache_name: str = "wd_cache.db"                           # SQLite cache (replaces wd_cache.json)

    # ---- endpoints / etiquette ----
    wd_api: str = WD_API
    sparql: str = SPARQL
    user_agent: str = "radio-bulletins-entity-linking/2.0 (research; contact: you@example.org)"
    timeout_api: float = 30.0
    timeout_sparql: float = 120.0
    rate_api: float = 8.0        # max requests/s to the MediaWiki API (search/cirrus/entities)
    rate_sparql: float = 1.5     # max requests/s to the SPARQL endpoint (it is slower and rate-limited)
    sparql_batch: int = 300      # QIDs per SPARQL facts query (P31 + gate dates)
    retries: int = 3             # retries per request on 429/5xx/network errors (exponential backoff)

    # ---- languages / concurrency ----
    langs: tuple = ("en", "nl", "de", "fr")      # languages fetched for labels/aliases/descriptions
    search_langs: tuple = ("nl", "en")           # wbsearchentities languages per query form
    workers: int = 6             # clusters processed concurrently (each fans out searches to queries_workers)
    queries_workers: int = 12    # concurrent search requests shared by all clusters
    cluster_retries: int = 3     # in-run retries per cluster on network/API errors before [skip]
    retry_backoff: float = 5.0   # base seconds between cluster retries (multiplied by attempt)

    # ---- entity types to link ----
    # Which of the six coarse cluster types get linked (PERSON, PLACE,
    # ORGANIZATION, VEHICLE, EVENT, LEGAL_DOCUMENT).  Clusters of other types
    # in the input are emitted as nil records with blocker
    # "type_not_selected" (no network calls, same record key set) and are NOT
    # written to the pass-1 file, so enabling a type later re-links them
    # without --restart.
    entity_types: list = field(default_factory=lambda: ["PERSON", "PLACE", "ORGANIZATION", "VEHICLE"])
    embed_batch: int = 64        # texts per embedding forward pass
    use_embeddings: bool = True  # context feature (sentence-transformers); False -> feature = null

    # ---- retrieval (recall knobs) ----
    search_limit: int = 20       # hits per (form, language) from wbsearchentities (API max 50)
    use_cirrus: bool = True      # second retrieval channel: Cirrus full-text search
    cirrus_limit: int = 20       # hits per form
    fuzzy_cirrus: bool = True    # "word~" queries for single-word forms (>= 6 chars)
    fuzzy_limit: int = 10
    max_query_forms: int = 8     # distinct strings searched per cluster
    max_candidates: int = 60     # distinct candidates kept after fusion (before the gates)
    max_stored_candidates: int = 40  # candidates written to the pass-1 and final files
    rrf_k: int = 15              # reciprocal rank fusion constant
    channel_weight: dict = field(default_factory=lambda: {"wb": 1.0, "cirrus": 1.0, "fuzzy": 0.8})

    # ---- candidate expansion (clusters with too few candidates) ----
    expand: bool = True
    min_candidates: int = 5      # fewer valid candidates than this -> search more
    expansion_limit: int = 30    # hits per query when the strongest forms are searched again
    expansion_langs: tuple = ("nl", "en", "de")  # wbsearchentities languages for the new forms
    expansion_forms: int = 6     # new query strings per cluster
    max_expansion_candidates: int = 30
    expansion_factor: float = 0.7  # cap of the retrieval feature of expansion candidates

    # ---- scoring (tune on gold data!) ----
    w: dict = field(default_factory=lambda: {        # feature weights (normalised over applicable ones)
        "label": 0.30, "context": 0.25, "retrieval": 0.15, "title": 0.10,
        "nationality": 0.10, "coherence": 0.15, "prior": 0.05, "domain": 0.05,
        "expansion": 0.10,
    })
    penalty: dict = field(default_factory=lambda: {"ended": 0.10, "anachronistic": 0.05})
    accept: float = 0.70         # score needed (with margin, no blocker) to accept
    review: float = 0.55         # score above which a non-accept goes to review
    margin: float = 0.05         # required gap best vs second-best
    label_min: float = 0.80      # best candidate must match a name at least this well
    ctx_floor: float = 0.75      # context = clip((e5 cosine - floor) / span, 0, 1)
    ctx_span: float = 0.20       # e5 cosine of unrelated text is already ~0.7-0.8
    ctx_max_mentions: int = 10   # sentences embedded per cluster (top-3 mean is the feature)

    # ---- cue support (PERSON titles / nationalities) ----
    cue_min_count: int = 3       # cue used if it appears in >= this many documents
    cue_min_share: float = 0.20  # ... or in >= this share of the cluster's documents

    # ---- cache ----
    cache_ttl_days: dict = field(default_factory=lambda: {
        # 'search' answers change and are the bulk of the entries: short TTL.
        # Entities/type checks/labels/closures are stable: long TTL.
        "search": 14, "entity": 90, "label": 90, "facts": 90, "closure": 90,
        "summary": 30, "misc": 30,
    })
    cache_max_mb: int = 512      # prune oldest shortest-TTL entries beyond this size

    # ------------------------------------------------------------------ API
    @property
    def pass1_path(self) -> str:
        return str(Path(self.work_dir) / self.pass1_name)

    @property
    def cache_path(self) -> str:
        return str(Path(self.work_dir) / self.cache_name)

    def ttl(self, kind: str) -> int:
        return self.cache_ttl_days.get(kind, self.cache_ttl_days["misc"])

    def apply_json(self, path: str) -> "LinkingConfig":
        """Override fields from a JSON file (only keys that exist as fields)."""
        p = Path(path)
        if not p.exists():
            return self
        data = json.loads(p.read_text(encoding="utf-8"))
        names = {f.name for f in fields(self)}
        for k, v in data.items():
            if k in names:
                cur = getattr(self, k)
                if isinstance(cur, tuple) and isinstance(v, list):
                    v = tuple(v)
                setattr(self, k, v)
        return self