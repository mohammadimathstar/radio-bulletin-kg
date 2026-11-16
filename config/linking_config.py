"""
Configuration for the Wikidata entity-linking pipeline.

All tunable parameters of :mod:`radio_bulletins_kg.linking.link_wikidata` live here
as a ``LinkingConfig`` dataclass, mirroring the pattern of
:class:`radio_bulletins_kg.clustering.config.ClusteringConfig`.  The linker module
instantiates a default ``cfg = LinkingConfig()`` at import time and binds its fields
to module-level names (``SEARCH_LIMIT = cfg.search_limit``, …) so the rest of the
code reads them at call time; CLI flags in ``link_wikidata.__main__`` override those
bindings.

Static lookup tables (``ALLOWED_CLASSES``, ``TYPE_MAP``, ``EVENT_SUBTYPE_MAP``,
``RELATION_PROPS``, ``TYPE_FILTER``, ``FIN_RE`` / ``FIN_WORDS``) are domain
knowledge, not tunable parameters, and stay as module-level constants in
``link_wikidata`` itself.
"""
from dataclasses import dataclass, field


@dataclass
class LinkingConfig:
    """All tunable parameters for linking NER clusters to Wikidata QIDs.

    Files
    -----
    in_path : str
        Input ``global_clusters.json`` (output of the clustering stage).
    out_path : str
        Final per-cluster links file (JSONL).
    pass1_path : str
        Provisional pass-1 file (no coherence bonus yet); written incrementally
        so the run is resumable.
    cache_path : str
        JSON cache for every Wikidata / Wikipedia response (makes reruns cheap).

    Wikidata API
    ------------
    wd_api : str
        Wikidata Action API endpoint.
    sparql : str
        SPARQL query endpoint (used for the P31/P279* type gate).
    headers : dict
        HTTP headers (a descriptive User-Agent is required by Wikidata policy).
    langs : list
        Languages fetched for labels / aliases / descriptions (wbgetentities).

    Retrieval (Reciprocal Rank Fusion over search channels)
    -------------------------------------------------------
    search_langs : tuple
        Languages tried by ``wbsearchentities`` (prefix match on labels/aliases).
    search_limit : int
        Hits per (form, language) from ``wbsearchentities`` (API max 50).
    use_cirrus : bool
        If True, also run Cirrus full-text search (searches descriptions too).
    cirrus_limit : int
        Hits per form from Cirrus (API max 50).
    fuzzy_cirrus : bool
        If True, also run a fuzzy (``word~``) Cirrus query for single-word forms
        of >= 6 characters (OCR tolerance).
    fuzzy_limit : int
        Hits per fuzzy query.
    max_query_forms : int
        Maximum distinct strings searched per cluster (surface forms + variants).
    max_candidates : int
        Distinct candidates kept after RRF fusion (before the type/time gates).
    max_stored_candidates : int
        Candidates written to the pass-1 and final files (what the judge step
        can choose from).
    rrf_k : int
        Reciprocal Rank Fusion constant (larger = flatter ranking).
    channel_weight : dict
        Weight per retrieval channel (``wb``, ``cirrus``, ``fuzzy``).

    Candidate expansion (clusters with too few type/time-valid candidates)
    ---------------------------------------------------------------------
    expand : bool
        If True, run a second search round when a cluster has fewer than
        ``min_candidates`` valid candidates.
    min_candidates : int
        Threshold below which expansion triggers.
    expansion_limit : int
        Hits per query when the strongest forms are searched again (API max 50).
    expansion_langs : tuple
        Extra languages tried for the new query forms (e.g. German for "Danzig").
    expansion_forms : int
        New query strings generated per cluster (modernised spelling, surname,
        particles removed, …).
    max_expansion_candidates : int
        New candidates kept per cluster.
    expansion_factor : float
        Cap on the retrieval feature of expansion candidates (so they cannot
        outrank normal-search candidates on retrieval alone).

    Scoring
    -------
    w : dict
        Weights of the applicable features (normalised by their sum, so they
        need not add to 1).  Features: label, context, retrieval, prior, domain,
        coherence, expansion, title, nationality.
    penalty : dict
        Penalties subtracted after normalisation (ended, anachronistic).
    accept : float
        Score a candidate needs (with a sufficient margin and no blocker) to be
        auto-accepted.
    review : float
        Score above which a non-accepted candidate is sent to review (else nil).
    margin : float
        Required gap between the best and second-best candidate for acceptance.
    label_min : float
        Best candidate whose label similarity is below this is always nil
        (unless it matches an ellipsis sibling name).
    ctx_floor : float
        Context-similarity rescale floor: e5 cosine of unrelated text is already
        ~0.7-0.8, so ``ctx = clip((cosine - ctx_floor) / ctx_span, 0, 1)``.
    ctx_span : float
        Context-similarity rescale span (see ``ctx_floor``).

    Cue support (PERSON titles / nationalities)
    --------------------------------------------
    cue_min_count : int
        A title or nationality cue is used only if it appears in at least this
        many mentions (or meets ``cue_min_share``).
    cue_min_share : float
        Alternatively, the cue is used if it appears in at least this share of
        the scanned mentions.
    """

    # ---- files ----
    in_path: str = "./data/interim/global_clusters.json"
    out_path: str = "./data/links/wikidata_links.jsonl"
    pass1_path: str = "./data/links/wikidata_links.pass1.jsonl"
    cache_path: str = "./data/links/wd_cache.json"

    # ---- Wikidata API ----
    wd_api: str = "https://www.wikidata.org/w/api.php"
    sparql: str = "https://query.wikidata.org/sparql"
    headers: dict = field(default_factory=lambda: {
        "User-Agent": "radio1939-linker/0.1 (research; contact: you@example.org)"})
    langs: list = field(default_factory=lambda: ["nl", "en", "de", "fr"])

    # ---- retrieval ----
    search_langs: tuple = ("nl", "en", "fr")
    search_limit: int = 25
    use_cirrus: bool = True
    cirrus_limit: int = 30
    fuzzy_cirrus: bool = True
    fuzzy_limit: int = 10
    max_query_forms: int = 8
    max_candidates: int = 80
    max_stored_candidates: int = 50
    rrf_k: int = 60
    channel_weight: dict = field(default_factory=lambda: {
        "wb": 1.0, "cirrus": 1.0, "fuzzy": 0.5})

    # ---- candidate expansion ----
    expand: bool = True
    min_candidates: int = 5
    expansion_limit: int = 50
    expansion_langs: tuple = ("nl", "en", "fr", "de")
    expansion_forms: int = 5
    max_expansion_candidates: int = 40
    expansion_factor: float = 0.5

    # ---- scoring ----
    w: dict = field(default_factory=lambda: dict(
        label=0.30, context=0.25, retrieval=0.10, prior=0.10, domain=0.10,
        coherence=0.15, expansion=0.05, title=0.05, nationality=0.08))
    penalty: dict = field(default_factory=lambda: dict(ended=0.10, anachronistic=0.05))
    accept: float = 0.70
    review: float = 0.45
    margin: float = 0.12
    label_min: float = 0.80
    ctx_floor: float = 0.70
    ctx_span: float = 0.20

    # ---- cue support ----
    cue_min_count: int = 2
    cue_min_share: float = 0.34