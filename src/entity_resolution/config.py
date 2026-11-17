"""Pipeline configuration (self-contained replacement for ``config.cls_config``).

``ClusteringConfig`` carries every tunable of the two-stage entity resolution:

Resolution strategies
---------------------
``strategy_by_type`` maps each canonical entity type to its resolution strategy:

  - ``STRATEGY_STRING_SIMILARITY`` (default for **Person, Place, LegalDocument,
    Vehicle**): the known two-level clustering — within-document fuzzy
    clustering with cannot-link guards (stage 1) and constrained corpus-wide
    merging (stage 2).
  - ``STRATEGY_EXACT_EVENT_ID`` (default for **Event**): mentions are grouped
    within one bulletin by exact ``event_id`` (``entity_id`` accepted as a
    fallback field name); events are NEVER merged across bulletins.

Types absent from the mapping (e.g. ``"Organization"``) are not resolved: each
mention becomes its own document-local singleton.  Pass a custom mapping to
change this — the keys must be canonical types (see :mod:`.entity_types`).

Subtype policy
--------------
The IE ``subtype`` fields are never consulted for entity resolution (no
subtype-based occupation mapping, no subtype constraints).  Person
nationality, extracted from TEXT by :mod:`cues`, remains the one cannot-link
signal, as in the original pipeline.

Thresholds
----------
``threshold`` (stage 1) and ``corpus_threshold`` (stage 2, ``None`` = fall back
to ``threshold``) apply to all string-similarity types.  Per-type overrides can
be given in ``stage1_threshold_by_type`` / ``stage2_threshold_by_type`` (e.g.
a lower threshold for short Place names).

Asymmetric strictness
---------------------
The two levels are deliberately different:

  - **Stage 1 (within one bulletin) is RELAXED**: clustering uses the plain
    character-level similarity (``fuzz.ratio``), as before.  Inside one bulletin
    two different entities with similar names are rare, and recall matters more
    than precision here.  Set ``stage1_token_guard=True`` to apply the strict
    token-aware score at stage 1 as well (protects against two *differently
    named* entities sharing a generic tail inside one bulletin — such a
    stage-1 merge cannot be undone at stage 2).
  - **Stage 2 (across the corpus) is ALWAYS strict**: the token-aware score
    (:func:`.cluster.name_similarity`) vetoes merges that would only happen
    through a shared generic tail ("... communistische partij").

Paragraphs
----------
``short_para_max_words`` (default 4): paragraphs with at most this many words
(datelines / kickers such as ``"Soesterberg."``) are merged into the FOLLOWING
paragraph before context extraction, so a mention's ``context`` is never a
bare title; the last paragraph merges into the previous one instead.  ``0``
disables the rule.
"""
from .entity_types import DEFAULT_STRATEGY_BY_TYPE, STRATEGY_EXACT_EVENT_ID, STRATEGY_STRING_SIMILARITY  # noqa: F401

__all__ = ["ClusteringConfig", "STRATEGY_STRING_SIMILARITY", "STRATEGY_EXACT_EVENT_ID"]


class ClusteringConfig:
    """All tunables of the entity resolution pipeline (keyword-initializable)."""

    def __init__(
        self,
        # --- strategies ---
        strategy_by_type: dict[str, str] | None = None,
        # types whose names are merged into the one longer name that contains
        # them, within one document ("Chamberlain" -> "Neville Chamberlain")
        containment_types: tuple[str, ...] = ("Person",),
        # --- thresholds ---
        threshold: float = 0.85,
        corpus_threshold: float | None = None,   # None -> same as `threshold`
        stage1_threshold_by_type: dict[str, float] | None = None,
        stage2_threshold_by_type: dict[str, float] | None = None,
        # --- constraints ---
        # cannot-link for Person clusters with different TEXT-extracted nationalities
        nationality_constraint: bool = True,
        # --- asymmetric strictness between the two levels ---
        # stage 1 (within bulletin) clusters on the relaxed character-level
        # similarity by default; True applies the strict token-aware guard of
        # stage 2 at stage 1 as well
        stage1_token_guard: bool = False,
        # paragraphs with at most this many words (datelines / kickers such as
        # "Soesterberg.") merge into the following paragraph before context
        # extraction; 0 disables
        short_para_max_words: int = 4,
        # --- cue extraction (Person metadata; nationality feeds the constraint) ---
        cue_window: int = 8,
        flexible: bool = True,
        nationality_strict: bool = True,
        # --- context windows ---
        min_sentence_chars: int = 120,
        max_sentence_chars: int = 400,
        local_chars: int = 120,
        # --- OCR consistency gate ---
        # minimum rapidfuzz ratio (0-100) between the OCR slice and the IE
        # mention text for the slice to be trusted; None/0 disables the gate
        ocr_min_similarity: float | None = 80.0,
    ):
        # Canonical-type -> strategy; a copy so callers cannot mutate the default.
        self.strategy_by_type = dict(strategy_by_type or DEFAULT_STRATEGY_BY_TYPE)
        self.containment_types = tuple(containment_types)

        self.threshold = float(threshold)
        self.corpus_threshold = (None if corpus_threshold is None
                                 else float(corpus_threshold))
        self.stage1_threshold_by_type = dict(stage1_threshold_by_type or {})
        self.stage2_threshold_by_type = dict(stage2_threshold_by_type or {})

        self.nationality_constraint = bool(nationality_constraint)
        self.stage1_token_guard = bool(stage1_token_guard)
        self.short_para_max_words = int(short_para_max_words)

        self.cue_window = int(cue_window)
        self.flexible = bool(flexible)
        self.nationality_strict = bool(nationality_strict)

        self.min_sentence_chars = int(min_sentence_chars)
        self.max_sentence_chars = int(max_sentence_chars)
        self.local_chars = int(local_chars)
        self.ocr_min_similarity = ocr_min_similarity

    # ------------------------------------------------------------------ lookups

    def strategy_for(self, etype: str) -> str | None:
        """Resolution strategy of a canonical type (None = not resolved)."""
        return self.strategy_by_type.get(etype)

    @property
    def string_similarity_types(self) -> set[str]:
        """Canonical types that go through both clustering stages."""
        return {t for t, s in self.strategy_by_type.items()
                if s == STRATEGY_STRING_SIMILARITY}

    def stage1_threshold(self, etype: str) -> float:
        """Within-document threshold for one type (per-type override supported)."""
        return self.stage1_threshold_by_type.get(etype, self.threshold)

    def stage2_threshold(self, etype: str) -> float:
        """Corpus-wide threshold for one type (per-type override supported)."""
        if etype in self.stage2_threshold_by_type:
            return self.stage2_threshold_by_type[etype]
        return (self.corpus_threshold if self.corpus_threshold is not None
                else self.threshold)