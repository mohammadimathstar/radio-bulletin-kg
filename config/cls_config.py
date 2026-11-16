"""
Configuration for the clustering pipeline.
"""
from dataclasses import dataclass, field


@dataclass
class ClusteringConfig:
    """All tunable parameters for two-stage entity resolution.

    Type scope
    ----------
    clustered_types : tuple
        Entity types that go through BOTH stages of clustering (within-document
        typo/containment + corpus-wide constrained single-linkage).  Default
        (PERSON, PLACE, ORGANIZATION).  Types are upper-cased in ``__post_init__``
        so ``("person", "place")`` works too.

    EVENT mentions are NOT in ``clustered_types``: they are grouped by ``event_id``
    *within one document* (same id -> one cluster; different id or missing id ->
    separate clusters) and are never merged across documents.  Every type not in
    ``clustered_types`` and not EVENT becomes a singleton cluster (one mention per
    cluster), also kept document-local.

    Stage 1 (within-document)
    -------------------------
    threshold : float
        Fuzzy-name similarity threshold for typo-level clustering (average linkage).
        Default 0.85.
    containment_types : tuple
        Entity types that get the containment-attachment step (a short name contained
        in exactly one longer name of the same document is merged).  Default PERSON.

    Stage 2 (corpus-wide)
    ---------------------
    corpus_threshold : float | None
        Similarity threshold for corpus-wide clustering (closest-pair / single linkage
        with cannot-link constraints).  Defaults to ``threshold`` when None.

    Cue extraction
    --------------
    cue_window : int
        How many tokens before the name (and after a comma-apposition) to scan for
        titles, roles, occupations and nationality.  Default 8.
    flexible : bool
        If True, non-cue content words within the window are skipped (so
        ``"minister van Onderwijs Prof. X"`` yields both ``minister`` and ``prof``).
        If False, a content word breaks the scan (strict behaviour).
    nationality_strict : bool
        If True, a nationality adjective found beyond a content-word skip is rejected,
        preventing ``"Fransche regeering. Generaal SMITH"`` from attaching ``French``.

    Context windows (only used when full OCR text is available)
    ----------------------------------------------------------
    context_chars : int
        Character window on each side of the mention for the ``context`` field.
    local_chars : int
        Window for ``context_before`` / ``context_after`` used by cue extraction.
    min_sentence_chars : int
        Sentences shorter than this are merged with neighbours (OCR repair).
    max_sentence_chars : int
        Upper bound on a merged sentence.

    Constraints
    -----------
    nationality_constraint : bool
        PERSON mentions with different known nationalities are never merged.
    occupation_constraint : bool
        (Experimental) PERSON mentions with different top occupations are never merged.
        Default False — occupations are a soft cue only.
    """

    threshold: float = 0.85
    clustered_types: tuple = ("PERSON", "PLACE", "ORGANIZATION")
    containment_types: tuple = ("PERSON",)
    corpus_threshold: float | None = None

    cue_window: int = 5
    flexible: bool = True
    nationality_strict: bool = True

    context_chars: int = 400
    local_chars: int = 160
    min_sentence_chars: int = 120
    max_sentence_chars: int = 700

    nationality_constraint: bool = True
    occupation_constraint: bool = False

    def __post_init__(self) -> None:
        # Normalise type tuples to upper-case so filtering is case-insensitive
        # (the IE JSONL may carry "Person" / "Place" / "Organization").
        object.__setattr__(
            self, "clustered_types", tuple(t.upper() for t in self.clustered_types))
        object.__setattr__(
            self, "containment_types", tuple(t.upper() for t in self.containment_types))