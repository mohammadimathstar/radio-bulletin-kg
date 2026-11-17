"""Entity resolution for Dutch 1939 radio bulletins: surface normalization, person
cues, and two-stage mention clustering with per-type strategies.

Types (canonical, capitalized): ``Person``, ``Place``, ``LegalDocument``,
``Event``, ``Vehicle`` — raw labels (``PER``, ``LOCATION``, ``Legal_Document``,
...) are standardized by :func:`canonical_type`.  Resolution strategies:
string-similarity clustering for Person/Place/LegalDocument/Vehicle; exact
within-bulletin ``event_id`` grouping for Event (never cross-bulletin).  The IE
``subtype`` fields are not used for resolution.

Typical use::

    from entity_resolution import (load_dataset, add_context,
                                  get_sentence_splitter, resolve_entities,
                                  doc_record, build_report, ClusteringConfig)
    mentions, ocr = load_dataset("ie.jsonl", "ocr.jsonl")
    cfg = ClusteringConfig()
    add_context(mentions, ocr, cfg, nlp=get_sentence_splitter())
    doc_clusters, global_resolved = resolve_entities(mentions, cfg)
"""

from .entity_types import (
    CANONICAL_TYPES, TYPE_ALIASES, DEFAULT_STRATEGY_BY_TYPE,
    STRATEGY_STRING_SIMILARITY, STRATEGY_EXACT_EVENT_ID,
    canonical_type, is_resolved_type,
)
from ..utils.cues import mention_cues, person_cues, cues_from_mention
from .normalize import normalize_name, norm_ws
from .context import add_context, get_sentence_splitter, normalize_type, paragraph_spans
from .config import ClusteringConfig
from .cluster import (
    resolve_entities, doc_record, build_report, reset_diagnostics,
    choose_representative, choose_longest,
)
from ..utils.io import (
    load_dataset, load_ie_mentions, load_ie_relations, load_ocr_texts,
    write_json, write_jsonl,
)

__all__ = [
    # types
    "CANONICAL_TYPES", "TYPE_ALIASES", "DEFAULT_STRATEGY_BY_TYPE",
    "STRATEGY_STRING_SIMILARITY", "STRATEGY_EXACT_EVENT_ID",
    "canonical_type", "is_resolved_type",
    # cues
    "mention_cues", "person_cues", "cues_from_mention",
    # normalize
    "normalize_name", "norm_ws",
    # context
    "add_context", "get_sentence_splitter", "normalize_type", "paragraph_spans",
    # config
    "ClusteringConfig",
    # clustering
    "resolve_entities", "doc_record", "build_report", "reset_diagnostics",
    "choose_representative", "choose_longest",
    # io
    "load_dataset", "load_ie_mentions", "load_ie_relations", "load_ocr_texts",
    "write_json", "write_jsonl",
]
