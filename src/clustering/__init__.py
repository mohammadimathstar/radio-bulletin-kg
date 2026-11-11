"""Clustering subpackage: surface normalization, person cues, and two-stage
entity resolution for Dutch 1939 radio bulletin mentions."""

from .cues import mention_cues, person_cues, cues_from_mention, occupation_from_subtype
from .normalize import normalize_name, norm_ws
from .context import add_context, get_sentence_splitter, normalize_type
from .cluster import (
    resolve_entities, doc_record, build_report, reset_diagnostics,
    choose_representative, choose_longest,
)

__all__ = [
    "mention_cues", "person_cues", "cues_from_mention", "occupation_from_subtype",
    "normalize_name", "norm_ws",
    "add_context", "get_sentence_splitter", "normalize_type",
    "resolve_entities", "doc_record", "build_report", "reset_diagnostics",
    "choose_representative", "choose_longest",
]
