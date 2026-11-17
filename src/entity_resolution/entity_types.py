"""Canonical entity types and type-name standardization.

The IE output does not always use the exact type labels of the target schema:
persons arrive as ``"PER"``, ``"PERSON"`` or ``"Person"``; places as ``"PLACE"``,
``"LOCATION"``, ``"LOC"`` or ``"Location"``; legal documents as
``"LEGAL_DOCUMENT"`` or ``"Legal Document"``.  Every downstream stage of this
package compares types against the CANONICAL capitalized forms, so all raw type
strings must pass through :func:`canonical_type` exactly once, at the boundary
(entry point / loader / context stage).

Canonical types
---------------
``"Person"``, ``"Place"``, ``"LegalDocument"``, ``"Event"``, ``"Vehicle"`` —
identical to the ``type`` literals of the Pydantic IE models.

``"Organization"`` is kept in the alias table for robustness (the IE data may
still contain it), but it is NOT one of the resolved types: unless it is added
to a resolution strategy in the configuration, its mentions become passthrough
singletons (see :data:`.config.ClusteringConfig.strategy_by_type`).

Resolution strategies
---------------------
``STRATEGY_STRING_SIMILARITY`` — the default: the two-level fuzzy clustering
(stage 1 within a bulletin, constrained stage 2 across the corpus).

``STRATEGY_EXACT_EVENT_ID`` — events only: mentions are grouped within one
bulletin by exact ``event_id`` (``entity_id`` is accepted as a fallback field
name) and are never merged across bulletins.

Unknown type labels are passed through unchanged (so no information is lost)
and are treated as singletons by the clustering stages.
"""
import re

# ----------------------------------------------------------------- canonical types

#: The resolved entity types, in canonical capitalized form.
CANONICAL_TYPES: tuple[str, ...] = ("Person", "Place", "Organization", "LegalDocument", "Event", "Vehicle")


#: Fallback label for mentions whose type is missing or empty.
UNKNOWN_TYPE = "Unknown"

# ----------------------------------------------------------------- alias table
# Alias keys are compared AFTER normalization: upper-cased with spaces,
# hyphens and underscores removed ("Legal Document" -> "LEGALDOCUMENT").

TYPE_ALIASES: dict[str, str] = {
    # Person: PER / PERSON / Person
    "PERSON": "Person",
    "PER": "Person",
    "person": "Person",
    # Place: PLACE / LOCATION / LOC / Place / Location (GPE = standard NER
    # label for geopolitical entities: countries, provinces, cities)
    "PLACE": "Place",
    "LOCATION": "Place",
    "LOC": "Place",
    "GPE": "Place",
    # LegalDocument: LEGAL_DOCUMENT / LEGALDOCUMENT / Legal Document / LegalDoc
    "LEGALDOCUMENT": "LegalDocument",
    "LEGALDOC": "LegalDocument",
    "LEGAL": "LegalDocument",
    # Event: EVENT / EVENTS / Event
    "EVENT": "Event",
    "EVENTS": "Event",
    # Vehicle: VEHICLE / VEHICLES / Vehicle
    "VEHICLE": "Vehicle",
    "VEHICLES": "Vehicle",
    # Organization (compatibility only — not resolved by default)
    "ORGANIZATION": "Organization",
    "ORGANISATION": "Organization",
    "ORG": "Organization",
    "organization": "Organization",
}

_STRIP_RE = re.compile(r"[\s_\-]+")

# ----------------------------------------------------------------- strategies

#: Default strategy: two-level string-similarity clustering (stage 1 + stage 2).
STRATEGY_STRING_SIMILARITY = "string_similarity"

#: Event strategy: within-bulletin grouping by exact event_id, never cross-corpus.
STRATEGY_EXACT_EVENT_ID = "exact_event_id"

#: Which strategy resolves which canonical type.  Types absent from this mapping
#: (e.g. "Organization") are not resolved: their mentions become singletons.
DEFAULT_STRATEGY_BY_TYPE: dict[str, str] = {
    "Person": STRATEGY_STRING_SIMILARITY,
    "Place": STRATEGY_STRING_SIMILARITY,
    "Organization": STRATEGY_STRING_SIMILARITY,
    "LegalDocument": STRATEGY_STRING_SIMILARITY,
    "Vehicle": STRATEGY_STRING_SIMILARITY,
    "Event": STRATEGY_EXACT_EVENT_ID,
}


def canonical_type(raw) -> str | None:
    """Standardize a raw type label to its canonical capitalized form.

    ``"PER"``, ``"PERSON"``, ``"Person"``, ``" person "`` -> ``"Person"``;
    ``"LEGAL_DOCUMENT"``, ``"Legal Document"`` -> ``"LegalDocument"``.

    Known aliases map to the canonical name; unknown non-empty labels are
    returned stripped-but-unchanged (so custom types keep their label and are
    treated as singletons downstream); ``None``/empty input returns ``None``.
    """
    s = str(raw or "").strip()
    if not s:
        return None
    key = _STRIP_RE.sub("", s).upper()
    return TYPE_ALIASES.get(key, s)


def is_resolved_type(etype: str) -> bool:
    """True if ``etype`` (canonical) has a resolution strategy in the defaults."""
    return etype in DEFAULT_STRATEGY_BY_TYPE
