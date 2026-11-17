"""Person-cue tables and extractors, shared between the clustering and linking stages.

Re-exports the cue tables from the NEW clustering package (``entity_resolution.cues``)
so both stages stay in sync — the old shim pointed at the archived clustering code.
Also defines ``_NAT_RE``: compiled keyword regexes per nationality group, used by
the linking stage to match a candidate's Wikidata descriptions against the cluster's
nationality cue.
"""
import re
from collections import Counter

from src.utils.cues import (  # noqa: F401  (re-exports)
    NAT_GROUPS,
    SEARCH_KINDS,
    TITLES,
    match_title,
    norm_tok,
    person_cues,
)

# Compiled keyword regex per nationality group: searches a candidate's Wikidata
# descriptions for the group's EN/NL/DE/FR keyword stems (case-insensitive).
_NAT_RE: dict[str, re.Pattern] = {
    g: re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keywords) + r")", re.IGNORECASE)
    for g, (_demonym, keywords, _adjs) in NAT_GROUPS.items()
}


def supported_cues(counter: Counter, n_scan: int, min_count: int, min_share: float) -> Counter:
    """Keep cues seen in >= min_count documents or in >= min_share of the scanned documents.

    One noisy document no longer decides: a cue only feeds queries and features
    when enough of the cluster's documents support it.
    """
    return Counter({k: v for k, v in counter.items()
                    if v >= min_count or v / max(n_scan, 1) >= min_share})