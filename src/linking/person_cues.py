"""
Person-cue tables and extractors, shared between clustering and Wikidata linking.

This module re-exports the cue tables (``TITLES``, ``NAT_GROUPS``, ``SEARCH_KINDS``)
and extractors (``match_title``, ``norm_tok``, ``person_cues``) from
:mod:`radio_bulletins_kg.clustering.cues`, so the two stages stay in sync.  It also
defines ``_NAT_RE`` — compiled keyword regexes per nationality group — which the
linking stage uses to match a candidate's Wikidata description against the cluster's
nationality cue.
"""
import re

from ..clustering.cues import (  # noqa: F401  (re-exports)
    NAT_GROUPS,
    SEARCH_KINDS,
    TITLES,
    match_title,
    norm_tok,
    person_cues,
)

# Compiled keyword regex per nationality group, used by link_wikidata's
# nationality_feature(): searches a candidate's Wikidata descriptions for the
# group's EN/NL/DE/FR stems (whole-word, case-insensitive).
_NAT_RE: dict[str, re.Pattern] = {
    g: re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keywords) + r")", re.IGNORECASE)
    for g, (_demonym, keywords, _adjs) in NAT_GROUPS.items()
}
