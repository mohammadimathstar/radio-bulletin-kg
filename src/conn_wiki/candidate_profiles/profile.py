"""build_profile(): assemble the profile dict of one candidate.

Common to all types: qid, label, other-language labels, description (en + nl),
aliases (nl/en/de/fr), instance_of (P31 labels), sitelinks, wikipedia_summary.
Type-specific fields come from types.spec_for(): item-valued properties are
resolved to labels, dates are formatted with their precision, quantities are
rounded.  Empty fields are omitted, which keeps the profile (and the judge
input) short.  See ../candidate_profiles/README.md for the field list.
"""

from . import api, config
from .extract import claim_ids, coordinates, positions, quantity, time_str
from .types import normalize_type, spec_for


def build_profile(qid: str, etype: str, with_summary: bool = True, summary_chars: int = 400) -> dict:
    """Profile of one candidate item (see the package docstring). Empty fields are omitted."""
    cfg = config.CONFIG
    e = api.wd_entities([qid]).get(qid) or {}
    labels, descs = e.get("labels", {}), e.get("descriptions", {})
    spec = spec_for(etype)
    norm = normalize_type(etype)

    item_qids = {"instance_of": claim_ids(e, "P31")[:cfg.max_values]}
    for name, pid in spec["items"].items():
        item_qids[name] = claim_ids(e, pid)[:cfg.max_values]
    pos = positions(e) if norm == "PERSON" else []
    lab = api.wd_labels([q for v in item_qids.values() for q in v] + [p[0] for p in pos])

    main = (labels.get("en") or labels.get("nl") or next(iter(labels.values()), {})).get("value", qid)
    others = list(dict.fromkeys(v["value"] for k, v in labels.items()
                                if isinstance(v, dict) and v.get("value") and v["value"] != main))
    aliases = list(dict.fromkeys(
        a["value"] for lang in ("nl", "en", "de", "fr") for a in e.get("aliases", {}).get(lang, [])
        if a["value"] != main and a["value"] not in others))

    prof = {"qid": qid, "type": norm,
            "label": main, "label_other_languages": others[:4],
            "description": descs.get("en", {}).get("value"),
            "description_nl": descs.get("nl", {}).get("value"),
            "aliases": aliases[:cfg.max_aliases]}
    for name, qs in item_qids.items():
        prof[name] = [lab[q] for q in qs]
    if pos:
        prof["positions_held"] = [f"{lab[q]} ({a or '?'}-{b or ''})" if (a or b) else lab[q]
                                  for q, a, b in pos]
    for name, pid in spec["dates"].items():
        prof[name] = time_str(e, pid)
    if norm == "PLACE":
        prof["coordinates"] = coordinates(e)
        prof["population"] = quantity(e, "P1082")
    prof["sitelinks"] = len(e.get("sitelinks", {}))
    if with_summary:
        prof["wikipedia_summary"] = api.wiki_summary(e, summary_chars)
    return {k: v for k, v in prof.items() if v not in (None, "", [], {})}
