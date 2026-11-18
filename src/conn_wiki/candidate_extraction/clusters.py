"""Load corpus clusters (the NEW ``entity_resolution`` output) for the linking stage.

Input: ``global_clusters.json`` written by ``entity_resolution.run_clustering``:

    {key: {TYPE: {representative, n_mentions, n_documents, surface_forms: {form: n},
                  documents: {doc_id: [mention, ...]},
                  # Person only: nationality, nationality_counts, title_counts,
                  #             role_counts, occupation_counts
                  # Event only: event_id}}}

Adaptations to the new output (compared with the old clustering):
* types arrive in canonical capitalization ("Person") and are mapped to the
  linker's UPPER-case types ("PERSON").  The coarse types are the ONLY types:
  the mention ``subtype`` ("Ship", "DiplomaticMeeting", ...) is NOT used as a
  type — SHIP/SUBMARINE/AIRCRAFT gate on the combined VEHICLE roots and event
  subtypes on the combined EVENT roots (see features.ALLOWED_CLASSES);
* person cues arrive as DOCUMENT counts (``title_counts``/``role_counts``/
  ``nationality_counts``); per-mention ``cues_local`` exists too but the
  cluster-level counts are what the support rule uses;
* the new pipeline never emits name-less clusters into this file, but a
  defensive skip is kept for degenerate names ("—", OCR junk).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date

from .cues import SEARCH_KINDS, NAT_GROUPS, TITLES, match_title, supported_cues

# canonical type of the new clustering -> linker type
TYPE_MAP = {
    "Person": "PERSON", "Organization": "ORGANIZATION", "Place": "PLACE",
    "Event": "EVENT", "LegalDocument": "LEGAL_DOCUMENT", "Vehicle": "VEHICLE",
}

# Cues that a sentence comes from the stock-exchange report (fractions, price words).
FIN_RE = re.compile(r"\b\d+\s+\d/\d\b")
FIN_WORDS = {"sloot", "koers", "hooger", "lager", "staalwaarden", "spoorwegen",
             "beurs", "punt", "punten"}

# OCR line-break hyphenation inside a name: "Dala- dier" (hyphen + space).
# A real compound ("Jean-Pierre") has NO space after the hyphen and is kept.
OCR_HYPHEN = re.compile(r"(\w)- +(\w)")


# ------------------------------------------------------------------ helpers
def compact(s: str) -> str:
    """Lowercase, strip diacritics and everything except [a-z0-9].

    Makes OCR splits comparable: "CHAMBER LAIN" -> "chamberlain".
    """
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def clean(s: str) -> str:
    """Collapse whitespace; Title-Case ALL-CAPS strings (better for Wikidata search)."""
    s = re.sub(r"\s+", " ", s).strip()
    return s.title() if s.isupper() else s


def doc_date(doc_id: str) -> date:
    """Parse the bulletin date from an id like 'anp:1939:07:22:6:mpeg21'."""
    p = doc_id.split(":")
    return date(int(p[-5]), int(p[-4]), int(p[-3]))


def build_titled_forms(forms: Counter, title_counts: Counter, max_titles: int = 2,
                       max_names: int = 2) -> list[str]:
    """Extra query strings "Title Name" for a PERSON cluster.

    Uses the most frequent searchable titles (noble/church/military) and the most
    frequent surface forms, plus the English equivalent of the title
    ("Prins Bernhard" and "Prince Bernhard"). Returns [] if no searchable title.
    """
    titles = [t for t, _ in title_counts.most_common() if TITLES[t][0] in SEARCH_KINDS][:max_titles]
    names = [f for f, _ in forms.most_common(max_names)]
    out = []
    for t in titles:
        eng = TITLES[t][1][0] if TITLES[t][1] else None
        for n in names:
            if match_title(n.split()[0]) == t:        # name already starts with the title
                continue
            out.append(f"{t.title()} {n}")
            if eng and eng != t:
                out.append(f"{eng.title()} {n}")
    return out


# ------------------------------------------------------------------ loader
def load_clusters(path: str, cue_min_count: int, cue_min_share: float) -> list[dict]:
    """Flatten ``global_clusters.json`` into one dict per (cluster key, entity type).

    Each cluster dict contains:
        id, key, type            identity "<key>||<TYPE>"
        forms                    Counter of distinct cleaned surface forms
        mentions                 [{doc, text, sentence, context, has_ctx}]
        docs, n_mentions, n_docs
        min_date, max_date       first/last bulletin date of the cluster
        finance_ratio            share of mentions (with a sentence) that look like stock reports
        titles / nationalities   Counters of SUPPORTED titles+roles and nationality groups
                                 (PERSON only); *_raw = all cues; counts are DOCUMENTS
        titled_forms             extra "Title Name" query strings (PERSON only)
    Mentions whose sentence is empty (no context resolved) still count as
    mentions and surface forms but give no context evidence.
    """
    import json

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    clusters, skipped = [], 0
    for key, by_type in data.items():
        for raw_t, body in by_type.items():
            t = TYPE_MAP.get(raw_t, raw_t.upper())
            rep = body.get("representative", key)   # representative is the name; key the unique id
            forms = Counter()
            for f, n in body["surface_forms"].items():
                v = clean(f)
                # PERSON names: a hyphen followed by a space is a scan line-break,
                # the parts are ONE name ("Dala- dier" -> "Daladier").  The joined
                # variant is added BEFORE the broken form, so it wins the compact()
                # dedup in retrieve.query_forms (searching the broken spelling is
                # pointless).  Without the space ("Jean-Pierre") the hyphen is real.
                if t == "PERSON" and OCR_HYPHEN.search(v):
                    forms[OCR_HYPHEN.sub(r"\1\2", v)] += n
                forms[v] += n
            if not forms:
                forms[clean(rep)] += 1
            forms.pop("", None)
            if not forms:                           # degenerate name ("—", OCR junk): not linkable
                skipped += 1
                continue
            mentions, dates = [], []
            for doc_id, ms in body["documents"].items():
                dates.append(doc_date(doc_id))
                for m in ms:
                    sent = m.get("sentence") or ""
                    mentions.append(dict(doc=doc_id,
                                         text=" ".join((m.get("text") or "").split()),
                                         sentence=sent, context=m.get("context") or "",
                                         has_ctx=bool(sent)))
                    # the IE stage already normalizes OCR names ("Dala- dier" ->
                    # "Daladier"); that form is the best search query there is
                    norm = " ".join((m.get("normalized_name") or "").split())
                    if norm:
                        v = clean(norm)
                        if OCR_HYPHEN.search(v):
                            forms[OCR_HYPHEN.sub(r"\1\2", v)] += 1
                        forms[v] += 1
            titles_raw = nat_raw = Counter()
            if t == "PERSON":
                titles_raw = Counter({k: v for k, v in
                                      {**body.get("title_counts", {}),
                                       **body.get("role_counts", {})}.items() if k in TITLES})
                nat_raw = Counter({k: v for k, v in body.get("nationality_counts", {}).items()
                                   if k in NAT_GROUPS})
            n_scan = body.get("n_documents", len(body["documents"]))
            sup_titles = supported_cues(titles_raw, n_scan, cue_min_count, cue_min_share)
            sup_nat = supported_cues(nat_raw, n_scan, cue_min_count, cue_min_share)
            with_ctx = [m for m in mentions if m["has_ctx"]]
            fin = sum(1 for m in with_ctx
                      if FIN_RE.search(m["sentence"])
                      or any(w in m["sentence"].lower() for w in FIN_WORDS))
            clusters.append(dict(
                id=f"{key}||{t}", key=rep, type=t, forms=forms, mentions=mentions,
                docs=set(body["documents"]), n_mentions=len(mentions),
                n_docs=len(body["documents"]),
                min_date=min(dates), max_date=max(dates),
                finance_ratio=fin / max(len(with_ctx), 1),
                titles=sup_titles, titles_raw=titles_raw,
                nationalities=sup_nat, nationalities_raw=nat_raw, n_scan=n_scan,
                titled_forms=build_titled_forms(forms, sup_titles) if t == "PERSON" else []))
    if skipped:
        print(f"  [clusters] skipped {skipped} cluster(s) without a usable name")
    return clusters


def expansion_hints(clusters: list[dict]) -> dict[str, list[str]]:
    """Detect ellipsis between ORGANIZATION clusters: "United States" -> "United States Steel".

    A short ORG name whose tokens are a prefix of a longer ORG name, both clusters
    looking like stock reports, gets the longer name as an extra query string and
    expansion feature. Clusters are NOT merged.
    """
    hints = defaultdict(list)
    orgs = [c for c in clusters if c["type"] == "ORGANIZATION"]
    for s in orgs:
        st = s["key"].lower().split()
        if not st:
            continue
        for l in orgs:
            lt = l["key"].lower().split()
            if len(lt) > len(st) and lt[:len(st)] == st \
                    and s["finance_ratio"] > 0.5 and l["finance_ratio"] > 0.5:
                hints[s["id"]].append(l["key"])
    return hints