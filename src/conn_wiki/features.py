"""Hard gates and features for Wikidata candidates.

Two changes against the previous version, both for speed and correctness:

* GATES USE FACTS, NOT CLAIM PAYLOADS: the type gate checks the candidate's
  direct P31 values against the locally cached P279* subclass closure of the
  allowed root classes, and the temporal gate reads P569/P570/P571/P576/...
  dates from the compact SPARQL facts (:mod:`linking.wikidata`).  Same gate
  semantics as before, no per-candidate SPARQL, no claims in every payload.
* CONTEXT EMBEDDINGS ARE BATCHED PER CLUSTER: all candidate "label. description"
  texts and all mention sentences are embedded in ONE call (the old code
  re-encoded the mention sentences once per candidate).  Embeddings are cached
  in memory for the run — deliberately NOT on disk, so the persistent cache
  stays small.
"""
from __future__ import annotations

import re
import threading

import numpy as np
from rapidfuzz import fuzz

from .clusters import compact
from .cues import NAT_GROUPS, TITLES, _NAT_RE
from .wikidata import best_label, claim_ids, parse_wd_time

# ------------------------------------------------------------------ entity model
# VERIFY THESE QIDs.  Matching is transitive: the candidate's direct P31 values
# must reach one of these roots via P279* (checked locally against the cached
# closure, see WDClient.closure).
#
# The corpus has exactly six coarse entity types (person, place, organization,
# event, vehicle, legal document) and the mention subtype is NOT used as a
# type: the former SHIP/SUBMARINE/AIRCRAFT types gate on the combined VEHICLE
# roots below, and battles/summits/meetings on the combined EVENT roots.
ALLOWED_CLASSES = {
    "PERSON": ["Q5"],
    "ORGANIZATION": ["Q43229", "Q4830453", "Q327333", "Q7278"],
    "PLACE": ["Q2221906", "Q6256", "Q515", "Q618123", "Q41176", "Q811979"],
    "VEHICLE": ["Q42889", "Q11446", "Q2811", "Q11436", "Q15056995"],
    "EVENT": ["Q350604", "Q198", "Q178561", "Q625994", "Q1656682"],
    "LEGAL_DOCUMENT": ["Q131569", "Q820655", "Q49848"],
}
BUSINESS = ["Q4830453"]                   # for the stock-report "domain" cue
COUNTRY_CLASSES = ["Q6256", "Q3624078"]   # VERIFY: country, sovereign state (P31/P279*)
# Properties whose item values are compared for the coherence bonus (pass 2).
RELATION_PROPS = ["P17", "P131", "P159", "P361", "P27", "P463", "P749", "P176", "P495", "P1344"]
# Types to which the "ended" temporal penalty never applies: a 1939 bulletin
# legitimately mentions historical events and (old) laws/treaties by name.
EVENT_TYPES = {"EVENT", "LEGAL_DOCUMENT"}


# ------------------------------------------------------------------ embeddings
class Embedder:
    """multilingual-e5 embeddings, batched and memoized in memory for the run.

    None-availability (sentence-transformers missing, no model, ...) degrades to
    context feature = null, exactly like the old ``embed()``.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._model = None
        self._failed = False
        self._lock = threading.Lock()
        self._cache: dict[str, list] = {}     # text -> vector (memory only)

    def encode(self, texts: list[str]):
        """Normalized e5 vectors for ``texts`` (deduplicated), or None if unavailable."""
        texts = list(dict.fromkeys(texts))
        if not texts:
            return []
        if self._failed or not self.cfg.use_embeddings:
            return None
        todo = [t for t in texts if t not in self._cache]
        if todo:
            try:
                with self._lock:
                    if self._model is None:
                        from sentence_transformers import SentenceTransformer
                        self._model = SentenceTransformer("intfloat/multilingual-e5-base")
                    vecs = self._model.encode(todo, normalize_embeddings=True,
                                              batch_size=self.cfg.embed_batch,
                                              show_progress_bar=False)
            except Exception as ex:               # noqa: BLE001 (any failure -> feature off)
                if not self._failed:
                    print(f"[warn] embeddings unavailable ({ex!r}); context feature = null")
                    self._failed = True
                return None
            for t, v in zip(todo, (v.tolist() if hasattr(v, "tolist") else v for v in vecs)):
                self._cache[t] = v
        return [self._cache[t] for t in texts]


# ------------------------------------------------------------------ gates
def temporal_status(facts: dict, c: dict, is_country: bool = False):
    """Temporal check of a candidate against the cluster's bulletin dates.

    Returns
        None             impossible: did not exist yet (start after the last
                         bulletin). Hard reject.
        "ended"          existed earlier but died/ended before the first bulletin.
                         Kept with a score penalty + flag, because 1939 bulletins
                         can mention commemorations, streets named after dead
                         people, historical states, ...
        "anachronistic"  only for PLACE candidates that are countries/states: the
                         item's inception is after the bulletins (the Italian
                         Republic, 1946, for "Italië" in 1939), but a country
                         keeps its identity across regimes. Kept, small penalty.
        "ok"             fine, or no usable dates (unknown != wrong).
    Events, treaties and conflicts are never "ended" (historical events are
    legitimately mentioned later).  Dates: PERSON P569/P570; others
    P571|P729|P580 (start), P576|P730|P582 (end).  Facts carry dates as ISO
    strings (cache-safe); with several claims per property the EARLIEST start /
    LATEST end is used (the lenient reading: "existed at some point"), instead
    of the old first-claim-in-JSON order, which was arbitrary.
    """
    from datetime import date

    def d(s):
        try:
            return date.fromisoformat(s)
        except (ValueError, TypeError):
            return None

    dates = facts.get("dates", {})
    if c["type"] == "PERSON":
        starts = [x for x in (d(s) for s in dates.get("P569", [])) if x]
        ends = [x for x in (d(s) for s in dates.get("P570", [])) if x]
    else:
        starts = [x for p in ("P571", "P729", "P580") for s in dates.get(p, []) if (x := d(s))]
        ends = [x for p in ("P576", "P730", "P582") for s in dates.get(p, []) if (x := d(s))]
    start = min(starts) if starts else None
    end = max(ends) if ends else None
    if start and start > c["max_date"]:
        return "anachronistic" if (c["type"] == "PLACE" and is_country) else None
    if end and end < c["min_date"] and c["type"] not in EVENT_TYPES:
        return "ended"
    return "ok"


# ------------------------------------------------------------------ text features
def label_sim(e: dict, c: dict) -> float:
    """Best fuzzy similarity (0..1) between any cluster form and any label/alias.

    Cluster forms are the surface forms plus, for persons, the "Title Name"
    forms.  Comparison uses compact strings.  For PERSON also compares against
    the last token of each name (surname-only mentions: "Chamberlain").
    """
    names = [v["value"] for v in e.get("labels", {}).values()]
    names += [a["value"] for al in e.get("aliases", {}).values() for a in al]
    best = 0.0
    for f in list(c["forms"]) + list(c.get("titled_forms", [])):
        cf = compact(f)
        for name in names:
            cands = [compact(name)]
            if c["type"] == "PERSON" and name.split():
                cands.append(compact(name.split()[-1]))
            for cn in cands:
                best = max(best, fuzz.ratio(cf, cn) / 100)
    return best


def _title_re(t: str, cache: dict) -> "re.Pattern":
    """Whole-word regex (optional plural) of a title's keywords ("major" must not match "majority")."""
    if t not in cache:
        cache[t] = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in TITLES[t][1]) + r")(?:s|es)?\b")
    return cache[t]


def title_feature(e: dict, c: dict, re_cache: dict):
    """Share of the title votes that match the candidate (None if no supported title).

    Every supported title votes with its document count; a vote matches when a
    keyword of the title occurs as a whole word in the candidate's labels,
    aliases or descriptions.
    """
    text = " ".join(
        [v["value"] for v in e.get("labels", {}).values()]
        + [v["value"] for v in e.get("descriptions", {}).values()]
        + [a["value"] for al in e.get("aliases", {}).values() for a in al]).lower()
    total_votes = hits = 0
    for t, n in c.get("titles", {}).items():
        if TITLES[t][0] == "honorific" or not TITLES[t][1]:
            continue
        total_votes += n
        hits += n * bool(_title_re(t, re_cache).search(text))
    return hits / total_votes if total_votes else None


def nationality_feature(e: dict, c: dict):
    """Share of the nationality votes matching the candidate's descriptions.

    None if the cluster has no supported nationality, or the candidate has no
    description (no information is not a mismatch).
    """
    text = " ".join(v["value"] for v in e.get("descriptions", {}).values()).lower()
    nats = c.get("nationalities", {})
    if not nats or not text:
        return None
    votes = sum(nats.values())
    return sum(n * bool(_NAT_RE[g].search(text)) for g, n in nats.items()) / votes


# ------------------------------------------------------------------ scorer
class Scorer:
    """Applies the gates and computes the feature vector of surviving candidates."""

    def __init__(self, wd, cfg, embedder: Embedder):
        self.wd, self.cfg, self.embedder = wd, cfg, embedder
        self._title_re_cache: dict = {}

    def score_cluster(self, c: dict, qids: list[str], rrf: dict, found: dict,
                      hint_forms: list[str]):
        """Returns (candidates, rejected, n_retrieved).

        Gates: type (direct P31 reaches ALLOWED_CLASSES[type] via P279*) and
        "not yet existing" (temporal_status is None).  Entities that already
        ended are kept with ended=1.  Features that do not apply to the cluster
        are None.  rejected = [{qid, label, why, found_by}] with why in {"type",
        "not_yet_existing"} — kept so a nil can be explained, and found_by
        (channel, form, rank) shows WHERE a rejected candidate came from.
        """
        cfg = self.cfg
        ents = self.wd.entities(qids)
        facts_by_qid = {q: self.wd.facts(q) for q in ents}

        allowed = ALLOWED_CLASSES.get(c["type"], [])
        ok_type = {q for q, f in facts_by_qid.items()
                   if allowed and self.wd.is_a(f["p31"], allowed)}
        domain_on = c["type"] == "ORGANIZATION" and c["finance_ratio"] > 0.5
        biz = {q for q, f in facts_by_qid.items() if self.wd.is_a(f["p31"], BUSINESS)} if domain_on else set()
        countries = ({q for q, f in facts_by_qid.items()
                      if self.wd.is_a(f["p31"], COUNTRY_CLASSES)} if c["type"] == "PLACE" else set())

        # context similarity, batched: one encode call for the whole cluster
        ctx = self._context_features(c, ents)

        out, rejected = [], []
        for q in qids:
            e = ents.get(q)
            if e is None:
                continue
            lab = best_label(e)
            facts = facts_by_qid[q]
            if q not in ok_type:
                rejected.append(dict(qid=q, label=lab, why="type",
                                     found_by=found.get(q, [])[:2]))
                continue
            ts = temporal_status(facts, c, is_country=q in countries)
            if ts is None:
                rejected.append(dict(qid=q, label=lab, why="not_yet_existing",
                                     found_by=found.get(q, [])[:2]))
                continue
            cs, per_mention = ctx.get(q, (None, []))
            feats = dict(
                label=label_sim(e, c),
                context=cs if per_mention else None,
                retrieval=rrf.get(q, 0.0),
                prior=min(len(e.get("sitelinks", {})) / 100, 1.0),
                domain=(1.0 if q in biz else 0.0) if domain_on else None,
                expansion=(max([fuzz.ratio(compact(h), compact(n["value"])) / 100
                                for h in hint_forms for n in e.get("labels", {}).values()]
                               or [0.0]) if hint_forms else None),
                coherence=None,
                title=title_feature(e, c, self._title_re_cache),
                nationality=nationality_feature(e, c),
                ended=1.0 if ts == "ended" else 0.0,
                anachronistic=1.0 if ts == "anachronistic" else 0.0)
            out.append(dict(qid=q, label=lab, feats=feats, per_mention=per_mention,
                            found_by=found.get(q, [])))
        return out, rejected, len(qids)

    def _context_features(self, c: dict, ents: dict) -> dict:
        """(aggregate, per-mention) context similarity for every candidate with a description.

        The candidate's "label. description" and each mention sentence (up to
        ctx_max_mentions, only mentions that have one) are embedded; cosine
        similarities are rescaled with ctx_floor/ctx_span; the aggregate is the
        mean of the top 3 per candidate.  Returns {} when nothing can be embedded.
        """
        cfg = self.cfg
        sents = [m["sentence"] for m in c["mentions"] if m["has_ctx"]][:cfg.ctx_max_mentions]
        if not sents:
            return {}
        cand_texts, cand_qids = [], []
        for q, e in ents.items():
            lab = best_label(e) or ""
            desc = next(iter(e.get("descriptions", {}).values()), {}).get("value", "")
            cand_texts.append(f"passage: {lab}. {desc}".strip())
            cand_qids.append(q)
        vecs = self.embedder.encode([f"query: {s}" for s in sents] + cand_texts)
        if vecs is None:
            return {}
        svecs, cvecs = vecs[:len(sents)], vecs[len(sents):]
        out = {}
        for q, cv in zip(cand_qids, cvecs):
            raw = [float(_dot(cv, sv)) for sv in svecs]
            sims = [min(max((x - cfg.ctx_floor) / cfg.ctx_span, 0.0), 1.0) for x in raw]
            top = sorted(sims, reverse=True)[:3]
            out[q] = (sum(top) / len(top), sims)
        return out


def _dot(a, b) -> float:
    return float(np.dot(a, b))


# re-exported for run_linking / pass 2
__all__ = ["ALLOWED_CLASSES", "BUSINESS", "COUNTRY_CLASSES", "RELATION_PROPS",
           "EVENT_TYPES", "Embedder", "Scorer", "temporal_status", "label_sim",
           "title_feature", "nationality_feature", "claim_ids", "parse_wd_time",
           "NAT_GROUPS", "TITLES"]