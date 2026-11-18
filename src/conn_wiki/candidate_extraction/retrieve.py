"""Candidate retrieval: query forms, RRF fusion over channels, expansion round.

Channels per query form (weights from ``cfg.channel_weight``):
    wb:<lang>   wbsearchentities (prefix match, per language)
    cirrus      Cirrus full-text over labels+aliases+descriptions of all languages
    fuzzy       Cirrus fuzzy ("word~") for single-word forms >= 6 characters
Every hit contributes weight/(rrf_k + rank); candidates found by several
channels/forms — and found high — rise to the top.  This is the v8 retrieval
fusion, unchanged in behaviour; what changed is that all searches of a cluster
run CONCURRENTLY on a shared thread pool (the old code slept 0.1s after every
serial request).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from .clusters import clean, compact

_CONS = "bcdfghjklmnpqrtvwxz"
EXP_PARTICLES = {"van", "de", "der", "den", "von", "di", "du", "le", "la", "el",
                 "al", "ten", "ter", "te", "het", "t"}
LEGAL_FORMS = {"n.v.", "nv", "n.v", "maatschappij", "mij", "mij.", "co", "co.",
               "corporation", "corp", "corp.", "company", "ltd", "ltd.", "limited",
               "inc", "inc.", "gmbh", "ag", "a.g.", "s.a."}

# Cirrus filters on DIRECT statements only (no subclass closure; the P31 gate
# still runs afterwards).  Only PERSON has one obvious class.
TYPE_FILTER = {"PERSON": "haswbstatement:P31=Q5"}

# Dutch organisation-type word in a cluster name -> English phrase used in
# Wikidata labels/descriptions.  The ORGANIZATION analogue of the person
# cue queries ("Name British"): when the cluster name contains one of these
# words, an extra Cirrus query "{name without the word} {english phrase}"
# bridges translation gaps lexical search cannot cross
# ("New-Yorksche Beurs" -> "New Yorks Stock Exchange" -> Q13677).
ORG_CUE_WORDS = {
    "beurs": "stock exchange", "maatschappij": "company", "mij": "company",
    "bank": "bank", "spoorwegen": "railway", "spoorweg": "railway",
    "rederij": "shipping company", "stoomvaart": "shipping company",
    "lijn": "shipping line", "krant": "newspaper", "blad": "newspaper",
    "hotel": "hotel", "universiteit": "university", "hoogeschool": "university",
    "museum": "museum", "theater": "theatre", "bond": "federation",
    "vereniging": "association", "comité": "committee", "comite": "committee",
    "raad": "council", "ministerie": "ministry", "departement": "ministry",
    "ambassade": "embassy", "vliegveld": "airport", "haven": "port",
}

_ADJ_SUFFIX = re.compile(r"(che|e)$")


def modernize(word: str) -> str:
    """Old Dutch -sch spelling to modern: Duitschland -> Duitsland, Poolsche -> Poolse.

    Only after a consonant, so Bosch or Schiphol stay as they are.
    """
    w = re.sub(rf"(?i)(?<=[{_CONS}])sch(e|en)?\b", lambda m: "s" + (m.group(1) or ""), word)
    return re.sub(rf"(?i)(?<=[{_CONS}])sch(?=[{_CONS}])", "s", w)


def strip_adjective(tok: str) -> str:
    """Dutch adjectival inflection off a token: New-Yorksche -> New-Yorks,
    New-Yorkse -> New-Yorks, Amerikaansche -> Amerikaans.

    The trailing "-e" (modern) / "-che" (old spelling) is the INFLECTION; the
    final -s belongs to the adjective stem, which the Cirrus stemmer maps back
    to the base word — confirmed live: "stock exchange Yorks" -> 165 hits with
    the NYSE at rank 1, while the unstripped "Yorksche" matches NOTHING
    (CirrusSearch effectively requires every word of the query to match
    somewhere, so one old spelling poisons the whole query).  Only strips when
    the remaining stem keeps >= 5 characters ("Nieuwe" -> "Nieuw" is fine, a
    shorter stem would over-strip).
    """
    stem = _ADJ_SUFFIX.sub("", tok)
    return stem if len(stem) >= 5 else tok


def fuzzy_form(form: str) -> str:
    """Cirrus fuzzy query: every word of >= 5 characters gets '~' ("Vrije Stad" -> "Vrije~ Stad")."""
    return " ".join(t + "~" if len(t) >= 5 else t for t in form.split())


def cirrus_term(form: str) -> str:
    """Cirrus-safe query string: hyphens become spaces.

    Hyphenated OCR forms are everywhere in 1939 Dutch ("New-Yorksche"); a
    leading hyphen token is an EXCLUSION operator in CirrusSearch, and hyphens
    add nothing (the tokenizer splits on them anyway).
    """
    return form.replace("-", " ")


class Retriever:
    """Searches Wikidata for a cluster's candidate names (concurrent, fused)."""

    def __init__(self, wd, cfg, nat_groups: dict, titles: dict, query_pool):
        self.wd, self.cfg = wd, cfg
        self.NAT_GROUPS, self.TITLES = nat_groups, titles
        self.pool = query_pool            # shared ThreadPoolExecutor for the search fan-out

    # ------------------------------------------------------------------ query forms
    def query_forms(self, c: dict, extra=()) -> list[str]:
        """Distinct strings to search for a cluster (at most max_query_forms).

        Order: top 4 frequent surface forms, the cleaned cluster key, expansion
        hints, "Title Name" forms (PERSON), each followed by a despaced variant
        for two-token forms whose second token has <= 4 characters
        ("AM STERDAM" -> "AMSTERDAM").  Deduplicated on the compact form, so
        each distinct spelling is searched once, not once per mention.
        """
        forms = ([f for f, _ in c["forms"].most_common(4)] + [clean(c["key"])]
                 + list(extra) + list(c.get("titled_forms", [])))
        out, seen = [], set()

        def add(s):
            if s and compact(s) not in seen:
                seen.add(compact(s))
                out.append(s)

        for f in forms:
            add(f)
            parts = f.split()
            if len(parts) == 2 and len(parts[1]) <= 4:
                add(re.sub(r"\s+", "", f))
        return out[:self.cfg.max_query_forms]

    def cue_queries(self, c: dict) -> list[str]:
        """Extra Cirrus-only queries bridging word/translation gaps.

        PERSON clusters: "Name British", "Name general" — the English demonym of
        the most frequent supported nationality and the English keyword of the
        most frequent supported non-honorific title, with the two most frequent
        surface forms.  Cirrus searches descriptions too ("British diplomat"),
        which wbsearchentities cannot.
        """
        if c["type"] != "PERSON":
            return []
        names = [f for f, _ in c["forms"].most_common(2)]
        words = [self.NAT_GROUPS[g][0] for g, _ in c.get("nationalities", Counter()).most_common(1)]
        words += [self.TITLES[t][1][0] for t, _ in c.get("titles", Counter()).most_common()
                  if self.TITLES[t][0] != "honorific" and self.TITLES[t][1]][:1]
        return [cirrus_term(f"{n} {w}") for n in names for w in words]

    def org_cue_queries(self, c: dict) -> list[str]:
        """Extra Cirrus-only queries for ORGANIZATION clusters (ORG_CUE_WORDS).

        When the name contains a Dutch organisation-type word ("beurs",
        "maatschappij", "spoorwegen", ...), query "{name without it, adjectival
        suffixes stripped, hyphens spaced} {english phrase}".  Every word must
        match somewhere in Cirrus (AND semantics), so old spellings and the
        Dutch word itself are dropped rather than poisoning the query:
        "New-Yorksche Beurs" -> "New Yorks Stock Exchange" (-> NYSE, rank 1).
        """
        if c["type"] != "ORGANIZATION":
            return []
        out, seen = [], set()
        for f, _ in c["forms"].most_common(3):
            toks = f.split()
            cue = next((t for t in toks
                        if t.lower().strip(".-") in ORG_CUE_WORDS), None)
            if cue is None:
                continue
            en = ORG_CUE_WORDS[cue.lower().strip(".-")]
            name = " ".join(strip_adjective(t) for t in toks if t.lower() != cue.lower())
            q = cirrus_term(f"{name} {en}".strip())
            if name and q not in seen:
                seen.add(q)
                out.append(q)
        return out[:2]

    # ------------------------------------------------------------------ channels + fusion
    def _channels(self, c: dict, forms: list[str], limit: int, prefix: str = ""):
        """(channel, form, search-callable) for every search this form set needs."""
        cfg = self.cfg
        filt = TYPE_FILTER.get(c["type"], "")
        for f in forms:
            for lang in cfg.search_langs:
                yield (f"{prefix}wb:{lang}", f,
                       lambda t=f, l=lang, lm=limit: self.wd.search(t, l, lm))
            if not cfg.use_cirrus:
                continue
            yield (f"{prefix}cirrus", f,
                   lambda t=cirrus_term(f): self.wd.cirrus(t, filt, limit))
            if cfg.fuzzy_cirrus and " " not in f and len(f) >= 6:
                yield (f"{prefix}fuzzy", f,
                       lambda t=cirrus_term(f): self.wd.cirrus(t + "~", filt, cfg.fuzzy_limit))

    def _fuse(self, c: dict, tasks, keep: int):
        """Run the channel tasks concurrently and fuse their ranked hits with RRF.

        A channel task is (channel, form, callable -> [qid, ...] in rank order).
        Exceptions propagate: one failed (retried-out) search fails the cluster,
        which is then retried on the next run instead of stored as a wrong nil.
        """
        fused, found = defaultdict(float), defaultdict(list)
        futures = [(chan, form, self.pool.submit(fn)) for chan, form, fn in tasks]
        for chan, form, fut in futures:
            for rank, q in enumerate(fut.result(), 1):
                base = chan.split(":")[1] if chan.startswith("x:") else chan.split(":")[0]
                fused[q] += self.cfg.channel_weight.get(base, 1.0) / (self.cfg.rrf_k + rank)
                found[q].append([chan, form, rank])
        order = sorted(fused, key=lambda q: -fused[q])[:keep]
        top = fused[order[0]] if order else 1.0
        return order, {q: fused[q] / top for q in order}, {q: found[q][:8] for q in order}

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, c: dict, extra=()):
        """Normal retrieval: fused candidate order, relative RRF score, found-by trail."""
        forms = self.query_forms(c, extra)
        tasks = list(self._channels(c, forms, self.cfg.search_limit))
        if self.cfg.use_cirrus:
            filt = TYPE_FILTER.get(c["type"], "")
            tasks += [("cirrus", cq, lambda t=cq: self.wd.cirrus(t, filt))
                      for cq in self.cue_queries(c) + self.org_cue_queries(c)]
        return self._fuse(c, tasks, self.cfg.max_candidates)

    def retrieve_more(self, c: dict, extra, known: set):
        """Second round for clusters with few candidates (see expansion_forms).

        The strongest forms again with deeper result lists, plus NEW forms; more
        languages and a fuzzy variant.  New candidates get a retrieval feature of
        at most expansion_factor, so they cannot outrank normal candidates on
        retrieval alone.  Candidates in ``known`` are left out.  Channels carry
        the "x:" prefix so found_by can mark them.
        """
        cfg = self.cfg
        tasks = list(self._channels(c, self.query_forms(c, extra)[:3],
                                    cfg.expansion_limit, prefix="x:"))
        for f in self.expansion_forms(c, extra):
            for lang in cfg.expansion_langs:
                tasks.append((f"x:wb:{lang}", f,
                              lambda t=f, l=lang: self.wd.search(t, l)))
            if cfg.use_cirrus:
                filt = TYPE_FILTER.get(c["type"], "")
                tasks.append(("x:cirrus", f, lambda t=cirrus_term(f): self.wd.cirrus(t, filt)))
                tasks.append(("x:fuzzy", f,
                              lambda t=cirrus_term(f): self.wd.cirrus(fuzzy_form(t), filt,
                                                                      cfg.fuzzy_limit)))
        order, rrf, found = self._fuse(c, tasks,
                                       cfg.max_candidates + cfg.max_expansion_candidates)
        order = [q for q in order if q not in known][:cfg.max_expansion_candidates]
        top = rrf.get(order[0], 1.0) if order else 1.0
        return (order,
                {q: cfg.expansion_factor * rrf[q] for q in order},
                {q: found[q] for q in order})

    # ------------------------------------------------------------------ expansion forms
    def expansion_forms(self, c: dict, extra=()) -> list[str]:
        """New query strings for a cluster with few candidates, at most expansion_forms:
        modern spelling of old Dutch, adjectival suffix stripped ("New-Yorksche"
        -> "New Yorks"), hyphenated form, without particles / legal forms ("N.V."),
        surname or last word, first word of a longer name."""
        base = self.query_forms(c, extra)
        seen, out = {f.lower() for f in base}, []

        def add(s: str) -> None:
            s = s.strip()
            if len(s) >= 3 and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)

        for f in base:
            toks = f.split()
            add(" ".join(modernize(t) for t in toks))
            add(" ".join(strip_adjective(t) for t in toks))
            if len(toks) >= 2:
                if c["type"] != "PERSON":
                    add("-".join(toks))                            # "Vrije-Stad"
                add(" ".join(t for t in toks
                             if t.lower() not in EXP_PARTICLES and t.lower() not in LEGAL_FORMS))
                if len(toks[-1]) >= 4 and (c["type"] == "PERSON" or len(toks) > 2):
                    add(toks[-1])                                  # surname / last word
                if c["type"] != "PERSON" and len(toks[0]) >= 4:
                    add(toks[0])                                   # first word
        return out[:self.cfg.expansion_forms]