"""
Link NER clusters to Wikidata QIDs  (v8).

Usage
-----
    python link_wikidata_v8.py              # full run (resumes automatically)
    python link_wikidata_v8.py --limit 30   # quick test on the 30 biggest clusters
    python link_wikidata_v8.py --restart    # ignore/delete an existing pass-1 file
    python link_wikidata_v8.py --max-candidates 120 --stored-candidates 40 \
                               --search-limit 30 --cirrus-limit 40   # recall knobs

Input format (output of resolve_entities_v2.py)
-----------------------------------------------
Per cluster body: `representative` (used as the cluster name; the dict key is only the unique
id, e.g. "Wilson [british]"), `surface_forms` (form -> mention count), `n_documents`, and for
PERSON `nationality_counts`, `title_counts`, `role_counts` (number of DOCUMENTS in which the
cue stands next to the name, already aggregated per document cluster).
* If these fields exist they are used directly: forms come from `surface_forms`, cue counts from
  the three count fields, and a cue is supported when it appears in >= CUE_MIN_COUNT documents
  or in >= CUE_MIN_SHARE of the cluster's documents.
* Otherwise (older files) the cues are extracted per mention from context_before/context_after
  and counted per mention, as in earlier versions.
Mentions also carry `text` (cleaned surface form); sentences are longer than before
(>= 120 characters), which improves the `context` feature.

What is new in v8 (compared with v7): person cues
-------------------------------------------------
* Nationality: an adjective right before the name or its title ("de britsche ambassadeur
  CRAIGIE") or in an apposition after it ("CRAIGIE, de britsche ambassadeur") is detected
  (OCR-tolerant). It feeds a new feature `nationality`, and an extra Cirrus query
  "Name British".
* Cue support across mentions: titles and nationalities are counted per MENTION. A cue is
  only used (queries and features) when it appears in at least CUE_MIN_COUNT mentions or in
  at least CUE_MIN_SHARE of the scanned mentions, so one noisy mention no longer decides.
  Raw counts are kept (`titles_raw`, `nationalities_raw`) and shown to JEV.
* The `title` and `nationality` features are weighted by support: every supported cue votes
  with its mention count, and the feature is the share of the votes that match the candidate.
  Title keywords are matched as whole words (so "major" no longer matches "majority").

What is new in v7 (compared with v6)
------------------------------------
* Retrieval fusion: results of wbsearchentities (per language), Cirrus full-text and
  fuzzy Cirrus are fused with Reciprocal Rank Fusion (RRF) instead of concatenated.
  The fused rank becomes the feature `retrieval`, and `found_by` records which
  channel/form/rank found each candidate.
* Score normalisation: the score is a weighted average over the features that are
  APPLICABLE to the cluster (features that do not apply are None and ignored).
  In v6 non-applicable features counted as 0, which made "accepted" nearly unreachable.
* `blockers`: every record lists why it was not accepted.
* Titles for PERSON clusters ("Lord Snell", "Prins Bernhard"): titles found right
  before the name are used for extra search queries, for label matching and as the
  feature `title`.
* All candidate-count knobs are constants/CLI flags (see "Candidate limits").
* The final file also contains the ranked candidate list (qid, label, score), which
  jev_resolve.py uses.

Pipeline
--------
Pass 1 (network, slow, resumable, written incrementally):
    flatten clusters (+ titles) -> retrieval (RRF over channels) -> hard gates
    (type, not-yet-existing) -> features -> decision.  One line per cluster is appended
    to PASS1_PATH right after the cluster is finished (tail -f works).
Pass 2 (cached, fast):
    coherence bonus using accepted clusters in the same documents -> re-decide
    -> consistency checks across clusters -> final file OUT_PATH (JSONL).

Candidate expansion (clusters with too few candidates)
-----------------------------------------------------
If a cluster has fewer than MIN_CANDIDATES type/time-valid candidates after the normal search, a second
round looks for more (expand_cluster), with
  * the strongest query forms again, but with deeper result lists (EXPANSION_LIMIT hits);
  * new forms: modern spelling of old Dutch ("Duitschland" -> "Duitsland", "Poolsche" -> "Poolse"), hyphenated
    form, without particles / legal forms ("N.V."), the surname or the first/last word of a longer name;
  * more languages (EXPANSION_LANGS, e.g. German for "Danzig") and a fuzzy variant ("Vrije~ Stad").
New candidates get a retrieval feature of at most EXPANSION_FACTOR, so they cannot outrank candidates of the
normal search on retrieval alone. The record gets `expansion: {before, after}` (null if no expansion was
needed). Existing pass-1 records without that key are expanded in place when the script runs again, so a
finished run does not have to start over. --no-expand switches it off, --min-candidates changes the threshold.

Candidate limits (all can be changed above or via CLI flags)
------------------------------------------------------------
    SEARCH_LIMIT          hits per (form, language) from wbsearchentities (API max 50)
    CIRRUS_LIMIT          hits per form from Cirrus full-text search
    FUZZY_LIMIT           hits per fuzzy ("word~") Cirrus query
    MAX_QUERY_FORMS       maximum number of distinct strings searched per cluster
    MAX_CANDIDATES        distinct candidates kept after fusion (before the gates)
    MAX_STORED_CANDIDATES candidates written to the pass-1 and final files (what the
                          JEV step can choose from)

Output files (JSONL: one JSON object per line)
----------------------------------------------
PASS1_PATH  (provisional: no coherence yet):
    {"cluster_id", "cluster_key", "type", "n_mentions", "n_docs",
     "result": {...same fields as the final record, without cluster ids...},
     "candidates": [{"qid", "label", "feats": {...}, "per_mention": [...],
                     "found_by": [[channel, query form, rank], ...]}],
     "rejected":   [{"qid", "label", "why": "type" | "not_yet_existing"}]}

OUT_PATH  (final), every record has exactly these keys:
    cluster_id, cluster_key, type, n_mentions, n_docs,
    status          "accepted" | "review" | "nil"
    qid             chosen QID, or null if status == "nil"
    label           label of qid (or of best_candidate)
    score           score in about [0, 1] of the best candidate (see below)
    margin          score(best) - score(second best)
    evidence        feature values of the best candidate (null = not applicable)
    flags           single_mention | impure_cluster | ended_before_bulletin |
                    anachronistic_state | weak_label | type_conflict |
                    possible_duplicate_clusters
    blockers        reasons why status is not "accepted" (empty when accepted):
                    score<ACCEPT, margin<MARGIN, single_mention, impure_cluster,
                    ended_not_full_name, weak_label, no_candidate
    alternatives    up to 3 runner-up candidates [{qid, label, score}]
    candidates      ranked candidates [{qid, label, score}], at most MAX_STORED_CANDIDATES
    expanded        true if the candidate expansion ran for this cluster
    best_candidate  best QID even when status == "nil"
    reason          short text for nils without candidates / weak labels

Features (evidence).  Each is in [0, 1] or null when not applicable:
    label          best fuzzy similarity between a surface form (also "Title Name"
                   forms for persons) and any label/alias of the candidate (nl/en/de/fr).
                   PERSON also compares with the surname alone. 1.0 = exact name.
    context        similarity between the mention sentences and "label. description"
                   of the candidate (multilingual-e5, per mention, then mean of the
                   top 3; cosine rescaled with CTX_FLOOR/CTX_SPAN). null if the
                   cluster has no sentences or sentence-transformers is missing.
    retrieval      RRF score of the candidate divided by the best RRF score in the
                   cluster (how consistently the search channels ranked it).
    prior          min(number of sitelinks / 100, 1): weak popularity prior.
    domain         1.0 if an ORGANIZATION cluster looks like stock-exchange reports
                   and the candidate is a business, else 0.0; null for other clusters.
    expansion      similarity to a longer sibling ORG name (ellipsis); null if none.
    coherence      null in pass 1; in pass 2 min(k/3, 1) with k = links to accepted
                   clusters of the same documents; null when there is no such cluster.
    title          PERSON clusters with supported titles next to the name: share of the
                   title votes (mention counts) whose keywords occur as whole words in the
                   candidate's labels/aliases/descriptions (lord -> "lord"/"baron",
                   generaal -> "general", ...). null if no supported title.
    nationality    PERSON clusters with a supported nationality cue: share of the votes
                   whose keywords (English/Dutch/German/French stems) occur in the
                   candidate's descriptions. null if no supported cue or the candidate
                   has no description.
    ended          0/1 penalty: candidate died/ended before the first bulletin.
    anachronistic  0/1 penalty: modern state item (inception after the bulletins).

Score
-----
    base  = sum(W[f] * value[f]) / sum(W[f])        over applicable features only
    score = base - 0.10*ended - 0.05*anachronistic  (PENALTY)
Status: accepted if score >= ACCEPT, margin >= MARGIN and no blocker; review if
score >= REVIEW; else nil. A best candidate whose label similarity is below LABEL_MIN
is always nil (unless it matches an ellipsis sibling name).

Requirements
------------
    pip install requests rapidfuzz
    optional: pip install sentence-transformers  (context similarity; else feature = null)

NOTE: verify every class QID in ALLOWED_CLASSES / COUNTRY_CLASSES before trusting results.
"""
import argparse
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import requests
from rapidfuzz import fuzz, process

# ------------------------------------------------------------------ configuration
IN_PATH = "./data/processed/openai/resolution/resolved_names_v3.json"
OUT_PATH = "./data/processed/openai/links/wikidata_links_v4.jsonl"
PASS1_PATH = "./data/processed/openai/links/wikidata_links.pass1_v3.jsonl"
CACHE_PATH = "./data/processed/wd_cache.json"

WD_API = "https://www.wikidata.org/w/api.php"
SPARQL = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "radio1939-linker/0.1 (research; contact: you@example.org)"}
LANGS = ["nl", "en", "de", "fr"]          # languages fetched for labels/aliases/descriptions

TYPE_MAP = {"ORG": "ORGANIZATION"}        # JSON uses ORG, the enum uses ORGANIZATION

# VERIFY THESE QIDs. Matching is transitive: P31/P279*.
ALLOWED_CLASSES = {
    "PERSON": ["Q5"],
    "ORGANIZATION": ["Q43229", "Q4830453", "Q327333", "Q7278"],
    "PLACE": ["Q2221906", "Q6256", "Q515", "Q618123", "Q41176", "Q811979"],
    "SHIP": ["Q11446"],
    "SUBMARINE": ["Q2811"],
    "AIRCRAFT": ["Q11436", "Q15056995"],
    "ARMED_CONFLICT": ["Q350604", "Q198", "Q178561"],
    "DIPLOMATIC_MEETING": ["Q625994", "Q1656682"],
    "LEGAL_DOCUMENT": ["Q131569", "Q820655", "Q49848"],
}
BUSINESS = ["Q4830453"]                   # used for the stock-report "domain" cue
COUNTRY_CLASSES = ["Q6256", "Q3624078"]   # VERIFY: country, sovereign state (P31/P279*)
# Properties whose item values are compared for the coherence bonus.
RELATION_PROPS = ["P17", "P131", "P159", "P361", "P27", "P463", "P749", "P176", "P495", "P1344"]

# ---- candidate limits (see module docstring) ----
SEARCH_LANGS = ("nl", "en", "fr")          # wbsearchentities languages
SEARCH_LIMIT = 25                          # per (form, language); API max 50
USE_CIRRUS = True                          # second retrieval channel: full-text search
CIRRUS_LIMIT = 30                          # per form
FUZZY_CIRRUS = True                        # also "word~" for single-word forms (>= 6 chars)
FUZZY_LIMIT = 10
MAX_QUERY_FORMS = 8                        # distinct strings searched per cluster
MAX_CANDIDATES = 80                        # distinct candidates kept after fusion
MAX_STORED_CANDIDATES = 30                 # candidates stored in pass-1/final files
RRF_K = 60                                 # reciprocal rank fusion constant
CHANNEL_WEIGHT = {"wb": 1.0, "cirrus": 1.0, "fuzzy": 0.5}
# ---- candidate expansion (clusters with too few candidates) ----
EXPAND = True                              # --no-expand switches it off
MIN_CANDIDATES = 5                         # fewer valid candidates than this -> search more
EXPANSION_LIMIT = 50                       # hits per query when the strongest forms are searched again (API max 50)
EXPANSION_LANGS = ("nl", "en", "fr", "de") # wbsearchentities languages for the new forms
EXPANSION_FORMS = 5                        # new query forms per cluster
MAX_EXPANSION_CANDIDATES = 40              # new candidates kept per cluster
EXPANSION_FACTOR = 0.5                     # cap of the retrieval feature of new candidates
# Cirrus filters on DIRECT statements only (no subclass closure). Only PERSON has one
# obvious class; the P31/P279* type gate still runs afterwards.
TYPE_FILTER = {"PERSON": "haswbstatement:P31=Q5"}

# ---- scoring ----
# Weights of the applicable features (normalised by their sum, so they need not add to 1).
W = dict(label=0.30, context=0.25, retrieval=0.10, prior=0.10, domain=0.10,
         coherence=0.15, expansion=0.05, title=0.05, nationality=0.08)
PENALTY = dict(ended=0.10, anachronistic=0.05)     # subtracted after normalisation
ACCEPT, REVIEW, MARGIN = 0.70, 0.45, 0.12          # decision thresholds
LABEL_MIN = 0.80                                   # best candidate must match a name this well
# context similarity is rescaled: e5 cosine values of unrelated text are already ~0.7-0.8
CTX_FLOOR, CTX_SPAN = 0.70, 0.20                   # ctx = clip((cosine - FLOOR) / SPAN, 0, 1)

# A title/nationality cue is used only if it appears in >= CUE_MIN_COUNT mentions, or in
# >= CUE_MIN_SHARE of the PERSON mentions that have text around the name.
CUE_MIN_COUNT, CUE_MIN_SHARE = 2, 0.34

# Cues that a sentence comes from the stock-exchange report (fractions, price words).
FIN_RE = re.compile(r"\b\d+\s+\d/\d\b")
FIN_WORDS = {"sloot", "koers", "hooger", "lager", "staalwaarden", "spoorwegen", "beurs", "punt", "punten"}

# Titles, roles and nationality tables/extractors live in person_cues.py (shared with the
# entity-resolution step).
from .person_cues import (NAT_GROUPS, SEARCH_KINDS, TITLES, _NAT_RE, _TITLE_RE,  # noqa: F401
                         extract_titles, match_title, norm_tok, person_cues)


def supported_cues(counter: Counter, n_scan: int) -> Counter:
    """Keep cues seen in >= CUE_MIN_COUNT mentions or in >= CUE_MIN_SHARE of the scanned mentions."""
    return Counter({k: v for k, v in counter.items()
                    if v >= CUE_MIN_COUNT or v / max(n_scan, 1) >= CUE_MIN_SHARE})


def cue_queries(c: dict) -> list[str]:
    """Extra Cirrus-only queries for PERSON clusters: "Name British" and "Name general".

    Cirrus also searches descriptions ("British diplomat"), which wbsearchentities cannot.
    Uses the most frequent supported nationality and the most frequent supported title
    (English keyword) with the two most frequent surface forms.
    """
    if c["type"] != "PERSON":
        return []
    names = [f for f, _ in c["forms"].most_common(2)]
    words = [NAT_GROUPS[g][0] for g, _ in c.get("nationalities", Counter()).most_common(1)]
    words += [TITLES[t][1][0] for t, _ in c.get("titles", Counter()).most_common()
              if TITLES[t][0] != "honorific" and TITLES[t][1]][:1]
    return [f"{n} {w}" for n in names for w in words]


def build_titled_forms(forms: Counter, title_counts: Counter, max_titles: int = 2, max_names: int = 2):
    """Extra query strings "Title Name" for a PERSON cluster.

    Uses the most frequent searchable titles (noble/church/military) and the most
    frequent surface forms, plus the English equivalent of the title
    ("Prins Bernhard" and "Prince Bernhard"). Returns [] if no searchable title was seen.
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
    """Parse the bulletin date from an id like 'anp:anp:1939:07:22:6:mpeg21'."""
    p = doc_id.split(":")
    return date(int(p[2]), int(p[3]), int(p[4]))


class Cache:
    """Tiny JSON-file key/value cache for all Wikidata responses.

    Keys: 's|lang|limit|term' (wbsearchentities), 'c|limit|query' (Cirrus),
    'e|QID' (entity), 't|classes|QID' (type check), 'l|QID' (label, profiles),
    'w|lang|title' (Wikipedia summary, profiles).
    Saved periodically and on exit; makes reruns and resumes cheap.
    """

    def __init__(self, path):
        self.path, self.d = Path(path), {}
        if self.path.exists():
            self.d = json.loads(self.path.read_text())

    def get(self, k): return self.d.get(k)
    def set(self, k, v): self.d[k] = v

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.d))


cache = Cache(CACHE_PATH)


def append_jsonl(fh, obj: dict) -> None:
    """Write one JSON line and flush so the file can be read while the script runs."""
    fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
    fh.flush()


def read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file; a truncated last line (crash mid-write) is ignored."""
    p, out = Path(path), []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


# ------------------------------------------------------------------ Wikidata access
def check_api(r: requests.Response) -> dict:
    """Parse a Wikidata API response and raise if it is an error.

    Raises requests.RequestException on HTTP errors (429, 5xx, ...), non-JSON bodies,
    and HTTP-200 responses carrying {"error": ...}. The caller then skips the cluster
    (it is not written to the pass-1 file) and a rerun retries it, instead of storing a
    wrong "nil" caused by an empty/error response.
    """
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "error" in data:
        raise requests.RequestException(f"API error: {data['error']}")
    return data


def wd_search(term: str, lang: str, limit: int | None = None) -> list[str]:
    """wbsearchentities: prefix match on labels/aliases in ONE language. Cached.

    Returns up to `limit` (default SEARCH_LIMIT, API max 50) QIDs in Wikidata's own order.
    """
    limit = min(limit or SEARCH_LIMIT, 50)
    k = f"s|{lang}|{limit}|{term}"
    if (v := cache.get(k)) is not None:
        return v
    r = requests.get(
        WD_API, headers=HEADERS, timeout=30,
        params=dict(action="wbsearchentities", search=term, language=lang, uselang=lang,
                    type="item", limit=limit, format="json"))
    ids = [x["id"] for x in check_api(r).get("search", [])]
    cache.set(k, ids)
    time.sleep(0.1)
    return ids


def wd_cirrus(term: str, filt: str = "", limit: int | None = None) -> list[str]:
    """Full-text search (action=query&list=search, CirrusSearch) in the item namespace.

    Unlike wbsearchentities this searches labels, aliases and descriptions of ALL
    languages, ranks by text relevance plus popularity, and supports operators:
        haswbstatement:P31=Q5   keep only items with that direct statement
        word~                   fuzzy match (edit distance), useful for OCR noise
    Returns QIDs (page titles of items). Cached.
    """
    limit = min(limit or CIRRUS_LIMIT, 50)
    q = f"{term} {filt}".strip()
    k = f"c|{limit}|{q}"
    if (v := cache.get(k)) is not None:
        return v
    r = requests.get(
        WD_API, headers=HEADERS, timeout=30,
        params=dict(action="query", list="search", srsearch=q, srnamespace=0,
                    srlimit=limit, srprop="", format="json"))
    ids = [x["title"] for x in check_api(r).get("query", {}).get("search", [])
           if re.fullmatch(r"Q\d+", x.get("title", ""))]
    cache.set(k, ids)
    time.sleep(0.1)
    return ids


def wd_entities(qids: list[str]) -> dict[str, dict]:
    """Fetch labels, descriptions, aliases, claims and sitelinks for QIDs (cached)."""
    need = [q for q in qids if cache.get(f"e|{q}") is None]
    for i in range(0, len(need), 50):
        r = requests.get(
            WD_API, headers=HEADERS, timeout=60,
            params=dict(action="wbgetentities", ids="|".join(need[i:i + 50]), format="json",
                        props="labels|descriptions|aliases|claims|sitelinks",
                        languages="|".join(LANGS)))
        for q, e in check_api(r).get("entities", {}).items():
            cache.set(f"e|{q}", e)
        time.sleep(0.1)
    return {q: cache.get(f"e|{q}") for q in qids if cache.get(f"e|{q}")}


def in_classes(qids: list[str], classes: list[str]) -> set[str]:
    """Return the subset of `qids` whose P31/P279* closure reaches any of `classes`.

    Uses the SPARQL endpoint in batches of 100 and caches every answer.
    """
    ck = "|".join(sorted(classes))
    todo = [q for q in qids if cache.get(f"t|{ck}|{q}") is None]
    for i in range(0, len(todo), 100):
        chunk = todo[i:i + 100]
        query = ("SELECT DISTINCT ?q WHERE { VALUES ?q {%s} VALUES ?c {%s} ?q wdt:P31/wdt:P279* ?c }"
                 % (" ".join(f"wd:{q}" for q in chunk), " ".join(f"wd:{c}" for c in classes)))
        r = requests.get(SPARQL, headers=HEADERS, timeout=90, params=dict(query=query, format="json"))
        hit = {b["q"]["value"].rsplit("/", 1)[-1] for b in check_api(r)["results"]["bindings"]}
        for q in chunk:
            cache.set(f"t|{ck}|{q}", q in hit)
        time.sleep(0.5)
    return {q for q in qids if cache.get(f"t|{ck}|{q}")}


def claim_ids(e: dict, pid: str) -> list[str]:
    """QIDs used as values of property `pid` in entity `e` (item-valued claims only)."""
    return [c["mainsnak"]["datavalue"]["value"]["id"] for c in e.get("claims", {}).get(pid, [])
            if c["mainsnak"].get("datavalue") and c["mainsnak"]["datavalue"]["type"] == "wikibase-entityid"]


def claim_date(e: dict, pid: str):
    """First time-valued claim of `pid` as a `date` (missing month/day -> 1), or None."""
    for c in e.get("claims", {}).get(pid, []):
        dv = c["mainsnak"].get("datavalue")
        if dv and dv["type"] == "time":
            t = dv["value"]["time"]
            try:
                return date(int(t[1:5]), int(t[6:8]) or 1, int(t[9:11]) or 1)
            except ValueError:
                return None
    return None


# ------------------------------------------------------------------ clusters
def load_clusters(path: str) -> list[dict]:
    """Flatten the resolved JSON into one dict per (cluster key, entity type).

    Each cluster dict contains:
        id, key, type            identity ("<key>||<TYPE>")
        forms                    Counter of distinct cleaned surface forms
        mentions                 [{doc, text, sentence, context, has_ctx}] (sentence may be "")
        docs                     set of document ids
        n_mentions, n_docs
        min_date, max_date       first/last bulletin date of the cluster
        finance_ratio            share of mentions (with a sentence) that look like stock reports
        titles / nationalities   Counters of SUPPORTED titles+roles and nationality groups next to
                                 the name (PERSON only); *_raw = all cues. Counts are documents
                                 (new input format) or mentions (older files)
        n_scan                   denominator of the share rule: documents of the cluster (new
                                 format) or PERSON mentions with text around the name (older)
        titled_forms             extra "Title Name" query strings (PERSON only)
    Mentions flagged "sentence_not_found" have no sentence; they still count as
    mentions and as surface forms but give no context evidence.
    """
    data = json.load(open(path, encoding="utf-8"))
    clusters = []
    for key, by_type in data.items():
        for t, body in by_type.items():
            t = TYPE_MAP.get(t, t)
            rep = body.get("representative", key)   # resolve_entities_v2 adds it; key stays the unique id
            new_fmt = "surface_forms" in body
            cue_fields = t == "PERSON" and any(k in body for k in ("nationality_counts", "title_counts", "role_counts"))
            forms, mentions, dates, title_counts = Counter(), [], [], Counter()
            nat_counts, n_scan = Counter(), 0
            for doc_id, ms in body["documents"].items():
                dates.append(doc_date(doc_id))
                for m in ms:
                    if not new_fmt:
                        forms[clean(m["text"])] += 1
                    sent = m.get("sentence") or ""
                    mentions.append(dict(doc=doc_id, text=" ".join((m.get("text") or "").split()),
                                         sentence=sent, context=m.get("context") or "", has_ctx=bool(sent)))
                    if t == "PERSON" and not cue_fields:           # older files: extract per mention
                        before, after = m.get("context_before") or "", m.get("context_after") or ""
                        if before or after:
                            n_scan += 1
                        tts, nat = person_cues(before, after)
                        for tt in tts:
                            title_counts[tt] += 1
                        if nat:
                            nat_counts[nat] += 1
            if new_fmt:
                for f, n in body["surface_forms"].items():
                    forms[clean(f)] += n
            if cue_fields:                                         # new format: counts are documents per cue
                title_counts = Counter({k: v for k, v in
                                        {**body.get("title_counts", {}), **body.get("role_counts", {})}.items()
                                        if k in TITLES})
                nat_counts = Counter({k: v for k, v in body.get("nationality_counts", {}).items() if k in NAT_GROUPS})
                n_scan = body.get("n_documents", len(body["documents"]))
            with_ctx = [m for m in mentions if m["has_ctx"]]
            sup_titles, sup_nat = supported_cues(title_counts, n_scan), supported_cues(nat_counts, n_scan)
            fin = sum(1 for m in with_ctx
                      if FIN_RE.search(m["sentence"]) or any(w in m["sentence"].lower() for w in FIN_WORDS))
            clusters.append(dict(
                id=f"{key}||{t}", key=rep, type=t, forms=forms, mentions=mentions,
                docs=set(body["documents"]), n_mentions=len(mentions), n_docs=len(body["documents"]),
                min_date=min(dates), max_date=max(dates), finance_ratio=fin / max(len(with_ctx), 1),
                titles=sup_titles, titles_raw=title_counts,
                nationalities=sup_nat, nationalities_raw=nat_counts, n_scan=n_scan,
                titled_forms=build_titled_forms(forms, sup_titles) if t == "PERSON" else []))
    return clusters


def expansion_hints(clusters: list[dict]) -> dict[str, list[str]]:
    """Detect ellipsis between ORG clusters, e.g. "United States" -> "United States Steel".

    A short ORG name whose tokens are a prefix of a longer ORG name, with both
    clusters looking like stock reports, gets the longer name as an extra query
    string and expansion feature. Clusters are NOT merged.
    """
    hints = defaultdict(list)
    orgs = [c for c in clusters if c["type"] == "ORGANIZATION"]
    for s in orgs:
        st = s["key"].lower().split()
        if not st:
            continue
        for l in orgs:
            lt = l["key"].lower().split()
            if len(lt) > len(st) and lt[:len(st)] == st and s["finance_ratio"] > 0.5 and l["finance_ratio"] > 0.5:
                hints[s["id"]].append(l["key"])
    return hints


# ------------------------------------------------------------------ retrieval
def query_forms(c: dict, extra=()) -> list[str]:
    """Distinct strings to send to Wikidata search for a cluster (at most MAX_QUERY_FORMS).

    Order: top 4 frequent surface forms, the cleaned cluster key, expansion hints,
    "Title Name" forms (PERSON), each followed by a despaced variant for two-token forms
    whose second token has <= 4 characters ("AM STERDAM" -> "AMSTERDAM").
    Deduplicated on the `compact` form: each distinct spelling is searched once,
    not once per mention.
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
    return out[:MAX_QUERY_FORMS]


def retrieve(c: dict, extra=()):
    """Candidate generation with Reciprocal Rank Fusion over all channels and forms.

    For each query form: wbsearchentities in SEARCH_LANGS; Cirrus full-text (PERSON
    clusters filtered to humans); fuzzy Cirrus ("word~") for single-word forms of >= 6
    characters. Every hit adds weight/(RRF_K + rank) to the candidate, so candidates
    found by several channels/forms, and found high, rise to the top. Kept: the best
    MAX_CANDIDATES.
    Returns (qids ordered by fused score,
             {qid: fused score relative to the best (0..1)},
             {qid: [[channel, form, rank], ...]}  first 8 hits).
    """
    filt = TYPE_FILTER.get(c["type"], "")
    fused, found = defaultdict(float), defaultdict(list)

    def add(chan, form, ids):
        w = CHANNEL_WEIGHT[chan.split(":")[0]]
        for rank, q in enumerate(ids, 1):
            fused[q] += w / (RRF_K + rank)
            found[q].append([chan, form, rank])

    for f in query_forms(c, extra):
        for lang in SEARCH_LANGS:
            add(f"wb:{lang}", f, wd_search(f, lang))
        if USE_CIRRUS:
            add("cirrus", f, wd_cirrus(f, filt))
            if FUZZY_CIRRUS and " " not in f and len(f) >= 6:
                add("fuzzy", f, wd_cirrus(f + "~", filt, FUZZY_LIMIT))
    if USE_CIRRUS:                                   # "Name British", "Name general" (PERSON)
        for cq in cue_queries(c):
            add("cirrus", cq, wd_cirrus(cq, filt))
    order = sorted(fused, key=lambda q: -fused[q])[:MAX_CANDIDATES]
    top = fused[order[0]] if order else 1.0
    return order, {q: fused[q] / top for q in order}, {q: found[q][:8] for q in order}


# ------------------------------------------------------------------ candidate expansion
_CONS = "bcdfghjklmnpqrtvwxz"
EXP_PARTICLES = {"van", "de", "der", "den", "von", "di", "du", "le", "la", "el", "al", "ten", "ter", "te", "het", "t"}
LEGAL_FORMS = {"n.v.", "nv", "n.v", "maatschappij", "mij", "mij.", "co", "co.", "corporation", "corp", "corp.",
               "company", "ltd", "ltd.", "limited", "inc", "inc.", "gmbh", "ag", "a.g.", "s.a."}


def modernize(word: str) -> str:
    """Old Dutch -sch spelling to modern: Duitschland -> Duitsland, Poolsche -> Poolse, Nederlandsch -> Nederlands.
    Only after a consonant, so Bosch or Schiphol stay as they are."""
    w = re.sub(rf"(?i)(?<=[{_CONS}])sch(e|en)?\b", lambda m: "s" + (m.group(1) or ""), word)
    return re.sub(rf"(?i)(?<=[{_CONS}])sch(?=[{_CONS}])", "s", w)


def fuzzy_form(form: str) -> str:
    """Cirrus fuzzy query: every word of >= 5 characters gets '~' ('Vrije Stad' -> 'Vrije~ Stad')."""
    return " ".join(t + "~" if len(t) >= 5 else t for t in form.split())


def expansion_forms(c: dict, extra=()) -> list[str]:
    """New query strings for a cluster with few candidates (see module docstring), at most EXPANSION_FORMS."""
    base = query_forms(c, extra)
    seen, out = {f.lower() for f in base}, []              # exact strings: "Vrije-Stad" differs from "Vrije Stad"

    def add(s: str) -> None:
        s = s.strip()
        if len(s) >= 3 and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)

    for f in base:
        toks = f.split()
        add(" ".join(modernize(t) for t in toks))
        if len(toks) >= 2:
            if c["type"] != "PERSON":
                add("-".join(toks))                            # "Vrije-Stad"
            add(" ".join(t for t in toks if t.lower() not in EXP_PARTICLES and t.lower() not in LEGAL_FORMS))
            if len(toks[-1]) >= 4 and (c["type"] == "PERSON" or len(toks) > 2):
                add(toks[-1])                                  # surname / last word
            if c["type"] != "PERSON" and len(toks[0]) >= 4:
                add(toks[0])                                   # first word of a longer place/organisation name
    return out[:EXPANSION_FORMS]


def retrieve_more(c: dict, extra, known: set[str]):
    """Second search round. Returns (new qids, {qid: retrieval feature <= EXPANSION_FACTOR}, {qid: found_by})
    in the same shape as retrieve(); candidates in `known` are left out."""
    filt = TYPE_FILTER.get(c["type"], "")
    fused, found = defaultdict(float), defaultdict(list)

    def add(chan, form, ids):
        w = CHANNEL_WEIGHT[chan.split(":")[1]]
        for rank, q in enumerate(ids, 1):
            fused[q] += w / (RRF_K + rank)
            found[q].append([chan, form, rank])

    for f in query_forms(c, extra)[:3]:                        # the strongest forms again, with deeper lists
        for lang in SEARCH_LANGS:
            add(f"x:wb:{lang}", f, wd_search(f, lang, EXPANSION_LIMIT))
        if USE_CIRRUS:
            add("x:cirrus", f, wd_cirrus(f, filt, EXPANSION_LIMIT))
    for f in expansion_forms(c, extra):                        # new spellings and parts of the name
        for lang in EXPANSION_LANGS:
            add(f"x:wb:{lang}", f, wd_search(f, lang))
        if USE_CIRRUS:
            add("x:cirrus", f, wd_cirrus(f, filt))
            add("x:fuzzy", f, wd_cirrus(fuzzy_form(f), filt, FUZZY_LIMIT))
    order = [q for q in sorted(fused, key=lambda q: -fused[q]) if q not in known][:MAX_EXPANSION_CANDIDATES]
    top = fused[order[0]] if order else 1.0
    return order, {q: EXPANSION_FACTOR * fused[q] / top for q in order}, {q: found[q][:8] for q in order}


def expand_cluster(c: dict, extra, known: set[str]):
    """Search more candidates for a cluster and score them. Returns (new candidates, rejected)."""
    qids, rrf, found = retrieve_more(c, extra, known)
    if not qids:
        return [], []
    return score_candidates(c, wd_entities(qids), extra, rrf, found)


def upgrade_records(records: dict[str, dict], by_id: dict[str, dict], hints: dict) -> int:
    """Expand, in place, the pass-1 records that have fewer than MIN_CANDIDATES candidates and were written
    without the `expansion` key (an earlier run). Returns the number of records changed."""
    todo = [cid for cid, r in records.items() if cid in by_id and "expansion" not in r and len(r["candidates"]) < MIN_CANDIDATES]
    for i, cid in enumerate(todo, 1):
        rec, c = records[cid], by_id[cid]
        known = {x["qid"] for x in rec["candidates"]} | {x["qid"] for x in rec.get("rejected", [])}
        try:
            more, rej = expand_cluster(c, hints.get(cid, []), known)
        except requests.RequestException as ex:
            print(f"[skip expansion] {cid}: {ex!r}")
            continue
        cands = sorted(rec["candidates"] + more, key=lambda x: total(x["feats"]), reverse=True)
        reason = rec["result"].get("reason")
        rec["result"] = decide(c, cands)
        if not cands:
            rec["result"]["reason"] = reason
        rec["expansion"] = dict(before=len(rec["candidates"]), after=len(cands))
        rec["candidates"] = cands[:MAX_STORED_CANDIDATES]
        rec["rejected"] = (rec.get("rejected", []) + rej)[:10]
        if i % 25 == 0:
            cache.save()
            print(f"  expansion {i}/{len(todo)}")
    return len(todo)


# ------------------------------------------------------------------ gates
EVENT_TYPES = {"ARMED_CONFLICT", "DIPLOMATIC_MEETING", "LEGAL_DOCUMENT"}


def temporal_status(e: dict, c: dict, is_country: bool = False):
    """Temporal check of a candidate against the cluster's bulletin dates.

    Returns
        None     impossible: the entity did not exist yet (born/founded after the
                 last bulletin). Hard reject.
        "ended"  existed earlier but died/ended before the first bulletin. Kept, with
                 a score penalty and the flag `ended_before_bulletin`, because 1939
                 bulletins can mention commemorations, streets named after dead
                 people, historical states, ...
        "anachronistic"  only for PLACE candidates that are countries/states
                 (`is_country`): the item's inception is after the bulletins (e.g. the
                 Italian Republic, 1946, for "Italië" in 1939), but a country keeps its
                 identity across regimes. Kept with a small penalty.
        "ok"     fine, or no usable dates (unknown != wrong).
    Events, treaties and conflicts are never "ended": historical events are
    legitimately mentioned later (e.g. the First World War, a 1919 treaty).
    Dates used: PERSON P569/P570; others P571|P729|P580 (start), P576|P730|P582 (end).
    """
    if c["type"] == "PERSON":
        start, end = claim_date(e, "P569"), claim_date(e, "P570")
    else:
        start = claim_date(e, "P571") or claim_date(e, "P729") or claim_date(e, "P580")
        end = claim_date(e, "P576") or claim_date(e, "P730") or claim_date(e, "P582")
    if start and start > c["max_date"]:
        return "anachronistic" if (c["type"] == "PLACE" and is_country) else None
    if end and end < c["min_date"] and c["type"] not in EVENT_TYPES:
        return "ended"
    return "ok"


# ------------------------------------------------------------------ features
_model = None
_warned = False


def embed(texts: list[str]):
    """Embed texts with multilingual-e5 (normalized). Returns None if unavailable."""
    global _model, _warned
    try:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("intfloat/multilingual-e5-base")
        return _model.encode(texts, normalize_embeddings=True)
    except Exception as ex:
        if not _warned:
            print(f"[warn] embeddings unavailable ({ex!r}); context feature will be null")
            _warned = True
        return None


def label_sim(e: dict, c: dict) -> float:
    """Best fuzzy similarity (0..1) between any cluster form and any label/alias.

    Cluster forms are the surface forms plus, for persons, the "Title Name" forms.
    Comparison uses `compact` strings. For PERSON also compares against the last token
    of each name to handle surname-only mentions ("Chamberlain").
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


def context_sim(e: dict, c: dict, max_m: int = 10) -> tuple[float, list[float]]:
    """Per-mention context similarity to a candidate, aggregated.

    Each mention is evaluated SEPARATELY (embeddings are not averaged): the candidate's
    "label. description" is embedded once, each mention sentence (up to `max_m`, only
    mentions that have a sentence) is embedded and compared by cosine similarity, the
    cosine is rescaled to 0..1 (CTX_FLOOR/CTX_SPAN). Aggregation: mean of the top 3.
    Returns (aggregate, per-mention list); the list is also used to detect impure clusters.
    """
    label = next(iter(e.get("labels", {}).values()), {}).get("value", "")
    desc = next(iter(e.get("descriptions", {}).values()), {}).get("value", "")
    ms = [m["sentence"] for m in c["mentions"] if m["has_ctx"]][:max_m]
    if not ms:
        return 0.0, []
    vecs = embed([f"passage: {label}. {desc}"] + [f"query: {s}" for s in ms])
    if vecs is None:
        return 0.0, []
    raw = (vecs[1:] @ vecs[0]).tolist()
    sims = [min(max((x - CTX_FLOOR) / CTX_SPAN, 0.0), 1.0) for x in raw]
    top = sorted(sims, reverse=True)[:3]
    return sum(top) / len(top), sims


def _title_re(t: str) -> re.Pattern:
    """Whole-word regex (optional plural) of the keywords of title `t` ("major" must not match "majority")."""
    if t not in _TITLE_RE:
        _TITLE_RE[t] = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in TITLES[t][1]) + r")(?:s|es)?\b")
    return _TITLE_RE[t]


def title_feature(e: dict, c: dict):
    """Share of the title votes that match the candidate (None if no supported title).

    Every supported title votes with its mention count; a vote matches when a keyword of
    the title occurs as a whole word in the candidate's labels, aliases or descriptions.
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
        hits += n * bool(_title_re(t).search(text))
    return hits / total_votes if total_votes else None


def nationality_feature(e: dict, c: dict):
    """Share of the nationality votes that match the candidate's descriptions.

    None if the cluster has no supported nationality, or the candidate has no description
    (no information is not a mismatch).
    """
    text = " ".join(v["value"] for v in e.get("descriptions", {}).values()).lower()
    nats = c.get("nationalities", {})
    if not nats or not text:
        return None
    votes = sum(nats.values())
    return sum(n * bool(_NAT_RE[g].search(text)) for g, n in nats.items()) / votes


def score_candidates(c: dict, ents: dict, hint_forms: list[str], rrf: dict, found: dict):
    """Apply hard gates and compute the feature vector of every surviving candidate.

    Gates: type (P31/P279* in ALLOWED_CLASSES[type]) and "not yet existing"
    (temporal_status is None). Entities that already ended are kept with ended=1.
    Features that do not apply to the cluster are None (see module docstring).
    Returns (candidates, rejected); rejected = [{qid, label, why}] with why in
    {"type", "not_yet_existing"}, kept so that a nil can be explained.
    """
    qids = list(ents)
    ok_type = in_classes(qids, ALLOWED_CLASSES[c["type"]])
    domain_on = c["type"] == "ORGANIZATION" and c["finance_ratio"] > 0.5
    biz = in_classes(qids, BUSINESS) if domain_on else set()
    countries = in_classes(qids, COUNTRY_CLASSES) if c["type"] == "PLACE" else set()
    out, rejected = [], []
    for q in qids:
        e = ents[q]
        labels = e.get("labels", {})
        lab = labels.get("en", labels.get("nl", {})).get("value")
        if q not in ok_type:
            rejected.append(dict(qid=q, label=lab, why="type"))
            continue
        ts = temporal_status(e, c, is_country=q in countries)
        if ts is None:
            rejected.append(dict(qid=q, label=lab, why="not_yet_existing"))
            continue
        cs, per_mention = context_sim(e, c)
        feats = dict(
            label=label_sim(e, c),
            context=cs if per_mention else None,
            retrieval=rrf.get(q, 0.0),
            prior=min(len(e.get("sitelinks", {})) / 100, 1.0),
            domain=(1.0 if q in biz else 0.0) if domain_on else None,
            expansion=(max([fuzz.ratio(compact(h), compact(n["value"])) / 100
                            for h in hint_forms for n in labels.values()] or [0.0])
                       if hint_forms else None),
            coherence=None,
            title=title_feature(e, c),
            nationality=nationality_feature(e, c),
            ended=1.0 if ts == "ended" else 0.0,
            anachronistic=1.0 if ts == "anachronistic" else 0.0)
        out.append(dict(qid=q, label=lab, feats=feats, per_mention=per_mention,
                        found_by=found.get(q, [])))
    return out, rejected


def total(f: dict) -> float:
    """Score of a feature dict: weighted average over applicable features minus penalties."""
    num = den = 0.0
    for k, w in W.items():
        v = f.get(k)
        if v is not None:
            num += w * v
            den += w
    base = num / den if den else 0.0
    return base - sum(p * (f.get(k) or 0.0) for k, p in PENALTY.items())


# ------------------------------------------------------------------ decision
def decide(c: dict, cands: list[dict]) -> dict:
    """Turn scored candidates into a result record (always the same keys).

    accepted: score >= ACCEPT, margin >= MARGIN and no blocker.
    review:   score >= REVIEW otherwise.
    nil:      everything else, or best label similarity < LABEL_MIN (qid is null, the best
              candidate is kept in best_candidate).
    Blockers (why not accepted): score<ACCEPT, margin<MARGIN, single_mention (1 mention
    without near-exact label + strong context), impure_cluster (per-mention context spread
    > 0.25), ended_not_full_name (candidate ended before the bulletin and the match is not a
    full-name match), weak_label.
    """
    ranked = sorted(cands, key=lambda x: total(x["feats"]), reverse=True)
    if not ranked:
        return dict(status="nil", qid=None, label=None, score=0.0, margin=0.0, evidence={},
                    flags=[], blockers=["no_candidate"], alternatives=[], best_candidate=None,
                    reason="no type/time-valid candidate")
    best = ranked[0]
    f = best["feats"]
    s = total(f)
    s2 = total(ranked[1]["feats"]) if len(ranked) > 1 else 0.0
    margin = s - s2
    pm = best["per_mention"]
    weak = c["n_mentions"] < 2 and not (f["label"] >= 0.95 and (f.get("context") or 0) >= 0.5)
    impure = len(pm) >= 4 and (max(pm) - min(pm)) > 0.25
    ended = (f.get("ended") or 0) > 0
    anachronistic = (f.get("anachronistic") or 0) > 0
    full_name = f["label"] >= 0.95 and any(len(x.split()) >= 2 for x in c["forms"])
    weak_label = f["label"] < LABEL_MIN and (f.get("expansion") or 0) < 0.9

    blockers = []
    if s < ACCEPT: blockers.append(f"score<{ACCEPT}")
    if margin < MARGIN: blockers.append(f"margin<{MARGIN}")
    if weak: blockers.append("single_mention")
    if impure: blockers.append("impure_cluster")
    if ended and not full_name: blockers.append("ended_not_full_name")
    if weak_label: blockers.append("weak_label")

    if weak_label:
        status = "nil"
    elif not blockers:
        status = "accepted"
    elif s >= REVIEW:
        status = "review"
    else:
        status = "nil"
    return dict(
        status=status, qid=best["qid"] if status != "nil" else None, label=best["label"],
        score=round(s, 3), margin=round(margin, 3),
        evidence={k: (None if v is None else round(v, 3)) for k, v in f.items()},
        flags=[n for n, on in [("single_mention", weak), ("impure_cluster", impure),
                               ("ended_before_bulletin", ended), ("anachronistic_state", anachronistic),
                               ("weak_label", weak_label)] if on],
        blockers=blockers,
        alternatives=[dict(qid=r["qid"], label=r["label"], score=round(total(r["feats"]), 3))
                      for r in ranked[1:4]],
        best_candidate=best["qid"],
        reason=f"best label similarity {f['label']:.2f} < LABEL_MIN" if weak_label else None)


# ------------------------------------------------------------------ pass 1
def pass1(clusters: list[dict], hints: dict, restart: bool) -> dict[str, dict]:
    """Retrieve + score + decide every cluster; append each result to PASS1_PATH.

    Resumable: clusters whose id already appears in PASS1_PATH are skipped. A network or
    API error on a cluster is printed and the cluster is left for the next run (it is not
    written). Returns {cluster_id: pass-1 record} for all finished clusters.
    """
    p1 = Path(PASS1_PATH)
    p1.parent.mkdir(parents=True, exist_ok=True)
    if restart and p1.exists():
        p1.unlink()
    records = {r["cluster_id"]: r for r in read_jsonl(PASS1_PATH)}
    if EXPAND and records:                                        # earlier runs: expand clusters with few candidates
        if upgrade_records(records, {c["id"]: c for c in clusters}, hints):
            with open(PASS1_PATH, "w", encoding="utf-8") as fh:
                for r in records.values():
                    append_jsonl(fh, r)
            cache.save()
    todo = [c for c in clusters if c["id"] not in records]
    print(f"pass 1: {len(records)} done, {len(todo)} to do")

    with open(PASS1_PATH, "a", encoding="utf-8") as fh:
        try:
            for i, c in enumerate(todo, 1):
                try:
                    h = hints.get(c["id"], [])
                    qids, rrf, found = retrieve(c, extra=h)
                    ents = wd_entities(qids)
                    cands, rejected = score_candidates(c, ents, h, rrf, found)
                    expansion = None
                    if EXPAND and len(cands) < MIN_CANDIDATES:            # too few candidates: search more
                        more, rej = expand_cluster(c, h, set(qids))
                        expansion = dict(before=len(cands), after=len(cands) + len(more))
                        cands, rejected = cands + more, rejected + rej
                except requests.RequestException as ex:
                    print(f"[skip] {c['id']}: {ex!r}")
                    continue
                cands.sort(key=lambda x: total(x["feats"]), reverse=True)
                res = decide(c, cands)
                if not cands:                 # explain the nil
                    res["reason"] = (f"no type/time-valid candidate ({len(ents)} retrieved; "
                                     f"rejected: {dict(Counter(r['why'] for r in rejected))})")
                rec = dict(cluster_id=c["id"], cluster_key=c["key"], type=c["type"],
                           n_mentions=c["n_mentions"], n_docs=c["n_docs"], result=res,
                           candidates=cands[:MAX_STORED_CANDIDATES], rejected=rejected[:10], expansion=expansion)
                append_jsonl(fh, rec)
                records[c["id"]] = rec
                if i % 25 == 0:
                    cache.save()
                    print(f"  {i}/{len(todo)}")
        finally:
            cache.save()                      # also on Ctrl-C / crash
    return records


# ------------------------------------------------------------------ pass 2
def pass2(clusters: list[dict], records: dict[str, dict]) -> list[dict]:
    """Coherence bonus + consistency checks; returns the final records.

    Coherence: for each cluster, take QIDs of ACCEPTED clusters sharing a document. A
    candidate gets min(k/3, 1) where k counts how many of its RELATION_PROPS values are
    those sibling QIDs or the siblings' own relation values (same country, part of the
    same body, ...). Then decide() is re-run. Clusters without accepted siblings keep
    coherence = null. Consistency: one QID used by several clusters is flagged
    (type_conflict if the types differ -> accepted is demoted to review, else
    possible_duplicate_clusters).
    """
    clusters = [c for c in clusters if c["id"] in records]
    by_doc = defaultdict(list)
    for c in clusters:
        for d in c["docs"]:
            by_doc[d].append(c["id"])

    results = {c["id"]: records[c["id"]]["result"] for c in clusters}
    cands = {c["id"]: records[c["id"]]["candidates"] for c in clusters}

    new_results = {}
    for c in clusters:
        cs = cands[c["id"]]
        if not cs:
            continue
        sibs = {results[s]["qid"] for d in c["docs"] for s in by_doc[d]
                if s != c["id"] and results[s]["status"] == "accepted"}
        if not sibs:
            continue
        sib_rel = set()
        for s in sibs:
            e = cache.get(f"e|{s}") or {}
            for p in RELATION_PROPS:
                sib_rel |= set(claim_ids(e, p))
        ents = wd_entities([x["qid"] for x in cs])
        for cand in cs:
            e = ents.get(cand["qid"], {})
            rel = {x for p in RELATION_PROPS for x in claim_ids(e, p)}
            cand["feats"]["coherence"] = min((len(rel & sibs) + len(rel & sib_rel)) / 3, 1.0)
        new_results[c["id"]] = decide(c, cs)
    results.update(new_results)               # update after the loop: siblings use pass-1 status

    by_qid = defaultdict(list)
    for c in clusters:
        r = results[c["id"]]
        if r["qid"] and r["status"] != "nil":
            by_qid[r["qid"]].append(c)
    for cs_ in by_qid.values():
        if len(cs_) > 1:
            kind = "type_conflict" if len({x["type"] for x in cs_}) > 1 else "possible_duplicate_clusters"
            for x in cs_:
                r = results[x["id"]]
                r["flags"].append(kind)
                if kind == "type_conflict" and r["status"] == "accepted":
                    r["status"] = "review"
                    r["blockers"].append("type_conflict")

    final = []
    for c in clusters:
        ranked = sorted(cands[c["id"]], key=lambda x: total(x["feats"]), reverse=True)
        final.append(dict(
            cluster_id=c["id"], cluster_key=c["key"], type=c["type"],
            n_mentions=c["n_mentions"], n_docs=c["n_docs"], **results[c["id"]],
            candidates=[dict(qid=x["qid"], label=x["label"], score=round(total(x["feats"]), 3))
                        for x in ranked[:MAX_STORED_CANDIDATES]],
            expanded=bool(records[c["id"]].get("expansion"))))
    return final


# ------------------------------------------------------------------ main
def main(limit: int | None = None, restart: bool = False) -> None:
    """Run pass 1 (incremental, resumable) then pass 2 and write OUT_PATH."""
    clusters = load_clusters(IN_PATH)
    clusters.sort(key=lambda c: -c["n_mentions"])      # biggest clusters first
    if limit:
        clusters = clusters[:limit]
    hints = expansion_hints(clusters)

    records = pass1(clusters, hints, restart)

    missing = [c["id"] for c in clusters if c["id"] not in records]
    if missing:
        print(f"WARNING: {len(missing)} clusters not linked yet (network/API errors). "
              f"Rerun the script to process only those. First few: {missing[:5]}")

    final = pass2(clusters, records)

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        for rec in final:
            append_jsonl(fh, rec)
    cache.save()
    print(Counter(r["status"] for r in final), f"-> {OUT_PATH}")
    print("most common blockers:", Counter(b for r in final for b in r["blockers"]).most_common(6))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Link NER clusters to Wikidata QIDs")
    ap.add_argument("--limit", type=int, default=None, help="only the N clusters with most mentions")
    ap.add_argument("--restart", action="store_true", help="delete existing pass-1 file and start over")
    ap.add_argument("--max-candidates", type=int, default=None, help=f"default {MAX_CANDIDATES}")
    ap.add_argument("--stored-candidates", type=int, default=None, help=f"default {MAX_STORED_CANDIDATES}")
    ap.add_argument("--search-limit", type=int, default=None, help=f"default {SEARCH_LIMIT} (max 50)")
    ap.add_argument("--cirrus-limit", type=int, default=None, help=f"default {CIRRUS_LIMIT} (max 50)")
    ap.add_argument("--min-candidates", type=int, default=None, help=f"search more when a cluster has fewer valid candidates (default {MIN_CANDIDATES})")
    ap.add_argument("--no-expand", action="store_true", help="do not search more candidates for clusters with few")
    a = ap.parse_args()
    if a.no_expand:
        EXPAND = False
    globals().update({k: v for k, v in dict(
        MAX_CANDIDATES=a.max_candidates, 
        MAX_STORED_CANDIDATES=a.stored_candidates,
        SEARCH_LIMIT=a.search_limit, 
        CIRRUS_LIMIT=a.cirrus_limit, 
        MIN_CANDIDATES=a.min_candidates
    ).items() if v})
    main(limit=a.limit, restart=a.restart)