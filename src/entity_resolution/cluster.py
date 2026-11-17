"""Two-stage entity resolution with per-type strategies.

Every mention's type is standardized to a canonical capitalized form (``"Person"``,
``"Place"``, ``"LegalDocument"``, ``"Event"``, ``"Vehicle"`` — see
:mod:`.entity_types`).  Which type is resolved HOW is decided by
``ClusteringConfig.strategy_by_type``; the IE ``subtype`` fields are never used.

STRATEGY_STRING_SIMILARITY (default: Person, Place, LegalDocument, Vehicle)
---------------------------------------------------------------------------
Type scope
----------
Only the types mapped to this strategy go through both stages.  Every type with
no strategy becomes a singleton cluster (one mention per cluster) and is kept
document-local.

Stage 1 — within one bulletin (``cluster_document``)
----------------------------------------------------
Inside one document two different entities with similar names are rare, so matching
is deliberately RELAXED:

  A. **Typo level**: average linkage on the plain character-level similarity
     (``fuzz.ratio``) — the original behaviour.  With ``cfg.stage1_token_guard``
     the strict token-aware :func:`name_similarity` of stage 2 is applied here
     too (protects against same-bulletin false merges through shared generic
     tails; such a stage-1 merge could never be undone at stage 2).
     Cannot-link guards for numbers / Roman numerals and (for PERSON)
     nationality always apply; identical names with different known
     nationalities are never merged.
  B. **Containment** (``attach_contained``): a cluster whose name is contained in the
     name of exactly ONE other cluster of the same document is merged into it,
     tolerating typos and initials ("Chamberlain" -> "Neville Chamberlain",
     "N. Chamberlain" -> "Neville Chamberlain").  Ambiguous containment (two or more
     hosts) is left unmerged.

The representative of a document cluster is the LONGEST surface form (ties: most
frequent, alphabetical).  PERSON text cues (nationality, title, role, occupation)
are aggregated per cluster and written to every mention; ``cues_local`` keeps the
mention's own evidence.

Guarantee: every mention of a string-similarity type lands in exactly one cluster.
Mentions whose normalized name is empty (e.g. a name consisting only of stripped
titles) become singleton clusters and are logged under the
``empty_name_singletons`` diagnostic — they are never silently dropped, and they
stay document-local: they are never cross-bulletin merged in stage 2.

Stage 2 — over the whole corpus (``cluster_corpus``)
----------------------------------------------------
Each document cluster is a node with ALL its normalised forms as elements.  The
similarity of two nodes is the CLOSEST pair of elements, scored by the token-aware
:func:`name_similarity` (not plain character similarity: names that would only
merge through a shared generic tail such as ``"... communistische partij"`` are
vetoed by the token guard and logged under ``token_guard_blocks``).
Closest-pair (single) linkage chains easily, so merging is constrained:

  - **cannot-link**: PERSON nodes with different known nationalities; nodes with
    different numbers or Roman numerals; PERSON nodes whose full names conflict
    (same surname, different given name).
  - **ambiguity**: a node linked to two nodes that cannot be linked to each other is
    not merged with either; it can still merge with equally underspecified nodes.
  - merging goes from strongest link to weakest; two groups merge only if no
    cannot-link exists between any of their members.

Identical nodes (same forms, same nationality) are collapsed first, so a name in
thousands of documents does not create millions of links.

STRATEGY_EXACT_EVENT_ID (default: Event)
----------------------------------------
Event mentions are grouped by exact ``event_id`` *within one bulletin* (an
``entity_id`` field is accepted as a fallback name):

  - same non-empty id -> one cluster; different ids -> separate clusters;
    missing id -> singleton cluster;
  - stage 2 NEVER merges events across bulletins — each event cluster becomes
    its own corpus cluster keyed by bulletin and id.

Output
------
``resolve_entities`` returns ``(doc_clusters, global_resolved)``:

  - ``doc_clusters`` — ``{doc_id: [document cluster dicts]}``; each cluster carries
    ``representative``, ``elements``, ``nationality``, ``occupation``, cue counts,
    ``mentions``, ``event_id`` (Event) and ``global_cluster`` (the corpus cluster id).
  - ``global_resolved`` — ``{key: {type: body}}``; each body has ``representative``,
    ``n_mentions``, ``n_documents``, ``surface_forms``, ``doc_representatives``,
    nationality / title / role / occupation counts (PERSON) and ``documents``.
"""
import re
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.cluster import AgglomerativeClustering

from .config import ClusteringConfig, STRATEGY_EXACT_EVENT_ID  # noqa: F401
from .normalize import normalize_name, norm_ws
from .entity_types import canonical_type

# ----------------------------------------------------------------- diagnostics

DIAG: dict = {}


def reset_diagnostics() -> None:
    DIAG.clear()
    DIAG.update(
        nationality_splits=[], doc_nationality_conflicts=[], empty_name_singletons=[],
        containment_merges=[], ambiguous_containment=[], ambiguous_links_dropped=[],
        blocked_merges=[], token_guard_blocks=[],
    )


def _log(key: str, item: dict, cap: int = 300) -> None:
    if len(DIAG[key]) < cap:
        DIAG[key].append(item)


reset_diagnostics()


# ----------------------------------------------------------------- representatives

def choose_representative(counter: Counter) -> str:
    """Most frequent form; ties -> longest; then alphabetical (deterministic)."""
    return sorted(counter, key=lambda s: (-counter[s], -len(s), s))[0]


def choose_longest(counter: Counter) -> str:
    """Longest form; ties -> most frequent, then alphabetical."""
    return sorted(counter, key=lambda s: (-len(s), -counter[s], s))[0]


# ----------------------------------------------------------------- identity helpers

ROMAN = {"ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii",
         "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx"}


def number_key(name: str) -> frozenset:
    """Digits and Roman numerals of a normalised name ("u 47" -> {"47"}, "willem iii" -> {"iii"})."""
    return frozenset(re.findall(r"\d+", name) + [t for t in name.split() if t in ROMAN])


def top_unique(counter: Counter):
    """Most frequent key, or None if empty or tied."""
    ranked = counter.most_common(2)
    if not ranked or (len(ranked) == 2 and ranked[0][1] == ranked[1][1]):
        return None
    return ranked[0][0]


# ----------------------------------------------------------------- cue accessors

def mention_nat(m: dict):
    """Text-extracted nationality of a mention (from ``cues_local``), or None."""
    return (m.get("cues_local") or {}).get("nationality")


def mention_event_id(m: dict):
    """Event id of a mention: ``event_id`` first, ``entity_id`` as fallback."""
    return m.get("event_id") or m.get("entity_id")


def unit_nationality(items: list, nat_of, weight_of=None):
    """Weighted most frequent nationality of a group (None if unknown or tied)."""
    cnt = Counter()
    for it in items:
        nat = nat_of(it)
        if nat:
            cnt[nat] += weight_of(it) if weight_of else 1
    return top_unique(cnt)


# ----------------------------------------------------------------- token-aware similarity
#
# Why not plain fuzz.ratio?  Character-level similarity overrates multi-word names
# that share a long generic tail.  Real corpus case (both Organization, threshold
# 0.9, merged by plain fuzz.ratio at 0.918):
#
#     fuzz.ratio("fransche communistische partij",
#                "spaansche communistische partij")  =  0.918
#
# although the two parties are obviously different — 24 of the 30 characters (the
# whole tail " communistische partij" plus the shared "ansche") match.  What
# distinguishes the names is the token that changed.  :func:`name_similarity`
# therefore takes the MINIMUM of two views:
#
#   * full-string ``fuzz.ratio`` — tolerant to typos spread over the name;
#   * worst per-token Jaro-Winkler — every token of the shorter name must find a
#     similar partner token in the longer name.  Jaro-Winkler rewards the shared
#     PREFIX of historical spelling variants ("fransche"/"franse" -> 0.95) and
#     stays low for different words ("fransche"/"spaansche" -> 0.81).
#
# Because it is a minimum, the score can only be LOWER than fuzz.ratio: it vetoes
# merges, it never creates them.  Names whose tokens concatenate to the same string
# ("chamber lain" / "chamberlain": an OCR word split in two) count as identical.
#
# The two levels use it differently: stage 2 ALWAYS scores edges with
# :func:`name_similarity`; stage 1 uses the relaxed plain fuzz.ratio unless
# ``cfg.stage1_token_guard`` is set (inside one bulletin two different entities
# with similar names are rare, and a stage-1 merge cannot be undone at stage 2).

GUARD_WINDOW = 0.15   # re-score candidate pairs within this margin below the threshold


@lru_cache(maxsize=65536)
def _token_sim(a: str, b: str) -> float:
    """Cached Jaro-Winkler similarity of two tokens (0..1)."""
    return float(JaroWinkler.similarity(a, b))


@lru_cache(maxsize=65536)
def min_token_similarity(a: str, b: str) -> float:
    """Worst per-token Jaro-Winkler similarity of two names (0..1).

    For every (multi-character) token of the shorter name the BEST match among
    the longer name's tokens is taken; the minimum over those is returned.
    A low value means "at least one token has no counterpart in the other name"
    — the signature of two different entities sharing only generic words.
    Single-character tokens are wildcards (initials are matched by containment
    and ``name_compat``, not here).
    """
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return 0.0
    shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    scores = [max(_token_sim(t, u) for u in longer) for t in shorter if len(t) > 1]
    return min(scores) if scores else 1.0


@lru_cache(maxsize=262144)
def name_similarity(a: str, b: str) -> float:
    """Token-aware similarity of two normalised names (0..1) — the merge score.

    ``min(fuzz.ratio of the full strings, worst per-token Jaro-Winkler)``, with an
    exact-despace shortcut for OCR words split in two.  Examples:

        ("fransche communistische partij", "spaansche ...")  ->  0.81  vetoed
        ("fransche communistische partij", "franse ...")     ->  0.95  merges
        ("chamber lain", "chamberlain")                      ->  1.00  OCR split
    """
    if a == b:
        return 1.0
    if a.replace(" ", "") == b.replace(" ", ""):
        return 1.0
    return min(fuzz.ratio(a, b) / 100.0, min_token_similarity(a, b))


# ----------------------------------------------------------------- distance matrix + stage-1 clustering

def distance_matrix(names: list[str], nat_ids, num_ids, thr: float,
                    guard: bool = True) -> np.ndarray:
    """1 - similarity, forced to 1.0 for pairs that must not be merged.

    The full fuzz.ratio matrix is computed with the fast C implementation.  With
    ``guard=True`` (stage 2, or stage 1 with ``stage1_token_guard``) candidate
    pairs (fuzz ratio within ``GUARD_WINDOW`` below ``thr``) are re-scored with
    the (slower, cached) :func:`name_similarity`, which can only LOWER the
    similarity — pairs outside that window keep their fuzz score, which is then
    an optimistic lower bound that could never reach the merge threshold anyway.
    Guard vetoes of otherwise-merging pairs are logged under ``token_guard_blocks``.
    With ``guard=False`` (default stage 1) the plain fuzz.ratio distances are kept.
    """
    fuzz_sim = process.cdist(names, names, scorer=fuzz.ratio, dtype=np.float32, workers=-1) / 100.0
    dist = 1.0 - fuzz_sim
    if guard:
        lo = max(thr - GUARD_WINDOW, 0.0)
        for i, j in np.argwhere(fuzz_sim >= lo):
            if j <= i:
                continue
            s = name_similarity(names[i], names[j])
            if s < float(fuzz_sim[i, j]):
                dist[i, j] = dist[j, i] = 1.0 - s
                if fuzz_sim[i, j] >= thr and s < thr:
                    _log("token_guard_blocks",
                         dict(a=names[i], b=names[j], fuzz=round(float(fuzz_sim[i, j]), 3),
                              score=round(s, 3), stage=1))
    if nat_ids is not None:
        dist[(nat_ids[:, None] != nat_ids[None, :]) & (nat_ids[:, None] > 0) & (nat_ids[None, :] > 0)] = 1.0
    dist[(num_ids[:, None] != num_ids[None, :]) & (num_ids[:, None] > 0) & (num_ids[None, :] > 0)] = 1.0
    np.fill_diagonal(dist, 0.0)
    return dist


def repair_nationality(labels, nats: list, dist: np.ndarray, thr: float, names: list[str]) -> np.ndarray:
    """Enforce the nationality constraint after average linkage.

    Average linkage only discourages the forced 1.0 distances (a big cluster dilutes
    them), so a cluster containing two different known nationalities is split, and a
    unit with unknown nationality that is close to known units of two different
    nationalities is moved out as ambiguous.
    """
    labels = np.array(labels)
    nxt = int(labels.max()) + 1
    close = 1.0 - thr
    ambiguous: dict[int, int] = {}

    def amb_label(orig: int) -> int:
        nonlocal nxt
        if orig not in ambiguous:
            ambiguous[orig] = nxt
            nxt += 1
        return ambiguous[orig]

    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        groups = defaultdict(list)
        for i in idx:
            if nats[i]:
                groups[nats[i]].append(i)
        if len(groups) < 2:
            continue
        order = sorted(groups, key=lambda k: (-len(groups[k]), k))
        new = {order[0]: int(lab)}
        for nat in order[1:]:
            new[nat] = nxt
            nxt += 1
        for nat, members in groups.items():
            for i in members:
                labels[i] = new[nat]
        for i in idx:
            if not nats[i]:
                near = [nat for nat, ms in groups.items() if min(dist[i, j] for j in ms) <= close]
                labels[i] = new[near[0]] if len(near) == 1 else amb_label(int(lab))
        _log("nationality_splits", dict(names=sorted({names[i] for i in idx}),
                                        nationalities={n: sorted({names[i] for i in m}) for n, m in groups.items()}))

    known = [j for j, n in enumerate(nats) if n]
    for i, n in enumerate(nats):
        if n or int(labels[i]) in ambiguous.values():
            continue
        if len({nats[j] for j in known if dist[i, j] <= close}) >= 2:
            labels[i] = amb_label(int(labels[i]))
    return labels


def cluster_units(names: list[str], nats: list, thr: float, use_nat: bool,
                  guard: bool = True) -> list[int]:
    """Cluster normalised unique names; returns one label per name.

    ``guard=False`` uses the relaxed plain fuzz.ratio (default stage 1);
    ``guard=True`` applies the token-aware merge score (stage 2, or opt-in).
    """
    n = len(names)
    if n == 1:
        return [0]
    nat_vals = {v: i + 1 for i, v in enumerate(sorted({x for x in nats if x}))}
    nat_ids = np.array([nat_vals.get(x, 0) for x in nats]) if use_nat else None
    keys, num_ids = {}, []
    for s in names:
        k = number_key(s)
        num_ids.append(keys.setdefault(k, len(keys) + 1) if k else 0)
    dist = distance_matrix(names, nat_ids, np.array(num_ids), thr, guard)
    labels = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                     distance_threshold=1.0 - thr).fit_predict(dist)
    if use_nat:
        labels = repair_nationality(labels, nats, dist, thr, names)
    return [int(x) for x in labels]


def entity_disambiguation(items: list, thr: float, thr_of=None, nat_of=None,
                          use_nat: bool = True, field: str = "text",
                          guard: bool = True) -> list[dict]:
    """Cluster items (mentions) independently PER TYPE (string-similarity strategy).

    The clustering unit is (normalised name, nationality) for PERSON: identical names
    with different known nationalities are separate units that are never merged.
    ``thr_of(etype)`` overrides the threshold per type when given.  ``guard=False``
    clusters on the relaxed plain fuzz.ratio (default stage 1); ``guard=True`` on
    the token-aware score.

    Every item lands in exactly one returned cluster: items whose normalized name is
    empty are wrapped in their own singleton cluster (and logged), never dropped.
    Returns ``[{"type", "norm_names", "members"}]``.
    """
    by_type = defaultdict(list)
    for it in items:
        if it.get(field) and it.get("type"):
            by_type[it["type"]].append(it)

    clusters = []
    for etype, group in by_type.items():
        is_person = etype == "Person"
        use = use_nat and nat_of is not None and is_person
        t = thr_of(etype) if thr_of else thr
        by_unit = defaultdict(list)
        empty: list = []
        for it in group:
            nm = normalize_name(it[field])
            if nm:
                by_unit[(nm, nat_of(it) if use else None)].append(it)
            else:
                empty.append(it)
        units = sorted(by_unit, key=lambda u: (u[0], u[1] or ""))
        if units:
            names = [u[0] for u in units]
            nats = [u[1] for u in units]
            labels = cluster_units(names, nats, t, use, guard)
            buckets = defaultdict(list)
            for u, lab in zip(units, labels):
                buckets[lab].append(u)
            for us in buckets.values():
                clusters.append(dict(type=etype, norm_names=sorted({u[0] for u in us}),
                                     members=[it for u in us for it in by_unit[u]]))
        if empty:
            # no-silent-drop guarantee: unwrapped mentions stay in the output
            clusters.append(dict(type=etype, norm_names=[], members=empty))
            _log("empty_name_singletons", dict(type=etype, count=len(empty),
                                               examples=[e.get("text", "")[:40] for e in empty[:5]]))
    return clusters


# ----------------------------------------------------------------- containment (stage 1)

PARTICLES = {"van", "de", "der", "den", "von", "di", "du", "le", "la", "el", "al",
             "ten", "ter", "te", "het", "t", "d"}


def tokens_match(a: str, b: str, thr: float) -> bool:
    """Equal tokens, an initial vs a name starting with it, or fuzzy equal tokens.

    Fuzzy equality uses Jaro-Winkler, which rewards a shared prefix: historical
    spelling variants ("fransche"/"franse", "chamberlain"/"chamberlin") match,
    different words that merely share a suffix do not.
    """
    if a == b:
        return True
    if len(a) == 1 or len(b) == 1:
        return a.startswith(b) or b.startswith(a)
    return _token_sim(a, b) >= thr


def contained(short: str, long: str, thr: float, anchor_last: bool = True) -> bool:
    """Is the normalised name ``short`` contained in ``long``?

    All tokens of ``short`` must match distinct tokens of ``long`` in order;
    ``short`` must have fewer tokens, or the same number with an initial standing for
    a longer token.  With ``anchor_last`` (PERSON) the surname must match the last
    token of ``long``.
    """
    ts, tl = short.split(), long.split()
    if not ts or len(ts) > len(tl):
        return False
    if len(ts) == len(tl) and not any(len(t) == 1 and len(u) > 1 for t, u in zip(ts, tl)):
        return False
    if anchor_last:
        if len(ts[-1]) < 3 or not tokens_match(ts[-1], tl[-1], thr):
            return False
        ts, tl = ts[:-1], tl[:-1]
    j = 0
    for t in ts:
        while j < len(tl) and not tokens_match(t, tl[j], thr):
            j += 1
        if j == len(tl):
            return False
        j += 1
    return True


def name_compat(a: str, b: str, thr: float) -> bool:
    """Can two multi-token PERSON names be the same person? Same surname and compatible first names.

    Shares :func:`name_similarity`'s despace identity: OCR breaks a surname
    across a line ("VON RIBBEN- TROP" / "VON RIBBENTROP"), so the token sequences
    disagree while the concatenated characters are identical.  Without this
    shortcut the cannot-link below vetoes the very edges that
    :func:`name_similarity` scores 1.0, and the ambiguity-drop pass then severs
    every link at such clusters — a hard corpus-wide split of one person.
    """
    if a.replace(" ", "") == b.replace(" ", ""):
        return True
    ta, tb = a.split(), b.split()
    if not tokens_match(ta[-1], tb[-1], thr):
        return False
    ga = [t for t in ta[:-1] if t not in PARTICLES]
    gb = [t for t in tb[:-1] if t not in PARTICLES]
    return not ga or not gb or tokens_match(ga[0], gb[0], 0.75)


def attach_contained(clusters: list[dict], cfg: ClusteringConfig, doc_id: str = "") -> list[dict]:
    """Merge a cluster into the ONE cluster of the same document that contains its name.

    Only types listed in ``cfg.containment_types`` participate; matching uses the
    stage-1 threshold ``cfg.threshold``.
    """
    types = set(cfg.containment_types)
    while True:
        cand = sorted((c for c in clusters if c["type"] in types),
                      key=lambda c: (max(len(n.split()) for n in c["norm_names"] or [""]), c["norm_names"]))
        merged = False
        for s in cand:
            if not s["norm_names"]:
                continue
            anchor = s["type"] == "Person"
            hosts = []
            for c in clusters:
                if c is s or c["type"] != s["type"] or not c["norm_names"]:
                    continue
                if not any(contained(sn, cn, cfg.threshold, anchor)
                           for sn in s["norm_names"] for cn in c["norm_names"]):
                    continue
                ns = unit_nationality(s["members"], mention_nat)
                nc = unit_nationality(c["members"], mention_nat)
                if ns and nc and ns != nc:
                    continue
                hosts.append(c)
            if len(hosts) == 1:
                h = hosts[0]
                _log("containment_merges", dict(doc=doc_id, name=s["norm_names"], into=h["norm_names"]))
                h["members"].extend(s["members"])
                h["norm_names"] = sorted(set(h["norm_names"]) | set(s["norm_names"]))
                clusters = [c for c in clusters if c is not s]
                merged = True
                break
            if len(hosts) > 1:
                _log("ambiguous_containment",
                     dict(doc=doc_id, name=s["norm_names"], hosts=[h["norm_names"] for h in hosts]))
        if not merged:
            return clusters


# ----------------------------------------------------------------- events + singletons (stage 1)

def cluster_events_by_id(events: list) -> list[dict]:
    """Group EVENT mentions by exact ``event_id`` *within one bulletin*.

    ``event_id`` is read first, ``entity_id`` is accepted as a fallback field name.

    - Events with the same non-empty id -> one cluster.
    - Events with different ids -> separate clusters.
    - Events without an id -> each its own singleton cluster.

    No cross-bulletin merging is ever performed for events (stage 2 skips them).
    """
    by_eid: dict = defaultdict(list)
    no_eid: list = []
    for e in events:
        eid = mention_event_id(e)
        if eid:
            by_eid[eid].append(e)
        else:
            no_eid.append(e)

    out = []
    for eid, members in by_eid.items():
        norm_names = sorted({normalize_name(m.get("text", "")) for m in members
                             if m.get("text")})
        out.append(dict(type="Event", members=members, norm_names=norm_names,
                        event_id=eid))
    for e in no_eid:
        nm = normalize_name(e.get("text", ""))
        out.append(dict(type="Event", members=[e],
                        norm_names=[nm] if nm else [], event_id=None))
    return out


def make_singleton_cluster(entity: dict) -> dict:
    """Wrap a single mention as its own cluster (no merging, stage 1 + stage 2)."""
    nm = normalize_name(entity.get("text", ""))
    return dict(type=canonical_type(entity.get("type")) or "Unknown",
                members=[entity], norm_names=[nm] if nm else [])


# ----------------------------------------------------------------- stage 1 entry point

def cluster_document(entities: list[dict], cfg: ClusteringConfig, doc_id: str = "") -> list[dict]:
    """Stage 1: within-bulletin clustering for ALL entity types.

    Types are canonicalized defensively here as well, so the function behaves
    correctly even without the context stage.  Dispatch by strategy:

    - string-similarity types (``cfg.string_similarity_types``): typo-level fuzzy
      clustering (:func:`entity_disambiguation`) + containment attachment
      (:func:`attach_contained`).
    - Event (``STRATEGY_EXACT_EVENT_ID``): grouped by exact ``event_id`` within this
      bulletin via :func:`cluster_events_by_id`.
    - Types with no strategy: each mention becomes its own singleton cluster via
      :func:`make_singleton_cluster`.
    """
    for e in entities:
        e["type"] = canonical_type(e.get("type")) or "Unknown"

    sim = cfg.string_similarity_types
    clustered = [e for e in entities if e.get("type") in sim]
    clusters = entity_disambiguation(
        clustered, cfg.threshold, thr_of=cfg.stage1_threshold,
        nat_of=mention_nat, use_nat=cfg.nationality_constraint,
        guard=cfg.stage1_token_guard)
    clusters = attach_contained(clusters, cfg, doc_id)

    events = [e for e in entities if e.get("type") == "Event"]
    clusters.extend(cluster_events_by_id(events))

    passthrough = sim | {"Event"}
    others = [e for e in entities if e.get("type") not in passthrough]
    for e in others:
        clusters.append(make_singleton_cluster(e))

    return clusters


# ----------------------------------------------------------------- stage-1 finalization

def finalize_doc_cluster(cl: dict, doc_id: str) -> dict:
    """Document-level cluster: longest representative, cues propagated to every mention."""
    mentions = cl["members"]
    out = dict(
        type=cl["type"], doc_id=doc_id, mentions=mentions,
        representative=choose_longest(Counter(norm_ws(m["text"]) for m in mentions)),
        elements=sorted(set(cl["norm_names"])),
        nationality=None, nationality_counts=Counter(),
        title_counts=Counter(), role_counts=Counter(), occupation_counts=Counter(),
        occupation=None,
    )
    if cl["type"] == "Person":
        def _add_each_once(counter: Counter, cues) -> None:
            # count each distinct cue once per mention, in reading order —
            # a set() here would order ties by per-process string hash
            for cue in dict.fromkeys(cues or []):
                counter[cue] += 1

        nat, ttl, rol, occ = Counter(), Counter(), Counter(), Counter()
        for m in mentions:
            c = m.get("cues_local") or {}
            if c.get("nationality"):
                nat[c["nationality"]] += 1
            _add_each_once(ttl, c.get("title"))
            _add_each_once(rol, c.get("role"))
            _add_each_once(occ, c.get("occupation"))
        if len(nat) > 1:
            _log("doc_nationality_conflicts", dict(doc=doc_id, name=out["representative"], counts=dict(nat)))
        out.update(
            nationality=top_unique(nat), nationality_counts=nat,
            title_counts=ttl, role_counts=rol, occupation_counts=occ,
            occupation=top_unique(occ),
        )
        for m in mentions:
            m["nationality"] = out["nationality"]
            m["title"] = [k for k, _ in ttl.most_common()]
            m["role"] = [k for k, _ in rol.most_common()]
            m["occupation"] = [k for k, _ in occ.most_common()]
    if cl["type"] == "Event":
        out["event_id"] = cl.get("event_id")
    return out


# ----------------------------------------------------------------- stage 2: corpus-wide

def cluster_corpus(dcs: list[dict], thr: float, use_nat: bool, person: bool) -> list[list[dict]]:
    """Stage 2 for ONE type: groups of document clusters (see module docstring).

    ``thr`` is the corpus threshold for this type; ``use_nat``/``person`` enable the
    nationality cannot-link and the full-name compatibility guard respectively
    (both apply to Person only).
    """
    sigs = defaultdict(list)
    for dc in dcs:
        key = (tuple(dc["elements"]), dc["nationality"] if use_nat else None)
        sigs[key].append(dc)
    keys = sorted(sigs, key=lambda k: (k[0], k[1] or ""))
    n = len(keys)
    if n == 1:
        return [sigs[keys[0]]]

    elems = [list(k[0]) for k in keys]
    uniq = sorted({e for el in elems for e in el})
    pos = {e: i for i, e in enumerate(uniq)}
    order = np.array([pos[e] for el in elems for e in el])
    starts = np.cumsum([0] + [len(el) for el in elems[:-1]])
    node = [dict(
        nat=keys[i][1],
        num=frozenset().union(*(number_key(e) for e in elems[i])),
        multi=frozenset(e for e in elems[i] if len(e.split()) >= 2),
    ) for i in range(n)]

    # candidate links: closest pair of elements, token-guarded (no n x n matrix stored).
    # cells below the fuzz cutoff are 0, so they can never reach the threshold;
    # candidate cells are re-scored with the token-aware similarity, which can only
    # lower them — a pair that survives via a shared generic tail is vetoed here.
    edges = []
    for i in range(n):
        sims = process.cdist(elems[i], uniq, scorer=fuzz.ratio, dtype=np.float32,
                             score_cutoff=thr * 100, workers=-1)
        if not sims.any():
            continue
        for x, y in np.argwhere(sims > 0):
            s = name_similarity(elems[i][x], uniq[y]) * 100.0
            if s < float(sims[x, y]):
                if s < thr * 100.0:
                    _log("token_guard_blocks",
                         dict(a=elems[i][x], b=uniq[y], fuzz=round(float(sims[x, y]) / 100, 3),
                              score=round(s / 100, 3), stage=2))
                sims[x, y] = s
        vec = np.maximum.reduceat(sims.max(axis=0)[order], starts)
        edges += [(float(vec[j]), i, int(j)) for j in np.where(vec >= thr * 100)[0] if j > i]

    @lru_cache(maxsize=None)
    def compat(a: str, b: str) -> bool:
        return name_compat(a, b, thr)

    def cannot(a: dict, b: dict) -> bool:
        if use_nat and a["nat"] and b["nat"] and a["nat"] != b["nat"]:
            return True
        if a["num"] and b["num"] and a["num"].isdisjoint(b["num"]):
            return True
        return bool(person and a["multi"] and b["multi"]
                    and not any(compat(x, y) for x in a["multi"] for y in b["multi"]))

    adj = defaultdict(dict)
    ok = []
    for s, i, j in edges:
        if not cannot(node[i], node[j]):
            ok.append((s, i, j))
            adj[i][j] = adj[j][i] = s
    drop = set()
    for u, nb in adj.items():
        js = list(nb)
        conflicting = {x for a_ in range(len(js)) for b_ in range(a_ + 1, len(js))
                       if cannot(node[js[a_]], node[js[b_]]) for x in (js[a_], js[b_])}
        if conflicting:
            _log("ambiguous_links_dropped",
                 dict(node=elems[u][:3], neighbours=[elems[x][:2] for x in sorted(conflicting)[:4]]))
            drop.update((u, x) for x in conflicting)
            drop.update((x, u) for x in conflicting)
    ok = [e for e in ok if (e[1], e[2]) not in drop]

    # constrained single-linkage: strongest links first
    parent = list(range(n))
    comp = [dict(nat=node[i]["nat"], num=set(node[i]["num"]),
                 multi=set(node[i]["multi"]), members=[i]) for i in range(n)]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s, i, j in sorted(ok, key=lambda e: (-e[0], e[1], e[2])):
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        if cannot(comp[ri], comp[rj]):
            _log("blocked_merges", dict(a=elems[i][:2], b=elems[j][:2], similarity=round(s / 100, 3)))
            continue
        if len(comp[ri]["members"]) < len(comp[rj]["members"]):
            ri, rj = rj, ri
        parent[rj] = ri
        c, d = comp[ri], comp[rj]
        c["members"] += d["members"]
        c["nat"] = c["nat"] or d["nat"]
        c["num"] |= d["num"]
        c["multi"] |= d["multi"]

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].extend(sigs[keys[i]])
    return list(groups.values())


# ----------------------------------------------------------------- assembly

def build_body(group: dict) -> dict:
    """Corpus-level cluster body from a group of document clusters."""
    dcs = [u["dc"] for u in group["members"]]
    forms = Counter(norm_ws(m["text"]) for dc in dcs for m in dc["mentions"])
    documents, doc_reps = {}, {}
    for dc in dcs:
        documents.setdefault(dc["doc_id"], []).extend(dc["mentions"])
        doc_reps.setdefault(dc["doc_id"], dc["representative"])
    body = dict(
        representative=choose_representative(forms),
        n_mentions=sum(forms.values()),
        n_documents=len(documents),
        surface_forms=dict(forms.most_common()),
        doc_representatives=doc_reps,
    )
    if group["type"] == "Person":
        nat, ttl, rol, occ = Counter(), Counter(), Counter(), Counter()
        for dc in dcs:
            if dc["nationality"]:
                nat[dc["nationality"]] += 1
            ttl.update(dc["title_counts"].keys())
            rol.update(dc["role_counts"].keys())
            occ.update(dc["occupation_counts"].keys())
        body.update(
            nationality=top_unique(nat), nationality_counts=dict(nat.most_common()),
            title_counts=dict(ttl.most_common()), role_counts=dict(rol.most_common()),
            occupation_counts=dict(occ.most_common()),
        )
    body["documents"] = documents
    return body


def assemble(bodies: list[tuple[str, dict]]) -> dict:
    """{key: {type: body}}; key = representative, suffixed with nationality if needed."""
    bodies = sorted(bodies, key=lambda tb: (-tb[1]["n_mentions"], tb[1]["representative"], tb[0]))
    by_rep = defaultdict(list)
    for etype, b in bodies:
        by_rep[(b["representative"], etype)].append(b)
    out = {}
    for (rep, etype), lst in by_rep.items():
        for i, b in enumerate(lst):
            key = rep if len(lst) == 1 else f"{rep} [{b.get('nationality') or 'unknown'}]"
            while etype in out.get(key, {}):
                key = f"{key} #{i + 2}"
            out.setdefault(key, {})[etype] = b
    return out


# ----------------------------------------------------------------- events + singletons (stage 2)

def build_singleton_body(dc: dict) -> dict:
    """Corpus-level body for a singleton or event doc-cluster.

    These clusters never participate in cross-bulletin merging, so the body just
    describes the single document cluster (one or more mentions of the same
    ``event_id``, or exactly one mention for other singleton types).
    """
    mentions = dc["mentions"]
    forms = Counter(norm_ws(m["text"]) for m in mentions)
    body = dict(
        representative=choose_longest(forms),
        n_mentions=len(mentions),
        n_documents=1,
        surface_forms=dict(forms.most_common()),
        doc_representatives={dc["doc_id"]: dc["representative"]},
        documents={dc["doc_id"]: mentions},
    )
    if dc["type"] == "Event":
        body["event_id"] = dc.get("event_id")
    return body


def singleton_global_key(dc: dict, body: dict) -> str:
    """Unique global key for a singleton/event doc-cluster.

    Events: ``"{rep} [event:{doc_id}:{event_id or 'nil'}]"``  — the doc_id and
    event_id make the key unique per bulletin, so two events with the same
    ``event_id`` in different bulletins stay separate.

    Other singletons: ``"{rep} [{doc_id}#{first_start}]"``.
    """
    rep = body["representative"]
    if dc["type"] == "Event":
        eid = dc.get("event_id") or "nil"
        return f"{rep} [event:{dc['doc_id']}:{eid}]"
    first_start = min((m.get("start") or 0) for m in dc["mentions"])
    return f"{rep} [{dc['doc_id']}#{first_start}]"


# ----------------------------------------------------------------- orchestration

def resolve_entities(mentions_by_doc: dict, cfg: ClusteringConfig):
    """Two-stage resolution with per-type strategies.

    Types mapped to ``STRATEGY_STRING_SIMILARITY`` (default Person, Place,
    LegalDocument, Vehicle) go through both clustering stages.  Event mentions
    (``STRATEGY_EXACT_EVENT_ID``) are grouped by exact ``event_id`` within their
    bulletin only and are never merged across bulletins.  Types with no strategy
    become singleton clusters (one mention per cluster, document-local).  All
    clusters — clustered, event, or singleton — receive a unique ``global_cluster``
    key.

    Returns ``(doc_clusters, global_resolved)``:
      - ``doc_clusters`` — ``{doc_id: [document cluster dicts]}`` (stage 1)
      - ``global_resolved`` — ``{key: {type: body}}`` (stage 2 for string-similarity
        types; one entry per event/singleton doc-cluster otherwise)
    """
    reset_diagnostics()

    # Stage 1: within-bulletin clustering for ALL types
    doc_clusters = {}
    for doc_id, entities in mentions_by_doc.items():
        doc_clusters[doc_id] = [finalize_doc_cluster(c, doc_id)
                                for c in cluster_document(entities, cfg, doc_id)]

    # Partition doc-clusters: string-similarity types go to stage 2; events and
    # singletons are kept document-local (each becomes its own global cluster).
    # Empty-name doc clusters (``empty_name_singletons``: OCR garbage whose text
    # normalizes to "") must NOT enter ``cluster_corpus`` — they have no elements
    # to compare, so they would crash the similarity pass and could never be
    # linked anyway.  They stay document-local singletons.
    by_type: dict = defaultdict(list)
    passthrough_dcs: list = []
    for dcs in doc_clusters.values():
        for dc in dcs:
            if cfg.strategy_for(dc["type"]) == STRATEGY_EXACT_EVENT_ID:
                passthrough_dcs.append(dc)
            elif dc["type"] in cfg.string_similarity_types and dc["elements"]:
                by_type[dc["type"]].append(dc)
            else:
                passthrough_dcs.append(dc)

    # Stage 2: corpus-wide for string-similarity types only
    groups = []
    for etype, dcs in by_type.items():
        person = etype == "Person"
        for members in cluster_corpus(
                dcs, cfg.stage2_threshold(etype),
                cfg.nationality_constraint and person, person):
            groups.append(dict(type=etype, members=[dict(dc=dc) for dc in members]))

    bodies = [(g["type"], build_body(g), g) for g in groups]
    global_resolved = assemble([(t, b) for t, b, _ in bodies])
    key_of = {id(b): f"{key}||{t}" for key, by_type_ in global_resolved.items()
              for t, b in by_type_.items()}
    for _, b, g in bodies:
        for u in g["members"]:
            u["dc"]["global_cluster"] = key_of[id(b)]

    # Events and singletons: each doc-cluster is its own global cluster.
    for dc in passthrough_dcs:
        body = build_singleton_body(dc)
        key = singleton_global_key(dc, body)
        etype = dc["type"]
        final_key = key
        i = 2
        while etype in global_resolved.get(final_key, {}):
            final_key = f"{key} #{i}"
            i += 1
        global_resolved.setdefault(final_key, {})[etype] = body
        dc["global_cluster"] = f"{final_key}||{etype}"

    return doc_clusters, global_resolved


# ----------------------------------------------------------------- per-document output

MENTION_FIRST = ("text", "normalized_text", "raw_text", "start", "end", "type",
                 "sentence", "context", "paragraph_id")


def doc_record(doc_id: str, dcs: list[dict]) -> dict:
    """JSON-friendly record of one document: its clusters with mentions.

    ``clusters`` holds only clusters with at least one non-empty normalized
    name — this is the wikidata-facing field.  Name-less clusters (the stage-1
    ``empty_name_singletons``, e.g. OCR garbage whose text normalizes to "") are
    tracked in ``empty_name_clusters`` instead, with the same per-cluster
    structure: kept for auditing, but never used for wikidata linking.
    ``n_mentions`` counts both lists; ``n_clusters`` counts ``clusters`` only.
    """
    clusters: list = []
    empty_name: list = []
    for i, dc in enumerate(sorted(dcs, key=lambda d: min(m.get("start") or 0 for m in d["mentions"])), 1):
        mentions = []
        for m in sorted(dc["mentions"], key=lambda m: m.get("start") or 0):
            out = {"normalized_text": normalize_name(m["text"])}
            for k in MENTION_FIRST:
                if k == "normalized_text":
                    continue
                if k in m:
                    out[k] = m[k]
            out.update({k: v for k, v in m.items() if k not in out})
            mentions.append(out)
        c = dict(
            cluster_id=f"{doc_id}#{i}", type=dc["type"],
            representative=dc["representative"],
            normalized_representative=normalize_name(dc["representative"]),
            normalized_forms=dc["elements"],
            surface_forms=dict(Counter(norm_ws(m["text"]) for m in dc["mentions"]).most_common()),
            n_mentions=len(mentions), global_cluster=dc.get("global_cluster"),
        )
        if dc["type"] == "Person":
            c.update(
                nationality=dc["nationality"],
                nationality_counts=dict(dc["nationality_counts"].most_common()),
                title=[k for k, _ in dc["title_counts"].most_common()],
                role=[k for k, _ in dc["role_counts"].most_common()],
                occupation=[k for k, _ in dc["occupation_counts"].most_common()],
                title_counts=dict(dc["title_counts"].most_common()),
                role_counts=dict(dc["role_counts"].most_common()),
                occupation_counts=dict(dc["occupation_counts"].most_common()),
            )
        if dc["type"] == "Event":
            c["event_id"] = dc.get("event_id")
        c["mentions"] = mentions
        if dc.get("elements"):
            clusters.append(c)      # has a normalized name -> wikidata-facing field
        else:
            empty_name.append(c)    # name-less -> tracked, excluded from wikidata linking
    return dict(
        doc_id=doc_id,
        n_mentions=sum(c["n_mentions"] for c in clusters + empty_name),
        n_clusters=len(clusters),
        n_empty_name_clusters=len(empty_name),
        clusters=clusters,
        empty_name_clusters=empty_name,
    )


# ----------------------------------------------------------------- diagnostics report

def build_report(global_resolved: dict, cfg: ClusteringConfig, top: int = 50) -> dict:
    """Cluster counts, cue coverage and the decisions of each step (samples)."""
    per_type, cov, loose = Counter(), Counter(), []
    clustered_set = cfg.string_similarity_types
    for key, by_type in global_resolved.items():
        for etype, b in by_type.items():
            per_type[etype] += 1
            # The loose-similarity check only makes sense for types that go through
            # fuzzy clustering; event/singleton clusters can legitimately contain
            # very different surface forms (same event_id, different phrasings).
            if etype not in clustered_set:
                continue
            names = [normalize_name(f) for f in list(b["surface_forms"])[:150]]
            names = sorted({n for n in names if n})
            if len(names) >= 2:
                sims = process.cdist(names, names, scorer=fuzz.ratio, dtype=np.float32) / 100.0
                m = float(sims.min())
                if m < cfg.stage2_threshold(etype) - 0.15:
                    loose.append(dict(key=key, type=etype, min_similarity=round(m, 2), names=names[:12]))
            if etype == "Person":
                for ms in b["documents"].values():
                    for m_ in ms:
                        cov["person_mentions"] += 1
                        cov["local_nationality"] += bool((m_.get("cues_local") or {}).get("nationality"))
                        cov["propagated_nationality"] += bool(m_.get("nationality"))
                        cov["any_title"] += bool(m_.get("title"))
                        cov["any_role"] += bool(m_.get("role"))
                        cov["any_occupation"] += bool(m_.get("occupation"))
    loose.sort(key=lambda d: d["min_similarity"])
    return dict(
        config=vars(cfg),
        clusters_per_type=dict(per_type),
        person_cue_coverage=dict(cov),
        nationality_splits=DIAG["nationality_splits"][:top],
        doc_nationality_conflicts=DIAG["doc_nationality_conflicts"][:top],
        empty_name_singletons=DIAG["empty_name_singletons"][:top],
        containment_merges=len(DIAG["containment_merges"]),
        containment_merge_examples=DIAG["containment_merges"][:top],
        ambiguous_containment_examples=DIAG["ambiguous_containment"][:top],
        ambiguous_links_dropped=len(DIAG["ambiguous_links_dropped"]),
        ambiguous_link_examples=DIAG["ambiguous_links_dropped"][:top],
        blocked_merges=len(DIAG["blocked_merges"]),
        blocked_merge_examples=DIAG["blocked_merges"][:top],
        token_guard_blocks=len(DIAG["token_guard_blocks"]),
        token_guard_examples=DIAG["token_guard_blocks"][:top],
        loosely_similar_clusters=loose[:top],
    )