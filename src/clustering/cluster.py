"""
Two-stage entity resolution: within-document clustering, then corpus-wide.

Type scope
----------
Only the types in ``ClusteringConfig.clustered_types`` (default PERSON, PLACE,
ORGANIZATION) go through both stages.  EVENT mentions are grouped by ``event_id``
*within one document* (same event_id -> one cluster; different event_id or no
event_id -> separate clusters) and are NEVER merged across documents.  Every
other type becomes a singleton cluster (one mention per cluster) and is also
kept document-local.  See :func:`cluster_document` and :func:`resolve_entities`.

Stage 1 — within one document (``cluster_document``)
----------------------------------------------------
Inside one document two different entities with similar names are rare, so matching
is generous:

  A. **Typo level**: fuzzy ratio + average linkage, with cannot-link guards for
     numbers / Roman numerals and (for PERSON) nationality.  Identical names with
     different known nationalities are never merged.
  B. **Containment** (``attach_contained``): a cluster whose name is contained in the
     name of exactly ONE other cluster of the same document is merged into it,
     tolerating typos and initials ("Chamberlain" -> "Neville Chamberlain",
     "N. Chamberlain" -> "Neville Chamberlain").  Ambiguous containment (two or more
     hosts) is left unmerged.

The representative of a document cluster is the LONGEST surface form (ties: most
frequent, alphabetical).  PERSON cues (nationality, title, role, occupation) are
aggregated per cluster and written to every mention; ``cues_local`` keeps the
mention's own evidence.

Stage 2 — over the whole corpus (``cluster_corpus``)
----------------------------------------------------
Each document cluster is a node with ALL its normalised forms as elements.  The
similarity of two nodes is the CLOSEST pair of elements (max fuzzy ratio).
Closest-pair (single) linkage chains easily, so merging is constrained:

  - **cannot-link**: PERSON nodes with different known nationalities (and optionally
    different top occupations when ``occupation_constraint``); nodes with different
    numbers or Roman numerals; PERSON nodes whose full names conflict (same surname,
    different given name).
  - **ambiguity**: a node linked to two nodes that cannot be linked to each other is
    not merged with either; it can still merge with equally underspecified nodes.
  - merging goes from strongest link to weakest; two groups merge only if no
    cannot-link exists between any of their members.

Identical nodes (same forms, same nationality) are collapsed first, so a name in
thousands of documents does not create millions of links.

Output
------
``resolve_entities`` returns ``(doc_clusters, global_resolved)``:

  - ``doc_clusters`` — ``{doc_id: [document cluster dicts]}``; each cluster carries
    ``representative``, ``elements``, ``nationality``, ``occupation``, cue counts,
    ``mentions`` and ``global_cluster`` (the stage-2 cluster id).
  - ``global_resolved`` — ``{key: {type: body}}``; each body has ``representative``,
    ``n_mentions``, ``n_documents``, ``surface_forms``, ``doc_representatives``,
    nationality / title / role / occupation counts (PERSON) and ``documents``.
"""
import re
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
from rapidfuzz import fuzz, process
from sklearn.cluster import AgglomerativeClustering

from .normalize import normalize_name, norm_ws
from config.cls_config import ClusteringConfig

# ----------------------------------------------------------------- diagnostics

DIAG: dict = {}


def reset_diagnostics() -> None:
    DIAG.clear()
    DIAG.update(
        nationality_splits=[], doc_nationality_conflicts=[], surname_merges=[],
        containment_merges=[], ambiguous_containment=[], ambiguous_links_dropped=[],
        blocked_merges=[],
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

ROMAN = {"ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii"}


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
    return (m.get("cues_local") or {}).get("nationality")


def mention_occ(m: dict):
    occs = (m.get("cues_local") or {}).get("occupation") or []
    return occs[0] if occs else None


def unit_nationality(items: list, nat_of, weight_of=None):
    """Weighted most frequent nationality of a group (None if unknown or tied)."""
    cnt = Counter()
    for it in items:
        nat = nat_of(it)
        if nat:
            cnt[nat] += weight_of(it) if weight_of else 1
    return top_unique(cnt)


def unit_occupation(items: list, occ_of, weight_of=None):
    """Weighted most frequent occupation of a group (None if unknown or tied)."""
    cnt = Counter()
    for it in items:
        occ = occ_of(it)
        if occ:
            cnt[occ] += weight_of(it) if weight_of else 1
    return top_unique(cnt)


# ----------------------------------------------------------------- distance matrix + stage-1 clustering

def distance_matrix(names: list[str], nat_ids, num_ids, occ_ids=None) -> np.ndarray:
    """1 - fuzzy similarity, forced to 1.0 for pairs that must not be merged."""
    dist = 1.0 - process.cdist(names, names, scorer=fuzz.ratio, dtype=np.float32, workers=-1) / 100.0
    if nat_ids is not None:
        dist[(nat_ids[:, None] != nat_ids[None, :]) & (nat_ids[:, None] > 0) & (nat_ids[None, :] > 0)] = 1.0
    if occ_ids is not None:
        dist[(occ_ids[:, None] != occ_ids[None, :]) & (occ_ids[:, None] > 0) & (occ_ids[None, :] > 0)] = 1.0
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
                  occs: list | None = None, use_occ: bool = False) -> list[int]:
    """Cluster normalised unique names; returns one label per name."""
    n = len(names)
    if n == 1:
        return [0]
    nat_vals = {v: i + 1 for i, v in enumerate(sorted({x for x in nats if x}))}
    nat_ids = np.array([nat_vals.get(x, 0) for x in nats]) if use_nat else None
    occ_vals = {v: i + 1 for i, v in enumerate(sorted({x for x in (occs or []) if x}))}
    occ_ids = np.array([occ_vals.get(x, 0) for x in (occs or [])]) if (use_occ and occs) else None
    keys, num_ids = {}, []
    for s in names:
        k = number_key(s)
        num_ids.append(keys.setdefault(k, len(keys) + 1) if k else 0)
    dist = distance_matrix(names, nat_ids, np.array(num_ids), occ_ids)
    labels = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                     distance_threshold=1.0 - thr).fit_predict(dist)
    if use_nat:
        labels = repair_nationality(labels, nats, dist, thr, names)
    return [int(x) for x in labels]


def entity_disambiguation(items: list, thr: float, nat_of=None, use_nat: bool = True,
                          occ_of=None, use_occ: bool = False, field: str = "text") -> list[dict]:
    """Cluster items (mentions or document clusters) independently PER TYPE.

    The clustering unit is (normalised name, nationality) for PERSON: identical names
    with different known nationalities are separate units that are never merged.
    Returns ``[{"type", "norm_names", "members"}]``.
    """
    by_type = defaultdict(list)
    for it in items:
        if it.get(field) and it.get("type"):
            by_type[it["type"]].append(it)

    clusters = []
    for etype, group in by_type.items():
        is_person = etype == "PERSON"
        use = use_nat and nat_of is not None and is_person
        use_o = use_occ and occ_of is not None and is_person
        by_unit = defaultdict(list)
        for it in group:
            nm = normalize_name(it[field])
            if nm:
                by_unit[(nm, nat_of(it) if use else None)].append(it)
        units = sorted(by_unit, key=lambda u: (u[0], u[1] or ""))
        if not units:
            continue
        names = [u[0] for u in units]
        nats = [u[1] for u in units]
        occs = [occ_of(it) if use_o else None for it in group]  # not per-unit; see below
        # Build per-unit occupation (top occupation among the unit's members)
        if use_o:
            unit_occs = []
            for u in units:
                unit_occs.append(unit_occupation(by_unit[u], occ_of))
        else:
            unit_occs = [None] * len(units)
        labels = cluster_units(names, nats, thr, use, unit_occs, use_o)
        buckets = defaultdict(list)
        for u, lab in zip(units, labels):
            buckets[lab].append(u)
        for us in buckets.values():
            clusters.append(dict(type=etype, norm_names=sorted({u[0] for u in us}),
                                 members=[it for u in us for it in by_unit[u]]))
    return clusters


# ----------------------------------------------------------------- containment (stage 1)

PARTICLES = {"van", "de", "der", "den", "von", "di", "du", "le", "la", "el", "al",
             "ten", "ter", "te", "het", "t", "d"}


def tokens_match(a: str, b: str, thr: float) -> bool:
    """Equal tokens, an initial vs a name starting with it, or fuzzy equal tokens."""
    if a == b:
        return True
    if len(a) == 1 or len(b) == 1:
        return a.startswith(b) or b.startswith(a)
    return fuzz.ratio(a, b) >= thr * 100


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
    """Can two multi-token PERSON names be the same person? Same surname and compatible first names."""
    ta, tb = a.split(), b.split()
    if not tokens_match(ta[-1], tb[-1], thr):
        return False
    ga = [t for t in ta[:-1] if t not in PARTICLES]
    gb = [t for t in tb[:-1] if t not in PARTICLES]
    return not ga or not gb or tokens_match(ga[0], gb[0], 0.75)


def attach_contained(clusters: list[dict], cfg: ClusteringConfig, doc_id: str = "") -> list[dict]:
    """Merge a cluster into the ONE cluster of the same document that contains its name."""
    types = set(cfg.containment_types)
    while True:
        cand = sorted((c for c in clusters if c["type"] in types),
                      key=lambda c: (max(len(n.split()) for n in c["norm_names"]), c["norm_names"]))
        merged = False
        for s in cand:
            anchor = s["type"] == "PERSON"
            hosts = []
            for c in clusters:
                if c is s or c["type"] != s["type"]:
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
    """Group EVENT mentions by ``event_id`` *within one document*.

    - Events with the same non-empty ``event_id`` -> one cluster.
    - Events with different ``event_id`` -> separate clusters.
    - Events without ``event_id`` -> each its own singleton cluster.

    No cross-document merging is ever performed for events (stage 2 skips them).
    """
    by_eid: dict = defaultdict(list)
    no_eid: list = []
    for e in events:
        eid = e.get("event_id")
        if eid:
            by_eid[eid].append(e)
        else:
            no_eid.append(e)

    out = []
    for eid, members in by_eid.items():
        norm_names = sorted({normalize_name(m.get("text", "")) for m in members
                             if m.get("text")})
        out.append(dict(type="EVENT", members=members, norm_names=norm_names,
                        event_id=eid))
    for e in no_eid:
        nm = normalize_name(e.get("text", ""))
        out.append(dict(type="EVENT", members=[e],
                        norm_names=[nm] if nm else [], event_id=None))
    return out


def make_singleton_cluster(entity: dict) -> dict:
    """Wrap a single mention as its own cluster (no merging, stage 1 + stage 2)."""
    nm = normalize_name(entity.get("text", ""))
    return dict(type=(entity.get("type") or "").upper() or "UNKNOWN",
                members=[entity], norm_names=[nm] if nm else [])


# ----------------------------------------------------------------- stage 1 entry point

def cluster_document(entities: list[dict], cfg: ClusteringConfig, doc_id: str = "") -> list[dict]:
    """Stage 1: within-document clustering for ALL entity types.

    - ``cfg.clustered_types`` (PERSON/PLACE/ORGANIZATION): typo-level fuzzy
      clustering (:func:`entity_disambiguation`) + containment attachment
      (:func:`attach_contained`).
    - EVENT: grouped by ``event_id`` within this document via
      :func:`cluster_events_by_id` (same id -> one cluster; different id or missing
      id -> separate clusters).  Never merged across documents.
    - Other types: each mention becomes its own singleton cluster via
      :func:`make_singleton_cluster`.
    """
    up = lambda t: (t or "").upper()
    clustered_set = set(cfg.clustered_types)

    clustered = [e for e in entities if up(e.get("type")) in clustered_set]
    clusters = entity_disambiguation(
        clustered, cfg.threshold,
        nat_of=mention_nat, use_nat=cfg.nationality_constraint,
        occ_of=mention_occ, use_occ=cfg.occupation_constraint)
    clusters = attach_contained(clusters, cfg, doc_id)

    events = [e for e in entities if up(e.get("type")) == "EVENT"]
    clusters.extend(cluster_events_by_id(events))

    passthrough = clustered_set | {"EVENT"}
    others = [e for e in entities if up(e.get("type")) not in passthrough]
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
    if cl["type"] == "PERSON":
        nat, ttl, rol, occ = Counter(), Counter(), Counter(), Counter()
        for m in mentions:
            c = m.get("cues_local") or {}
            if c.get("nationality"):
                nat[c["nationality"]] += 1
            ttl.update(set(c.get("title", [])))
            rol.update(set(c.get("role", [])))
            occ.update(set(c.get("occupation", [])))
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
    if cl["type"] == "EVENT":
        out["event_id"] = cl.get("event_id")
    return out


# ----------------------------------------------------------------- stage 2: corpus-wide

def cluster_corpus(dcs: list[dict], thr: float, use_nat: bool, person: bool,
                   use_occ: bool = False) -> list[list[dict]]:
    """Stage 2 for ONE type: groups of document clusters (see module docstring)."""
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
        occ=top_unique(Counter(
            dc.get("occupation") or [None] for dc in sigs[keys[i]] if dc.get("occupation")
        )) if use_occ else None,
        num=frozenset().union(*(number_key(e) for e in elems[i])),
        multi=frozenset(e for e in elems[i] if len(e.split()) >= 2),
    ) for i in range(n)]

    # candidate links: closest pair of elements (no n x n matrix stored)
    edges = []
    for i in range(n):
        sims = process.cdist(elems[i], uniq, scorer=fuzz.ratio, dtype=np.float32, workers=-1)
        vec = np.maximum.reduceat(sims.max(axis=0)[order], starts)
        edges += [(float(vec[j]), i, int(j)) for j in np.where(vec >= thr * 100)[0] if j > i]

    @lru_cache(maxsize=None)
    def compat(a: str, b: str) -> bool:
        return name_compat(a, b, thr)

    def cannot(a: dict, b: dict) -> bool:
        if use_nat and a["nat"] and b["nat"] and a["nat"] != b["nat"]:
            return True
        if use_occ and a["occ"] and b["occ"] and a["occ"] != b["occ"]:
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
    comp = [dict(nat=node[i]["nat"], occ=node[i]["occ"], num=set(node[i]["num"]),
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
        c["occ"] = c["occ"] or d["occ"]
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
    if group["type"] == "PERSON":
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

    These clusters never participate in cross-document merging, so the body just
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
    if dc["type"] == "EVENT":
        body["event_id"] = dc.get("event_id")
    return body


def singleton_global_key(dc: dict, body: dict) -> str:
    """Unique global key for a singleton/event doc-cluster.

    Events: ``"{rep} [event:{doc_id}:{event_id or 'nil'}]"``  — the doc_id and
    event_id make the key unique per document, so two events with the same
    ``event_id`` in different documents stay separate.

    Other singletons: ``"{rep} [{doc_id}#{first_start}]"``.
    """
    rep = body["representative"]
    if dc["type"] == "EVENT":
        eid = dc.get("event_id") or "nil"
        return f"{rep} [event:{dc['doc_id']}:{eid}]"
    first_start = min((m.get("start") or 0) for m in dc["mentions"])
    return f"{rep} [{dc['doc_id']}#{first_start}]"


# ----------------------------------------------------------------- orchestration

def resolve_entities(mentions_by_doc: dict, cfg: ClusteringConfig):
    """Two-stage resolution with type-scoped clustering.

    ``cfg.clustered_types`` (PERSON/PLACE/ORGANIZATION) go through both stages.
    EVENT mentions are grouped by ``event_id`` within their document only and are
    never merged across documents.  Every other type becomes a singleton cluster
    (one mention per cluster, also document-local).  All clusters — clustered,
    event, or singleton — receive a unique ``global_cluster`` key.

    Returns ``(doc_clusters, global_resolved)``:
      - ``doc_clusters`` — ``{doc_id: [document cluster dicts]}`` (stage 1)
      - ``global_resolved`` — ``{key: {type: body}}`` (stage 2 for clustered types;
        one entry per event/singleton doc-cluster otherwise)
    """
    reset_diagnostics()

    # Stage 1: within-doc clustering for ALL types
    doc_clusters = {}
    for doc_id, entities in mentions_by_doc.items():
        doc_clusters[doc_id] = [finalize_doc_cluster(c, doc_id)
                                for c in cluster_document(entities, cfg, doc_id)]

    # Partition doc-clusters: clustered types go to stage 2; events + singletons
    # are kept document-local (each becomes its own global cluster).
    by_type: dict = defaultdict(list)
    passthrough_dcs: list = []
    for dcs in doc_clusters.values():
        for dc in dcs:
            if dc["type"] in cfg.clustered_types:
                by_type[dc["type"]].append(dc)
            else:
                passthrough_dcs.append(dc)

    # Stage 2: corpus-wide for clustered_types only
    thr = cfg.corpus_threshold if cfg.corpus_threshold is not None else cfg.threshold
    groups = []
    for etype, dcs in by_type.items():
        person = etype == "PERSON"
        for members in cluster_corpus(
                dcs, thr,
                cfg.nationality_constraint and person, person,
                cfg.occupation_constraint and person):
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
                 "sentence", "context")


def doc_record(doc_id: str, dcs: list[dict]) -> dict:
    """JSON-friendly record of one document: its clusters with mentions."""
    clusters = []
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
        if dc["type"] == "PERSON":
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
        if dc["type"] == "EVENT":
            c["event_id"] = dc.get("event_id")
        c["mentions"] = mentions
        clusters.append(c)
    return dict(
        doc_id=doc_id,
        n_mentions=sum(c["n_mentions"] for c in clusters),
        n_clusters=len(clusters),
        clusters=clusters,
    )


# ----------------------------------------------------------------- diagnostics report

def build_report(global_resolved: dict, cfg: ClusteringConfig, top: int = 50) -> dict:
    """Cluster counts, cue coverage and the decisions of each step (samples)."""
    per_type, cov, loose = Counter(), Counter(), []
    clustered_set = set(cfg.clustered_types)
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
                if m < cfg.threshold - 0.15:
                    loose.append(dict(key=key, type=etype, min_similarity=round(m, 2), names=names[:12]))
            if etype == "PERSON":
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
        containment_merges=len(DIAG["containment_merges"]),
        containment_merge_examples=DIAG["containment_merges"][:top],
        ambiguous_containment_examples=DIAG["ambiguous_containment"][:top],
        ambiguous_links_dropped=len(DIAG["ambiguous_links_dropped"]),
        ambiguous_link_examples=DIAG["ambiguous_links_dropped"][:top],
        blocked_merges=len(DIAG["blocked_merges"]),
        blocked_merge_examples=DIAG["blocked_merges"][:top],
        loosely_similar_clusters=loose[:top],
    )