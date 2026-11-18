"""The judge run loop: document by document, entity by entity.

Ported from the uploaded decide_common.py.  Inputs are the three pipeline
files: resolved documents (one line per document, from the IE/resolution
step), the links file of candidate extraction (candidates per corpus cluster,
BOTH record shapes: the final file with flat decision fields and the pass-1
intermediate with the decision nested under 'result'), and the profiles file
of candidate_profiles (keyed (qid, normalize_type(cluster type))).  Output is
decisions.jsonl, one line per judged entity, appended — a rerun skips the
(doc_id, cluster_id) pairs already in the file, and a network/model error
leaves that entity for the next run.

The model-specific part (payload build, the call) is injected: main(build,
send, name, errors) — jev.py provides it; to judge with another backend, add
a sibling module and call main() the same way.

Per entity:
  1. take the candidates of its corpus cluster; only clusters whose status is
     in config.statuses, whose (normalized) type is in config.entity_types
     (default: PERSON, ORGANIZATION, PLACE, VEHICLE — the types with
     candidates; aliases count, SHIP -> VEHICLE) and that have at least one
     candidate;
  2. build the text snippets (texts.py);
  3. build(payload) -> payload, send(payload, model) -> {option: probability};
  4. link / nil / uncertain via judge(), append the line.

Every cluster NOT judged is counted with its reason in the printed summary
(already_done, no_link, status_skip, no_candidates, other_type, no_gold,
no_usable_mention, error), so an empty run always says WHY.  A missing or
empty documents file, or an empty links file, raises SystemExit.  The
documents/links join tries the exact cluster id first, then a normalized
id (case/whitespace/type-casing: 'Spanje||Place' == 'spanje||PLACE',
counted as norm_join) — see candidate_judge/diagnose_join.py to inspect
exactly how the two files' ids differ.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

# make utils/io.py and candidate_profiles importable whether the package lives
# at repo root or nested (same bootstrap as candidate_profiles/collect.py)
_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.io import append_jsonl, read_jsonl  # noqa: E402
from candidate_profiles.types import normalize_type  # noqa: E402

from . import config  # noqa: E402
from .jev import NONE_KEY  # noqa: E402
from .texts import snippets  # noqa: E402

QID = re.compile(r"Q\d+", re.I)


def _field(link: dict, name: str):
    """Field of a links record, either shape: flat in the final file, nested
    under 'result' in the pass-1 intermediate."""
    return link.get(name, (link.get("result") or {}).get(name))


# ------------------------------------------------------------------ small experiments: which documents / entities
def gold_spans(path: str, missing: str = "nil") -> dict:
    """{(doc_id, start, end): set of gold qids (empty set = NIL)} from the gold file
    (resolution format with a `qid` per cluster, JSON or JSONL).  Clusters without
    a `qid` key count as NIL (missing="nil") or are left out ("skip")."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        objs = [json.loads(text)]
    except json.JSONDecodeError:
        objs = [json.loads(l) for l in text.splitlines() if l.strip()]
    spans = {}
    for obj in objs:
        for by_type in obj.values():
            for body in by_type.values():
                if "qid" not in body and missing == "skip":
                    continue
                value = body.get("qid")
                items = value if isinstance(value, list) else re.split(r"[|,;]", str(value or ""))
                qids = {m.group(0).upper() for s in items if (m := QID.search(str(s)))}
                for doc_id, mentions in body["documents"].items():
                    for m in mentions:
                        spans[(doc_id, m["start"], m["end"])] = qids
    return spans


def select_docs(docs, doc_ids=None, gold=None, limit=None):
    """Documents to process: only `doc_ids`, only documents with gold annotations,
    and at most `limit` of them."""
    gold_docs = {d for d, _, _ in gold} if gold else None
    n = 0
    for doc in docs:
        if (doc_ids and doc["doc_id"] not in doc_ids) or (gold_docs is not None and doc["doc_id"] not in gold_docs):
            continue
        if limit and n >= limit:
            return
        n += 1
        yield doc


def doc_date(doc_id: str) -> str:
    """YYYY-MM-DD from a colon-separated doc_id, wherever the 4-digit year sits:
    'radio:bulletin:1939:09:01' (year at position 2) and 'anp:1939:01:11:4:mpeg21'
    (year at position 1) both parse; no year part -> ''."""
    parts = doc_id.split(":")
    for i, p in enumerate(parts):
        if re.fullmatch(r"(?:19|20)\d{2}", p or ""):
            if i + 2 < len(parts) and re.fullmatch(r"\d{1,2}", parts[i + 1] or "") \
                    and re.fullmatch(r"\d{1,2}", parts[i + 2] or ""):
                return f"{p}-{int(parts[i + 1]):02d}-{int(parts[i + 2]):02d}"
            return ""
    return ""


def _join_key(raw) -> tuple[str, str]:
    """Normalized join key of a corpus cluster id: (name part squashed+casefolded,
    type part via normalize_type).  'Spanje||Place' and 'spanje||PLACE' share a
    key, so a links file whose run cased the type differently still joins."""
    name, _, t = str(raw or "").rpartition("||")
    return re.sub(r"\s+", " ", name).strip().casefold(), normalize_type(t)


# ------------------------------------------------------------------ policy
def judge(probs: dict, min_p: float, min_ratio: float) -> tuple[str, str | None]:
    """(outcome, qid) from {qid or NONE_KEY: probability}."""
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    top, p1 = ranked[0]
    p2 = ranked[1][1] if len(ranked) > 1 else 0.0
    if p1 < min_p:
        return "uncertain", None
    if top == NONE_KEY:
        return "nil", None
    if p2 and p1 / p2 < min_ratio:
        return "uncertain", None
    return "link", top


# ------------------------------------------------------------------ the run
def main(build, send, name: str, errors=(RuntimeError, OSError, ValueError)) -> None:
    """Document by document, entity by entity.  `build(entity, cfg)` -> payload;
    `send(payload, model)` -> {qid or NONE_KEY: probability}."""
    c = config.CONFIG
    statuses = set(c.statuses)
    entity_types = {normalize_type(t) for t in (c.entity_types or [])}   # "person" -> PERSON, SHIP -> VEHICLE
    links = {r["cluster_id"]: r for r in read_jsonl(c.links_path)}
    if not links:
        raise SystemExit(f"{c.links_path} is empty or missing: run candidate extraction first")
    buckets = {}
    for r in links.values():
        buckets.setdefault(_join_key(r["cluster_id"]), []).append(r)
    norm = {k: v[0] for k, v in buckets.items() if len(v) == 1}   # ambiguous keys are not used
    docs = read_jsonl(c.documents_path)
    if not docs:
        raise SystemExit(f"{c.documents_path} is empty or missing: this must be the RESOLVED "
                         "documents file (clusters with global_cluster) of the IE/resolution step")
    profiles = {(r["qid"], r["type"]): r["profile"] for r in read_jsonl(c.profiles_path)}
    if c.restart and Path(c.out_path).exists():
        Path(c.out_path).unlink()
    done = {(r["doc_id"], r["cluster_id"]) for r in read_jsonl(c.out_path)}
    gold = gold_spans(c.gold_path) if c.gold_path else None
    print(f"documents {len(docs)}, links records {len(links)}, already judged {len(done)}"
          + (f", gold spans {len(gold)}" if gold else ""))
    Path(c.out_path).parent.mkdir(parents=True, exist_ok=True)
    out = None if c.dry_run else open(c.out_path, "a", encoding="utf-8")
    stats, shown, stop = Counter(), 0, False

    for doc in select_docs(docs, c.doc_ids, gold, c.limit_docs):                          # document by document
        doc_id = doc["doc_id"]
        date = doc_date(doc_id)
        # entities of this document that the pipeline already accepted: context for the others
        known = {}
        for cl in doc["clusters"]:
            g = cl.get("global_cluster")
            link = (links.get(g)                                   # exact id first,
                    or (norm.get(_join_key(g)) if g else None)     # then normalized (no counter here)
                    or {})
            if _field(link, "status") == "accepted" and _field(link, "label"):
                known[cl["cluster_id"]] = f"{_field(link, 'label')} ({cl['type']})"

        for cluster in doc["clusters"]:                                                   # entity by entity
            if (doc_id, cluster["cluster_id"]) in done:
                stats["already_done"] += 1
                continue
            gid = cluster.get("global_cluster")
            link = links.get(gid)
            if not link and gid:
                link = norm.get(_join_key(gid))     # same id modulo case/whitespace/type casing
                if link:
                    stats["norm_join"] += 1
            if not link:
                stats["no_link"] += 1                                     # documents vs links mismatch
                continue
            if _field(link, "status") not in statuses:
                stats["status_skip"] += 1                                 # e.g. already accepted
                continue
            if not link.get("candidates"):
                stats["no_candidates"] += 1
                continue
            if entity_types and normalize_type(link.get("type") or cluster.get("type")) not in entity_types:
                stats["other_type"] += 1                  # only clusters that would be judged reach this
                continue
            spans = [(doc_id, m["start"], m["end"]) for m in cluster["mentions"] if m.get("start") is not None]
            if gold is not None and not any(s in gold for s in spans):
                stats["no_gold"] += 1                                     # no gold annotation for this entity
                continue
            texts = snippets(cluster, c)
            if not texts:
                stats["no_usable_mention"] += 1
                continue
            candidates = []
            for cand in link["candidates"][:c.topk]:
                profile = profiles.get((cand["qid"], normalize_type(link["type"])))
                if profile and c.no_summary:
                    profile = {k: v for k, v in profile.items() if k != "wikipedia_summary"}
                candidates.append(dict(cand, profile=profile))
            entity = dict(doc_id=doc_id, date=date, cluster=cluster, link=link, texts=texts, candidates=candidates,
                          known=[v for k, v in known.items() if k != cluster["cluster_id"]][:c.max_known])
            payload = build(entity, c)

            if c.dry_run:
                print(json.dumps(dict(doc_id=doc_id, cluster_id=cluster["cluster_id"], payload=payload),
                                 ensure_ascii=False, indent=2, default=str))
                shown += 1
                if shown >= c.dry_run:
                    return
                continue

            try:
                probs = send(payload, c.model)
            except errors as ex:                                  # not written: the next run tries again
                print(f"[skip] {cluster['cluster_id']}: {ex}")
                stats["error"] += 1
                continue
            unknown = set(probs) - {cand["qid"] for cand in candidates} - {NONE_KEY}
            if unknown or not probs:                              # the model must only answer with the options given
                print(f"[skip] {cluster['cluster_id']}: unexpected options {sorted(unknown)[:3]}")
                stats["error"] += 1
                continue
            outcome, qid = judge(probs, c.min_p, c.min_ratio)
            if out:
                append_jsonl(out, dict(
                    doc_id=doc_id, cluster_id=cluster["cluster_id"], global_cluster=cluster["global_cluster"],
                    type=cluster["type"], representative=cluster["representative"],
                    pipeline_status=_field(link, "status"), pipeline_qid=_field(link, "qid"),
                    top_candidate=candidates[0]["qid"], options=[cand["qid"] for cand in candidates],
                    outcome=outcome, qid=qid,
                    probabilities=dict(sorted(probs.items(), key=lambda kv: -kv[1])),
                    backend=name, model=c.model, mention_spans=[[s[1], s[2]] for s in spans]))
            stats[outcome] += 1
            if qid:                                               # later entities of the document can use it
                label = next((cand.get("label") for cand in candidates if cand["qid"] == qid), qid)
                known[cluster["cluster_id"]] = f"{label} ({cluster['type']})"
            if c.max_entities and sum(stats[k] for k in ("link", "nil", "uncertain")) >= c.max_entities:
                stop = True
                break
        if stop:
            break
    if out:
        out.close()
    print(f"judged: {dict(stats)} -> {c.out_path}")
    if stats["no_link"] and not (stats["link"] or stats["nil"] or stats["uncertain"]):
        print("WARNING: no document cluster matched a links record — is --documents-path the RESOLVED "
              "documents file, and does every cluster's global_cluster equal a cluster_id of the "
              "links file (wikidata_pass1.jsonl / wikidata_links.jsonl)?")
    if stats["norm_join"]:
        print(f"note: {stats['norm_join']} entities joined only after case/whitespace/type-casing "
              "normalization of the cluster ids — check that documents and links come from the SAME "
              "clustering run (candidate_judge/diagnose_join.py shows the exact differences)")