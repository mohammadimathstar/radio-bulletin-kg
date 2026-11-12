"""
decide_common.py - the shared driver for the LLM judge step.

Reads   document_clusters.jsonl   entities per document          (radio_bulletins_kg.clustering)
        wikidata_links.jsonl      candidates per corpus cluster   (link_wikidata)
        candidate_profiles.jsonl  facts about each candidate      (candidate_profiles)
Writes  decisions.jsonl           one line per judged entity (appended at once; a rerun skips finished entities)

For every entity of every document (document by document, entity by entity):
  1. take the candidates of its corpus cluster; only clusters with status review/nil and at least one candidate
  2. build the text: mentions WITHOUT a sentence are ignored; a sentence shorter than MIN_SENTENCE_CHARS is replaced
     by the longer context; the mention is marked [[like this]]; then every newline/double space becomes one space
  3. hand the entity to the model-specific code: build(entity) -> payload, send(payload) -> probabilities
  4. link / nil / uncertain from the probabilities, write the line

The model-specific part (prompt format, how candidates are described, the API call) lives in a
driver module (e.g. decide_jev or decide_laya) that calls ``run(args, build, send, name)``;
to use another model, copy one of them and supply its own ``build`` / ``send``.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

BASE = "./data/output"
MIN_SENTENCE_CHARS = 80        # shorter sentences are replaced by the context
MAX_CHARS = 500                # characters of one text snippet
MAX_SNIPPETS = 5               # snippets per entity
MAX_KNOWN = 8                  # already resolved entities of the document shown as context
NONE_KEY = "NONE_OF_THE_ABOVE"

INSTRUCTIONS = (
    "Which Wikidata item does the marked mention [[...]] refer to? The mention is of type {type}. The text comes "
    "from Dutch radio news bulletins of 1939 (OCR errors, old spelling; names may be misspelled or incomplete). "
    "Use the text, the date, the nationality and titles next to the name and the other entities of the bulletin. "
    "Choose '{none}' if no listed item fits."
)


def read_jsonl(path):
    p = Path(path)
    if p.exists():
        for line in p.open(encoding="utf-8"):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


# ------------------------------------------------------------------ prepare the text
def squash(text):
    """Every whitespace run (newlines included) becomes one space."""
    return re.sub(r"\s+", " ", text or "").strip()


def snippet(m):
    """Text of one mention with [[mention]] marked, or None if the mention has no sentence."""
    sentence = squash(m.get("sentence"))
    if not sentence:
        return None                                              # no sentence: ignore the mention
    text, base = sentence, m.get("sentence_start")
    context = squash(m.get("context"))
    if len(sentence) < MIN_SENTENCE_CHARS and len(context) > len(sentence):
        text, base = context, m.get("context_start")             # short sentence: use the context

    name = squash(m.get("text")).lower()                         # mark the occurrence closest to the offset
    hits = [i for i in range(len(text)) if name and text.lower().startswith(name, i)]
    if hits:
        expected = m["start"] - base if base is not None and m.get("start") is not None else 0
        i = min(hits, key=lambda p: abs(p - expected))
        text = f"{text[:i]}[[{text[i:i + len(name)]}]]{text[i + len(name):]}"
    text = squash(text)                                          # no newline or double space after marking

    if len(text) > MAX_CHARS:                                    # cut around the mention
        a = max(0, max(text.find("[["), 0) - MAX_CHARS // 2)
        text = text[a:a + MAX_CHARS]
    return text


def snippets(cluster):
    out = []
    for m in sorted(cluster["mentions"], key=lambda m: m.get("start") or 0):
        s = snippet(m)
        if s and s not in out:
            out.append(s)
    return out[:MAX_SNIPPETS]


# ------------------------------------------------------------------ small experiments: which documents / entities
QID = re.compile(r"Q\d+", re.I)


def gold_spans(path, missing="nil"):
    """{(doc_id, start, end): set of gold qids (empty set = NIL)} from the gold file (resolution format with a `qid`
    per cluster, JSON or JSONL). Clusters without a `qid` key count as NIL (missing="nil") or are left out ("skip")."""
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
    """Documents to process: only `doc_ids`, only documents with gold annotations, and at most `limit` of them."""
    gold_docs = {d for d, _, _ in gold} if gold else None
    n = 0
    for doc in docs:
        if (doc_ids and doc["doc_id"] not in doc_ids) or (gold_docs is not None and doc["doc_id"] not in gold_docs):
            continue
        if limit and n >= limit:
            return
        n += 1
        yield doc


# ------------------------------------------------------------------ decide
def judge(probs, min_p, min_ratio):
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


def make_parser():
    ap = argparse.ArgumentParser()
    g = ap.add_argument_group("files")
    g.add_argument("--documents", default=f"{BASE}/document_clusters.jsonl")
    g.add_argument("--links", default=f"{BASE}/links/wikidata_links.jsonl")
    g.add_argument("--profiles", default=f"{BASE}/links/candidate_profiles.jsonl")
    g.add_argument("--out", default=f"{BASE}/links/decisions.jsonl")
    g.add_argument("--restart", action="store_true", help="delete the output file first")
    g = ap.add_argument_group("what is judged")
    g.add_argument("--statuses", default="review,nil", help="pipeline statuses of the corpus cluster to judge")
    g.add_argument("--topk", type=int, default=10, help="candidates shown to the model")
    g.add_argument("--model", default=None, help="model name (recorded in the decision line; set by the driver)")
    g.add_argument("--no-summary", action="store_true", help="leave Wikipedia summaries out of the profiles")
    g.add_argument("--min-p", type=float, default=0.50, help="probability the chosen option needs")
    g.add_argument("--min-ratio", type=float, default=2.0, help="chosen option / runner-up needed to link")
    g = ap.add_argument_group("small experiments")
    g.add_argument("--limit-docs", type=int, default=110, help="only the first N documents")
    g.add_argument("--doc-ids", nargs="*", help="only these documents")
    g.add_argument("--gold", help="only documents and entities that have gold annotations")
    g.add_argument("--max-entities", type=int, help="stop after N judged entities")
    g.add_argument("--dry-run", type=int, default=0, metavar="N", help="print the first N requests, call no model")
    return ap


def run(args, build, send, name, errors=(RuntimeError, ValueError, OSError)):
    """Document by document, entity by entity. `build(entity)` -> payload; `send(payload)` -> {qid or NONE_KEY: p}."""
    statuses = set(args.statuses.split(","))
    links = {r["cluster_id"]: r for r in read_jsonl(args.links)}
    profiles = {(r["qid"], r["type"]): r["profile"] for r in read_jsonl(args.profiles)}
    if args.restart and Path(args.out).exists():
        Path(args.out).unlink()
    done = {(r["doc_id"], r["cluster_id"]) for r in read_jsonl(args.out)}
    gold = gold_spans(args.gold) if args.gold else None
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = None if args.dry_run else open(args.out, "a", encoding="utf-8")
    stats, shown, stop = Counter(), 0, False

    for doc in select_docs(read_jsonl(args.documents), args.doc_ids, gold, args.limit_docs):    # document by document
        doc_id = doc["doc_id"]
        parts = doc_id.split(":")
        date = "-".join(parts[2:5]) if len(parts) >= 5 else ""
        # entities of this document that the pipeline already accepted: context for the others
        known = {c["cluster_id"]: f"{links[c['global_cluster']]['label']} ({c['type']})" for c in doc["clusters"]
                 if (links.get(c.get("global_cluster")) or {}).get("status") == "accepted"
                 and links[c["global_cluster"]].get("label")}
        for cluster in doc["clusters"]:                                                          # entity by entity
            link = links.get(cluster.get("global_cluster"))
            if (doc_id, cluster["cluster_id"]) in done or not link or link["status"] not in statuses or not link["candidates"]:
                continue
            spans = [(doc_id, m["start"], m["end"]) for m in cluster["mentions"] if m.get("start") is not None]
            if gold is not None and not any(s in gold for s in spans):
                continue                                                                         # no gold annotation
            texts = snippets(cluster)
            if not texts:
                stats["no_usable_mention"] += 1
                continue
            candidates = []
            for c in link["candidates"][:args.topk]:
                profile = profiles.get((c["qid"], link["type"]))
                if profile and args.no_summary:
                    profile = {k: v for k, v in profile.items() if k != "wikipedia_summary"}
                candidates.append(dict(c, profile=profile))
            entity = dict(doc_id=doc_id, date=date, cluster=cluster, link=link, texts=texts, candidates=candidates,
                          known=[v for k, v in known.items() if k != cluster["cluster_id"]][:MAX_KNOWN])
            payload = build(entity)

            if args.dry_run:
                print(json.dumps(dict(doc_id=doc_id, cluster_id=cluster["cluster_id"], payload=payload),
                                 ensure_ascii=False, indent=2, default=str))
                shown += 1
                if shown >= args.dry_run:
                    return
                continue

            try:
                probs = send(payload)
            except errors as ex:                                  # not written: the next run tries again
                print(f"[skip] {cluster['cluster_id']}: {ex}")
                stats["error"] += 1
                continue
            unknown = set(probs) - {c["qid"] for c in candidates} - {NONE_KEY}
            if unknown or not probs:                              # the model must only answer with the options given
                print(f"[skip] {cluster['cluster_id']}: unexpected options {sorted(unknown)[:3]}")
                stats["error"] += 1
                continue
            outcome, qid = judge(probs, args.min_p, args.min_ratio)
            out.write(json.dumps(dict(
                doc_id=doc_id, cluster_id=cluster["cluster_id"], global_cluster=cluster["global_cluster"],
                type=cluster["type"], representative=cluster["representative"], pipeline_status=link["status"],
                pipeline_qid=link.get("qid"), top_candidate=candidates[0]["qid"], options=[c["qid"] for c in candidates],
                outcome=outcome, qid=qid, probabilities=dict(sorted(probs.items(), key=lambda kv: -kv[1])),
                backend=name, model=args.model, mention_spans=[[s[1], s[2]] for s in spans]),
                ensure_ascii=False) + "\n")
            out.flush()
            stats[outcome] += 1
            if qid:                                               # later entities of the document can use it
                label = next((c.get("label") for c in candidates if c["qid"] == qid), qid)
                known[cluster["cluster_id"]] = f"{label} ({cluster['type']})"
            if args.max_entities and sum(stats[k] for k in ("link", "nil", "uncertain")) >= args.max_entities:
                stop = True
                break
        if stop:
            break
    print(f"judged: {dict(stats)} -> {args.out}")
