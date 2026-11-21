"""
final_qids.py - final qids per entity = accepted links of the pipeline + decisions of the JEV AI model.
Successor of combine_results.py without the gold evaluation (no gold file, no errors.jsonl, no metrics).

Reads   resolved_documents.jsonl   entities per document             (resolve_entities_v3.py, step 1)
        wikidata_links.jsonl       status + candidates per corpus cluster (link_wikidata.py; pass-1 files with a
                                   nested `result` dict and final flat files both work)
        decisions.jsonl            model decisions per entity        (decide_jev.py / decide_laya.py / candidate_judge)
Writes  final_qids.jsonl           ONE LINE PER ENTITY (document cluster) with its final qid, where it came from and
                                   the label of the qid when known; both sides are always kept in the record
                                   (pipeline_status / pipeline_qid and model_outcome / model_qid)

Where the final qid of an entity comes from (`source`):
    model_link                the JEV AI model linked the entity to a candidate
    model_nil                 the JEV AI model decided "none of the candidates" and nothing overrides it
                              (--priority jev: an accepted pipeline link beats a model nil)
    pipeline_accepted         an accepted pipeline link was applied: the model did not judge the entity,
                              judged it only uncertainly, or said nil — or --priority pipeline was given
    fallback_pipeline_review  no accepted link and no usable model decision, cluster status review: the
                              pipeline's best candidate (--uncertain pipeline, default); --uncertain nil
                              turns it into NIL
    fallback_nil              everything else (nil cluster not judged, uncertain nil cluster, no candidates)
    no_qid_type               entity types without Wikidata qids (not in --qid-types: events, ...); no qid is
                              computed, no decision is applied

--priority jev (default)   a JEV AI "link" beats an accepted pipeline link; an accepted pipeline link beats
                           a JEV AI "nil"; entities the JEV AI never judged fall back to the pipeline
--priority pipeline        an accepted pipeline link beats any JEV AI decision

Usage:
    python -m <package>.final_qids
    python -m <package>.final_qids --priority pipeline --uncertain nil \\
        --documents ./data/output/resolution/resolved_documents.jsonl \\
        --links ./data/output/links/wikidata.pass1.jsonl \\
        --decisions ./data/output/links/decisions.jsonl
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from src.utils.io import read_jsonl, select_docs

# same mapping as context.normalize_type, plus the vehicle aliases of the links step
TYPE_MAP = {"person": "PERSON", "per": "PERSON", "organization": "ORGANIZATION", "org": "ORGANIZATION",
            "place": "PLACE", "loc": "PLACE", "location": "PLACE", "ship": "VEHICLE", "vehicle": "VEHICLE",
            "event": "EVENT"}


def normalize_type(t: str) -> str:
    return TYPE_MAP.get(str(t or "").lower().strip(), str(t or "").upper())


def field(link, key):
    """`status`/`qid`/... of a links record: pass-1 files nest them under `result`, final files are flat."""
    return (link.get("result") or {}).get(key, link.get(key)) if link else None


def cluster_key(cid):
    """Join key between document clusters and corpus links: the corpus ids uppercase the type
    ('ROME||PLACE') while the doc-side global_cluster keeps the raw type ('ROME||Place') — the
    join must not miss on case, so compare the whole key case-insensitively."""
    return str(cid or "").strip().upper()


# ------------------------------------------------------------------ combine
def combine(doc, links, decisions, priority, uncertain, qid_types):
    """One final-qids record per entity (document cluster) of the document."""
    for cluster in doc["clusters"]:
        link = links.get(cluster_key(cluster.get("global_cluster")))
        dec = decisions.get((doc["doc_id"], cluster["cluster_id"]))
        rec = dict(doc_id=doc["doc_id"], cluster_id=cluster["cluster_id"], global_cluster=cluster.get("global_cluster"),
                   type=cluster["type"], representative=cluster["representative"], qid=None, source=None, label=None,
                   pipeline_status=field(link, "status"), pipeline_qid=field(link, "qid"),
                   model_outcome=dec["outcome"] if dec else None, model_qid=dec.get("qid") if dec else None,
                   mention_spans=[[m["start"], m["end"]] for m in cluster["mentions"] if m.get("start") is not None])
        if qid_types and normalize_type(cluster["type"]) not in qid_types:
            rec["source"] = "no_qid_type"                      # events etc.: no qids are computed, nothing applied
            yield rec
            continue
        acc = field(link, "qid") if link and field(link, "status") == "accepted" else None   # a pipeline DECISION
        model = dec.get("qid") if dec and dec["outcome"] == "link" else None
        model_nil = bool(dec) and dec["outcome"] == "nil"      # the model explicitly said: none of the candidates

        def pick(first):
            """(qid, source): try the priority side, then the other side, then the fallbacks.

            A model nil only decides when neither side produced a qid: under --priority jev an accepted
            pipeline link overrides a model nil (the model was only asked about the uncertain ones), and
            a model nil also stands above the review fallback (the model looked and said: no candidate).
            """
            for side in (first, "pipeline" if first == "model" else "model"):
                if side == "model":
                    if model is not None:
                        return model, "model_link"
                elif acc is not None:
                    return acc, "pipeline_accepted"
            if model_nil:
                return None, "model_nil"
            if uncertain == "pipeline" and link and field(link, "status") == "review" and field(link, "qid"):
                return field(link, "qid"), "fallback_pipeline_review"
            return None, "fallback_nil"

        rec["qid"], rec["source"] = pick("model" if priority == "jev" else "pipeline")
        if rec["qid"]:
            cands = (link or {}).get("candidates") or []
            rec["label"] = next((c.get("label") for c in cands if c.get("qid") == rec["qid"]), None)
            if rec["label"] is None and field(link, "qid") == rec["qid"]:
                rec["label"] = field(link, "label")
        yield rec


def main():
    ap = argparse.ArgumentParser(description="Final qids per entity: accepted pipeline links + JEV AI decisions")
    ap.add_argument("--documents", default="data/interim/document_clusters.jsonl")
    ap.add_argument("--links", default="data/output/links/wikidata_links.jsonl")
    ap.add_argument("--decisions", default="data/output/links/decisions.jsonl")
    ap.add_argument("--out", default="data/output/links/final_qids.jsonl")
    ap.add_argument("--priority", choices=["jev", "pipeline"], default="jev",
                    help="when BOTH an accepted pipeline link and a JEV decision exist: which one wins (default jev)")
    ap.add_argument("--uncertain", choices=["pipeline", "nil"], default="pipeline",
                    help="entities the model left uncertain or did not judge: the pipeline's best candidate "
                         "(review clusters) or NIL")
    ap.add_argument("--qid-types", default="person,organization,place,vehicle",
                    help="entity types that get qids, comma separated; '' = all types (events etc. never have any)")
    ap.add_argument("--limit-docs", type=int)
    ap.add_argument("--doc-ids", nargs="*")
    ap.add_argument("--only-decided-docs", action="store_true",
                    help="only documents that appear in the decisions file (small experiments)")
    a = ap.parse_args()

    links = {cluster_key(r["cluster_id"]): r for r in read_jsonl(a.links)}
    decisions = {(d["doc_id"], d["cluster_id"]): d for d in read_jsonl(a.decisions)}
    qid_types = {normalize_type(t) for t in a.qid_types.split(",") if t.strip()}
    doc_ids = set(a.doc_ids) if a.doc_ids else None
    if a.only_decided_docs:
        decided = {d for d, _ in decisions}
        doc_ids = decided if doc_ids is None else doc_ids & decided

    recs = [rec for doc in select_docs(read_jsonl(a.documents), doc_ids, None, a.limit_docs)
            for rec in combine(doc, links, decisions, a.priority, a.uncertain, qid_types)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for rec in recs:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{len(recs)} entities in {len({r['doc_id'] for r in recs})} documents -> {a.out}")
    print("sources:", dict(Counter(r["source"] for r in recs)))
    print("with qid:", sum(1 for r in recs if r["qid"]))
    print("accepted pipeline links kept over a model nil:",
          sum(1 for r in recs if r["source"] == "pipeline_accepted" and r["model_outcome"] == "nil"))


if __name__ == "__main__":
    main()