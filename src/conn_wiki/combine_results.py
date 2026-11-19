"""
combine_results.py - final links = accepted links of the pipeline + decisions of the model, then evaluation.

Reads   resolved_documents.jsonl   entities per document            (resolve_entities_v3.py, step 1)
        wikidata_links.jsonl       candidates per corpus cluster     (link_wikidata_v8.py)
        decisions.jsonl            model decisions per entity        (decide_jev.py / decide_laya.py)
        gold file (optional)       resolution format with a `qid` per cluster (null/missing = no Wikidata item)
Writes  final_links.jsonl          ONE LINE PER ENTITY (document cluster) with its final qid and where it came from
        errors.jsonl               entities whose qid differs from the gold qid, for further investigation
        metrics.json               the numbers printed on the screen

Where the final qid of an entity comes from (`source`):
    pipeline_accepted         its corpus cluster was accepted by link_wikidata_v10.py
    model_link                the JEV AI model linked it to a candidate
    model_nil                 the JEV AI model said: none of the candidates
    fallback_pipeline_review  the model was uncertain / did not judge it, cluster status review: the pipeline's best
                              candidate (--uncertain pipeline, default); with --uncertain nil it is NIL instead
    fallback_nil              everything else (nil cluster not judged, uncertain nil cluster, no candidates)

Evaluation (only entities that have gold annotations; matched by the character offsets of the mentions):
    tp = the gold qid was predicted, wrong = another qid, missed = gold has a qid but the answer is NIL,
    spurious = gold is NIL but a qid was predicted, tn = both NIL.
    precision = tp / (tp + wrong + spurious)   recall = tp / (tp + wrong + missed)   accuracy = (tp + tn) / n
    Once per entity (`entities`) and once weighted by the number of gold mentions (`mentions`); per type, overall and
    per source (how good are the accepted links, how good the model).
Every error line of errors.jsonl has a `diagnosis`:
    gold_is_nil               gold says there is no Wikidata item, but a qid was linked
    gold_not_in_candidates    the gold qid was never a candidate: a retrieval problem (see candidate expansion)
    gold_not_shown_to_model   it is a candidate but beyond the top-k shown to the model
    pipeline_accepted_wrong / model_chose_wrong / model_said_nil / uncertain_review_fallback_wrong /
    undecided_or_nil_cluster  the gold qid was available and the decision went wrong

Usage:
    python combine_results.py --gold gold_clusters.json
    python combine_results.py --gold gold_clusters.json --decisions small.jsonl --only-decided-docs   # small experiment
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .decide_common import BASE, gold_spans, read_jsonl, select_docs, snippets


# ------------------------------------------------------------------ combine
def combine(doc, links, decisions, uncertain):
    """Yield (record, cluster) for every entity of a document."""
    for cluster in doc["clusters"]:
        link = links.get(cluster.get("global_cluster") or "")
        dec = decisions.get((doc["doc_id"], cluster["cluster_id"]))
        if link and link["status"] == "accepted":
            qid, source = link["qid"], "pipeline_accepted"
        elif dec and dec["outcome"] == "link":
            qid, source = dec["qid"], "model_link"
        elif dec and dec["outcome"] == "nil":
            qid, source = None, "model_nil"
        elif uncertain == "pipeline" and link and link["status"] == "review" and link.get("qid"):
            qid, source = link["qid"], "fallback_pipeline_review"      # uncertain or not judged: the pipeline's best guess
        else:
            qid, source = None, "fallback_nil"
        yield dict(doc_id=doc["doc_id"], cluster_id=cluster["cluster_id"], global_cluster=cluster.get("global_cluster"),
                   type=cluster["type"], representative=cluster["representative"], qid=qid, source=source,
                   pipeline_status=link["status"] if link else None, pipeline_qid=link.get("qid") if link else None,
                   model_outcome=dec["outcome"] if dec else None, model_qid=dec.get("qid") if dec else None,
                   mention_spans=[[m["start"], m["end"]] for m in cluster["mentions"] if m.get("start") is not None]), cluster


# ------------------------------------------------------------------ evaluate
def outcome(gold: frozenset, pred):
    if gold:
        return "missed" if pred is None else ("tp" if pred in gold else "wrong")
    return "tn" if pred is None else "spurious"


def scores(c: Counter) -> dict:
    tp, wrong, missed, spur, tn = (c[k] for k in ("tp", "wrong", "missed", "spurious", "tn"))
    ratio = lambda a, b: round(a / b, 3) if b else None
    p, r = ratio(tp, tp + wrong + spur), ratio(tp, tp + wrong + missed)
    return dict(n=tp + wrong + missed + spur + tn, precision=p, recall=r,
                f1=round(2 * p * r / (p + r), 3) if p and r else (0.0 if p is not None and r is not None else None),
                accuracy=ratio(tp + tn, tp + wrong + missed + spur + tn), tp=tp, wrong=wrong, missed=missed, spurious=spur, tn=tn)


def diagnose(oc, source, in_candidates, in_options):
    if oc == "spurious":
        return "gold_is_nil"
    if not in_candidates:
        return "gold_not_in_candidates"
    if source.startswith("model") and in_options is False:
        return "gold_not_shown_to_model"
    return {"pipeline_accepted": "pipeline_accepted_wrong", "model_link": "model_chose_wrong", "model_nil": "model_said_nil",
            "fallback_pipeline_review": "uncertain_review_fallback_wrong"}.get(source, "undecided_or_nil_cluster")


def evaluate(pairs, gold, links, decisions):
    """-> (metrics, error records). pairs: [(record, cluster)]."""
    ent, men, errors = defaultdict(Counter), defaultdict(Counter), []
    for r, cluster in pairs:
        golds = [frozenset(gold[(r["doc_id"], s, e)]) for s, e in r["mention_spans"] if (r["doc_id"], s, e) in gold]
        if not golds:
            continue
        g = Counter(golds).most_common(1)[0][0]                        # gold qids of the entity (empty = NIL)
        oc, w = outcome(g, r["qid"]), len(golds)
        for key in (r["type"], "ALL", f"source={r['source']}"):
            ent[key][oc] += 1
            men[key][oc] += w
        if oc in ("wrong", "missed", "spurious"):
            link, dec = links.get(r["global_cluster"] or ""), decisions.get((r["doc_id"], r["cluster_id"]))
            cands = [c["qid"] for c in (link or {}).get("candidates", [])]
            in_cands = bool(g & set(cands))
            in_options = bool(g & set(dec["options"])) if dec else None
            errors.append(dict(
                error=oc, diagnosis=diagnose(oc, r["source"], in_cands, in_options), doc_id=r["doc_id"], cluster_id=r["cluster_id"],
                global_cluster=r["global_cluster"], type=r["type"], representative=r["representative"], gold_qids=sorted(g),
                pred_qid=r["qid"], source=r["source"], n_gold_mentions=w, pipeline_status=r["pipeline_status"],
                pipeline_score=(link or {}).get("score"), model_outcome=r["model_outcome"],
                model_probabilities=dict(list(dec["probabilities"].items())[:3]) if dec else None,
                gold_in_candidates=in_cands, gold_rank=next((i + 1 for i, q in enumerate(cands) if q in g), None),
                gold_in_options=in_options, candidates=(link or {}).get("candidates", [])[:5],
                texts=snippets(cluster)[:2], mention_spans=r["mention_spans"]))
    metrics = {level: {k: scores(c) for k, c in sorted(tab.items())} for level, tab in (("entities", ent), ("mentions", men))}
    errors.sort(key=lambda e: (-e["n_gold_mentions"], e["doc_id"]))
    return metrics, errors


def print_table(title, table):
    print(f"\n{title}")
    print(f"{'':28}{'n':>6}{'P':>7}{'R':>7}{'F1':>7}{'acc':>7}{'tp':>6}{'wrong':>6}{'missed':>7}{'spur':>6}")
    f = lambda x: "-" if x is None else f"{x:.3f}"
    for k in [k for k in table if not k.startswith("source=") and k != "ALL"] + ["ALL"] + [k for k in table if k.startswith("source=")]:
        v = table[k]
        print(f"{k:28}{v['n']:>6}{f(v['precision']):>7}{f(v['recall']):>7}{f(v['f1']):>7}{f(v['accuracy']):>7}"
              f"{v['tp']:>6}{v['wrong']:>6}{v['missed']:>7}{v['spurious']:>6}")


def main():
    ap = argparse.ArgumentParser(description="Combine pipeline and model results, evaluate, list the errors")
    ap.add_argument("--documents", default=f"{BASE}/resolution/resolved_documents.jsonl")
    ap.add_argument("--links", default=f"{BASE}/links/wikidata_links_v4.jsonl")
    ap.add_argument("--decisions", default=f"{BASE}/links/decisions.jsonl")
    ap.add_argument("--out", default=f"{BASE}/links/eval/final_links.jsonl")
    ap.add_argument("--errors", default=f"{BASE}/links/eval/errors.jsonl")
    ap.add_argument("--metrics", default=f"{BASE}/links/eval/metrics.json")
    ap.add_argument("--gold", help="gold file; without it only the combined file is written", 
                    default='./data/processed/openai/resolution/resolved_names_with_qid.json')
    ap.add_argument("--missing-qid", choices=["nil", "skip"], default="nil", help="gold clusters without a qid: no Wikidata item, or not annotated")
    ap.add_argument("--uncertain", choices=["pipeline", "nil"], default="pipeline",
                    help="entities the model left uncertain or did not judge: the pipeline's best candidate (review clusters) or NIL")
    ap.add_argument("--limit-docs", type=int)
    ap.add_argument("--doc-ids", nargs="*")
    ap.add_argument("--only-decided-docs", action="store_true", help="only documents that appear in the decisions file (small experiments)")
    a = ap.parse_args()

    links = {r["cluster_id"]: r for r in read_jsonl(a.links)}
    decisions = {(d["doc_id"], d["cluster_id"]): d for d in read_jsonl(a.decisions)}
    doc_ids = set(a.doc_ids) if a.doc_ids else None
    if a.only_decided_docs:
        decided = {d for d, _ in decisions}
        doc_ids = decided if doc_ids is None else doc_ids & decided

    pairs = [p for doc in select_docs(read_jsonl(a.documents), doc_ids, None, a.limit_docs)
             for p in combine(doc, links, decisions, a.uncertain)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r, _ in pairs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    src = Counter(r["source"] for r, _ in pairs)
    print(f"{len(pairs)} entities in {len({r['doc_id'] for r, _ in pairs})} documents -> {a.out}")
    print("sources:", dict(src))
    if not a.gold:
        return

    metrics, errors = evaluate(pairs, gold_spans(a.gold, a.missing_qid), links, decisions)
    print_table("entities (every entity with gold counts once)", metrics["entities"])
    print_table("mentions (weighted by the number of gold mentions)", metrics["mentions"])
    Path(a.metrics).write_text(json.dumps(dict(sources=dict(src), **metrics), indent=2), encoding="utf-8")
    with open(a.errors, "w", encoding="utf-8") as fh:
        for e in errors:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"\n{len(errors)} errors -> {a.errors}: {dict(Counter(e['error'] for e in errors))}")
    print("diagnosis:", dict(Counter(e["diagnosis"] for e in errors).most_common()))


if __name__ == "__main__":
    main()