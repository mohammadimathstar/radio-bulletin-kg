"""
enrich_ner_re.py - put everything together: add the final Wikidata qids to the NER/RE output.

Reads   ner_re_fixed_offsets.jsonl  one bulletin per line: text, entities, relations, re  (the RE step)
        final_qids.jsonl            one line per entity with its final qid and mention_spans  (final_qids.py)
Writes  ner_re_with_qids.jsonl     the same records with two changes:
                                       every entity that has a final qid gets a `qid` field; entities without
                                       one (events, NIL decisions, no matching cluster, ...) have NO `qid` field
                                       `created_at` is set to the current time
                                   everything else (topics, language, text, model, relations, re, ...) is kept

Matching: an entity is matched to a final-qids record of the same bulletin (bulletin_id == doc_id) by the
exact character offsets (start, end) of one of its mention_spans.  Without an exact match, the record with
the largest span overlap is used (the counts of exact and overlap matches are printed; among equally good
matches, records that have a qid win).

Usage:
    python -m <package>.enrich_ner_re --ner-re <path>/ner_re_fixed_offsets.jsonl
"""
import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.utils.io import read_jsonl

def now_iso():
    """The timestamp format of the NER/RE file: `2026-11-08T08:32:20.342746Z`."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def match(spans, start, end):
    """Final-qids record for the entity at [start, end): exact span match, else the largest overlap."""
    exact = [r for s, e, r in spans if s == start and e == end]
    if exact:
        return sorted(exact, key=lambda r: (r["qid"] is None, r["cluster_id"]))[0]
    overlap = [(min(e, end) - max(s, start), r["qid"] is not None, r["cluster_id"], r)
               for s, e, r in spans if s < end and start < e]
    return max(overlap, key=lambda t: t[:3])[3] if overlap else None

BASE = "data/output"
def main():
    ap = argparse.ArgumentParser(description="Add the final qids to the NER/RE bulletins")
    ap.add_argument("--ner-re", default="data/inputs/full_data/ner_re_fixed_offsets.jsonl",
                    help="the NER/RE output, one bulletin per line")
    ap.add_argument("--qids", default="data/output/links/final_qids.jsonl", help="the final_qids.py output")
    ap.add_argument("--out", default="data/output/links/ner_re_with_qids.jsonl")
    a = ap.parse_args()

    spans_by_doc = defaultdict(list)
    for r in read_jsonl(a.qids):
        spans_by_doc[r["doc_id"]].append(r)
    stamp = now_iso()
    stats = Counter()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for rec in read_jsonl(a.ner_re):
            spans = [(s, e, r) for r in spans_by_doc.get(rec.get("bulletin_id"), []) for s, e in r["mention_spans"]]
            entities = []
            for ent in rec.get("entities", []):
                if ent.get("start") is None or ent.get("end") is None:
                    stats["no_matching_cluster"] += 1                    # no offsets: nothing to match
                elif (hit := match(spans, ent["start"], ent["end"])) is None:
                    stats["no_matching_cluster"] += 1
                elif hit["qid"]:
                    stats["with_qid"] += 1
                    ent = dict(ent, qid=hit["qid"])
                else:
                    stats["without_qid"] += 1                            # matched cluster has no qid (nil, ...)
                entities.append(ent)
            out = dict(rec)
            out["entities"] = entities
            out["created_at"] = stamp
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            stats["bulletins"] += 1
            stats["exact"] += sum(1 for ent in entities if "qid" in ent
                                  and any(s == ent["start"] and e == ent["end"] for s, e, _ in spans))
    n_ent = stats["with_qid"] + stats["without_qid"] + stats["no_matching_cluster"]
    print(f"{stats['bulletins']} bulletins, {n_ent} entities: {stats['with_qid']} with qid "
          f"({stats['exact']} via exact offsets), {stats['without_qid']} matched without qid, "
          f"{stats['no_matching_cluster']} without matching cluster -> {a.out}")


if __name__ == "__main__":
    main()
