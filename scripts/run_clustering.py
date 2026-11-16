#!/usr/bin/env python3
"""Run the clustering pipeline on an IE JSONL file.

Usage
-----
    python scripts/run_clustering.py --ie data/input/extracted.jsonl --output data/interim/

    # with OCR texts (enables full context windows + sentence segmentation):
    python scripts/run_clustering.py --ie data/input/extracted.jsonl --ocr data/input/ocr.jsonl --output data/interim/

    # tune parameters:
    python scripts/run_clustering.py --ie ... --threshold 0.85 --corpus-threshold 0.80 --cue-window 10
"""
import argparse
import json
import sys
from pathlib import Path

# Make the package importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clustering.cluster import (
    resolve_entities, doc_record, build_report,
)
from src.clustering.context import add_context, get_sentence_splitter
from src.clustering.io import load_dataset, write_json, write_jsonl
from config.cls_config import ClusteringConfig


def main():
    ap = argparse.ArgumentParser(description="Cluster entity mentions from IE output.")
    ap.add_argument("--ie", 
                    default="data/inputs/full_data/ner_re.jsonl",
                    # required=True, 
                    help="Path to IE JSONL (entities + relations).")
    ap.add_argument("--ocr", 
                    default="data/inputs/full_data/bulletins_processed.jsonl", 
                    help="Path to OCR text JSONL (optional).")
    ap.add_argument("--output", default="data/interim", help="Output directory.")
    ap.add_argument("--threshold", type=float, default=0.85, help="Stage-1 similarity threshold.")
    ap.add_argument("--corpus-threshold", type=float, default=0.9, help="Stage-2 threshold (default: --threshold).")
    ap.add_argument("--cue-window", type=int, default=5, help="Token window for cue extraction.")
    ap.add_argument("--no-flexible", action="store_true", help="Strict cue scan (break at content words).")
    ap.add_argument("--no-nat-strict", action="store_true", help="Allow nationality past content words.")
    ap.add_argument("--no-nat-constraint", action="store_true", help="Disable nationality cannot-link.")
    ap.add_argument("--occ-constraint", action="store_true", help="Enable occupation cannot-link (experimental).")
    ap.add_argument("--clustered-types", default="Person,Place,Organization",
                    help="Comma-separated types that get BOTH stages of clustering "
                         "(default PERSON,PLACE,ORGANIZATION). EVENT is always grouped "
                         "by event_id within a bulletin only; all other types are singletons.")
    ap.add_argument("--containment-types", default="PERSON", help="Types for containment attachment.")
    ap.add_argument("--spacy-model", default="nl_core_news_lg", help="spaCy model (if available).")
    args = ap.parse_args()

    cfg = ClusteringConfig(
        threshold=args.threshold,
        corpus_threshold=args.corpus_threshold,
        cue_window=args.cue_window,
        flexible=not args.no_flexible,
        nationality_strict=not args.no_nat_strict,
        nationality_constraint=not args.no_nat_constraint,
        occupation_constraint=args.occ_constraint,
        clustered_types=tuple(t.strip() for t in args.clustered_types.split(",") if t.strip()),
        containment_types=tuple(t.strip() for t in args.containment_types.split(",") if t.strip()),
    )

    print(f"Loading IE data from {args.ie} ...")
    mentions_by_doc, ocr_texts = load_dataset(args.ie, args.ocr)
    n_docs = len(mentions_by_doc)
    n_mentions = sum(len(v) for v in mentions_by_doc.values())
    print(f"  {n_docs} documents, {n_mentions} mentions")
    if ocr_texts:
        print(f"  OCR texts available for {len(ocr_texts)} documents (full mode)")
    else:
        print("  No OCR texts — using context field (simple mode)")

    splitter = get_sentence_splitter(args.spacy_model)
    print("Adding context + extracting person cues ...")
    add_context(mentions_by_doc, ocr_texts, cfg, nlp=splitter)

    print("Clustering (stage 1: within-document, stage 2: corpus-wide) ...")
    doc_clusters, global_resolved = resolve_entities(mentions_by_doc, cfg)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # corpus-level clusters
    global_path = out / "global_clusters.json"
    write_json(global_resolved, str(global_path))
    print(f"  wrote {global_path}")

    # per-document clusters (JSONL)
    doc_records = [doc_record(doc_id, doc_clusters[doc_id])
                   for doc_id in sorted(doc_clusters)]
    doc_path = out / "document_clusters.jsonl"
    write_jsonl(doc_records, str(doc_path))
    print(f"  wrote {doc_path} ({len(doc_records)} documents)")

    # diagnostics report
    report = build_report(global_resolved, cfg)
    report_path = out / "clustering_report.json"
    write_json(report, str(report_path))
    print(f"  wrote {report_path}")

    # summary
    print()
    print(f"Corpus clusters: {len(global_resolved)}")
    for key, by_type in list(global_resolved.items())[:10]:
        for t, b in by_type.items():
            print(f"  [{t}] {b['representative']}  ({b['n_mentions']} mentions, {b['n_documents']} docs)")
    if len(global_resolved) > 10:
        print(f"  ... and {len(global_resolved) - 10} more")


if __name__ == "__main__":
    main()
