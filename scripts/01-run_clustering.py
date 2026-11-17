"""Run the entity resolution pipeline on an IE JSONL file.

Usage
-----
    python run_clustering.py --ie ner_re_fixed.jsonl --output data/interim/

    # tune parameters:
    python run_clustering.py --ie ... --threshold 0.85 --corpus-threshold 0.80

The bulletin text comes from the IE records' own ``text`` field (full mode:
sentence segmentation + the containing PARAGRAPH as each mention's `context`).
Run fix_ie_offsets.py first so that field holds the PREPROCESSED bulletin text
and the mention offsets point into it — no separate text file is needed.

An external text file is only an optional override, e.g. when the IE records
carry no text or the offsets were computed against another variant:

    python run_clustering.py --ie data/input/extracted.jsonl \
                             --ocr data/input/bulletins_processed.jsonl \
                             --output data/interim/

Level strictness (asymmetric by default):
  - stage 1 (within bulletin): relaxed plain character similarity; the strict
    token-aware score can be turned on with --stage1-token-guard
  - stage 2 (corpus-wide): always the strict token-aware score (a shared
    generic tail such as "... communistische partij" cannot merge two clusters)

Resolution strategies (fixed defaults, see config.py):
  - string similarity, two-level clustering: Person, Place, LegalDocument, Vehicle
  - exact event_id matching within one bulletin only: Event
Use --string-similarity-types to change the first set (labels are canonicalized,
so "person,loc,legal_document" works too).  Types outside both sets (e.g.
"Organization") become document-local singletons.

Entity subtypes are NOT used for resolution.
"""

import argparse
import sys
from pathlib import Path

# Make the package importable when running from the repo root or from here:
# this script lives INSIDE the package directory, so add its parent.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.entity_resolution.cluster import ( 
    resolve_entities, doc_record, build_report,
)
from src.entity_resolution.context import add_context, get_sentence_splitter  
from src.utils.io import load_dataset, write_json, write_jsonl  
from src.entity_resolution.entity_types import canonical_type, STRATEGY_STRING_SIMILARITY  
from src.entity_resolution.config import ClusteringConfig, DEFAULT_STRATEGY_BY_TYPE  

def print_context_summary(mentions_by_doc: dict) -> None:
    """Report how many mentions actually got real context, with actionable hints.

    This is where a broken OCR/IE id match would otherwise hide: the run would
    complete normally while every mention silently stays in simple mode.
    """
    from collections import Counter
    counts: Counter = Counter()
    empty = 0
    total = 0
    for mentions in mentions_by_doc.values():
        for m in mentions:
            total += 1
            counts[m.get("context_status") or "none"] += 1
            if not (m.get("context") or "").strip():
                empty += 1
    parts = [f"{label}: {n}" for label, n in counts.most_common()]
    print(f"  context status — {', '.join(parts) or 'no mentions'}")
    if empty:
        print(f"  WARNING: {empty}/{total} mentions have an EMPTY context field")
        if counts.get("from_context_field"):
            print("    hint: mentions fell back to the IE 'context' field. If an "
                  "external --ocr file was given, its ids probably do not match "
                  "the IE ids (see the [io] match report above); otherwise the "
                  "IE records carry no text — run fix_ie_offsets.py.")
    if counts.get("ocr_mismatch", 0) > total * 0.05:
        print(f"    hint: {counts['ocr_mismatch']} mentions failed the OCR "
              f"consistency gate — the IE offsets were probably computed against "
              f"a different text variant; try --ocr-text-field (e.g. text_org).")


def main():
    ap = argparse.ArgumentParser(description="Cluster entity mentions from IE output.")
    ap.add_argument("--ie", default="data/inputs/full_data/ner_re_fixed_offsets.jsonl",
                    help="Path to IE JSONL (entities + relations).")
    ap.add_argument("--ocr", default=None,
                    help="Optional external bulletin-text JSONL. By default the "
                         "text already present on the IE records' own 'text' "
                         "field is used (full mode, no separate file needed); "
                         "pass this only to use a different text file/variant — "
                         "the mention offsets must have been computed against "
                         "exactly that variant. With it, each mention gets the "
                         "whole paragraph (blank-line separated, i.e. two "
                         "newlines) as context.")
    ap.add_argument("--ocr-text-field", default=None,
                    help="JSON field of the OCR file to use as the bulletin text "
                         "(default: first non-empty of text/raw_text/content). "
                         "Set this to the variant the IE offsets were computed "
                         "against — e.g. text_org — when the context summary "
                         "reports many ocr_mismatch mentions.")
    ap.add_argument("--output", default="data/interim", help="Output directory.")
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="Stage-1 (within-bulletin) similarity threshold.")
    ap.add_argument("--corpus-threshold", type=float, default=0.9,
                    help="Stage-2 (corpus-wide) threshold. Use 0 to fall back "
                         "to --threshold for a type without a per-type override.")
    ap.add_argument("--cue-window", type=int, default=8,
                    help="Token window for person cue extraction.")
    ap.add_argument("--no-flexible", action="store_true",
                    help="Strict cue scan (break at content words).")
    ap.add_argument("--no-nat-strict", action="store_true",
                    help="Allow nationality cues beyond content words.")
    ap.add_argument("--no-nat-constraint", action="store_true",
                    help="Disable the nationality cannot-link for Person.")
    ap.add_argument("--stage1-token-guard", action="store_true",
                    help="Apply the strict token-aware similarity at stage 1 "
                         "(within bulletin) too — default is relaxed there; "
                         "stage 2 always uses it.")
    ap.add_argument("--short-para-words", type=int, default=4,
                    help="Paragraphs with at most this many words (datelines / "
                         "kickers such as 'Soesterberg.') merge into the "
                         "following paragraph before context extraction. "
                         "Use 0 to disable.")
    ap.add_argument("--string-similarity-types",
                    default="Person,Place,LegalDocument,Vehicle",
                    help="Comma-separated types resolved by two-level string "
                         "similarity (labels are canonicalized: PER, LOCATION, "
                         "legal_document ... all work). Event always uses exact "
                         "event_id matching within a bulletin; types in neither "
                         "set stay singletons.")
    ap.add_argument("--spacy-model", default="nl_core_news_lg",
                    help="spaCy model (if available).")
    args = ap.parse_args()

    # Canonicalize the CLI type labels once, at the boundary (this is what
    # makes "person" vs "PERSON" vs "PER" irrelevant everywhere downstream).
    string_types = []
    for t in args.string_similarity_types.split(","):
        c = canonical_type(t)
        if c:
            string_types.append(c)

    strategy_by_type = dict(DEFAULT_STRATEGY_BY_TYPE)
    for t in string_types:
        strategy_by_type[t] = STRATEGY_STRING_SIMILARITY

    cfg = ClusteringConfig(
        threshold=args.threshold,
        corpus_threshold=(None if args.corpus_threshold == 0 else args.corpus_threshold),
        cue_window=args.cue_window,
        flexible=not args.no_flexible,
        nationality_strict=not args.no_nat_strict,
        nationality_constraint=not args.no_nat_constraint,
        stage1_token_guard=args.stage1_token_guard,
        short_para_max_words=args.short_para_words,
        strategy_by_type=strategy_by_type,
    )
    print(f"Strategies: string-similarity={sorted(cfg.string_similarity_types)}, "
          f"exact-event-id=Event, singletons="
          f"{sorted(set(cfg.strategy_by_type) - cfg.string_similarity_types - {'Event'})}")
    print(f"Strictness: stage 1 {'token-guarded' if cfg.stage1_token_guard else 'relaxed'}, "
          f"stage 2 token-guarded; short paragraphs <= {cfg.short_para_max_words} words "
          f"merge into the next (0 = off)")

    print(f"Loading IE data from {args.ie} ...")
    mentions_by_doc, ocr_texts = load_dataset(args.ie, args.ocr,
                                              ocr_text_field=args.ocr_text_field)
    n_docs = len(mentions_by_doc)
    n_mentions = sum(len(v) for v in mentions_by_doc.values())
    print(f"  {n_docs} bulletins, {n_mentions} mentions")
    if ocr_texts:
        src = (f"from {args.ocr}" if args.ocr
               else "from the IE records' own text field")
        print(f"  Bulletin texts available for {len(ocr_texts)} bulletins "
              f"({src}) — full mode")
    else:
        print("  No bulletin texts — using context field (simple mode); "
              "run fix_ie_offsets.py to put the preprocessed text on the IE records")

    splitter = get_sentence_splitter(args.spacy_model)
    print("Adding context + extracting person cues ...")
    add_context(mentions_by_doc, ocr_texts, cfg, nlp=splitter)
    print_context_summary(mentions_by_doc)

    print("Clustering (stage 1: within-bulletin, stage 2: corpus-wide) ...")
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
    print(f"  wrote {doc_path} ({len(doc_records)} bulletins)")

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
            print(f"  [{t}] {b['representative']}  ({b['n_mentions']} mentions, "
                  f"{b['n_documents']} bulletins)")
    if len(global_resolved) > 10:
        print(f"  ... and {len(global_resolved) - 10} more")


if __name__ == "__main__":
    main()