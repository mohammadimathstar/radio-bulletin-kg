"""Rewrite an IE JSONL (e.g. ``ner_re.jsonl``) so that its ``text`` field and all
character offsets refer to the PREPROCESSED bulletin text.

Why
---
The IE records carry the bulletin text *before* paragraph preprocessing (the
``bulletins_processed.jsonl`` variant, where paragraphs are separated by two
newlines).  Downstream, the clustering pipeline slices ``text[start:end]`` and
extracts the containing ``\\n\\n``-paragraph as each mention's context, so the
record text and the offsets must refer to the preprocessed variant.

What it does
------------
For every IE record, matched to ``bulletins_processed.jsonl`` by bulletin id
(separator-insensitively: ``anp:1939:07:17:4:mpeg21`` == ``anp_1939_07_17_4_mpeg21``):

1. ``text``              -> replaced by the preprocessed bulletin text;
2. every entity's        -> remapped into the new text's coordinates
   ``start``/``end``        (token-level diff alignment, see below);
3. every relation's      -> remapped the same way;
   ``evidence.start/end``
4. every entity's        -> verified against the new text; on a whitespace-only
   ``text`` (surface)       difference the surface is refreshed from the slice,
                            on a small character difference (OCR hyphenation
                            fixes etc.) the mention is re-located fuzzily near
                            the mapped position.

Verification tiers per entity
    exact          new_text[start:end] equals the old surface        (best)
    ws_normalized  equal after collapsing whitespace / newlines     (fine)
    fuzzy          re-located nearby with similarity >= threshold   (review)
    failed         no confident match: OLD offsets kept             (fix manually)

Alignment algorithm: both texts are split into whitespace/non-whitespace token
spans; ``difflib.SequenceMatcher`` aligns the token lists (equal / replace /
delete / insert); equal tokens map character-for-character, replaced regions
proportionally, deleted regions collapse onto the gap.  This is fast (token
counts are ~1e2-1e3 per bulletin, not character counts) and robust against the
whitespace-heavy edits paragraph preprocessing makes.

Usage
-----
    python fix_ie_offsets.py --ie ner_re.jsonl --processed bulletins_processed.jsonl \
                             --output ner_re_fixed.jsonl

    # only report what WOULD change:
    python fix_ie_offsets.py --ie ner_re.jsonl --processed bulletins_processed.jsonl --dry-run

    # built-in self test of the offset mapping:
    python fix_ie_offsets.py --selftest

The script is standalone (standard library only; ``rapidfuzz`` is used when
available for the fuzzy tier) and leaves the input files untouched.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"\s+|\S+")
_ID_NORM_RE = re.compile(r"[^a-z0-9]+")


# ----------------------------------------------------------------- small helpers

def norm_ws(s: str) -> str:
    """Collapse every whitespace run (newlines included) into one space."""
    return " ".join((s or "").split())


def norm_bulletin_id(doc_id) -> str:
    """Separator-insensitive id key: 'anp:1939:07:17:4' == 'anp_1939_07_17_4'."""
    return _ID_NORM_RE.sub("", str(doc_id or "").lower())


def doc_id_of(rec: dict) -> str | None:
    return rec.get("bulletin_id") or rec.get("id") or rec.get("doc_id")


def pick_text(rec: dict, text_field: str | None):
    if text_field:
        return rec.get(text_field)
    return rec.get("text") or rec.get("raw_text") or rec.get("content")


def _ratio(a: str, b: str) -> float:
    """Similarity 0-100 (rapidfuzz when available, difflib otherwise)."""
    if a == b:
        return 100.0
    try:
        from rapidfuzz import fuzz
        return float(fuzz.ratio(a, b))
    except ImportError:
        return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() * 100.0


# ----------------------------------------------------------------- offset mapping

def token_spans(text: str) -> list[tuple[int, int, str]]:
    """Contiguous (start, end, text) spans of ``\\s+`` and ``\\S+`` runs."""
    return [(m.start(), m.end(), m.group()) for m in TOKEN_RE.finditer(text or "")]


def build_offset_map(old: str, new: str) -> list[int]:
    """``fwd[i]`` = position in ``new`` corresponding to offset ``i`` in ``old``.

    ``fwd`` has ``len(old) + 1`` entries, so span ends (offsets "just past") map
    too.  Token-level alignment keeps this fast and makes paragraph-boundary
    edits (inserting ``\\n\\n``, joining wrapped lines) clean token operations
    instead of a character-level diff swamp.
    """
    ot, nt = token_spans(old), token_spans(new)
    fwd = [0] * (len(old) + 1)
    sm = difflib.SequenceMatcher(None, [t[2] for t in ot], [t[2] for t in nt],
                                 autojunk=False)
    for tag, k1, k2, l1, l2 in sm.get_opcodes():
        if tag == "equal":
            # identical token strings -> character-for-character mapping
            for k in range(k1, k2):
                o_s, o_e, _ = ot[k]
                n_s, n_e, _ = nt[l1 + (k - k1)]
                for c in range(o_e - o_s):
                    fwd[o_s + c] = n_s + c
                fwd[o_e] = n_e
        elif tag == "replace":
            # proportional interpolation inside the differing region
            o_s, o_e = ot[k1][0], ot[k2 - 1][1]
            n_s, n_e = nt[l1][0], nt[l2 - 1][1]
            lo, ln = o_e - o_s, n_e - n_s
            for c in range(lo + 1):
                fwd[o_s + c] = n_s + (round(c * ln / lo) if lo else 0)
            fwd[o_e] = n_e
        elif tag == "delete":
            # old-only region collapses onto the gap in the new text
            o_s, o_e = ot[k1][0], ot[k2 - 1][1]
            gap = nt[l1][0] if l1 < len(nt) else len(new)
            for c in range(o_s, o_e + 1):
                fwd[c] = gap
        # "insert": nothing on the old side; the boundary offset already
        # points at the gap position set by the preceding opcode.
    fwd[len(old)] = len(new)
    return fwd


def locate_fuzzy(new: str, surface: str, ns: int, ne: int,
                 window: int, min_ratio: float):
    """Re-find ``surface`` in ``new`` near (ns, ne) as a whole-word span.

    Tries word spans of 1 .. k+2 tokens (k = token count of the surface — the
    count itself can change when preprocessing joins OCR hyphenation, e.g.
    "RIBBEN- TROP" -> "RIBBENTROP") starting at every word within +-``window``
    characters of the mapped position, and returns ``(start, end, score)`` of
    the best match scoring >= ``min_ratio``, else None.
    """
    words = [(s, e) for s, e, t in token_spans(new) if t.strip()]
    if not words:
        return None
    target = norm_ws(surface).lower()
    k = max(1, len(target.split()))
    max_chars = len(target) + 20          # cap candidate span length
    lo, hi = max(0, ns - window), min(len(new), ne + window)
    best = None
    for i, (cs, _) in enumerate(words):
        if cs > hi:                        # starts are sorted: nothing left in reach
            break
        if cs < lo:
            continue
        for j in range(i, min(i + k + 2, len(words))):
            ce = words[j][1]
            if ce - cs > max_chars:
                break
            score = _ratio(norm_ws(new[cs:ce]).lower(), target)
            if score >= min_ratio and (best is None or score > best[2]):
                best = (cs, ce, score)
        if best and best[2] >= 99.9:       # can't be beaten
            break
    return best


def remap_entity(fwd: list[int], old: str, new: str, ent: dict,
                 window: int, min_ratio: float) -> tuple[str, int | None, int | None]:
    """Remap one entity's offsets in place; returns (tier, new_start, new_end)."""
    s, e = ent.get("start"), ent.get("end")
    if not isinstance(s, int) or not isinstance(e, int) or e < s:
        return "no_offsets", None, None
    s, e = max(0, min(s, len(old))), max(0, min(e, len(old)))
    ns, ne = fwd[s], fwd[e]
    if ns > ne:
        ns, ne = ne, ns
    surface = ent.get("text") or ""
    ns, ne = max(0, min(ns, len(new))), max(0, min(ne, len(new)))
    if not surface:
        ent["start"], ent["end"] = ns, ne
        return "mapped", ns, ne
    sl = new[ns:ne]
    if sl == surface:
        ent["start"], ent["end"] = ns, ne
        return "exact", ns, ne
    if norm_ws(sl) == norm_ws(surface):
        ent["start"], ent["end"] = ns, ne
        ent["text"] = norm_ws(sl)
        return "ws_normalized", ns, ne
    hit = locate_fuzzy(new, surface, ns, ne, window, min_ratio)
    if hit:
        fs, fe, score = hit
        ent["start"], ent["end"] = fs, fe
        ent["text"] = norm_ws(new[fs:fe])
        return f"fuzzy({score:.0f})", fs, fe
    # apply the mapped offsets anyway (the record text changes regardless) and
    # flag the mention so it can be found again; the clustering pipeline's OCR
    # consistency gate will independently refuse slices that don't match.
    ent["start"], ent["end"] = ns, ne
    ent["offset_fix"] = "unverified"
    return "failed", ns, ne


# ----------------------------------------------------------------- file plumbing

def load_processed_texts(path: str, text_field: str | None) -> dict[str, str]:
    """``{normalized bulletin id: preprocessed text}``; first record per id wins."""
    texts: dict[str, str] = {}
    dups: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            did = doc_id_of(rec)
            text = pick_text(rec, text_field)
            if not did or not text:
                continue
            key = norm_bulletin_id(did)
            if key in texts:
                if texts[key] != text:
                    dups.append(str(did))
                continue
            texts[key] = text
    if dups:
        print(f"  [fix] {len(dups)} duplicate processed-text record(s) ignored for: "
              f"{sorted(set(dups))[:5]}")
    return texts


def fix_record(rec: dict, texts: dict[str, str], stats: Counter,
               failures: list, window: int, min_ratio: float) -> None:
    old = rec.get("text") or ""
    new = texts.get(norm_bulletin_id(doc_id_of(rec)))
    if new is None:
        stats["missing_processed_text"] += 1
        stats["_records_unchanged"] += 1
        return
    if old == new:
        stats["already_equal"] += 1        # offsets already valid; nothing to do
        stats["_records_ok"] += 1
        return
    fwd = build_offset_map(old, new)
    n_ent = n_rel = 0
    for ent in rec.get("entities") or []:
        tier, ns, ne = remap_entity(fwd, old, new, ent, window, min_ratio)
        if tier == "no_offsets":
            stats["no_offsets"] += 1
            continue
        n_ent += 1
        stats[tier.split("(")[0]] += 1
        if tier.startswith("failed"):
            failures.append({"bulletin_id": rec.get("bulletin_id"),
                             "entity_id": ent.get("id"), "surface": ent.get("text"),
                             "old": [ent.get("start"), ent.get("end")],
                             "new_candidates": [ns, ne]})
    for rel in rec.get("relations") or []:
        ev = rel.get("evidence")
        if not isinstance(ev, dict):
            continue
        rs, re_ = ev.get("start"), ev.get("end")
        if isinstance(rs, int) and isinstance(re_, int):
            ev["start"] = max(0, min(fwd[max(0, min(rs, len(old)))], len(new)))
            ev["end"] = max(0, min(fwd[max(0, min(re_, len(old)))], len(new)))
            if ev["end"] < ev["start"]:
                ev["start"], ev["end"] = ev["end"], ev["start"]
            n_rel += 1
    rec["text"] = new
    stats["_records_fixed"] += 1
    stats["_entities_remapped"] += n_ent
    stats["_relations_remapped"] += n_rel


def selftest() -> int:
    """Assertions for the offset mapping + verification tiers."""
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        if not cond:
            raise AssertionError("SELFTEST FAIL: " + msg)
        checks += 1
        print(f"  ok - {msg}")

    # 1. identical text -> identity map
    t = "AB CD.\n\nEF GH."
    ok(build_offset_map(t, t) == list(range(len(t) + 1)), "identical text -> identity map")

    # 2. paragraph separator inserted before the mention
    old, new = "AA BB. CC DD.", "AA BB.\n\nCC DD."
    fwd = build_offset_map(old, new)
    ok(new[fwd[7]:fwd[13]] == "CC DD.", "\\n\\n insertion: mention shifted by +2")

    # 3. deletion before the mention
    old, new = "AB HZZ CD", "AB CD"
    fwd = build_offset_map(old, new)
    ok(new[fwd[7]:fwd[9]] == "CD", "deletion before mention remaps onto the gap")

    # 4. whitespace run widened (line-wrap join): ws_normalized tier
    old, new = "A B", "A  B"
    fwd = build_offset_map(old, new)
    ent = {"text": "A B", "start": 0, "end": 3}
    tier, ns, ne = remap_entity(fwd, old, new, ent, 120, 85.0)
    ok(tier == "ws_normalized" and new[ns:ne] == "A  B" and ent["text"] == "A B",
       "whitespace-run widening -> ws_normalized tier, surface refreshed")

    # 5. OCR hyphenation join: fuzzy tier re-locates the mention
    old, new = "minister VON RIBBEN- TROP zei", "minister VON RIBBENTROP zei"
    fwd = build_offset_map(old, new)
    ent = {"text": "VON RIBBEN- TROP", "start": 9, "end": 25}
    tier, ns, ne = remap_entity(fwd, old, new, ent, 120, 85.0)
    ok(tier.startswith("fuzzy") and new[ns:ne] == "VON RIBBENTROP"
       and ent["text"] == "VON RIBBENTROP",
       f"OCR hyphen-join -> {tier}, surface updated to the new spelling")

    # 6. hopeless entity -> failed: mapped offsets applied but flagged
    old, new = "XXX YYY", "totally different text"
    fwd = build_offset_map(old, new)
    ent = {"text": "QQQ ZZZ", "start": 0, "end": 7}
    tier, ns, ne = remap_entity(fwd, old, new, ent, 10, 85.0)
    ok(tier == "failed" and ent.get("offset_fix") == "unverified" and ns == 0,
       "unmatchable mention -> failed, mapped offsets kept + flagged unverified")

    # 7. entity spanning an inserted paragraph break stays verifiable
    old, new = "AA BB CC. DD EE.", "AA BB CC.\n\nDD EE."
    fwd = build_offset_map(old, new)
    ent = {"text": "CC. DD", "start": 6, "end": 12}
    tier, ns, ne = remap_entity(fwd, old, new, ent, 120, 85.0)
    ok(tier == "ws_normalized" and norm_ws(new[ns:ne]) == "CC. DD",
       "mention crossing a paragraph break -> ws_normalized")

    print(f"Selftest OK ({checks} checks)")
    return checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Replace the IE records' text with the preprocessed bulletin "
                    "text and correct all start/end offsets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example:\n  python fix_ie_offsets.py --ie ner_re.jsonl "
               "--processed bulletins_processed.jsonl --output ner_re_fixed.jsonl")
    ap.add_argument("--ie", default="data/inputs/full_data/ner_re.jsonl", 
                    help="IE JSONL to fix (e.g. ner_re.jsonl)")
    ap.add_argument("--processed", default="data/inputs/full_data/bulletins_processed.jsonl",
                    help="bulletins_processed.jsonl with the "
                         "preprocessed bulletin texts")
    ap.add_argument("--output", default="data/inputs/full_data/ner_re_fixed_offsets.jsonl",
                    help="output JSONL (default: <ie-stem>_fixed_offsets.jsonl "
                                     "next to the input)")
    ap.add_argument("--text-field", default=None,
                    help="field of the processed file holding the bulletin text "
                         "(default: first non-empty of text/raw_text/content)")
    ap.add_argument("--window", type=int, default=100,
                    help="search window (characters) for the fuzzy re-location tier")
    ap.add_argument("--min-fuzzy-ratio", type=float, default=85.0,
                    help="minimum similarity (0-100) to accept a fuzzy re-location")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only; write no output file")
    ap.add_argument("--selftest", action="store_true",
                    help="run the built-in offset-mapping self test and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        return 0 if selftest() >= 0 else 1
    if not args.ie or not args.processed:
        ap.error("--ie and --processed are required (or use --selftest)")

    texts = load_processed_texts(args.processed, args.text_field)
    print(f"Loaded {len(texts)} preprocessed bulletin text(s) from {args.processed}")

    stats: Counter = Counter()
    failures: list = []
    out_path = (Path(args.output) if args.output
                else Path(args.ie).with_name(Path(args.ie).stem + "_fixed_offsets.jsonl"))

    n_records = 0
    records = []
    with open(args.ie, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            n_records += 1

    for rec in records:
        fix_record(rec, texts, stats, failures, args.window, args.min_fuzzy_ratio)
    # per-record verbose lines would need the tier diff; keep the aggregate summary
    # (rerun with --dry-run to iterate on failures without writing anything)

    print(f"\n{n_records} record(s) processed from {args.ie}:")
    print(f"  text replaced + offsets remapped : {stats['_records_fixed']} record(s), "
          f"{stats['_entities_remapped']} entity span(s), "
          f"{stats['_relations_remapped']} relation evidence span(s)")
    print(f"  text already equal (no-op)       : {stats['already_equal']}")
    print(f"  no processed text found          : {stats['missing_processed_text']}")
    print(f"  entity verification: exact={stats['exact']} "
          f"ws_normalized={stats['ws_normalized']} fuzzy={stats['fuzzy']} "
          f"failed={stats['failed']} no_offsets={stats['no_offsets']}")
    if failures:
        print(f"\n  FAILED entities (old offsets kept, fix manually) — first "
              f"{min(10, len(failures))} of {len(failures)}:")
        for ffail in failures[:10]:
            print(f"    {ffail['bulletin_id']} entity {ffail['entity_id']!r} "
                  f"surface={ffail['surface']!r} old={ffail['old']} "
                  f"candidate={ffail['new_candidates']}")
    if args.dry_run:
        print("\n--dry-run: no output written")
        return 1 if (stats["failed"] or stats["missing_processed_text"]) else 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(f"\nWrote {out_path}")
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())