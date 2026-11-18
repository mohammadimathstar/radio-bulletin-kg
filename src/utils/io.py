"""I/O for the clustering pipeline.

Reads the IE JSONL produced by the ``radio-bulletins-ie`` repository (one record
per document, with ``entities`` and ``relations`` lists).  The bulletin text is
taken from the records' own top-level ``text`` field (full mode) — after
``fix_ie_offsets.py`` has replaced it with the preprocessed bulletin text and
corrected the offsets, no separate text file is needed.  An external OCR-text
file (``--ocr``) remains available as an override.

The IE schema uses ``bulletin_id`` as the document identifier and marks every entity
with a ``status`` field (``"kept"``, ``"removed"``, ...).  Both are handled here:
``bulletin_id`` is recognised as the doc id (with ``id`` / ``doc_id`` as fallbacks for
older schemas), and only entities whose ``status`` is in ``kept_statuses`` are loaded
(default ``{"kept"}``; pass ``None`` to load every entity regardless of status).

Entity types may be written in any convention (``"PER"``, ``"Person"``,
``"PERSON"``, ``"LOCATION"``, ``"Legal_Document"`` ...) — they are standardized to
the canonical capitalized forms by :func:`.entity_types.canonical_type` when context is
added (and again defensively in the clustering stage).

Robustness
----------
If the same ``bulletin_id`` appears on several JSONL records (chunked extraction),
the entity lists are ACCUMULATED instead of overwritten, and a warning is printed;
for OCR texts the first record wins (a later one would silently invalidate the
offsets already emitted downstream).
"""
import json
import re
from pathlib import Path

# Default: only entities the IE system explicitly kept.  Pass None to load all.
DEFAULT_KEPT_STATUSES = frozenset({"kept"})

_ID_NORM_RE = re.compile(r"[^a-z0-9]+")


def norm_bulletin_id(doc_id) -> str:
    """Canonical form of a bulletin id for cross-file matching.

    The IE output and the OCR text file frequently spell the same bulletin
    differently — the IE pipeline writes ``anp:1939:01:11:5:mpeg21`` while the
    OCR file writes ``anp_1939_01_11_5_mpeg21``.  All separators (colons,
    underscores, dashes, dots) and casing are dropped, so both collapse to
    ``anp193901115mpeg21`` and can be matched.
    """
    return _ID_NORM_RE.sub("", str(doc_id or "").lower())


def align_ocr_texts(mentions_by_doc: dict, ocr_texts: dict | None) -> dict:
    """Re-key ``ocr_texts`` to the IE document ids, matching exact or normalized ids.

    Without this, a formatting difference in the ids (``:`` vs ``_``) makes every
    ``ocr_texts.get(doc_id)`` miss, the whole corpus silently drops to simple
    mode, and ``context`` comes out empty.  The returned dict is keyed by the IE
    doc ids, ready for :func:`.context.add_context`.  A match report is printed;
    when nothing matches, sample ids from both sides are shown so the mismatch
    is visible immediately.
    """
    if not ocr_texts:
        return {}
    norm: dict[str, tuple[str, str]] = {}
    for oid, text in ocr_texts.items():
        key = norm_bulletin_id(oid)
        if key in norm:                       # two ids collapse to one key: first wins
            if norm[key][1] != text:
                print(f"  [io] WARNING: normalized OCR id collision for {key!r} "
                      f"({norm[key][0]!r} vs {oid!r}); keeping the first text")
        else:
            norm[key] = (oid, text)
    out: dict[str, str] = {}
    missed: list[str] = []
    for doc_id in mentions_by_doc:
        text = ocr_texts.get(doc_id)
        if text is None:
            hit = norm.get(norm_bulletin_id(doc_id))
            text = hit[1] if hit else None
        if text is None:
            missed.append(doc_id)
        else:
            out[doc_id] = text
    total = len(mentions_by_doc)
    if not out:
        print(f"  [io] WARNING: no OCR text matched any of the {total} IE bulletins — "
              "context falls back to the IE 'context' field (usually empty).")
        print(f"  [io]   IE ids look like:   {sorted(mentions_by_doc)[:2]}")
        print(f"  [io]   OCR ids look like:  {sorted(ocr_texts)[:2]}")
    elif missed:
        print(f"  [io] OCR texts matched {len(out)}/{total} IE bulletins "
              f"({len(missed)} unmatched stay in simple mode), e.g. {missed[:3]}")
    else:
        print(f"  [io] OCR texts matched all {total} IE bulletins")
    return out


def _doc_id(rec: dict) -> str | None:
    """Resolve the document id from an IE record, across schema variants."""
    return rec.get("bulletin_id") or rec.get("id") or rec.get("doc_id")


def load_ie_mentions(path: str, kept_statuses=DEFAULT_KEPT_STATUSES) -> dict:
    """Load IE mentions from a JSONL file.

    Each line is a record with ``bulletin_id`` (or ``id`` / ``doc_id``) and
    ``entities`` (a list of mention dicts with at least ``text``, ``type``,
    ``start``, ``end``).  Returns ``{doc_id: [mentions]}``.

    ``kept_statuses`` selects which entities to load based on their ``status``
    field.  The default ``{"kept"}`` loads only entities the IE system marked as
    kept; pass ``None`` to load every entity (useful for debugging).  Records
    whose entities have no ``status`` field are always loaded.

    Duplicate records for the same document are accumulated (not overwritten).
    """
    statuses = frozenset(kept_statuses) if kept_statuses is not None else None
    mentions_by_doc: dict[str, list] = {}
    extra_records: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            doc_id = _doc_id(rec)
            if not doc_id:
                continue
            entities = rec.get("entities") or []
            if statuses is not None:
                entities = [e for e in entities
                            if not e.get("status") or e["status"] in statuses]
            # carry doc_id onto each mention for downstream use
            for e in entities:
                e.setdefault("doc_id", doc_id)
            if doc_id in mentions_by_doc:
                extra_records.append(doc_id)
                mentions_by_doc[doc_id].extend(entities)
            else:
                mentions_by_doc[doc_id] = list(entities)
    if extra_records:
        print(f"  [io] {len(extra_records)} extra record(s) accumulated for: "
              f"{sorted(set(extra_records))[:10]}")
    return mentions_by_doc


def load_ie_relations(path: str) -> dict:
    """Load IE relations from the same JSONL.  Returns ``{doc_id: [relations]}``."""
    rels_by_doc: dict[str, list] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            doc_id = _doc_id(rec)
            if not doc_id:
                continue
            rels_by_doc.setdefault(doc_id, []).extend(rec.get("relations") or [])
    return rels_by_doc


def load_ocr_texts(path: str, text_field: str | None = None) -> dict:
    """Load OCR texts from a JSONL file with ``bulletin_id`` (or ``id``) and ``text`` fields.

    Returns ``{doc_id: text}``.  By default the first non-empty of ``text``,
    ``raw_text``, ``content`` is used; pass ``text_field`` to pin one field —
    needed when the IE mention offsets were computed against a different
    variant than the default (e.g. the raw ``text_org`` instead of the cleaned
    ``text``), which would otherwise surface as mass ``ocr_mismatch`` statuses.
    If a document appears on several records, the FIRST text is kept (later
    texts would silently invalidate the mention offsets computed against the
    first).
    """
    texts: dict[str, str] = {}
    extra: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            doc_id = _doc_id(rec)
            if text_field:
                text = rec.get(text_field)
            else:
                text = rec.get("text") or rec.get("raw_text") or rec.get("content")
            if doc_id and text:
                if doc_id in texts:
                    extra.append(doc_id)
                else:
                    texts[doc_id] = text
    if extra:
        print(f"  [io] {len(extra)} duplicate OCR text record(s) ignored for: "
              f"{sorted(set(extra))[:10]}")
    return texts


def load_ie_texts(path: str) -> dict:
    """Collect the top-level ``text`` field of every IE record: ``{doc_id: text}``.

    The IE JSONL normally carries the full bulletin text next to ``entities`` /
    ``relations``.  Once that text has been replaced by the preprocessed bulletin
    text and the offsets corrected (``fix_ie_offsets.py``), these texts are
    exactly what the context stage needs — the clustering pipeline then runs in
    full (paragraph) mode WITHOUT a separate ``--ocr`` file.  Keys are the IE
    doc ids, so no id alignment is necessary; first record wins for duplicates.
    """
    texts: dict[str, str] = {}
    extra: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            doc_id = _doc_id(rec)
            text = rec.get("text") or rec.get("raw_text") or rec.get("content")
            if doc_id and text:
                if doc_id in texts:
                    if texts[doc_id] != text:
                        extra.append(doc_id)
                else:
                    texts[doc_id] = text
    if extra:
        print(f"  [io] {len(extra)} IE record(s) with a DIFFERENT text for an "
              f"already-seen bulletin ignored (chunked records?): "
              f"{sorted(set(extra))[:5]}")
    return texts


def load_dataset(ie_path: str, ocr_path: str | None = None,
                 kept_statuses=DEFAULT_KEPT_STATUSES,
                 ocr_text_field: str | None = None) -> tuple[dict, dict | None]:
    """Load mentions and bulletin texts.

    Text source, in order of precedence:

    1. ``--ocr`` file (optional override) — aligned to the IE document ids via
       :func:`align_ocr_texts` (exact id match first, then separator-insensitive
       matching, since the IE output and an external text file may format ids
       differently).  The mention offsets must have been computed against THIS
       text variant.
    2. the IE records' own top-level ``text`` field (the usual case: after
       ``fix_ie_offsets.py`` has replaced it with the preprocessed bulletin text
       and corrected the offsets, no separate file is needed).

    ``kept_statuses`` is forwarded to :func:`load_ie_mentions`;
    ``ocr_text_field`` to :func:`load_ocr_texts`.
    """
    mentions_by_doc = load_ie_mentions(ie_path, kept_statuses=kept_statuses)
    ocr_texts = None
    if ocr_path:
        ocr_texts = align_ocr_texts(
            mentions_by_doc, load_ocr_texts(ocr_path, text_field=ocr_text_field))
        if not ocr_texts:
            ocr_texts = None
    else:
        ie_texts = load_ie_texts(ie_path)
        if ie_texts:
            have = [d for d in mentions_by_doc if d in ie_texts]
            missing = [d for d in mentions_by_doc if d not in ie_texts]
            print(f"  [io] using the IE records' own 'text' field: {len(have)}/"
                  f"{len(mentions_by_doc)} bulletins have text (full mode, "
                  "no --ocr file needed)")
            if missing:
                print(f"  [io]   {len(missing)} bulletins without text on their IE "
                      f"record stay in simple mode, e.g. {missing[:3]}")
            ocr_texts = ie_texts
    return mentions_by_doc, ocr_texts


def write_json(data, path: str) -> None:
    """Write ``data`` as pretty JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def write_jsonl(records: list, path: str) -> None:
    """Write ``records`` as JSONL (one JSON object per line)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file; a truncated last line (crash mid-write) is ignored."""
    p, out = Path(path), []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def append_jsonl(fh, obj: dict) -> None:
    """Write one JSON line and flush so the file can be read while the script runs."""
    fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
    fh.flush()


