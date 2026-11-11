"""
I/O for the clustering pipeline.

Reads the IE JSONL produced by the ``radio-bulletins-ie`` repository (one record per
document, with ``entities`` and ``relations`` lists) and optionally an OCR-text file.

The IE schema uses ``bulletin_id`` as the document identifier and marks every entity
with a ``status`` field (``"kept"``, ``"removed"``, ...).  Both are handled here:
``bulletin_id`` is recognised as the doc id (with ``id`` / ``doc_id`` as fallbacks for
older schemas), and only entities whose ``status`` is in ``kept_statuses`` are loaded
(default ``{"kept"}``; pass ``None`` to load every entity regardless of status).

Entity types may be any case (``"Person"``, ``"PLACE"``, ``"event"`` ...) — they are
normalised to upper-case later by :func:`context.normalize_type`.
"""
import json
from pathlib import Path

# Default: only entities the IE system explicitly kept.  Pass None to load all.
DEFAULT_KEPT_STATUSES = frozenset({"kept"})


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
    """
    statuses = frozenset(kept_statuses) if kept_statuses is not None else None
    mentions_by_doc: dict[str, list] = {}
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
            mentions_by_doc[doc_id] = entities
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
            rels_by_doc[doc_id] = rec.get("relations") or []
    return rels_by_doc


def load_ocr_texts(path: str) -> dict:
    """Load OCR texts from a JSONL file with ``bulletin_id`` (or ``id``) and ``text`` fields.

    Returns ``{doc_id: text}``.
    """
    texts: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            doc_id = _doc_id(rec)
            text = rec.get("text") or rec.get("raw_text") or rec.get("content")
            if doc_id and text:
                texts[doc_id] = text
    return texts


def load_dataset(ie_path: str, ocr_path: str | None = None,
                 kept_statuses=DEFAULT_KEPT_STATUSES) -> tuple[dict, dict | None]:
    """Load mentions and (optionally) OCR texts.

    If ``ocr_path`` is None but the IE JSONL records carry a top-level ``text`` field,
    those texts are used as OCR texts (so no separate file is needed).

    ``kept_statuses`` is forwarded to :func:`load_ie_mentions`.
    """
    mentions_by_doc = load_ie_mentions(ie_path, kept_statuses=kept_statuses)
    ocr_texts = None
    if ocr_path:
        ocr_texts = load_ocr_texts(ocr_path)
    else:
        # check whether the IE JSONL itself carries the full text
        with open(ie_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                doc_id = _doc_id(rec)
                text = rec.get("text") or rec.get("raw_text") or rec.get("content")
                if doc_id and text:
                    if ocr_texts is None:
                        ocr_texts = {}
                    ocr_texts[doc_id] = text
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