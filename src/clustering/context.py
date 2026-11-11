"""
Context extraction for entity mentions.

For each mention this adds:
  - ``text``         cleaned surface form (line breaks / edge punctuation removed)
  - ``context``      the sentence containing the mention (or a character window)
  - ``context_before`` / ``context_after``  local windows used for cue extraction
  - ``cues_local``   PERSON only: {nationality, title, role, occupation}

Two modes
---------
* **Full mode** (OCR text provided): flattens whitespace, segments sentences
  (spaCy if available, otherwise a regex splitter), repairs short sentences by
  merging neighbours, and builds character windows for cues.  Offsets stay valid
  because whitespace is replaced 1:1.
* **Simple mode** (no OCR text): derives ``context_before`` / ``context_after``
  from the IE output's ``context`` field by locating the mention text inside it.
  This is sufficient for cue extraction and needs no extra input file.
"""
import re
import unicodedata
from bisect import bisect_right

from .cues import mention_cues, cues_from_mention, occupation_from_subtype

# Whitespace -> " " one character for one character: offsets stay valid.
FLATTEN = str.maketrans({c: " " for c in "\n\r\t\x0b\x0c\u2028\u2029"})
_LEFT = " .,;:!?\"'()[]{}*_-\u2013\u2014\u00ab\u00bb"
_RIGHT = " ,;:!?\"'()[]{}*_-\u2013\u2014\u00ab\u00bb"

# Common abbreviations that should NOT trigger a sentence split.
_ABBREV = {
    "dr", "drs", "prof", "professor", "mr", "ir", "ing", "jhr", "vs",
    "min", "sec", "sts", "st", "etc", "jr", "sr", "no", "nr", "art",
    "evt", "bijl", "afb", "pag", "blz", "ca", "dd", "geb", " overl",
}


def norm_ws(s: str) -> str:
    """Collapse every whitespace run (newlines included) into one space."""
    return " ".join((s or "").split())


def clean_surface(raw: str) -> str:
    """Surface form without line breaks, double spaces or edge punctuation.

    ``".WESSELS"`` -> ``"WESSELS"``, ``"United\\nStates  Steel"`` -> ``"United States Steel"``.
    """
    s = norm_ws(raw).lstrip(_LEFT).rstrip(_RIGHT)
    while s.endswith(".") and len(s) > 1 and s[-2].islower():
        s = s[:-1].rstrip(_RIGHT)
    return s.strip()


# ----------------------------------------------------------------- sentence splitting

def regex_split_sentences(text: str) -> list[tuple[int, int]]:
    """Split ``text`` into (start, end) sentence spans.

    Splits at ``[.!?]+`` followed by whitespace + a capital letter, but skips
    common abbreviations (Dr., Prof., min., sec., ...).  Good enough for the
    ``pick_sentence`` repair; not a full parser.
    """
    spans = []
    start = 0
    for m in re.finditer(r"([.!?]+)\s+(?=[A-Z\u00c0-\u024F])", text):
        ws = text.rfind(" ", start, m.start(1))
        word = text[ws + 1:m.start(1)].lower().strip(".,;:()\"'")
        if word in _ABBREV:
            continue
        spans.append((start, m.start(1)))
        start = m.end()
    spans.append((start, len(text)))
    return [(s, e) for s, e in spans if s < e]


def get_sentence_splitter(model: str = "nl_core_news_lg"):
    """Return a callable ``f(text) -> [(start, end), ...]``.

    Uses spaCy if the model is installed, otherwise falls back to the regex splitter.
    """
    try:
        import spacy
        nlp = spacy.load(model, disable=["ner", "lemmatizer"])

        def split(text):
            return [(s.start_char, s.end_char) for s in nlp(text).sents]
        return split
    except (ImportError, OSError, ValueError):
        return regex_split_sentences


# ----------------------------------------------------------------- windows

def window(flat: str, a: int, b: int, left: int, right: int) -> tuple[int, int]:
    """Character window [a-left, b+right] snapped inwards to word boundaries."""
    n = len(flat)
    s, e = max(0, a - left), min(n, b + right)
    while 0 < s < a and not flat[s - 1].isspace():
        s += 1
    while b < e < n and not flat[e].isspace():
        e -= 1
    return s, e


def pick_sentence(spans, starts, n_chars, start, end, min_chars, max_chars):
    """Sentence span around a mention, repaired for bad segmentation.

    The sentence containing ``start`` is taken; if the mention runs past its end the
    following sentences are merged in; if the result is shorter than ``min_chars``
    neighbours are added as long as the total stays <= ``max_chars``.
    """
    if not spans:
        return max(0, start - min_chars), min(n_chars, end + min_chars), "window_fallback"
    lo = max(bisect_right(starts, start) - 1, 0)
    hi, status = lo, "segmented"
    while spans[hi][1] < end and hi + 1 < len(spans):
        hi += 1
        status = "merged"
    while spans[hi][1] - spans[lo][0] < min_chars and (lo > 0 or hi + 1 < len(spans)):
        grew = False
        if hi + 1 < len(spans) and spans[hi + 1][1] - spans[lo][0] <= max_chars:
            hi += 1
            grew = True
        if spans[hi][1] - spans[lo][0] >= min_chars:
            status = "expanded"
            break
        if lo > 0 and spans[hi][1] - spans[lo - 1][0] <= max_chars:
            lo -= 1
            grew = True
        if not grew:
            break
        status = "expanded"
    return min(spans[lo][0], start), max(spans[hi][1], end), status


# ----------------------------------------------------------------- local context from field

def derive_local_context(context: str, mention_text: str) -> tuple[str, str]:
    """Split the IE ``context`` field at the mention to get before / after text.

    Tries exact match, then case-insensitive, then first-token match.  Falls back to
    the whole context as ``before`` when the mention cannot be located.
    """
    ctx = context or ""
    mt = mention_text or ""
    if not mt:
        return ctx, ""
    idx = ctx.find(mt)
    if idx < 0:
        idx = ctx.lower().find(mt.lower())
    if idx < 0:
        first_tok = mt.split()[0] if mt.split() else mt
        idx = ctx.lower().find(first_tok.lower())
    if idx < 0:
        return ctx, ""
    return ctx[:idx], ctx[idx + len(mt):]


# ----------------------------------------------------------------- type normalization

_TYPE_MAP = {
    "person": "PERSON", "per": "PERSON",
    "organization": "ORGANIZATION", "org": "ORGANIZATION",
    "place": "PLACE", "loc": "PLACE", "location": "PLACE",
    "event": "EVENT",
}


def normalize_type(t: str) -> str:
    return _TYPE_MAP.get(str(t or "").lower().strip(), str(t or "").upper())


# ----------------------------------------------------------------- main entry point

def add_context(mentions_by_doc: dict, ocr_texts: dict | None, cfg, nlp=None) -> dict:
    """Add text / context / cue fields to every mention in ``mentions_by_doc``.

    When ``ocr_texts`` is provided and a document's text is available, full mode is
    used (sentence segmentation + character windows).  Otherwise simple mode derives
    ``context_before`` / ``context_after`` from the ``context`` field.
    ``nlp`` may be a callable ``f(text) -> [(start, end), ...]`` (from
    :func:`get_sentence_splitter`); if None a splitter is created automatically.
    """
    splitter = nlp or get_sentence_splitter()

    for doc_id, entities in mentions_by_doc.items():
        text = (ocr_texts or {}).get(doc_id)
        flat = None
        spans = []
        starts = []

        if text:
            flat = text.translate(FLATTEN)
            spans = splitter(flat)
            starts = [a for a, _ in spans]

        for e in entities:
            e["type"] = normalize_type(e.get("type"))
            start, end = e.get("start"), e.get("end")

            # --- surface form ---
            if flat is not None and start is not None and end is not None \
                    and 0 <= start < end <= len(text):
                raw = text[start:end]
                e["raw_text"] = raw
                e["text"] = clean_surface(raw)
            else:
                e["text"] = clean_surface(e.get("text", ""))

            if not e["text"]:
                e["context_status"] = "invalid"
                continue

            # --- context windows ---
            if flat is not None and start is not None and end is not None \
                    and 0 <= start < end <= len(flat):
                s, t_end, status = pick_sentence(spans, starts, len(flat), start, end,
                                                 cfg.min_sentence_chars, cfg.max_sentence_chars)
                e["sentence"] = norm_ws(flat[s:t_end])
                e["sentence_start"], e["sentence_end"] = s, t_end
                e["sentence_status"] = status

                cs, ce = window(flat, start, end, cfg.context_chars, cfg.context_chars)
                e["context"] = norm_ws(flat[cs:ce])

                bs, _ = window(flat, start, start, cfg.local_chars, 0)
                _, ae = window(flat, end, end, 0, cfg.local_chars)
                e["context_before"] = norm_ws(flat[bs:start])
                e["context_after"] = norm_ws(flat[end:ae])
                e["context_status"] = "resolved"
            else:
                e["context"] = e.get("context", "")
                before, after = derive_local_context(e["context"], e["text"])
                e["context_before"] = before
                e["context_after"] = after
                e["context_status"] = "from_context_field"

            # --- person cues ---
            if e["type"] == "PERSON":
                cues = mention_cues(e["context_before"], e["context_after"],
                                    cue_window=cfg.cue_window, flexible=cfg.flexible,
                                    nationality_strict=cfg.nationality_strict)
                # nationality from demonym inside the mention span (e.g. "den Bel HENDRICKX")
                if cues["nationality"] is None:
                    cues["nationality"] = cues_from_mention(e["text"])
                # occupation from the IE subtype (e.g. subtype="Athlete" -> "atleet")
                occ = occupation_from_subtype(e.get("subtype"))
                if occ and occ not in cues["occupation"]:
                    cues["occupation"] = [occ] + cues["occupation"]
                e["cues_local"] = cues

    return mentions_by_doc
