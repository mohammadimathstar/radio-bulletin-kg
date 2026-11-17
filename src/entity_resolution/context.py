"""Context extraction for entity mentions.

For each mention this adds:
  - ``text``          cleaned surface form (line breaks / edge punctuation removed)
  - ``sentence``      the sentence containing the mention, clamped to its paragraph
  - ``context``       the WHOLE PARAGRAPH containing the mention (full mode), or the
                      IE output's ``context`` field (simple mode)
  - ``paragraph_id``  index of the mention's paragraph within the bulletin (full mode)
  - ``cues_local``    PERSON only: {nationality, title, role, occupation}

The mention is the unit; the PARAGRAPH is the evidence unit.  The paragraph — not a
small character window — is stored as ``context`` because that is what the wikidata
linking stage reads for disambiguation: it carries the appositions, titles and
nationalities around the name.  Local before/after windows are still computed to
feed PERSON cue extraction, but only transiently: they are no longer written onto
the mention (in simple mode with an empty IE ``context`` field they were always
empty, and the paragraph supersedes them as evidence anyway).

Text structure
--------------
Bulletins arrive as OCR/plain text in which **paragraphs are separated by two
newlines** (``"\\n\\n"``).  Paragraph boundaries are respected everywhere:

  - sentences are segmented WITHIN a paragraph (never across a ``\\n\\n``);
  - ``sentence`` and ``context`` are clamped to the mention's paragraph, and the
    transient cue windows cannot reach into the previous or next paragraph.

Few-word paragraphs — datelines / paragraph titles such as ``"Soesterberg."`` —
are merged into the FOLLOWING paragraph first (:func:`merge_short_paragraphs`,
``cfg.short_para_max_words``, default 4 words); a few-word paragraph at the very
end merges into the previous one.  A mention's ``context`` is therefore never a
bare title, and ``paragraph_id`` does not inflate on title lines.

Offsets stay valid because whitespace is translated 1:1 (each ``\\n`` becomes
one space).

Two modes
---------
* **Full mode** (OCR text provided): flattens whitespace, splits into paragraphs
  (``\\n\\n``), segments sentences (spaCy if available, otherwise a regex
  splitter), repairs short sentences by merging neighbours, and stores the
  containing paragraph as ``context``.  The OCR slice is only trusted when it
  resembles the IE mention text (``ocr_min_similarity``); on mismatch the IE text
  and the IE ``context`` field are used instead (``context_status = "ocr_mismatch"``),
  because differing text means the stored offsets point at the wrong words.
* **Simple mode** (no OCR text): there is no text to extract a paragraph from, so
  ``context`` is the IE output's ``context`` field (possibly empty — supply OCR
  texts to get real paragraph context).  Person cues are still derived from local
  windows around the mention inside that field.

Type standardization
--------------------
``normalize_type`` delegates to :func:`.entity_types.canonical_type`, so every mention
type becomes a canonical capitalized form (``"PER"`` -> ``"Person"``) before
clustering.  The IE ``subtype`` fields are never used for entity resolution.
"""
import re
import unicodedata
from bisect import bisect_right

from rapidfuzz import fuzz

from ..utils.cues import mention_cues, cues_from_mention
from .entity_types import canonical_type

# Whitespace -> " " one character for one character: offsets stay valid.
FLATTEN = str.maketrans({c: " " for c in "\n\r\t\x0b\x0c\u2028\u2029"})
_LEFT = " .,;:!?\"'()[]{}*_-\u2013\u2014\u00ab\u00bb"
_RIGHT = " ,;:!?\"'()[]{}*_-\u2013\u2014\u00ab\u00bb"

# Common abbreviations that should NOT trigger a sentence split.
_ABBREV = {
    "dr", "drs", "prof", "professor", "mr", "ir", "ing", "jhr", "vs",
    "min", "sec", "sts", "st", "etc", "jr", "sr", "no", "nr", "art",
    "evt", "bijl", "afb", "pag", "blz", "ca", "dd", "geb", "overl",
}

# Paragraph separator: two or more consecutive newlines (the bulletin format
# separates paragraphs by "\n\n").  Offsets are identical in the flattened text.
PARAGRAPH_SEP_RE = re.compile(r"\n{2,}")


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


# ----------------------------------------------------------------- paragraphs

def paragraph_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) spans of the paragraphs of ``text``.

    Paragraphs are separated by two or more newlines (``"\\n\\n"``).  The spans
    are character offsets into the RAW text; because :data:`FLATTEN` translates
    every character 1:1, the same offsets are valid in the flattened text.
    """
    spans = []
    start = 0
    for m in PARAGRAPH_SEP_RE.finditer(text or ""):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text or "")))
    return [(s, e) for s, e in spans if e > s]


def merge_short_paragraphs(spans: list[tuple[int, int]], text: str,
                           max_words: int) -> list[tuple[int, int]]:
    """Merge few-word paragraphs (datelines / paragraph titles) into the next one.

    A paragraph with at most ``max_words`` whitespace-separated words — e.g. the
    kicker ``"Soesterberg."`` standing alone between two blank lines — is merged
    into the FOLLOWING paragraph; runs of consecutive short paragraphs chain into
    the first following paragraph.  A few-word paragraph at the very end has no
    next to absorb into, so it merges into the PREVIOUS paragraph instead.
    ``max_words <= 0`` disables the rule.  Offsets into ``text`` stay untouched.
    """
    if not spans or max_words is None or max_words <= 0:
        return list(spans)

    def wc(span: tuple[int, int]) -> int:
        return len(text[span[0]:span[1]].split())

    out: list[tuple[int, int]] = []
    i = 0
    while i < len(spans):
        start = spans[i][0]
        j = i
        while j < len(spans) - 1 and wc(spans[j]) <= max_words:
            j += 1
        out.append((start, spans[j][1]))
        i = j + 1
    # trailing short paragraph: no next paragraph -> absorb backward
    if len(out) > 1 and wc(out[-1]) <= max_words:
        out[-2] = (out[-2][0], out[-1][1])
        out.pop()
    return out


def _para_at(para_starts: list[int], para_spans: list[tuple[int, int]], pos: int) -> tuple[int, int]:
    """Paragraph (start, end) that contains character offset ``pos``."""
    i = bisect_right(para_starts, pos) - 1
    if i < 0:
        i = 0
    return para_spans[i]


def _clamp(span: tuple[int, int], bounds: tuple[int, int]) -> tuple[int, int]:
    """Clip a (start, end) span to the paragraph bounds."""
    return max(span[0], bounds[0]), min(span[1], bounds[1])


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


def sentence_spans(flat: str, paras: list[tuple[int, int]], splitter) -> list[tuple[int, int]]:
    """Sentence spans over the whole flattened text, segmented PER PARAGRAPH.

    Each paragraph is split independently and its sentence offsets are shifted
    back into global coordinates, so a sentence can never cross a ``\\n\\n``
    boundary.
    """
    spans: list[tuple[int, int]] = []
    for ps, pe in paras:
        for a, b in splitter(flat[ps:pe]):
            if b > a:
                spans.append((ps + a, ps + b))
    return sorted(spans)


# ----------------------------------------------------------------- windows

def window(flat: str, a: int, b: int, left: int, right: int) -> tuple[int, int]:
    """Character window [a-left, b+right] snapped inwards to word boundaries."""  # noqa: D401
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
    neighbours are added as long as the total stays <= ``max_chars``.  ``spans``
    are paragraph-local, so the repair can never cross a paragraph boundary.
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

def normalize_type(t: str) -> str | None:
    """Standardize a raw type label via :func:`.entity_types.canonical_type`.

    ``"PER"``/``"PERSON"`` -> ``"Person"``; ``"LOCATION"`` -> ``"Place"``;
    ``"LEGAL_DOCUMENT"`` -> ``"LegalDocument"``.  Unknown labels pass through.
    """
    return canonical_type(t)


# ----------------------------------------------------------------- main entry point

def add_context(mentions_by_doc: dict, ocr_texts: dict | None, cfg, nlp=None) -> dict:
    """Add text / context / cue fields to every mention in ``mentions_by_doc``.

    When ``ocr_texts`` is provided and a document's text is available, full mode is
    used (paragraph + sentence segmentation; ``context`` = the containing PARAGRAPH).
    Otherwise simple mode keeps the IE output's ``context`` field as-is (there is no
    text to extract a paragraph from) and derives transient local windows from it
    for person-cue extraction.

    ``nlp`` may be a callable ``f(text) -> [(start, end), ...]`` (from
    :func:`get_sentence_splitter`); if None a splitter is created automatically.

    Mentions get a canonical ``type`` (``"Person"``, ``"Place"``, ...).  The IE
    ``subtype`` field is intentionally left untouched and unused.
    """
    splitter = nlp or get_sentence_splitter()
    min_sim = getattr(cfg, "ocr_min_similarity", 80.0)

    for doc_id, entities in mentions_by_doc.items():
        text = (ocr_texts or {}).get(doc_id)
        flat = None
        paras: list[tuple[int, int]] = []
        para_starts: list[int] = []
        spans: list[tuple[int, int]] = []
        starts: list[int] = []

        if text:
            flat = text.translate(FLATTEN)
            paras = paragraph_spans(text)
            # datelines / kickers ("Soesterberg.") belong to the paragraph they
            # introduce: merge few-word paragraphs before anything else looks at them
            paras = merge_short_paragraphs(paras, text,
                                           getattr(cfg, "short_para_max_words", 4))
            para_starts = [p for p, _ in paras]
            spans = sentence_spans(flat, paras, splitter)
            starts = [a for a, _ in spans]

        for e in entities:
            e["type"] = canonical_type(e.get("type"))
            start, end = e.get("start"), e.get("end")
            offsets_ok = (flat is not None and start is not None and end is not None
                          and 0 <= start < end <= len(text))

            # --- surface form (with OCR consistency gate) ---
            ie_clean = norm_ws(e.get("text", "") or "")
            sliced = clean_surface(text[start:end]) if offsets_ok else None
            if offsets_ok and sliced:
                # Trust the OCR slice only when it looks like the text the IE
                # system processed; differing text means the stored offsets are
                # meaningless for this file.
                if not ie_clean or fuzz.ratio(sliced.lower(), ie_clean.lower()) >= min_sim:
                    e["raw_text"] = text[start:end]
                    e["text"] = sliced
                else:
                    e["ocr_slice"] = text[start:end]
                    e["context_status"] = "ocr_mismatch"
                    offsets_ok = False
            elif offsets_ok:
                e["raw_text"] = text[start:end]
                e["text"] = sliced

            if not e["text"]:
                e["text"] = clean_surface(e.get("text", ""))
            if not e["text"]:
                e["context_status"] = "invalid"
                continue

            # --- context: the containing paragraph (+ transient cue windows) ---
            if offsets_ok and start is not None and 0 <= start < end <= len(flat):
                bounds = _para_at(para_starts, paras, start)
                s, t_end, status = pick_sentence(spans, starts, len(flat), start, end,
                                                 cfg.min_sentence_chars, cfg.max_sentence_chars)
                s, t_end = _clamp((s, t_end), bounds)
                e["sentence"] = norm_ws(flat[s:t_end])
                e["sentence_start"], e["sentence_end"] = s, t_end
                e["sentence_status"] = status

                # the whole paragraph containing the mention is the evidence unit
                # the wikidata linker reads (bounds IS the mention's paragraph, so
                # a ``\n\n`` boundary can never be crossed)
                e["context"] = norm_ws(flat[bounds[0]:bounds[1]])
                e["paragraph_id"] = max(bisect_right(para_starts, start) - 1, 0)

                # transient local windows: used for PERSON cue extraction below,
                # deliberately NOT stored on the mention
                bs, _ = _clamp(window(flat, start, start, cfg.local_chars, 0), bounds)
                _, ae = _clamp(window(flat, end, end, 0, cfg.local_chars), bounds)
                before = norm_ws(flat[bs:start])
                after = norm_ws(flat[end:ae])
                if "context_status" not in e:
                    e["context_status"] = "resolved"
            else:
                if "context_status" not in e:
                    e["context_status"] = "from_context_field"
                e["context"] = e.get("context", "")
                before, after = derive_local_context(e["context"], e["text"])

            # --- person cues (text-derived only; the IE subtype is NOT used) ---
            if e["type"] == "Person":
                cues = mention_cues(before, after,
                                    cue_window=cfg.cue_window, flexible=cfg.flexible,
                                    nationality_strict=cfg.nationality_strict)
                # nationality from demonym inside the mention span (e.g. "den Bel HENDRICKX")
                if cues["nationality"] is None:
                    cues["nationality"] = cues_from_mention(e["text"])
                e["cues_local"] = cues

    return mentions_by_doc