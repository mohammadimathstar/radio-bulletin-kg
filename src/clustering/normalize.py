"""
Surface normalization for entity names.

Strips titles, accents and punctuation so that ``"Dr. CHAMBERLAIN"`` and
``"Chamberlain"`` normalize to the same key.  The title table is shared with
:mod:`cues` so there is one source of truth.
"""
import re
import unicodedata

from .cues import TITLES

_TITLE_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in sorted(TITLES, key=len, reverse=True)) + r")\b\.?",
    re.IGNORECASE,
)
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_name(raw: str, strip_titles: bool = True, strip_accents: bool = True) -> str:
    """Normalize a surface form for clustering.

    1. Flatten whitespace (newlines, tabs).
    2. NFKD + strip combining marks (accents).
    3. Lowercase.
    4. Strip titles (``Dr.``, ``Prof.``, ``minister`` ...) when ``strip_titles``.
    5. Strip punctuation.
    6. Collapse remaining whitespace.
    """
    s = str(raw)
    s = s.replace("\\n", " ").replace("\n", " ").replace("\t", " ")
    s = unicodedata.normalize("NFKD", s)
    if strip_accents:
        s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    if strip_titles:
        s = _TITLE_RE.sub(" ", s)
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def norm_ws(s: str) -> str:
    """Collapse every whitespace run (newlines included) into one space."""
    return " ".join((s or "").split())
