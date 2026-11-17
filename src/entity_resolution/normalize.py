"""Surface normalization for entity names.

Strips leading titles, accents and punctuation so that ``"Dr. CHAMBERLAIN"`` and
``"Chamberlain"`` normalize to the same key.  The title table is shared with
:mod:`cues` so there is one source of truth.

Title stripping is ANCHORED to the start of the name: only titles at the very
beginning are removed (optionally preceded by name particles such as ``van``,
``de``, ``den``).  This is deliberate.  The shared TITLES table contains tokens
that are also frequent Dutch surnames — *koning, prins, graaf, heer, bisschop,
hertog, keizer* — so an unanchored strip would destroy them:

    "Minister DE GEER"   -> "de geer"      (title stripped, as intended)
    "dr. H. KONING"      -> "h koning"     (leading title only; surname survives)
    "Koning"             -> "koning"       (would be empty -> fallback)
    "De Koning"          -> "de koning"    (would be particles-only -> fallback)
    "De Graaf"           -> "de graaf"     (distinct from "de koning": no collision)

If stripping would leave nothing (or only particles), the unstripped form is
kept, so a mention can never normalize to an empty key by title removal.
"""
import re
import unicodedata

from ..utils.cues import TITLES

# Particles that may precede a title at the head of a name ("den heer COLIJN").
_HEAD_PARTICLES = ("van", "de", "der", "den", "von", "di", "du", "le", "la", "el",
                   "al", "ten", "ter", "te", "het", "'t", "'s", "d'")
_PARTICLES_SET = {p.strip("'") for p in _HEAD_PARTICLES}

_TITLE_PREFIX_RE = re.compile(
    r"^(?:(?:" + "|".join(re.escape(p) for p in _HEAD_PARTICLES) + r")\s+)*"
    r"(?:" + "|".join(re.escape(t) for t in sorted(TITLES, key=len, reverse=True)) + r")\b\.?\s*",
    re.IGNORECASE,
)
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_name(raw: str, strip_titles: bool = True, strip_accents: bool = True) -> str:
    """Normalize a surface form for clustering.

    1. Flatten whitespace (newlines, tabs).
    2. NFKD + strip combining marks (accents).
    3. Lowercase.
    4. Strip LEADING titles (``Dr.``, ``Prof.``, ``minister`` ...), each optionally
       preceded by particles, repeatedly — but only when what remains is non-empty
       and not particles-only (surname protection; see module docstring).
    5. Strip punctuation.
    6. Collapse remaining whitespace.
    """
    s = str(raw)
    s = s.replace("\\n", " ").replace("\n", " ").replace("\t", " ")
    s = unicodedata.normalize("NFKD", s)
    if strip_accents:
        s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    if strip_titles:
        base = s
        while True:
            m = _TITLE_PREFIX_RE.match(s)
            if not m:
                break
            s = s[m.end():]
        rest = s.split()
        # Surname protection: an all-titles name ("Koning", "De Koning") would be
        # emptied or reduced to particles -> keep the unstripped form instead.
        if not rest or all(t in _PARTICLES_SET for t in rest):
            s = base
        else:
            s = " ".join(rest)
    return s.strip()


def norm_ws(s: str) -> str:
    """Collapse every whitespace run (newlines included) into one space."""
    return " ".join((s or "").split())
