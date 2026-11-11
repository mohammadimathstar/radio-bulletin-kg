"""
Person cues: titles, roles, occupations and nationality next to a person's name
in Dutch 1939 news text.

Two improvements over the original person_cues.py:

1. WINDOW-BASED EXTRACTION.  Instead of requiring cues to be in a contiguous run
   immediately adjacent to the name, the scanner collects cues within a configurable
   token window (``cue_window``, default 8).  Filler particles (van, de, der, ...) are
   always skipped.  In flexible mode (default) non-cue content words are also skipped,
   so ``"minister van Onderwijs Prof. VAN MOERKERKEN"`` yields both ``minister`` and
   ``prof``.  In strict mode (``flexible=False``) a content word breaks the scan.

   Nationality has an independent guard: ``nationality_strict`` (default True) rejects a
   nationality adjective found *beyond* a content word, so ``"Fransche regeering.
   Generaal SMITH"`` does not attach ``French`` to Smith even in flexible mode.  Set
   ``nationality_strict=False`` to allow it.

2. OCCUPATION DICTIONARY.  A broad ``OCCUPATIONS`` table covers sports (wielrenner,
   voetballer, schaker ...), arts (schilder, beeldhouwer ...), science, trade and more,
   mapped to English labels for Wikidata search queries.  Occupations appear in a new
   ``occupation`` field alongside ``title`` and ``role``.

Additionally, ``cues_from_mention(text)`` extracts a nationality from demonyms embedded
IN the mention span (``"den Bel HENDRICKX"`` -> ``belgian``), and
``occupation_from_subtype(subtype)`` maps the IE system's subtype (``Athlete``) to an
occupation cue.

    from radio_bulletins_kg.clustering.cues import mention_cues
    mention_cues("de britsche ambassadeur", ", die vandaag")
    # {"nationality": "british", "title": [], "role": ["ambassadeur"], "occupation": []}

Requirements: rapidfuzz.
"""
import re
import unicodedata

from rapidfuzz import fuzz, process

# ------------------------------------------------------------------ titles
# normalised title (lowercase, no diacritics/dots) -> (kind, English keywords).
# kinds: noble/church/military -> used for search queries AND for the `title` feature;
#        role -> only for the `title` feature (a query "Minister Chamberlain" does not help);
#        honorific -> consumed (so we can look further back) but otherwise ignored.
TITLES: dict[str, tuple[str, list[str]]] = {}


def _add(kind, keywords, *names):
    for n in names:
        TITLES[n] = (kind, keywords)


_add("noble", ["lord", "baron"], "lord")
_add("noble", ["sir", "knight"], "sir")
_add("noble", ["lady"], "lady")
_add("noble", ["prince", "prins"], "prins", "prince")
_add("noble", ["princess", "prinses"], "prinses", "princesse", "princess")
_add("noble", ["king", "koning"], "koning", "roi", "king")
_add("noble", ["queen", "koningin"], "koningin", "reine", "queen")
_add("noble", ["emperor", "keizer"], "keizer", "empereur", "kaiser")
_add("noble", ["empress", "keizerin"], "keizerin")
_add("noble", ["tsar", "czar"], "tsaar")
_add("noble", ["duke", "hertog"], "hertog", "duc", "duke")
_add("noble", ["duchess", "hertogin"], "hertogin")
_add("noble", ["count", "earl", "graaf"], "graaf", "comte", "count", "earl")
_add("noble", ["countess", "gravin"], "gravin")
_add("noble", ["baron"], "baron")
_add("noble", ["jonkheer"], "jonkheer")
_add("noble", ["marquess", "marquis", "markies"], "markies", "marquis")

_add("church", ["pope", "paus"], "paus")
_add("church", ["cardinal", "kardinaal"], "kardinaal")
_add("church", ["archbishop", "aartsbisschop"], "aartsbisschop")
_add("church", ["bishop", "bisschop"], "bisschop")

_add("military", ["general"], "generaal", "general", "generaal-majoor", "luitenant-generaal",
     "brigadegeneraal", "generaloberst")
_add("military", ["admiral"], "admiraal", "amiral", "admiral", "vice-admiraal", "schout-bij-nacht")
_add("military", ["colonel"], "kolonel", "colonel")
_add("military", ["major"], "majoor")
_add("military", ["captain"], "kapitein", "captain")
_add("military", ["lieutenant"], "luitenant")
_add("military", ["marshal"], "maarschalk", "marechal", "veldmaarschalk", "feldmarschall", "marshal")

_add("role", ["minister"], "minister")
_add("role", ["prime minister", "minister-president"], "ministerpresident", "minister-president",
     "premier", "eerste minister")
_add("role", ["president"], "president", "voorzitter")
_add("role", ["chancellor"], "kanselier", "rijkskanselier", "reichskanzler")
_add("role", ["mayor"], "burgemeester")
_add("role", ["governor"], "gouverneur", "gouverneur-generaal")
_add("role", ["ambassador", "envoy", "minister"], "ambassadeur", "gezant")
_add("role", ["professor"], "professor", "prof")
_add("role", ["senator"], "senator")
_add("role", ["secretary"], "staatssecretaris")
_add("honorific", [], "heer", "mevrouw", "mevr", "mej", "mejuffrouw", "dr", "drs", "ir", "ing",
     "mr", "jhr", "meester", "herr")

SEARCH_KINDS = {"noble", "church", "military"}
_FUZZY_TITLE_KEYS = [k for k in TITLES if len(k) >= 6 and " " not in k]


# ------------------------------------------------------------------ occupations
# Dutch occupation term -> English label (for Wikidata search queries).
# QID mapping is done in the linking stage's type_mapping, not here.
OCCUPATIONS: dict[str, str] = {
    # --- sports ---
    "wielrenner": "cyclist",
    "wielrijder": "cyclist",
    "coureur": "racing cyclist",
    "voetballer": "footballer",
    "schaker": "chess player",
    "schaatser": "speed skater",
    "schaatsenrijder": "speed skater",
    "zwemmer": "swimmer",
    "bokser": "boxer",
    "atleet": "athlete",
    "tennisser": "tennis player",
    "jockey": "jockey",
    "renner": "runner",
    # --- arts ---
    "kunstenaar": "artist",
    "schilder": "painter",
    "beeldhouwer": "sculptor",
    "architect": "architect",
    "musicus": "musician",
    "componist": "composer",
    "dirigent": "conductor",
    "zanger": "singer",
    "zangeres": "singer",
    "acteur": "actor",
    "actrice": "actress",
    "schrijver": "writer",
    "dichter": "poet",
    "journalist": "journalist",
    "fotograaf": "photographer",
    # --- academia / science ---
    "wetenschapper": "scientist",
    "geleerde": "scholar",
    "arts": "physician",
    "geneesheer": "physician",
    "dokter": "physician",
    "historicus": "historian",
    "wiskundige": "mathematician",
    "natuurkundige": "physicist",
    "scheikundige": "chemist",
    # --- business / trade ---
    "directeur": "director",
    "koopman": "merchant",
    "winkelier": "shopkeeper",
    "ondernemer": "entrepreneur",
    "bankier": "banker",
    "fabrikant": "manufacturer",
    # --- agriculture / trades ---
    "boer": "farmer",
    "visser": "fisherman",
    "timmerman": "carpenter",
    "smid": "blacksmith",
    "bakker": "baker",
    "slager": "butcher",
    "metselaar": "bricklayer",
    # --- transport ---
    "zeeman": "sailor",
    "matroos": "sailor",
    "machinist": "engine driver",
    "vlieger": "aviator",
    "piloot": "pilot",
}
_FUZZY_OCC_KEYS = [k for k in OCCUPATIONS if len(k) >= 6]


# IE subtype -> occupation key (for cues derived from the IE classifier, not from text).
SUBTYPE_TO_OCCUPATION: dict[str, str] = {
    "Athlete": "atleet",
    "Journalist": "journalist",
    "Businessperson": "ondernemer",
}

# Nationality demonyms that can appear inside the mention span
# ("den Bel HENDRICKX" -> belgian, "den Franschman CALLIEN" -> french).
NAT_NOUNS: dict[str, str] = {
    "bel": "belgian", "belg": "belgian", "belge": "belgian",
    "franschman": "french", "francais": "french", "fransman": "french",
    "engelschman": "british", "englishman": "british", "brit": "british",
    "duitscher": "german", "duitsman": "german", "deutscher": "german",
    "hollander": "dutch", "nederlander": "dutch",
    "amerikaan": "american",
    "italiaan": "italian",
    "rus": "russian",
    "japanner": "japanese",
    "pool": "polish",
    "china": "chinese", "chinees": "chinese",
    "zweed": "swedish",
    "dane": "danish", "deensche": "danish",
    "noor": "norwegian",
    "spaansche": "spanish", "spanjaard": "spanish",
    "oostenrijker": "austrian",
    "tsjech": "czech",
    "hongaarse": "hungarian", "hongaar": "hungarian",
    "griek": "greek",
    "turk": "turkish",
}

# Particles skipped during the cue scan without breaking the chain.
FILLERS = {
    # articles
    "de", "den", "het", "der", "des", "een",
    # demonstratives / possessives
    "die", "deze", "dezen", "zijn",
    # prepositions
    "van", "te", "ten", "ter", "tot", "uit", "voor", "als",
    "door", "met", "bij", "aan", "in", "op", "tegen", "en",
}


# ------------------------------------------------------------------ token helpers

def norm_tok(t: str) -> str:
    """Lowercase a token, strip diacritics and punctuation (hyphens are kept)."""
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return re.sub(r"[.,:;'\"()]", "", t.lower())


def match_title(tok: str):
    """Return the normalised TITLES key matching `tok`, or None.

    Exact match first, then the head of a hyphenated compound ("generaal-majoor" ->
    "generaal"), then an OCR-tolerant fuzzy match for tokens of >= 6 characters.
    """
    t = norm_tok(tok)
    if t in TITLES:
        return t
    if "-" in t and t.split("-")[0] in TITLES:
        return t.split("-")[0]
    if len(t) >= 6:
        m = process.extractOne(t, _FUZZY_TITLE_KEYS, scorer=fuzz.ratio, score_cutoff=90)
        if m:
            return m[0]
    return None


def match_occupation(tok: str):
    """Return the normalised OCCUPATIONS key matching `tok`, or None (fuzzy for >= 6 chars)."""
    t = norm_tok(tok)
    if t in OCCUPATIONS:
        return t
    if len(t) >= 6:
        m = process.extractOne(t, _FUZZY_OCC_KEYS, scorer=fuzz.ratio, score_cutoff=90)
        if m:
            return m[0]
    return None


# ------------------------------------------------------------------ nationalities
# group -> (English demonym for queries, keywords matched at the START of a word in candidate
# descriptions [English/Dutch/German/French stems], Dutch 1939 adjectives).
NAT_GROUPS: dict[str, tuple[str, list[str], list[str]]] = {
    "british": ("British", ["brit", "english", "engels", "englisch", "anglais", "scot", "schots", "welsh",
                            "united kingdom", "verenigd koninkrijk", "england", "engeland", "groot-brittanni"],
                ["britsche", "engelsche", "schotsche"]),
    "french": ("French", ["french", "frans", "fran\u00e7ais", "france", "frankrijk"], ["fransche"]),
    "german": ("German", ["german", "duits", "deutsch", "allemand", "duitsland", "prussian", "pruisisch"],
               ["duitsche"]),
    "dutch": ("Dutch", ["dutch", "nederlands", "niederl", "n\u00e9erlandais", "netherlands", "nederland", "holland"],
              ["nederlandsche", "hollandsche"]),
    "belgian": ("Belgian", ["belgi", "belge"], ["belgische"]),
    "italian": ("Italian", ["ital"], ["italiaansche"]),
    "american": ("American", ["american", "amerikaan", "united states", "verenigde staten", "\u00e9tats-unis"],
                 ["amerikaansche"]),
    "russian": ("Russian", ["russ", "soviet", "sovjet", "rusland", "ussr"], ["russische"]),
    "japanese": ("Japanese", ["japan"], ["japansche"]),
    "polish": ("Polish", ["polish", "pools", "polnisch", "polonais", "poland", "polen", "pologne"], ["poolsche"]),
    "chinese": ("Chinese", ["chin"], ["chineesche"]),
    "swedish": ("Swedish", ["swed", "zweed", "zweden", "schwed", "su\u00e9d"], ["zweedsche"]),
    "danish": ("Danish", ["danish", "deens", "d\u00e4nisch", "danois", "denmark", "denemarken"], ["deensche"]),
    "norwegian": ("Norwegian", ["norw", "noors", "noorwegen", "norv\u00e9g"], ["noorsche"]),
    "spanish": ("Spanish", ["spanish", "spaans", "spanisch", "espagnol", "spain", "spanje", "spanien"], ["spaansche"]),
    "swiss": ("Swiss", ["swiss", "zwitser", "schweiz", "suisse"], ["zwitsersche"]),
    "austrian": ("Austrian", ["austria", "oostenrijk", "\u00f6sterreich", "autrich"], ["oostenrijksche"]),
    "czech": ("Czech", ["czech", "tsjech", "tschech", "tch\u00e8que", "bohem"], ["tsjechische", "tsjechoslowaaksche"]),
    "hungarian": ("Hungarian", ["hungar", "hongaar", "hongarije", "ungar", "hongrois"], ["hongaarsche"]),
    "romanian": ("Romanian", ["romanian", "roemeen", "rum\u00e4n", "roumain", "romania"], ["roemeensche"]),
    "yugoslav": ("Yugoslav", ["yugoslav", "joegoslav", "jugoslaw", "yougoslave", "serbi"], ["joegoslavische"]),
    "greek": ("Greek", ["greek", "grieks", "griechisch", "grec", "greece", "griekenland"], ["grieksche"]),
    "turkish": ("Turkish", ["turk", "ottoman", "turquie"], ["turksche"]),
    "finnish": ("Finnish", ["finn", "fins", "finland", "finnois"], ["finsche"]),
    "portuguese": ("Portuguese", ["portug"], ["portugeesche"]),
    "irish": ("Irish", ["irish", "iers", "irland", "ierland", "irlandais", "ireland"], ["iersche"]),
    "canadian": ("Canadian", ["canad"], ["canadeesche"]),
    "australian": ("Australian", ["austral"], ["australische"]),
}
_ADJ_END = re.compile(r"(sche|schen|sch|sen|se)$")
_NAT_STEMS = {_ADJ_END.sub("", norm_tok(a)): g for g, (_, _, adjs) in NAT_GROUPS.items() for a in adjs}


def match_nationality(tok: str):
    """Nationality group of a Dutch adjective ("Britsche" -> "british"), or None.

    The adjective ending (-sche/-schen/-sch/-se/-sen) is stripped and the stem looked up;
    stems of >= 5 characters are also matched fuzzily to survive OCR errors.
    """
    t = norm_tok(tok)
    if len(t) < 5 or not _ADJ_END.search(t):
        return None
    stem = _ADJ_END.sub("", t)
    if stem in _NAT_STEMS:
        return _NAT_STEMS[stem]
    if len(stem) >= 5:
        m = process.extractOne(stem, list(_NAT_STEMS), scorer=fuzz.ratio, score_cutoff=88)
        if m:
            return _NAT_STEMS[m[0]]
    return None


# ------------------------------------------------------------------ cue scanning

_TOK_RE = re.compile(r"[^\s,;:()\"]+")


def _classify_cue(tok: str):
    """Classify a single token as ('filler',), ('occupation', key), ('title', key),
    ('nationality', group), or ('other',).  Bigrams are handled by the caller."""
    nt = norm_tok(tok)
    if nt in FILLERS:
        return ("filler",)
    occ = match_occupation(tok)
    if occ is not None:
        return ("occupation", occ)
    t = match_title(tok)
    if t is not None:
        return ("title", t)
    n = match_nationality(tok)
    if n is not None:
        return ("nationality", n)
    return ("other",)


def _record_cue(kind, key, title, role, occupation):
    if kind == "occupation":
        occupation.append(key)
    elif kind == "title":
        title_or_role = TITLES[key][0]
        if title_or_role == "role":
            role.append(key)
        else:
            title.append(key)


def _scan_backward(toks: list[str], cue_window: int, flexible: bool, nationality_strict: bool) -> dict:
    """Backward scan of the last ``cue_window`` tokens before the name.

    Collects titles, roles and occupations; records the closest nationality to the name.
    Fillers are always skipped.  Content words break the scan in strict mode, or are
    skipped (setting ``skipped_content``) in flexible mode.  If ``nationality_strict``
    is True, a nationality found after a content-word skip is rejected.
    """
    start_i = max(0, len(toks) - cue_window)
    title, role, occupation = [], [], []
    nat = None
    skipped_content = False
    i = len(toks)
    while i > start_i:
        if i >= 2 and i - 2 >= start_i:
            bg = f"{norm_tok(toks[i - 2])} {norm_tok(toks[i - 1])}"
            if bg in TITLES:
                _record_cue("title", bg, title, role, occupation)
                i -= 2
                continue
        kind, *rest = _classify_cue(toks[i - 1])
        if kind == "filler":
            i -= 1
            continue
        if kind == "occupation":
            _record_cue(kind, rest[0], title, role, occupation)
            i -= 1
            continue
        if kind == "title":
            _record_cue(kind, rest[0], title, role, occupation)
            i -= 1
            continue
        if kind == "nationality":
            if nat is None and not (nationality_strict and skipped_content):
                nat = rest[0]
            i -= 1
            continue
        # other: content word
        if flexible:
            skipped_content = True
            i -= 1
            continue
        break
    return dict(title=title[::-1], role=role[::-1], occupation=occupation[::-1], nationality=nat)


def _scan_after(after: str, cue_window: int, flexible: bool, nationality_strict: bool) -> dict:
    """Forward scan of a comma-apposition after the name.

    Only fires when the text after the name starts with a comma.  Accepts the result
    only when at least one title/role/occupation cue was found (otherwise
    ``", de Engelsche regeering"`` would be misread).
    """
    a = (after or "").lstrip()
    if not a.startswith(","):
        return dict(title=[], role=[], occupation=[], nationality=None)
    toks = _TOK_RE.findall(a[1:])[:cue_window]
    title, role, occupation = [], [], []
    nat = None
    skipped_content = False
    i = 0
    while i < len(toks):
        if i + 1 < len(toks):
            bg = f"{norm_tok(toks[i])} {norm_tok(toks[i + 1])}"
            if bg in TITLES:
                _record_cue("title", bg, title, role, occupation)
                i += 2
                continue
        kind, *rest = _classify_cue(toks[i])
        if kind == "filler":
            i += 1
            continue
        if kind in ("occupation", "title"):
            _record_cue(kind, rest[0], title, role, occupation)
            i += 1
            continue
        if kind == "nationality":
            if nat is None and not (nationality_strict and skipped_content):
                nat = rest[0]
            i += 1
            continue
        if flexible:
            skipped_content = True
            i += 1
            continue
        break
    if not title and not role and not occupation:
        nat = None
    return dict(title=title, role=role, occupation=occupation, nationality=nat)


def mention_cues(before: str, after: str, cue_window: int = 8,
                 flexible: bool = True, nationality_strict: bool = True) -> dict:
    """Cues of ONE mention: ``{"nationality": str|None, "title": [..], "role": [..], "occupation": [..]}``.

    ``before`` / ``after`` are the texts immediately before / after the mention.  Titles,
    roles and occupations are normalised keys in reading order (before-then-after),
    without duplicates.  Nationality is the closest adjective to the name (before takes
    priority over the comma-apposition).

    Parameters
    ----------
    cue_window : int
        How many tokens before (and after the comma) to scan.  Default 8.
    flexible : bool
        If True (default), non-cue content words are skipped, allowing cues separated by
        org names or portfolios (``"minister van Onderwijs Prof. X"``) to be found.  If
        False, a content word breaks the scan (original strict behaviour).
    nationality_strict : bool
        If True (default), a nationality found beyond a content-word skip is rejected,
        preventing ``"Fransche regeering. Generaal SMITH"`` from attaching ``French``.
    """
    before_toks = _TOK_RE.findall(before or "")
    b = _scan_backward(before_toks, cue_window, flexible, nationality_strict)
    a = _scan_after(after or "", cue_window, flexible, nationality_strict)
    ordered = list(dict.fromkeys(b["title"] + a["title"]))
    ordered_role = list(dict.fromkeys(b["role"] + a["role"]))
    ordered_occ = list(dict.fromkeys(b["occupation"] + a["occupation"]))
    return dict(
        nationality=b["nationality"] or a["nationality"],
        title=[t for t in ordered],
        role=ordered_role,
        occupation=ordered_occ,
    )


def person_cues(before: str, after: str, cue_window: int = 8,
                flexible: bool = True, nationality_strict: bool = True):
    """Backward-compatible wrapper: returns (set of title keys, nationality group)."""
    c = mention_cues(before, after, cue_window, flexible, nationality_strict)
    return set(c["title"]) | set(c["role"]), c["nationality"]


def cues_from_mention(text: str) -> str | None:
    """Nationality group embedded IN the mention span, or None.

    Catches demonyms that are part of the mention itself: ``"den Bel HENDRICKX"`` ->
    ``belgian``, ``"den Franschman CALLIEN"`` -> ``french``.
    """
    for tok in _TOK_RE.findall(text or ""):
        nt = norm_tok(tok)
        if nt in NAT_NOUNS:
            return NAT_NOUNS[nt]
        n = match_nationality(tok)
        if n:
            return n
    return None


def occupation_from_subtype(subtype: str) -> str | None:
    """Map an IE subtype (``Athlete``) to an occupation key, or None."""
    if not subtype:
        return None
    return SUBTYPE_TO_OCCUPATION.get(subtype)
