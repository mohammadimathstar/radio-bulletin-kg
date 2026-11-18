"""Type -> spec mapping: which Wikidata properties a profile contains.

The canonical upper-case type literals of the extraction stage
(``candidate_extraction.clusters.TYPE_MAP``): PERSON, ORGANIZATION, PLACE,
EVENT, LEGAL_DOCUMENT, VEHICLE.  Every literal a profile can be asked for
resolves through :data:`TYPE_ALIASES` onto one of these six
(:func:`normalize_type`), so the canonical names of the clustering
(``"Person"``, ``"Vehicle"``) and the historical fine-grained literals
(``"SHIP"``, ``"ARMED_CONFLICT"``) share one spec — and one stored profile
per (qid, type).  Pure data: the caps (max_values, max_aliases) and all other
tunables live in config.py.
"""

# type -> item-valued properties {field name: property} and date properties {field name: property}
PROFILE_SPEC = {
    "PERSON": dict(
        items={"occupations": "P106", "citizenship": "P27", "place_of_birth": "P19",
               "place_of_death": "P20", "member_of_party": "P102", "employer": "P108",
               "military_rank": "P410"},
        dates={"birth_date": "P569", "death_date": "P570"}),
    "ORGANIZATION": dict(
        items={"country": "P17", "headquarters": "P159", "industry": "P452",
               "chairperson": "P488", "parent_organization": "P749", "founded_by": "P112"},
        dates={"inception": "P571", "dissolved": "P576"}),
    "PLACE": dict(
        items={"country": "P17", "located_in": "P131"},
        dates={"inception": "P571", "dissolved": "P576"}),
    "VEHICLE": dict(
        items={"operator": "P137", "owner": "P127", "manufacturer": "P176",
               "country": "P17", "country_of_origin": "P495", "ship_class": "P289"},
        dates={"inception": "P571", "service_entry": "P729", "service_retirement": "P730",
               "first_flight": "P606"}),
    "EVENT": dict(
        items={"location": "P276", "participants": "P710", "part_of": "P361", "country": "P17"},
        dates={"point_in_time": "P585", "start": "P580", "end": "P582"}),
    "LEGAL_DOCUMENT": dict(
        items={"signatories": "P1891", "jurisdiction": "P1001", "country": "P17"},
        dates={"inception": "P571", "publication_date": "P577", "point_in_time": "P585"}),
}

# Historical / fine-grained literals -> canonical spec key (and stored type).
TYPE_ALIASES = {
    # canonical names of the new clustering (candidate_extraction.clusters.TYPE_MAP)
    "PERSON": "PERSON", "ORGANIZATION": "ORGANIZATION", "PLACE": "PLACE",
    "EVENT": "EVENT", "LEGAL_DOCUMENT": "LEGAL_DOCUMENT", "LEGALDOCUMENT": "LEGAL_DOCUMENT",
    "VEHICLE": "VEHICLE",
    # historical fine-grained literals of the old IE
    "SHIP": "VEHICLE", "SUBMARINE": "VEHICLE", "AIRCRAFT": "VEHICLE",
    "ARMED_CONFLICT": "EVENT", "DIPLOMATIC_MEETING": "EVENT",
}


def normalize_type(etype: str) -> str:
    """Any type literal (canonical 'Vehicle', linker 'VEHICLE', old 'SHIP') -> spec key."""
    return TYPE_ALIASES.get(str(etype or "").strip().upper(), str(etype or "").strip().upper())


def spec_for(etype: str) -> dict:
    """PROFILE_SPEC entry of a (possibly aliased) type literal; empty spec if unknown."""
    return PROFILE_SPEC.get(normalize_type(etype), dict(items={}, dates={}))
