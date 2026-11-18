"""profile_block(): the text view of a profile for the judge prompt.

The judge step reads profiles + mention context and picks a candidate.
profile_block() renders one profile deterministically: first line
"label (qid) - description", then 'aka: ...' when there are aliases or
other-language labels, then the type-specific facts in a fixed order
(BLOCK_FIELDS, with readable names: "positions:", "born:", "operator:"),
then sitelinks and the Wikipedia extract.  Empty fields and sections are
omitted; unknown/legacy fields are appended, so a profile stored by an older
version still renders completely.
"""

from ..candidate_profiles import config

#: render order per type: (profile key, display name)
BLOCK_FIELDS = {
    "PERSON": [("instance_of", "instance of"), ("occupations", "occupations"),
               ("citizenship", "citizenship"), ("birth_date", "born"),
               ("place_of_birth", "born in"), ("death_date", "died"),
               ("place_of_death", "died in"), ("positions_held", "positions"),
               ("member_of_party", "party"), ("employer", "employer"),
               ("military_rank", "military rank")],
    "ORGANIZATION": [("instance_of", "instance of"), ("industry", "industry"),
                     ("country", "country"), ("headquarters", "headquarters"),
                     ("chairperson", "chairperson"), ("parent_organization", "parent"),
                     ("founded_by", "founded by"), ("inception", "founded"),
                     ("dissolved", "dissolved")],
    "PLACE": [("instance_of", "instance of"), ("country", "country"),
              ("located_in", "located in"), ("population", "population"),
              ("coordinates", "coordinates"), ("inception", "founded"),
              ("dissolved", "dissolved")],
    "VEHICLE": [("instance_of", "instance of"), ("operator", "operator"), ("owner", "owner"),
                ("manufacturer", "manufacturer"), ("country", "country"),
                ("country_of_origin", "country of origin"), ("ship_class", "ship class"),
                ("inception", "built"), ("service_entry", "in service from"),
                ("service_retirement", "in service until"), ("first_flight", "first flight")],
    "EVENT": [("instance_of", "instance of"), ("location", "location"), ("country", "country"),
              ("participants", "participants"), ("part_of", "part of"),
              ("point_in_time", "date"), ("start", "start"), ("end", "end")],
    "LEGAL_DOCUMENT": [("instance_of", "instance of"), ("signatories", "signatories"),
                       ("jurisdiction", "jurisdiction"), ("country", "country"),
                       ("inception", "adopted"), ("publication_date", "published"),
                       ("point_in_time", "date")],
}


def _block_val(v) -> str:
    return ", ".join(v) if isinstance(v, list) else str(v)


def profile_block(prof: dict) -> str:
    """Deterministic text rendering of one profile for an LLM judge prompt."""
    lines = [f"{prof.get('label', prof.get('qid', '?'))} ({prof.get('qid', '?')})"
             + (f" - {prof['description']}" if prof.get("description") else "")]
    aka = list(prof.get("aliases") or []) + list(prof.get("label_other_languages") or [])
    if aka:
        lines.append("aka: " + ", ".join(aka[:config.CONFIG.max_aliases + 4]))
    fields = BLOCK_FIELDS.get(prof.get("type", ""), [])
    seen = {"qid", "type", "label", "description", "label_other_languages", "aliases",
            "sitelinks", "wikipedia_summary"}
    for key, disp in fields:
        if prof.get(key):
            lines.append(f"{disp}: {_block_val(prof[key])}")
        seen.add(key)
    for key, v in prof.items():                    # legacy/unknown fields: keep the information
        if key not in seen and key not in ("description_nl",):
            lines.append(f"{key}: {_block_val(v)}")
    if prof.get("description_nl"):
        lines.append(f"desc (nl): {prof['description_nl']}")
    if prof.get("sitelinks"):
        lines.append(f"sitelinks: {prof['sitelinks']}")
    if prof.get("wikipedia_summary"):
        lines.append(f"Wikipedia: \"{prof['wikipedia_summary']}\"")
    return "\n".join(lines)
