"""decide with JEV (TypeSafe System One) — the model-specific part of the judge.

Ported from the uploaded decide_jev.py; changes:
  * pip install typesafe-sdk, export TYPESAFE_API_KEY=... — the SDK is
    imported LAZILY inside send(), so --dry-run and the offline tests work on
    machines without the SDK;
  * the old argparse constants are config fields (config.CONFIG.model);
  * option_style (config): "json" sends every candidate with its FULL stored
    profile dict (the default, as before — JEV takes up to 255 options of any
    length, so every new profile feature of every entity type flows through
    automatically), "text" sends candidate_profiles.render.profile_block()
    strings instead, which also append unknown/legacy profile fields;
  * any SDK failure is wrapped in RuntimeError, so the run loop's error
    handling stays model-agnostic.

    python -m candidate_judge --dry-run 3                # show what would be sent, no API call
    python -m candidate_judge --limit-docs 5 --out decisions_small.jsonl
    python -m candidate_judge                            # everything (rerun skips finished entities)
"""
from .config import CONFIG, build_parser
from .texts import squash

NONE_KEY = "NONE_OF_THE_ABOVE"

INSTRUCTIONS = (
    "Which Wikidata item does the marked mention [[...]] refer to? The mention is of type {type}. The text comes "
    "from Dutch radio news bulletins of 1939 (OCR errors, old spelling; names may be misspelled or incomplete). "
    "Use the text, the date, the nationality and titles next to the name and the other entities of the bulletin. "
    "Choose '{none}' if no listed item fits."
)

NONE_DESCRIPTION = ("None of the listed Wikidata items is the entity meant in the text: the correct item is "
                    "missing from the list, or the entity has no Wikidata item.")

#: mention cluster fields that go into the state, {cluster key: state key}
_MENTION_FIELDS = (("nationality", "nationality_next_to_name"),
                   ("title", "titles_next_to_name"),
                   ("role", "roles_next_to_name"))

_CLIENT = None                                            # created on the first send()


def _option_value(cand: dict, cfg) -> object:
    """Value of one JEV option: the full profile dict, its text block, or the
    {qid, label} fallback when no profile was collected for the candidate."""
    prof = cand.get("profile")
    if not prof:
        return {"qid": cand["qid"], "label": cand.get("label")}
    if cfg.option_style == "text":
        from .render import profile_block   # lazy: keeps judge imports light
        return profile_block(prof)
    return prof                                               # "json": as stored, new fields included


def build(e: dict, cfg=CONFIG) -> dict:
    """Payload for JEV: a JSON state and the options {qid: profile}."""
    c = e["cluster"]
    mention = {"entity_type": c["type"], "name_in_text": c["representative"],
               "spellings_in_text": list(c.get("surface_forms", {}))[:5]}
    for key, name in _MENTION_FIELDS:
        if c.get(key):
            mention[name] = squash(c[key]) if isinstance(c[key], str) else c[key]
    state = {"mention": mention, "bulletin_date": e["date"], "context_sentences": e["texts"],
             "other_entities_in_this_bulletin": e["known"]}
    options = {cand["qid"]: _option_value(cand, cfg) for cand in e["candidates"]}
    options[NONE_KEY] = NONE_DESCRIPTION
    return {"instructions": INSTRUCTIONS.format(type=c["type"], none=NONE_KEY),
            "state": state, "options": options}


def send(payload: dict, model: str) -> dict[str, float]:
    """One JEV system_one call -> {qid or NONE_KEY: probability}."""
    global _CLIENT
    try:
        from typesafe_sdk import Choice, TypeSafeClient       # lazy: dry-run needs no SDK
    except ImportError as ex:
        raise RuntimeError("typesafe-sdk is not installed: pip install typesafe-sdk "
                           "and export TYPESAFE_API_KEY=...") from ex
    if _CLIENT is None:
        _CLIENT = TypeSafeClient()
    question = Choice(instructions=payload["instructions"], criteria=payload["options"])
    try:
        resp = _CLIENT.system_one(state=payload["state"], questions={"pick": question}, model=model)
        return {k: float(v) for k, v in dict(resp.answers["pick"].probabilities).items()}
    except Exception as ex:                                   # SDK error types: wrap for the run loop
        raise RuntimeError(f"JEV call failed: {ex!r}") from ex


def main() -> None:
    """CLI: --config JSON + one auto flag per config field, then the shared run loop."""
    ap = build_parser("Judge the entities of every resolved document against the Wikidata "
                      "candidates of their corpus cluster, with JEV AI (TypeSafe System One)")
    a = ap.parse_args()
    if a.config:
        CONFIG.apply_json(a.config)
    CONFIG.apply_args(a)
    from . import collect                                     # lazy: keeps --help light
    collect.main(build, send, "jev")


if __name__ == "__main__":
    main()
