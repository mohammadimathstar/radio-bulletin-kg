"""Candidate profiles: compact, type-specific facts about a Wikidata item, shown to
the judge (an LLM or JEV) that has to decide which candidate a mention refers to.

    from candidate_profiles import build_profile, profile_block
    prof = build_profile("Q8016", "PERSON")     # Winston Churchill
    block = profile_block(prof)                 # deterministic text for the judge prompt

Modules
    types      PROFILE_SPEC (per-type Wikidata properties), TYPE_ALIASES,
               normalize_type, spec_for
    extract    pure claim extractors: claim_ids, time_str, coordinates,
               quantity, positions (P39 + qualifier years)
    config     ALL tunable parameters in one Config dataclass (defaults <
               --config JSON < CLI flags); module singleton config.CONFIG
    api        network layer: wd_entities (batched, cached), wd_labels
               (batched, cached), wiki_summary — one JSON cache
    profile    build_profile: assemble the profile dict of one candidate
    render     profile_block: deterministic text view for the judge prompt
    collect    batch CLI: profiles for the candidates of the selected clusters;
               reads/writes JSONL via the repo's utils/io.py

Batch use (see collect.py):

    python -m candidate_profiles --topk 10 --statuses review   # defaults from config.py
    python -m candidate_profiles --config profiles.json --limit 20
"""
from .types import PROFILE_SPEC, normalize_type, spec_for
from .config import Config
from .profile import build_profile
from .collect import candidate_pairs, load_profiles, main, select_records
from . import api, config

__all__ = ["PROFILE_SPEC", "normalize_type", "spec_for", "Config", "build_profile",
           "candidate_pairs", "load_profiles", "main", "select_records",
           "api", "config"]
