"""Wikidata candidate-extraction stage (formerly linking/): corpus clusters -> candidate QIDs + accept/review/nil.

Modules
    config      all tunable parameters (LinkingConfig; JSON overrides + CLI flags)
    cache       bounded SQLite response cache (TTL per kind, size cap, auto-prune)
    wikidata    WDClient: search/Cirrus/entities/facts/closures with batching + rate limits
    clusters    loader for the NEW entity_resolution ``global_clusters.json`` (+ hints)
    cues        person-cue tables shared with the clustering stage
    retrieve    query forms, RRF fusion, candidate expansion (concurrent fan-out)
    features    hard gates (type/temporal) + features (label, context, prior, ...)
    decide      weighted scoring + accept/review/nil decision
    run_linking CLI: pass 1 (resumable, concurrent) + pass 2 (coherence) -> links file

Typical use::

    from candidate_extraction.config import LinkingConfig
    from candidate_extraction.run_linking import run
    cfg = LinkingConfig()               # or LinkingConfig().apply_json("linking.json")
    run(cfg)                            # writes cfg.out_path
"""
from .config import LinkingConfig

__all__ = ["LinkingConfig"]
