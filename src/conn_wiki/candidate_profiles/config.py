"""All tunable parameters of the profiles package in one dataclass.

Same pattern as candidate_extraction.config.LinkingConfig: the dataclass
carries the DEFAULTS, an optional JSON file (--config, keys = field names)
overrides any subset, and CLI flags override both.  Every field is
auto-generated as a CLI flag in collect.cli() — there is no separate flag
table to keep in sync.  Read at CALL time everywhere (config.CONFIG.field),
so overrides apply without reimporting anything.
"""
import json
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class Config:
    # ---- input / output ----
    final_path: str = field(default="./data/output/links/wikidata_links.jsonl",
                            metadata=dict(help="links file of candidate extraction (input)"))
    out_path: str = field(default="./data/output/links/candidate_profiles.jsonl",
                          metadata=dict(help="profiles output, one JSONL line per candidate (appended, resumable)"))
    cache_path: str = field(default="./data/output/links/profiles_wd_cache.json",
                            metadata=dict(help="JSON cache of entity payloads, labels and summaries"))

    # ---- selection (which clusters/candidates get profiles) ----
    topk: int = field(default=10,
                      metadata=dict(help="how many of the stored (best-first) candidates per cluster get profiles"))
    statuses: list = field(default_factory=lambda: ["review", "nil"],
                           metadata=dict(help="pipeline statuses to cover"))
    below: float = field(default=None,
                         metadata=dict(help="also cover clusters with score below this"))
    min_score: float = field(default=None,
                             metadata=dict(help="skip clusters whose score is below this"))
    limit: int = field(default=None,
                       metadata=dict(help="build at most N profiles in this run"))

    # ---- collection ----
    batch: int = field(default=50,
                       metadata=dict(help="profiles per chunk; entities are fetched per chunk"))
    api_batch: int = field(default=50,
                           metadata=dict(help="QIDs per wbgetentities call (MediaWiki max 50)"))
    no_summary: bool = field(default=False,
                             metadata=dict(help="skip Wikipedia summaries (much faster)"))
    summary_chars: int = field(default=400,
                               metadata=dict(help="trim length of the Wikipedia summary"))

    # ---- endpoints / etiquette ----
    wd_api: str = field(default="https://www.wikidata.org/w/api.php",
                        metadata=dict(help="Wikidata API endpoint"))
    user_agent: str = field(default="candidate-profiles/1.0 (research; contact: you@example.org)",
                            metadata=dict(help="User-Agent for all Wikidata/Wikipedia requests"))
    timeout: float = field(default=60.0,
                           metadata=dict(help="timeout per Wikidata API request (s)"))
    summary_timeout: float = field(default=20.0,
                                   metadata=dict(help="timeout per Wikipedia summary request (s)"))
    request_sleep: float = field(default=0.1,
                                 metadata=dict(help="pause between API calls (s)"))

    # ---- languages ----
    entity_langs: str = field(default="en|nl|de|fr",
                              metadata=dict(help="languages fetched per entity (labels/descriptions/aliases)"))
    label_langs: str = field(default="en|nl",
                             metadata=dict(help="label language order for resolved property values"))
    summary_langs: list = field(default_factory=lambda: ["en", "nl"],
                                metadata=dict(help="Wikipedia language order for summaries"))

    # ---- profile shape ----
    max_values: int = field(default=6,
                            metadata=dict(help="values kept per item-valued property"))
    max_aliases: int = field(default=8,
                             metadata=dict(help="aliases kept in the profile"))

    # ------------------------------------------------------------------ API
    def apply_json(self, path: str) -> "Config":
        """Override fields from a JSON file (keys = field names; unknown keys are ignored)."""
        p = Path(path)
        if not p.exists():
            return self
        data = json.loads(p.read_text(encoding="utf-8"))
        names = {f.name for f in fields(self)}
        for k, v in data.items():
            if k in names:
                cur = getattr(self, k)
                if isinstance(cur, list) and isinstance(v, str):   # "review,nil" -> ["review", "nil"]
                    v = [s.strip() for s in v.split(",") if s.strip()]
                setattr(self, k, v)
        return self

    def apply_args(self, ns) -> "Config":
        """Override fields from parsed CLI flags: only flags explicitly given (not None) apply.

        Flag names are the field names with underscores replaced by dashes
        (argparse generates them in collect.cli()); list fields take
        comma-separated strings.
        """
        for f in fields(self):
            v = getattr(ns, f.name, None)
            if v is None:
                continue
            if isinstance(getattr(self, f.name), list) and isinstance(v, str):
                v = [s.strip() for s in v.split(",") if s.strip()]
            setattr(self, f.name, v)
        return self


#: the module singleton every function reads at call time (api.py, collect.py,
#: extract/profile/render for the caps); tests swap it with one assignment
CONFIG = Config()
