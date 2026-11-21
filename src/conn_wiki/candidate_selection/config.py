"""All tunable parameters of the judge package in one dataclass.

Same pattern as candidate_profiles.config.Config and the old decide_common
argparse constants: the dataclass carries the DEFAULTS, an optional JSON file
(--config, keys = field names) overrides any subset, and CLI flags override
both.  Every field is auto-generated as a CLI flag (config.build_parser, used
by jev.main()) — there is no separate flag table to keep in sync.  Read at
CALL time everywhere (config.CONFIG.field), so overrides apply without
reimporting anything.
"""
import argparse
import json
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class Config:
    # ---- files ----
    documents_path: str = field(
        default="./data/interim/document_clusters.jsonl",
        metadata=dict(help="resolved documents of the IE step (input; one line per document)"))
    links_path: str = field(
        default="./data/output/links/wikidata_links.jsonl",
        metadata=dict(help="links file of candidate extraction (input): the final file, or "
                           "wikidata_pass1.jsonl (both record shapes are accepted)"))
    profiles_path: str = field(
        default="./data/output/links/candidate_profiles.jsonl",
        metadata=dict(help="profiles file of candidate_profiles (input; keyed (qid, type))"))
    out_path: str = field(
        default="./data/output/links/decisions.jsonl",
        metadata=dict(help="decisions output, one line per judged entity (appended, resumable)"))
    restart: bool = field(default=False,
                          metadata=dict(help="delete the output file first"))

    # ---- what is judged ----
    statuses: list = field(default_factory=lambda: ["review", "nil"],
                           metadata=dict(help="pipeline statuses of the corpus cluster to judge"))
    entity_types: list = field(default_factory=lambda: ["PERSON", "ORGANIZATION", "PLACE", "VEHICLE"],
                               metadata=dict(help="only judge clusters of these types (normalized, so casing "
                                                  "never matters and aliases count: SHIP -> VEHICLE); "
                                                  "empty = judge every type that has candidates"))
    topk: int = field(default=30,
                      metadata=dict(help="candidates shown to the model (<= the profiles --topk)"))
    no_summary: bool = field(default=False,
                             metadata=dict(help="leave Wikipedia summaries out of the option profiles"))
    option_style: str = field(default="json",
                              metadata=dict(help="'json' = full profile dict per option (new profile "
                                                 "fields flow through automatically); 'text' = "
                                                 "candidate_profiles.render.profile_block() strings"))
    min_p: float = field(default=0.50,
                         metadata=dict(help="probability the chosen option needs"))
    min_ratio: float = field(default=2.0,
                             metadata=dict(help="chosen option / runner-up needed to link"))

    # ---- text shaping (the old decide_common constants) ----
    min_sentence_chars: int = field(default=80,
                                    metadata=dict(help="shorter sentences are replaced by the context"))
    max_chars: int = field(default=500,
                           metadata=dict(help="characters of one text snippet"))
    max_snippets: int = field(default=5,
                              metadata=dict(help="snippets per entity"))
    max_known: int = field(default=8,
                           metadata=dict(help="already resolved entities of the document shown as context"))

    # ---- JEV ----
    model: str = field(default="jev-latest",
                       metadata=dict(help="JEV model name passed to system_one()"))

    # ---- small experiments ----
    limit_docs: int = field(default=None,
                            metadata=dict(help="only the first N documents"))
    doc_ids: list = field(default=None,
                          metadata=dict(help="only these documents (comma-separated)"))
    gold_path: str = field(default=None,
                           metadata=dict(help="gold file (resolution format): only documents and "
                                              "entities that have gold annotations"))
    max_entities: int = field(default=None,
                              metadata=dict(help="stop after N judged entities"))
    dry_run: int = field(default=0,
                         metadata=dict(help="print the first N requests, call no model"))

    # ------------------------------------------------------------------ API
    def _list_fields(self) -> set:
        """Names of the list-valued fields (declared `list`), e.g. statuses, doc_ids."""
        return {f.name for f in fields(self) if f.type is list}

    def apply_json(self, path: str) -> "Config":
        """Override fields from a JSON file (keys = field names; unknown keys are ignored)."""
        p = Path(path)
        if not p.exists():
            return self
        data = json.loads(p.read_text(encoding="utf-8"))
        names, list_fields = {f.name for f in fields(self)}, self._list_fields()
        for k, v in data.items():
            if k in names:
                if k in list_fields and isinstance(v, str):    # "review,nil" -> ["review", "nil"]
                    v = [s.strip() for s in v.split(",") if s.strip()]
                setattr(self, k, v)
        return self

    def apply_args(self, ns) -> "Config":
        """Override fields from parsed CLI flags: only flags explicitly given (not None) apply."""
        list_fields = self._list_fields()
        for f in fields(self):
            v = getattr(ns, f.name, None)
            if v is None:
                continue
            if f.name in list_fields and isinstance(v, str):   # list fields take comma-separated strings
                v = [s.strip() for s in v.split(",") if s.strip()]
            setattr(self, f.name, v)
        return self


#: the module singleton every function reads at call time (texts.py, jev.py,
#: collect.main); tests swap it with one assignment
CONFIG = Config()


def build_parser(description: str) -> argparse.ArgumentParser:
    """--config JSON + one auto-generated flag per Config field (same pattern as
    candidate_profiles.collect.cli()).  Flag name = field name with dashes."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", help="JSON file overriding any config field (keys = field names)")
    for f in fields(Config):
        t, h = f.type, f.metadata.get("help", "")
        flag = "--" + f.name.replace("_", "-")
        if t is bool:
            ap.add_argument(flag, action="store_true", help=h)
        elif t is list:
            ap.add_argument(flag, type=str, help=(h + " (comma-separated)").strip())
        else:
            ap.add_argument(flag, type=t, help=h)
    return ap
