"""Batch collection: build the profiles of the candidates of the selected clusters.

Independent of candidate extraction: its links JSONL file is the only input
(final_path in config.py, or the --config JSON / CLI flags).  JSONL reading
and writing use the repo's shared helpers (utils/io.py).

    python -m candidate_profiles --topk 10 --statuses review   # defaults from config.py
    python -m candidate_profiles --config profiles.json --limit 20   # JSON file + flags
    python -m candidate_profiles --statuses review,nil --min-score 0.3 --no-summary

Every config field is a CLI flag (auto-generated from the dataclass; flag name
= field name with dashes).  Precedence: config.py defaults < --config JSON <
CLI flags.  Writes one line per (qid, type) to out_path and can be rerun:
profiles already in the file are skipped, a network error leaves that profile
for the next run.  Use a --topk at least as large as the --topk of the judge
step, and the same --statuses/--below, so that every candidate the judge will
see has a stored profile.  The judge step then only READS the file
(load_profiles) and makes no Wikidata/Wikipedia calls.  Line format:
    {"qid": "Q30", "type": "PLACE", "profile": {...}}
A profile depends on the type (PERSON profiles list occupations, PLACE profiles
the country, ...), so the key is (qid, type); the stored type is
types.normalize_type() of the cluster type.
"""
import argparse
import sys
from dataclasses import fields
from pathlib import Path

# make the repo's utils/io.py importable whether the package lives at repo root
# or nested (same bootstrap as candidate_extraction/run_linking.py)
_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import requests  # noqa: E402
from utils.io import append_jsonl, read_jsonl  # noqa: E402

from . import api, config  # noqa: E402
from .profile import build_profile  # noqa: E402
from .types import normalize_type  # noqa: E402


def select_records(records: list[dict], statuses: set[str], below: float | None = None,
                   min_score: float | None = None) -> list[dict]:
    """Final-file records that have candidates, (status in `statuses` or score < `below`) and, if given,
    score >= `min_score` (skips hopeless clusters, e.g. --statuses review,nil --min-score 0.3).
    Clusters without candidates are never selected: there is nothing to choose from."""
    return [r for r in records if r["candidates"]
            and (r["status"] in statuses or (below is not None and r["score"] < below))
            and (min_score is None or r["score"] >= min_score)]


def candidate_pairs(records: list[dict], topk: int) -> list[tuple[str, str]]:
    """Unique (qid, normalized cluster type) of the first `topk` candidates of every record."""
    return list(dict.fromkeys((c["qid"], normalize_type(r["type"]))
                              for r in records for c in r["candidates"][:topk]))


def load_profiles(path: str) -> dict[tuple[str, str], dict]:
    """Stored profiles as {(qid, type): profile}; an empty dict if the file does not exist."""
    return {(d["qid"], d["type"]): d["profile"] for d in read_jsonl(path)}


def main() -> None:
    """Build and store the profiles of the candidates of the selected clusters (resumable)."""
    c = config.CONFIG
    records = read_jsonl(c.final_path)
    if not records:
        raise SystemExit(f"{c.final_path} is empty or missing: run candidate extraction first")
    chosen = select_records(records, set(c.statuses), c.below, c.min_score)
    pairs = candidate_pairs(chosen, c.topk)
    done = load_profiles(c.out_path)
    stored = sum(1 for p in pairs if p in done)
    todo = [p for p in pairs if p not in done]
    if c.limit:
        todo = todo[:c.limit]
    print(f"{len(chosen)} clusters selected, {len(pairs)} candidate profiles needed, "
          f"{stored} already stored, {len(todo)} to build now")

    Path(c.out_path).parent.mkdir(parents=True, exist_ok=True)
    ok = failed = 0
    with open(c.out_path, "a", encoding="utf-8") as fh:
        try:
            for i in range(0, len(todo), c.batch):
                chunk = todo[i:i + c.batch]
                try:
                    api.wd_entities([q for q, _ in chunk])    # fetch the entities of the chunk (batched)
                except (requests.RequestException, ValueError) as ex:
                    print(f"[skip chunk of {len(chunk)}] {ex!r}")
                    failed += len(chunk)
                    continue
                for qid, etype in chunk:
                    try:
                        prof = build_profile(qid, etype, with_summary=not c.no_summary,
                                             summary_chars=c.summary_chars)
                    except (requests.RequestException, ValueError) as ex:   # ValueError: non-JSON reply
                        print(f"[skip] {qid} ({etype}): {ex!r}")            # leave it for the next run
                        failed += 1
                        continue
                    append_jsonl(fh, dict(qid=qid, type=etype, profile=prof))
                    ok += 1
                api.get_cache().save()
                print(f"  {min(i + c.batch, len(todo))}/{len(todo)}")
        finally:
            api.get_cache().save()
    print(f"built {ok}, failed {failed} (rerun to retry) -> {c.out_path}")


def cli() -> None:
    """--config JSON + one auto-generated flag per config.Config field, then run."""
    ap = argparse.ArgumentParser(
        description="Collect Wikidata/Wikipedia profiles of the candidates of the selected "
                    "clusters (independent of candidate extraction: its links JSONL is the only input)")
    ap.add_argument("--config", help="JSON file overriding any config field (keys = field names)")
    for f in fields(config.Config):
        t, h = f.type, f.metadata.get("help", "")
        flag = "--" + f.name.replace("_", "-")
        if t is bool:
            ap.add_argument(flag, action="store_true", help=h)
        elif t is list:
            ap.add_argument(flag, type=str, help=(h + " (comma-separated)").strip())
        else:
            ap.add_argument(flag, type=t, help=h)
    a = ap.parse_args()
    if a.config:
        config.CONFIG.apply_json(a.config)
    config.CONFIG.apply_args(a)
    main()


if __name__ == "__main__":
    cli()
