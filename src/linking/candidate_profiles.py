"""
Candidate profiles: compact, type-specific facts about a Wikidata item, meant to be shown
to a judge (JEV or an LLM) that has to decide which candidate a mention refers to.

    from radio_bulletins_kg.linking.candidate_profiles import build_profile
    build_profile("Q462002", "PERSON")
    # {"qid": "Q462002", "label": "...", "description": "...", "aliases": [...],
    #  "instance_of": ["human"], "occupations": [...], "citizenship": [...],
    #  "positions_held": ["... (1918-1922)"], "birth_date": "1847-03-25", "death_date": "1925-12-05",
    #  "sitelinks": 12, "wikipedia_summary": "..."}

Batch use: collect the profiles once, separately from the judge step
-------------------------------------------------------------------
    python -m radio_bulletins_kg.linking.candidate_profiles --topk 10 --statuses review  # reads the links file
    python -m radio_bulletins_kg.linking.candidate_profiles --topk 15 --statuses review,nil --min-score 0.3 --no-summary
    # only for a few documents (a small experiment); the filters are those of decide_common.run():
    python -m radio_bulletins_kg.linking.candidate_profiles --statuses review,nil --limit-docs 5
    python -m radio_bulletins_kg.linking.candidate_profiles --statuses review,nil --gold gold_clusters.json --limit-docs 20
Writes one line per (qid, type) to PROFILES_PATH and can be rerun: profiles that are already in the file
are skipped, a network error leaves that profile for the next run. Use a --topk at least as large as the
--topk of the judge step, and the same --statuses/--below, so that every candidate the judge will see has a
stored profile. The judge step then only READS the file (load_profiles) and makes no Wikidata/Wikipedia
calls. Line format:  {"qid": "Q30", "type": "PLACE", "profile": {...}}
A profile depends on the type (PERSON profiles list occupations, PLACE profiles the country, ...), so the
key is (qid, type).

What a profile contains
-----------------------
Common to all types:
    qid, label, label_other_languages, description (en), description_nl, aliases (nl/en/de/fr, at most
    MAX_ALIASES), instance_of (labels of P31 values), sitelinks (number of Wikipedia/wiki links; a
    popularity hint), wikipedia_summary (first sentences of the English, else Dutch, article).
Type-specific fields (see PROFILE_SPEC): item-valued properties are resolved to labels,
date properties are formatted with their precision (year / month / day).
    PERSON              occupations, citizenship, place_of_birth, member_of_party,
                        positions_held (with start-end years), birth_date, death_date
    ORGANIZATION        country, headquarters, industry, parent_organization, founded_by,
                        inception, dissolved
    PLACE               country, located_in, coordinates, inception, dissolved
    SHIP / SUBMARINE    operator, owner, country, ship_class, builder, inception,
                        service_entry, service_retirement
    AIRCRAFT            manufacturer, operator, country_of_origin, inception, first_flight
    ARMED_CONFLICT      location, participants, part_of, country, start, end
    DIPLOMATIC_MEETING  location, participants, part_of, country, point_in_time, start, end
    LEGAL_DOCUMENT      signatories, jurisdiction, country, inception, publication_date, point_in_time
Empty fields are omitted, which keeps the profile (and the JEV input) short.

Data sources and caching
------------------------
* The entity itself comes from wd_entities() of link_wikidata (already cached by the pipeline).
* Labels of referenced items (occupations, countries, ...) are fetched in one batched
  wbgetentities call and cached as 'l|QID'.
* Wikipedia summaries use the REST endpoint /api/rest_v1/page/summary/<title> and are cached as
  'w|lang|title'. Network errors are raised (requests.RequestException) so the caller can retry
  later, instead of silently producing a profile without summary.
All caches live in the same JSON file as the pipeline cache (wd_cache.json).
"""
import argparse
import time
from pathlib import Path
from urllib.parse import quote

import requests

from .link_wikidata import (HEADERS, OUT_PATH, WD_API, append_jsonl, cache, check_api, claim_ids,
                              read_jsonl, wd_entities)

PROFILES_PATH = "./data/output/links/candidate_profiles.jsonl"
MAX_VALUES = 6        # values kept per item-valued property
MAX_ALIASES = 8
SUMMARY_LANGS = ("en", "nl")

# type -> item-valued properties {field name: property} and date properties {field name: property}
PROFILE_SPEC = {
    "PERSON": dict(
        items={"occupations": "P106", "citizenship": "P27", "place_of_birth": "P19",
               "member_of_party": "P102"},
        dates={"birth_date": "P569", "death_date": "P570"}),
    "ORGANIZATION": dict(
        items={"country": "P17", "headquarters": "P159", "industry": "P452",
               "parent_organization": "P749", "founded_by": "P112"},
        dates={"inception": "P571", "dissolved": "P576"}),
    "PLACE": dict(
        items={"country": "P17", "located_in": "P131"},
        dates={"inception": "P571", "dissolved": "P576"}),
    "SHIP": dict(
        items={"operator": "P137", "owner": "P127", "country": "P17", "ship_class": "P289",
               "builder": "P176"},
        dates={"inception": "P571", "service_entry": "P729", "service_retirement": "P730"}),
    "AIRCRAFT": dict(
        items={"manufacturer": "P176", "operator": "P137", "country_of_origin": "P495"},
        dates={"inception": "P571", "first_flight": "P606"}),
    "ARMED_CONFLICT": dict(
        items={"location": "P276", "participants": "P710", "part_of": "P361", "country": "P17"},
        dates={"start": "P580", "end": "P582"}),
    "DIPLOMATIC_MEETING": dict(
        items={"location": "P276", "participants": "P710", "part_of": "P361", "country": "P17"},
        dates={"point_in_time": "P585", "start": "P580", "end": "P582"}),
    "LEGAL_DOCUMENT": dict(
        items={"signatories": "P1891", "jurisdiction": "P1001", "country": "P17"},
        dates={"inception": "P571", "publication_date": "P577", "point_in_time": "P585"}),
}
PROFILE_SPEC["SUBMARINE"] = PROFILE_SPEC["SHIP"]


def wd_labels(qids: list[str]) -> dict[str, str]:
    """English (else Dutch) labels for QIDs, batched and cached ('l|QID'). Falls back to the QID."""
    uniq = list(dict.fromkeys(qids))
    need = [q for q in uniq if cache.get(f"l|{q}") is None]
    for i in range(0, len(need), 50):
        chunk = need[i:i + 50]
        r = requests.get(WD_API, headers=HEADERS, timeout=60, params=dict(
            action="wbgetentities", ids="|".join(chunk), props="labels",
            languages="en|nl", format="json"))
        ents = check_api(r).get("entities", {})
        for q in chunk:
            lab = ents.get(q, {}).get("labels", {})
            cache.set(f"l|{q}", (lab.get("en") or lab.get("nl") or {}).get("value", ""))
        time.sleep(0.1)
    return {q: cache.get(f"l|{q}") or q for q in uniq}


def time_str(e: dict, pid: str):
    """First time claim of `pid` as '1925-12-05', '1925-12' or '1925' according to its precision."""
    for c in e.get("claims", {}).get(pid, []):
        dv = c["mainsnak"].get("datavalue")
        if dv and dv["type"] == "time":
            t, prec = dv["value"]["time"], dv["value"].get("precision", 11)
            sign = "-" if t.startswith("-") else ""
            y, mo, d = t[1:5], t[6:8], t[9:11]
            if prec >= 11 and mo != "00" and d != "00":
                return f"{sign}{y}-{mo}-{d}"
            if prec == 10 and mo != "00":
                return f"{sign}{y}-{mo}"
            return f"{sign}{y}"
    return None


def coordinates(e: dict):
    """'lat, lon' (2 decimals) from P625, or None."""
    for c in e.get("claims", {}).get("P625", []):
        dv = c["mainsnak"].get("datavalue")
        if dv and dv["type"] == "globecoordinate":
            return f"{dv['value']['latitude']:.2f}, {dv['value']['longitude']:.2f}"
    return None


def positions(e: dict) -> list[tuple[str, str | None, str | None]]:
    """Positions held (P39) as (QID, start year, end year) using qualifiers P580/P582."""
    out = []
    for c in e.get("claims", {}).get("P39", [])[:MAX_VALUES]:
        dv = c["mainsnak"].get("datavalue")
        if not dv or dv["type"] != "wikibase-entityid":
            continue

        def yr(pid):
            for s in c.get("qualifiers", {}).get(pid, []):
                d = s.get("datavalue")
                if d and d["type"] == "time":
                    return d["value"]["time"][1:5]
            return None

        out.append((dv["value"]["id"], yr("P580"), yr("P582")))
    return out


def _trim(s: str, n: int) -> str:
    """Cut at a sentence end within n characters when possible."""
    if len(s) <= n:
        return s
    cut = s[:n]
    i = cut.rfind(". ")
    return cut[:i + 1] if i > n * 0.5 else cut.rstrip() + "..."


def wiki_summary(e: dict, max_chars: int = 400) -> str:
    """Intro of the English (else Dutch) Wikipedia article linked from the item, or ''."""
    for lang in SUMMARY_LANGS:
        title = (e.get("sitelinks", {}).get(f"{lang}wiki") or {}).get("title")
        if not title:
            continue
        k = f"w|{lang}|{title}"
        v = cache.get(k)
        if v is None:
            url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title.replace(' ', '_'), safe='')}"
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 404:
                v = ""
            else:
                r.raise_for_status()
                v = r.json().get("extract", "") or ""
            cache.set(k, v)
            time.sleep(0.1)
        if v:
            return _trim(v, max_chars)
    return ""


def build_profile(qid: str, etype: str, with_summary: bool = True, summary_chars: int = 400) -> dict:
    """Profile of one candidate item (see module docstring). Empty fields are omitted."""
    e = wd_entities([qid]).get(qid) or {}
    labels, descs = e.get("labels", {}), e.get("descriptions", {})
    spec = PROFILE_SPEC.get(etype, dict(items={}, dates={}))

    item_qids = {"instance_of": claim_ids(e, "P31")[:MAX_VALUES]}
    for name, pid in spec["items"].items():
        item_qids[name] = claim_ids(e, pid)[:MAX_VALUES]
    pos = positions(e) if etype == "PERSON" else []
    lab = wd_labels([q for v in item_qids.values() for q in v] + [p[0] for p in pos])

    main = (labels.get("en") or labels.get("nl") or next(iter(labels.values()), {})).get("value", qid)
    others = list(dict.fromkeys(v["value"] for k, v in labels.items() if v["value"] != main))
    aliases = list(dict.fromkeys(
        a["value"] for lang in ("nl", "en", "de", "fr") for a in e.get("aliases", {}).get(lang, [])
        if a["value"] != main and a["value"] not in others))

    prof = {"qid": qid, "label": main, "label_other_languages": others[:4],
            "description": descs.get("en", {}).get("value"),
            "description_nl": descs.get("nl", {}).get("value"),
            "aliases": aliases[:MAX_ALIASES]}
    for name, qs in item_qids.items():
        prof[name] = [lab[q] for q in qs]
    if pos:
        prof["positions_held"] = [f"{lab[q]} ({a or '?'}-{b or ''})" if (a or b) else lab[q] for q, a, b in pos]
    for name, pid in spec["dates"].items():
        prof[name] = time_str(e, pid)
    if etype == "PLACE":
        prof["coordinates"] = coordinates(e)
    prof["sitelinks"] = len(e.get("sitelinks", {}))
    if with_summary:
        prof["wikipedia_summary"] = wiki_summary(e, summary_chars)
    return {k: v for k, v in prof.items() if v not in (None, "", [], {})}


# ------------------------------------------------------------------ batch collection (separate from JEV)
def select_records(records: list[dict], statuses: set[str], below: float | None = None,
                   min_score: float | None = None) -> list[dict]:
    """Final-file records that have candidates, (status in `statuses` or score < `below`) and, if given,
    score >= `min_score` (skips hopeless clusters, e.g. --statuses review,nil --min-score 0.3).
    Clusters without candidates are never selected: there is nothing to choose from.
    The same selection is used by decide_common.run(), so both steps look at the same clusters."""
    return [r for r in records if r["candidates"]
            and (r["status"] in statuses or (below is not None and r["score"] < below))
            and (min_score is None or r["score"] >= min_score)]


def candidate_pairs(records: list[dict], topk: int) -> list[tuple[str, str]]:
    """Unique (qid, type) of the first `topk` candidates of every record, in order of appearance."""
    return list(dict.fromkeys((c["qid"], r["type"]) for r in records for c in r["candidates"][:topk]))


def load_profiles(path: str) -> dict[tuple[str, str], dict]:
    """Stored profiles as {(qid, type): profile}; an empty dict if the file does not exist."""
    return {(d["qid"], d["type"]): d["profile"] for d in read_jsonl(path)}


def main(a: argparse.Namespace) -> None:
    """Build and store the profiles of the candidates of the selected clusters (resumable)."""
    records = read_jsonl(a.final)
    if not records:
        raise SystemExit(f"{a.final} is empty or missing: run link_wikidata first")
    chosen = select_records(records, set(a.statuses.split(",")), a.below, a.min_score)
    if a.doc_ids or a.limit_docs or a.gold:                       # keep only the clusters of the selected documents
        from .decide_common import gold_spans, select_docs
        gold = gold_spans(a.gold) if a.gold else None
        wanted = {c.get("global_cluster") for doc in select_docs(read_jsonl(a.documents), a.doc_ids, gold, a.limit_docs)
                  for c in doc["clusters"]}
        chosen = [r for r in chosen if r["cluster_id"] in wanted]
    pairs = candidate_pairs(chosen, a.topk)
    done = load_profiles(a.out)
    todo = [p for p in pairs if p not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(chosen)} clusters selected, {len(pairs)} candidate profiles needed, "
          f"{len(pairs) - len([p for p in pairs if p not in done])} already stored, {len(todo)} to build now")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ok = failed = 0
    with open(a.out, "a", encoding="utf-8") as fh:
        try:
            for i in range(0, len(todo), a.batch):
                chunk = todo[i:i + a.batch]
                try:
                    wd_entities([q for q, _ in chunk])          # fetch the entities of the chunk in batches of 50
                except requests.RequestException as ex:
                    print(f"[skip chunk of {len(chunk)}] {ex!r}")
                    failed += len(chunk)
                    continue
                for qid, etype in chunk:
                    try:
                        prof = build_profile(qid, etype, with_summary=not a.no_summary, summary_chars=a.summary_chars)
                    except requests.RequestException as ex:      # leave it for the next run
                        print(f"[skip] {qid} ({etype}): {ex!r}")
                        failed += 1
                        continue
                    append_jsonl(fh, dict(qid=qid, type=etype, profile=prof))
                    ok += 1
                cache.save()
                print(f"  {min(i + a.batch, len(todo))}/{len(todo)}")
        finally:
            cache.save()
    print(f"built {ok}, failed {failed} (rerun to retry) -> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Collect Wikidata/Wikipedia profiles of the candidates")
    ap.add_argument("--final", default=OUT_PATH, help="final links file of link_wikidata")
    ap.add_argument("--out", default=PROFILES_PATH, help="profiles file (JSONL, appended, resumable)")
    ap.add_argument("--topk", type=int, default=10, help="candidates per cluster (>= --topk of the judge step)")
    ap.add_argument("--statuses", default="review", help="comma-separated pipeline statuses to cover")
    ap.add_argument("--below", type=float, default=None, help="also cover clusters with score below this")
    ap.add_argument("--min-score", type=float, default=None, help="skip clusters whose score is below this")
    ap.add_argument("--documents", default="./data/output/document_clusters.jsonl",
                    help="document-level clusters file (used with the three options below)")
    ap.add_argument("--limit-docs", type=int, default=None, help="only the clusters of the first N documents")
    ap.add_argument("--doc-ids", nargs="*", default=None, help="only the clusters of these documents")
    ap.add_argument("--gold", default=None, help="only documents that have gold annotations")
    ap.add_argument("--limit", type=int, default=None, help="build at most N profiles now")
    ap.add_argument("--batch", type=int, default=50, help="profiles per chunk (entities are fetched per chunk)")
    ap.add_argument("--no-summary", action="store_true", help="skip Wikipedia summaries (much faster)")
    ap.add_argument("--summary-chars", type=int, default=400)
    main(ap.parse_args())