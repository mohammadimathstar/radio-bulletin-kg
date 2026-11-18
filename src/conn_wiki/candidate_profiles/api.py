"""Network layer of the profiles: entity fetch, label resolution, Wikipedia
summaries.

All data goes through one JSON cache (config.Config.cache_path) with keys
'ep|QID' (entity payload with claims), 'l|QID' (label), 'w|lang|title'
(summary).  Endpoints, timeouts, batch sizes and language lists come from
config.CONFIG, read at CALL time — so JSON/CLI overrides apply without
reimporting.  Everything is module-level on purpose: tests (and you) can swap
any piece (api.wd_entities, api._cache, api.requests) with one assignment.
"""
import json
import time
from pathlib import Path
from urllib.parse import quote

import requests

from . import config


def check_api(r):
    """Parse a MediaWiki API reply; raise on transport errors and API errors."""
    data = r.json()
    if isinstance(data, dict) and "error" in data:
        raise requests.RequestException(f"API error: {data['error']}")
    return data


class JsonCache:
    """Minimal file cache with a get/set/save interface (loads once, saves the whole dict)."""

    def __init__(self, path: str):
        self.path = Path(path)
        try:
            self.d = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            self.d = {}

    def get(self, k):
        return self.d.get(k)

    def set(self, k, v):
        self.d[k] = v

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.d, ensure_ascii=False), encoding="utf-8")


_cache = None


def get_cache() -> JsonCache:
    """The cache, created on first use from config.CONFIG.cache_path (tests pre-seed api._cache)."""
    global _cache
    if _cache is None:
        _cache = JsonCache(config.CONFIG.cache_path)
    return _cache


def wd_entities(qids: list[str], claims: bool = True) -> dict[str, dict]:
    """Entities for QIDs via wbgetentities, batched (config.CONFIG.api_batch), cache-first.

    The payload carries labels/descriptions/aliases/sitelinks and (claims=True)
    the full claim dict the extractors in extract.py need.  Missing items are
    silently dropped from the result.
    """
    cfg = config.CONFIG
    uniq = list(dict.fromkeys(qids))
    out = {}
    for q in uniq:
        if (v := get_cache().get(f"ep|{q}")) is not None:
            out[q] = v
    need = [q for q in uniq if q not in out]
    for i in range(0, len(need), cfg.api_batch):
        chunk = need[i:i + cfg.api_batch]
        data = check_api(requests.get(
            cfg.wd_api, headers={"User-Agent": cfg.user_agent}, timeout=cfg.timeout,
            params=dict(action="wbgetentities", ids="|".join(chunk),
                        props="labels|descriptions|aliases|sitelinks" + ("|claims" if claims else ""),
                        languages=cfg.entity_langs, format="json")))
        for q, e in data.get("entities", {}).items():
            if "missing" not in e:
                get_cache().set(f"ep|{q}", e)
                out[q] = e
    return out


def wd_labels(qids: list[str]) -> dict[str, str]:
    """English (else Dutch, config.CONFIG.label_langs) labels for QIDs, batched and cached.

    Unknown labels fall back to the QID itself.
    """
    cfg = config.CONFIG
    uniq = list(dict.fromkeys(qids))
    need = [q for q in uniq if get_cache().get(f"l|{q}") is None]
    for i in range(0, len(need), cfg.api_batch):
        chunk = need[i:i + cfg.api_batch]
        r = requests.get(cfg.wd_api, headers={"User-Agent": cfg.user_agent}, timeout=cfg.timeout,
                         params=dict(action="wbgetentities", ids="|".join(chunk), props="labels",
                                     languages=cfg.label_langs, format="json"))
        ents = check_api(r).get("entities", {})
        for q in chunk:
            lab = ents.get(q, {}).get("labels", {})
            get_cache().set(f"l|{q}", (lab.get("en") or lab.get("nl") or {}).get("value", ""))
        time.sleep(cfg.request_sleep)
    return {q: get_cache().get(f"l|{q}") or q for q in uniq}


def _trim(s: str, n: int) -> str:
    """Cut at a sentence end within n characters when possible."""
    if len(s) <= n:
        return s
    cut = s[:n]
    i = cut.rfind(". ")
    return cut[:i + 1] if i > n * 0.5 else cut.rstrip() + "..."


def wiki_summary(e: dict, max_chars: int = 400) -> str:
    """Intro of the English (else Dutch, config.CONFIG.summary_langs) Wikipedia
    article linked from the item, or ''.

    Network errors are raised (requests.RequestException) so the caller can
    retry later, instead of silently producing a profile without summary.
    """
    cfg = config.CONFIG
    for lang in cfg.summary_langs:
        title = (e.get("sitelinks", {}).get(f"{lang}wiki") or {}).get("title")
        if not title:
            continue
        k = f"w|{lang}|{title}"
        v = get_cache().get(k)
        if v is None:
            url = (f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/"
                   f"{quote(title.replace(' ', '_'), safe='')}")
            r = requests.get(url, headers={"User-Agent": cfg.user_agent}, timeout=cfg.summary_timeout)
            if r.status_code == 404:
                v = ""
            else:
                r.raise_for_status()
                v = r.json().get("extract", "") or ""
            get_cache().set(k, v)
            time.sleep(cfg.request_sleep)
        if v:
            return _trim(v, max_chars)
    return ""
