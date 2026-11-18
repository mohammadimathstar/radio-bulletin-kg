"""Pure claim extractors: full entity payload -> python values.

No I/O, no network — every function takes the entity dict (wbgetentities WITH
claims) and returns plain values (QID lists, formatted strings, tuples), which
makes them trivially testable against fixtures.
"""

from . import config


def claim_ids(e: dict, pid: str) -> list[str]:
    """QIDs used as values of property ``pid`` (item-valued claims)."""
    return [c["mainsnak"]["datavalue"]["value"]["id"]
            for c in e.get("claims", {}).get(pid, [])
            if c["mainsnak"].get("datavalue") and c["mainsnak"]["datavalue"]["type"] == "wikibase-entityid"]


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


def quantity(e: dict, pid: str):
    """First quantity claim of `pid` as a thousand-separated integer, or None."""
    for c in e.get("claims", {}).get(pid, []):
        dv = c["mainsnak"].get("datavalue")
        if dv and dv["type"] == "quantity":
            try:
                return f"{round(float(dv['value']['amount'])):,}"
            except (KeyError, ValueError):
                return None
    return None


def positions(e: dict) -> list[tuple[str, str | None, str | None]]:
    """Positions held (P39) as (QID, start year, end year) using qualifiers P580/P582."""
    out = []
    for c in e.get("claims", {}).get("P39", [])[:config.CONFIG.max_values]:
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
