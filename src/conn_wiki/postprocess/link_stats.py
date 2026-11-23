"""
link_stats.py - statistics over final_qids.jsonl (the output of final_qids.py).

Two levels are reported:
  * doc-entities     one row of final_qids.jsonl = the entity as it occurs in ONE document
  * unique entities  distinct `global_cluster` values (the corpus clusters, e.g. FRANCO||Person).
                     Rows without a global_cluster are grouped per (doc_id, cluster_id) and flagged.

On entity level `with qid` means: at least one of its doc-entities carries a qid;
`full`        = every doc-entity of the cluster carries a qid;
`conflicting` = the doc-entities carry two or more DIFFERENT qids (a split cluster);
`partial`     = some doc-entities have a qid and some do not (e.g. JEV coverage gaps).

Usage:
    python link_stats.py data/output/links/final_qids.jsonl
    python link_stats.py final_qids.jsonl --out stats.txt --top 20
    python link_stats.py final_qids.jsonl --small      # compact report only

The compact report is also available as an importable function: link_stats.small_report(rows).
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# same mapping as final_qids.py (standalone copy: this script has no package imports)
TYPE_MAP = {"person": "PERSON", "per": "PERSON", "organization": "ORGANIZATION",
            "org": "ORGANIZATION", "place": "PLACE", "loc": "PLACE", "location": "PLACE",
            "ship": "VEHICLE", "vehicle": "VEHICLE", "event": "EVENT"}


def norm_type(t):
    t = str(t or "").strip()
    return TYPE_MAP.get(t.lower(), t.upper()) or "(none)"


def pct(n, d):
    return f"{100 * n / d:.1f}" if d else "-"


def dominant(counter):
    return max(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0] if counter else "(none)"


def load(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return [json.loads(l) for l in fh if l.strip()]
    except OSError as e:
        sys.exit(f"cannot read {path}: {e}")


def entity_facts(rows):
    """Group the doc-entities into unique entities (global_cluster) and compute their facts."""
    ents = defaultdict(list)
    for r in rows:
        key = r.get("global_cluster") or f"(no global cluster) {r.get('doc_id')}#{r.get('cluster_id')}"
        ents[key].append(r)
    e_qids, e_full, e_type, e_label, e_src, e_status, e_mentions, e_docs = {}, {}, {}, {}, {}, {}, {}, {}
    for key, rws in ents.items():
        e_qids[key] = sorted({r.get("qid") for r in rws if r.get("qid")})
        e_full[key] = all(r.get("qid") for r in rws)
        e_type[key] = norm_type(rws[0].get("type"))
        e_label[key] = next((r.get("label") for r in rws if r.get("label")), None)
        e_src[key] = Counter((r.get("source") or "(none)") for r in rws)
        e_status[key] = Counter((r.get("pipeline_status") or "(none)") for r in rws)
        e_mentions[key] = sum(len(r.get("mention_spans") or []) for r in rws)
        e_docs[key] = len({r.get("doc_id") for r in rws})
    e_state = {k: ("conflicting" if len(q) > 1 else ("qid" if q else "no_qid")) for k, q in e_qids.items()}
    return ents, dict(qids=e_qids, full=e_full, type=e_type, label=e_label, src=e_src,
                      status=e_status, mentions=e_mentions, docs=e_docs, state=e_state)


def small_report(rows, path=None):
    """Compact report: a few context lines plus the unique entities per type with their qid
    percentage (entity = global_cluster, computed along the whole corpus). Returns the text."""
    ents, F = entity_facts(rows)
    types = sorted({F["type"][k] for k in ents})
    per_t = {t: [0, 0] for t in types}
    for k in ents:
        d = per_t[F["type"][k]]
        d[0] += 1
        d[1] += F["state"][k] != "no_qid"
    tot = [sum(d[i] for d in per_t.values()) for i in (0, 1)]
    n_rows_q = sum(1 for r in rows if r.get("qid"))
    L = [f"final_qids small report - {path or '(input rows)'}",
         f"generated {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
         f"{'documents':<34}: {len({r.get('doc_id') for r in rows})}",
         f"{'doc-entities (rows)':<34}: {len(rows)}",
         f"{'doc-entities with qid':<34}: {n_rows_q} ({pct(n_rows_q, len(rows))}%)",
         "unique entities per type:",
         f"{'type':<14}{'entities':>9}{'with_qid':>10}{'pct':>8}"]
    for t in types:
        d = per_t[t]
        L.append(f"{t:<14}{d[0]:>9}{d[1]:>10}{pct(d[1], d[0]):>8}")
    L.append(f"{'TOTAL':<14}{tot[0]:>9}{tot[1]:>10}{pct(tot[1], tot[0]):>8}")
    return "\n".join(L) + "\n"


def save(text, path):
    """Write the report text to `path`, creating parent directories."""
    out_p = Path(path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Statistics over final_qids.jsonl")
    ap.add_argument("--final_qids", default='data/output/links/final_qids.jsonl', help="path to final_qids.jsonl")
    ap.add_argument("--out", default='data/output/links/report.txt', help="also write the report to this text file")
    ap.add_argument("--top", type=int, default=15, help="size of the top-entities table (default 15)")
    ap.add_argument("--small", action="store_true",
                    help="only the compact report: unique entities per type + qid percentage")
    a = ap.parse_args()

    rows = load(a.final_qids)
    if not rows:
        sys.exit(f"no rows in {a.final_qids}")

    if a.small:
        text = small_report(rows, a.final_qids)
        print(text, end="")
        if a.out:
            save(text, a.out)
            print(f"report also written to {a.out}")
        return

    ents, F = entity_facts(rows)
    e_qids, e_full, e_type = F["qids"], F["full"], F["type"]
    e_label, e_src, e_status = F["label"], F["src"], F["status"]
    e_mentions, e_docs, e_state = F["mentions"], F["docs"], F["state"]

    L = []

    def line(s=""):
        L.append(s)

    def section(title):
        line("")
        line(f"== {title}")

    n_rows_q = sum(1 for r in rows if r.get("qid"))
    n_ent_q = sum(1 for k in ents if e_state[k] != "no_qid")
    n_conf = sum(1 for k in ents if e_state[k] == "conflicting")
    n_part = sum(1 for k in ents if e_state[k] != "no_qid" and not e_full[k])

    line(f"final_qids statistics - {a.final_qids}")
    line(f"generated {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    line(f"{'documents':<34}: {len({r.get('doc_id') for r in rows})}")
    line(f"{'doc-entities (rows)':<34}: {len(rows)}")
    line(f"{'doc-entities with qid':<34}: {n_rows_q} ({pct(n_rows_q, len(rows))}%)")
    line(f"{'unique entities (corpus clusters)':<34}: {len(ents)}")
    line(f"{'unique entities with qid':<34}: {n_ent_q} ({pct(n_ent_q, len(ents))}%)")
    line(f"{'  conflicting qids':<34}: {n_conf}")
    line(f"{'  partial qid coverage':<34}: {n_part}")
    line(f"{'total mentions':<34}: {sum(e_mentions.values())}")
    line(f"{'rows without global_cluster':<34}: {sum(1 for r in rows if not r.get('global_cluster'))}")

    types = sorted({e_type[k] for k in ents})

    section("UNIQUE ENTITIES PER TYPE (entity = global_cluster)")
    line(f"{'type':<14}{'entities':>9}{'with_qid':>10}{'pct':>8}{'full':>7}{'conflicting':>13}")
    per_t = {t: [0, 0, 0, 0] for t in types}
    for k in ents:
        d = per_t[e_type[k]]
        d[0] += 1
        d[1] += e_state[k] != "no_qid"
        d[2] += e_full[k]
        d[3] += e_state[k] == "conflicting"
    tot = [0, 0, 0, 0]
    for t in types:
        d = per_t[t]
        line(f"{t:<14}{d[0]:>9}{d[1]:>10}{pct(d[1], d[0]):>8}{d[2]:>7}{d[3]:>13}")
        for i in range(4):
            tot[i] += d[i]
    line(f"{'TOTAL':<14}{tot[0]:>9}{tot[1]:>10}{pct(tot[1], tot[0]):>8}{tot[2]:>7}{tot[3]:>13}")

    section("DOC-ENTITIES PER TYPE (rows of final_qids.jsonl)")
    line(f"{'type':<14}{'rows':>9}{'with_qid':>10}{'pct':>8}")
    rper_t = {t: [0, 0] for t in types}
    for r in rows:
        d = rper_t[norm_type(r.get("type"))]
        d[0] += 1
        d[1] += bool(r.get("qid"))
    rtot = [0, 0]
    for t in types:
        d = rper_t[t]
        line(f"{t:<14}{d[0]:>9}{d[1]:>10}{pct(d[1], d[0]):>8}")
        rtot[0] += d[0]
        rtot[1] += d[1]
    line(f"{'TOTAL':<14}{rtot[0]:>9}{rtot[1]:>10}{pct(rtot[1], rtot[0]):>8}")

    section("SOURCES PER TYPE (doc-entities)")
    st = {t: Counter() for t in types}
    for r in rows:
        st[norm_type(r.get("type"))][r.get("source") or "(none)"] += 1
    src_names = sorted({s for c in st.values() for s in c})
    line(f"{'source':<26}" + "".join(f"{t:>12}" for t in types) + f"{'TOTAL':>10}")
    for s in src_names:
        cells = [st[t].get(s, 0) for t in types]
        line(f"{s:<26}" + "".join(f"{c:>12}" for c in cells) + f"{sum(cells):>10}")

    section("PIPELINE STATUS OF UNIQUE ENTITIES (dominant status of their doc-entities)")
    sc = Counter(dominant(e_status[k]) for k in ents)
    for s, n in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0])):
        line(f"  {s:<14}: {n}")

    section("ENTITIES WITHOUT QID, BY DOMINANT SOURCE")
    nc = Counter(dominant(e_src[k]) for k in ents if e_state[k] == "no_qid")
    for s, n in sorted(nc.items(), key=lambda kv: (-kv[1], kv[0])):
        line(f"  {s:<14}: {n}")
    if not nc:
        line("  (none)")

    section(f"TOP {a.top} UNIQUE ENTITIES BY MENTIONS")
    line(f"{'mentions':>9}{'docs':>6}  {'type':<12}{'qid':<16}label")
    for k in sorted(ents, key=lambda k: (-e_mentions[k], -e_docs[k], k))[:a.top]:
        qid_s = "/".join(e_qids[k]) or "-"
        line(f"{e_mentions[k]:>9}{e_docs[k]:>6}  {e_type[k]:<12}{qid_s:<16}{(e_label[k] or '-')[:36]}")

    section("DATA QUALITY")
    qid_owners = defaultdict(set)
    for k, qs in e_qids.items():
        for q in qs:
            qid_owners[q].add(k)
    shared = {q: o for q, o in qid_owners.items() if len(o) > 1}
    line(f"{'distinct qids used':<40}: {len(qid_owners)}")
    line(f"{'qids claimed by >1 entity (splits)':<40}: {len(shared)}")
    for q in sorted(shared)[:5]:
        line(f"  {q}: {', '.join(sorted(shared[q]))}")
    line(f"{'entities with conflicting qids':<40}: {n_conf}")
    for k in sorted((k for k in ents if e_state[k] == "conflicting"))[:5]:
        line(f"  {k}: {', '.join(e_qids[k])}")
    line(f"{'doc-entities with qid but no label':<40}: {sum(1 for r in rows if r.get('qid') and not r.get('label'))}")
    line(f"{'doc-entities with empty mention_spans':<40}: {sum(1 for r in rows if not (r.get('mention_spans') or []))}")
    line(f"{'doc-entities without a source':<40}: {sum(1 for r in rows if not r.get('source'))}")

    text = "\n".join(L) + "\n"
    print(text)
    if a.out:
        save(text, a.out)
        print(f"report also written to {a.out}")


if __name__ == "__main__":
    main()