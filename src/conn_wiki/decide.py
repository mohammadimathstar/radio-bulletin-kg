"""Scoring and the accept / review / nil decision.

    base  = sum(W[f] * value[f]) / sum(W[f])        over APPLICABLE features only
    score = base - 0.10*ended - 0.05*anachronistic  (PENALTY)

Status: accepted if score >= accept, margin >= margin and no blocker; review if
score >= review; else nil.  A best candidate whose label similarity is below
label_min is always nil (unless it matches an ellipsis sibling name), because a
score built on a name that does not match is meaningless even when everything
else lines up.  Every record lists its ``blockers`` — why it is not accepted.
"""
from __future__ import annotations


def total(f: dict, w: dict, penalty: dict) -> float:
    """Score of a feature dict: weighted average over applicable features minus penalties."""
    num = den = 0.0
    for k, weight in w.items():
        v = f.get(k)
        if v is not None:
            num += weight * v
            den += weight
    base = num / den if den else 0.0
    return base - sum(p * (f.get(k) or 0.0) for k, p in penalty.items())


def decide(c: dict, cands: list[dict], cfg) -> dict:
    """Turn scored candidates into a result record (always the same keys).

    accepted: score >= accept, margin >= margin, no blocker.
    review:   score >= review otherwise.
    nil:      everything else, or best label similarity < label_min
              (qid is null, the best candidate is kept in best_candidate).
    Blockers (why not accepted): score<..., margin<..., single_mention (1 mention
    without near-exact label + strong context), impure_cluster (per-mention
    context spread > 0.25), ended_not_full_name (candidate ended before the
    bulletins and the match is not a full-name match), weak_label.
    """
    ranked = sorted(cands, key=lambda x: total(x["feats"], cfg.w, cfg.penalty), reverse=True)
    if not ranked:
        return dict(status="nil", qid=None, label=None, score=0.0, margin=0.0,
                    evidence={}, flags=[], blockers=["no_candidate"],
                    alternatives=[], best_candidate=None,
                    reason="no type/time-valid candidate")
    best = ranked[0]
    f = best["feats"]
    s = total(f, cfg.w, cfg.penalty)
    s2 = total(ranked[1]["feats"], cfg.w, cfg.penalty) if len(ranked) > 1 else 0.0
    margin = s - s2
    pm = best["per_mention"]
    weak = c["n_mentions"] < 2 and not (f["label"] >= 0.95 and (f.get("context") or 0) >= 0.5)
    impure = len(pm) >= 4 and (max(pm) - min(pm)) > 0.25
    ended = (f.get("ended") or 0) > 0
    anachronistic = (f.get("anachronistic") or 0) > 0
    full_name = f["label"] >= 0.95 and any(len(x.split()) >= 2 for x in c["forms"])
    weak_label = f["label"] < cfg.label_min and (f.get("expansion") or 0) < 0.9

    blockers = []
    if s < cfg.accept: blockers.append(f"score<{cfg.accept}")
    if margin < cfg.margin: blockers.append(f"margin<{cfg.margin}")
    if weak: blockers.append("single_mention")
    if impure: blockers.append("impure_cluster")
    if ended and not full_name: blockers.append("ended_not_full_name")
    if weak_label: blockers.append("weak_label")

    if weak_label:
        status = "nil"
    elif not blockers:
        status = "accepted"
    elif s >= cfg.review:
        status = "review"
    else:
        status = "nil"
    return dict(
        status=status, qid=best["qid"] if status != "nil" else None, label=best["label"],
        score=round(s, 3), margin=round(margin, 3),
        evidence={k: (None if v is None else round(v, 3)) for k, v in f.items()},
        flags=[n for n, on in [("single_mention", weak), ("impure_cluster", impure),
                               ("ended_before_bulletin", ended),
                               ("anachronistic_state", anachronistic),
                               ("weak_label", weak_label)] if on],
        blockers=blockers,
        alternatives=[dict(qid=r["qid"], label=r["label"],
                           score=round(total(r["feats"], cfg.w, cfg.penalty), 3))
                      for r in ranked[1:4]],
        best_candidate=best["qid"],
        reason=(f"best label similarity {f['label']:.2f} < label_min" if weak_label else None))
