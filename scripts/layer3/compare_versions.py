"""Compare two versions' submissions on unlabelled data: what each puts at the top of its queue.

usage: python scripts/layer3/compare_versions.py OLD.csv NEW.csv [--names v1.0 v1.1] [--max-size 15] [--out FILE.csv]

Without labels, AP and IoU cannot be computed. This reports, for the first k cases of each list, the case
sizes, how many cases are oversized or single-record policy findings, and how many of the first k cases of
the new list match a case of the first k of the old one, one to one at IoU >= 0.5.
"""
import argparse

import pandas as pd

CUTOFFS = [10, 25, 50, 100, None]


def members(sub):
    return list(sub["member_record_ids"].astype(str).str.split("|").map(frozenset))


def shared(a, b, threshold=0.5):
    """cases of `a` matched one-to-one, in rank order, to a case of `b` at IoU >= threshold"""
    taken, n = set(), 0
    for m in a:
        best = max(((len(m & x) / len(m | x), j) for j, x in enumerate(b) if j not in taken), default=(0, None))
        if best[1] is not None and best[0] >= threshold:
            taken.add(best[1])
            n += 1
    return n


def profile(sub, other, k, max_size):
    top, ref = sub.head(k) if k else sub, other.head(k) if k else other
    size = top["member_record_ids"].astype(str).str.split("|").map(len)
    tags = top["typology_tags"].astype(str)
    return {"cases": len(top), "mean members": round(size.mean(), 1), "median members": size.median(),
            "max members": size.max(), f"cases > {max_size} members": int((size > max_size).sum()),
            "single-record cases": int((size == 1).sum()),
            "POLICY_VIOLATION cases": int(tags.str.contains("POLICY_VIOLATION").sum()),
            "single-record POLICY_VIOLATION": int(((size == 1) & tags.str.contains("POLICY_VIOLATION")).sum()),
            "OPEN_SET primary": int(tags.str.split("|").str[0].eq("OPEN_SET").sum()),
            "distinct primary tags": tags.str.split("|").str[0].nunique(),
            "records covered": len(frozenset().union(*members(top))),
            "shared with the other list (IoU >= 0.5)": shared(members(top), members(ref))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--names", nargs=2, default=["old", "new"])
    ap.add_argument("--max-size", type=int, default=15)
    ap.add_argument("--out")
    a = ap.parse_args()
    subs = dict(zip(a.names, [pd.read_csv(a.old), pd.read_csv(a.new)]))
    rows = []
    for k in CUTOFFS:
        for name, other in zip(a.names, reversed(a.names)):
            rows.append({"first k": k or "all", "version": name, **profile(subs[name], subs[other], k, a.max_size)})
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.set_index(["first k", "version"]).to_string())
    if a.out:
        out.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
