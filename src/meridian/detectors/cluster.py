"""DENSE_MULTI_RELATION_CLUSTER and RAPID_ONBOARDING_LINKED."""
from __future__ import annotations

from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

from ..context import Context
from .base import Candidate
from .evidence import item_edges, shared_item_edges
from .relations import IDENTITY_TYPES, LINK_TYPES, entity_country, entity_items, entity_onboarding, idf, pair_links


def _relations(ctx: Context):
    if "_relations" not in ctx.__dict__:
        items = entity_items(ctx)
        ctx.__dict__["_relations"] = (items, pair_links(ctx, items))
    return ctx.__dict__["_relations"]


def _records(ctx: Context, entities) -> frozenset:
    return frozenset().union(*[ctx.records_of[e] for e in entities])


def coverage(ctx: Context, members: set, items: dict, onb: pd.Series, window_days: int = 30) -> dict:
    """type -> (best item, share of members holding it). Window = onboarding within window_days of the median."""
    out = {}
    n = len(members)
    for t in LINK_TYPES + ("category",):
        cnt = Counter()
        for key, es in items.get(t, {}).items():
            k = len(es & members)
            if k:
                cnt[key] = k
        if cnt:
            key, k = cnt.most_common(1)[0]
            out[t] = (key, k / n)
    d = onb[list(members)]
    med = d.median()
    out["window"] = (str(med.date()), float((abs(d - med) <= pd.Timedelta(days=window_days)).mean()))
    return out


def detect_dense(ctx: Context) -> list[Candidate]:
    min_types = ctx.param("dense_cluster", "min_pair_types", 4)
    min_size = ctx.param("dense_cluster", "min_size", 4)
    min_cover = ctx.param("dense_cluster", "min_cover", 0.8)
    min_cover_types = ctx.param("dense_cluster", "min_covered_types", 4)
    items, pairs = _relations(ctx)
    onb = entity_onboarding(ctx)
    cats = {}
    for key, es in items.get("category", {}).items():
        for e in es:
            cats.setdefault(e, set()).add(key)

    g = nx.Graph()
    for (a, b), types in pairs.items():
        n_types = len(types)
        if cats.get(a, set()) & cats.get(b, set()):
            n_types += 1
        if abs(onb[a] - onb[b]) <= pd.Timedelta(days=30):
            n_types += 1
        if n_types >= min_types:
            g.add_edge(a, b)
    total = len(onb)
    out = []
    for comp in nx.connected_components(g):
        if len(comp) < min_size:
            continue
        cov = coverage(ctx, comp, items, onb)
        covered = {t: v for t, v in cov.items() if v[1] >= min_cover}
        if len(covered) < min_cover_types:
            continue
        rarity = np.mean([idf(len(items[t][v[0]]), total) for t, v in covered.items() if t in items])
        identity = [t for t in covered if t in IDENTITY_TYPES]
        members = _records(ctx, comp)
        v = ctx.vendors.loc[sorted(members)]
        grp = v["declared_parent_group"].dropna()
        benign = []
        if len(grp) and grp.nunique() == 1 and len(grp) >= 0.8 * len(v):
            benign.append(f"declared group {grp.iloc[0]}")
        if not identity:
            benign.append("only process relations shared (sponsor, approver, category, institution): "
                          "consistent with a standardised local business process")
        out.append(Candidate(
            members=members, tags=("DENSE_MULTI_RELATION_CLUSTER",),
            score=float(len(covered) + len(identity) + np.log2(len(comp)) + rarity), emerged_at=onb[list(comp)].max(),
            detector="dense_cluster",
            reason=(f"{len(comp)} vendors in {','.join(sorted(set(v['country'])))} overlap on {len(covered)} relation "
                    f"types ({', '.join(f'{t} {s:.0%}' for t, (_, s) in sorted(covered.items()))})"),
            evidence={"coverage": {t: {"item": k, "share": s} for t, (k, s) in cov.items()}},
            benign=benign,
            features={"size": len(comp), "covered_types": len(covered), "rarity": rarity},
            edges=tuple(shared_item_edges(ctx, {t: k for t, (k, _) in covered.items()}, members)),
        ))
    return out


STRUCTURAL_LINKS = ("sponsor", "address", "owner", "phone", "domain")


def detect_rapid_onboarding(ctx: Context) -> list[Candidate]:
    days = ctx.param("rapid_onboarding", "window_days", 14)
    min_linked = ctx.param("rapid_onboarding", "min_linked", 4)
    items, pairs = _relations(ctx)
    onb = entity_onboarding(ctx)
    country = entity_country(ctx)
    link = {k: v for k, v in pairs.items() if any(t in v for t in STRUCTURAL_LINKS)}
    adj: dict[str, set] = {}
    for a, b in link:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    found = []
    for mkt, es in country.groupby(country):
        d = onb[es.index].sort_values()
        dates = d.to_numpy()
        for i in range(len(d)):
            j = np.searchsorted(dates, dates[i] + np.timedelta64(days, "D"), side="right")
            if j - i < min_linked:
                continue
            win = list(d.index[i:j])
            pos = {e: p for p, e in enumerate(win)}
            g = nx.Graph()
            for a_i, a in enumerate(win):
                for b_i in sorted(pos[b] for b in adj.get(a, ()) if pos.get(b, -1) > a_i):
                    g.add_edge(a, win[b_i])
            comps = [c for c in nx.connected_components(g) if len(c) >= min_linked]
            for comp in comps:
                lo = d.iloc[i]
                prior = d[(d >= lo - pd.Timedelta(days=730)) & (d < lo)]
                expected = len(prior) / (730 / days)
                found.append((mkt, frozenset(comp), lo, onb[list(comp)].max(), len(win), expected))
    # One candidate per linked group: keep the strongest window, drop heavily overlapping repeats.
    found.sort(key=lambda x: (-len(x[1]), x[2]))
    kept = []
    for f in found:
        if any(len(f[1] & k[1]) / len(f[1] | k[1]) >= 0.5 for k in kept):
            continue
        kept.append(f)
    out = []
    for mkt, comp, lo, hi, n_win, expected in kept:
        types = Counter(t for a in sorted(comp) for b in sorted(comp) if a < b and (a, b) in link
                        for t in link[(a, b)] if t in STRUCTURAL_LINKS)
        ratio = len(comp) / max(expected, 0.5)
        members = _records(ctx, comp)
        e = ctx.engagements[ctx.engagements["record_id"].isin(members)]
        first = e.sort_values("start_date").groupby("record_id").first()
        benign = ctx.benign_events("onboarding", lo, hi, mkt, cats=zip(first["parent"], first["cat_primary"]))
        out.append(Candidate(
            members=members, tags=("RAPID_ONBOARDING_LINKED",),
            score=float(np.log2(len(comp)) + np.log2(1 + ratio)), emerged_at=hi, active_until=hi,
            detector="rapid_onboarding",
            reason=(f"{len(comp)} linked vendors onboarded in {mkt} within {(hi - lo).days + 1} days "
                    f"({lo.date()}–{hi.date()}); market baseline {expected:.1f} per {days} days "
                    f"({ratio:.1f}x); links: {', '.join(f'{t}×{n}' for t, n in types.most_common())}"),
            evidence={"market": mkt, "window": (lo, hi), "expected_per_window": expected,
                      "link_types": dict(types)},
            benign=benign,
            features={"n_linked": len(comp), "ratio": ratio},
            edges=tuple(eid for a in comp for b in comp if a < b and (a, b) in link
                        for t, keys in link[(a, b)].items() if t in STRUCTURAL_LINKS for k in keys
                        for eid in item_edges(ctx, t, k, ctx.records_of[a] | ctx.records_of[b])),
        ))
    return out
