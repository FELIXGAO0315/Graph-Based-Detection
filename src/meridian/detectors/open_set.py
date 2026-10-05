"""Open-set detection. Evaluated by leave-one-type-out in experiments.py.

The blind test contains patterns that are not in the 14-tag list. These
detectors do not target any known type; their cases carry the tag OPEN_SET:

  rare_sharing_community  communities of the multi-relation vendor graph whose
                          internal, rarity-weighted sharing far exceeds what
                          their members' degrees predict
  entity_change_point     one vendor changes several things at once (address,
                          sponsor, category, spend level) within 90 days
  duplicate_paid          duplicate master records of one entity paid in the same months
  entity_outlier          records far in the tail of one or two explicit features
                          (record attributes, relation degrees, ownership structure)
"""
from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from ..context import Context
from .base import OPEN_SET_TAG, Candidate
from .evidence import address_edges, engagement_edges, identity_edges, shared_item_edges
from .relations import IDENTITY_TYPES, LINK_TYPES, entity_items, idf


def _weighted_graph(ctx: Context, max_item_size: int = 30) -> nx.Graph:
    """Entity graph; each shared item adds its rarity (idf) to the edge weight."""
    items = entity_items(ctx)
    total = len(set(ctx.entity_of.values()))
    g = nx.Graph()
    for t in LINK_TYPES:
        for key, es in items.get(t, {}).items():
            if not 2 <= len(es) <= max_item_size:
                continue
            w = idf(len(es), total) * (1.5 if t in IDENTITY_TYPES else 1.0)
            es = sorted(es)
            for i, a in enumerate(es):
                for b in es[i + 1:]:
                    if g.has_edge(a, b):
                        g[a][b]["weight"] += w
                        g[a][b]["types"].add(t)
                    else:
                        g.add_edge(a, b, weight=w, types={t})
    return g


def detect_rare_sharing(ctx: Context) -> list[Candidate]:
    """Groups from Louvain at several resolutions, scored by rare shared items.

    Resolutions are fixed by the community sizes they produce (from market-level
    blobs to groups of a few vendors), not by labels. A group is kept when at
    least two relation types each have one item held by most members; the score
    sums those items' rarity, weighting identity relations higher.
    """
    resolutions = ctx.param("open_set", "resolutions", [3, 10, 30])
    min_size = ctx.param("open_set", "min_size", 3)
    max_size = ctx.param("open_set", "max_size", 15)
    min_cover = ctx.param("open_set", "min_cover", 0.6)
    min_types = ctx.param("open_set", "min_types", 2)
    g = _weighted_graph(ctx)
    if g.number_of_edges() == 0:
        return []
    items = entity_items(ctx)
    total = len(set(ctx.entity_of.values()))
    onb = ctx.vendors.assign(e=ctx.vendors["record_id"].map(ctx.entity_of)).groupby("e")["onboarding_date"].max()
    groups: list[frozenset] = []
    for res in resolutions:
        for comm in nx.community.louvain_communities(g, weight="weight", resolution=res, seed=ctx.cfg.get("seed", 0)):
            core = frozenset(nx.k_core(g.subgraph(comm), 2).nodes) if len(comm) > 2 else frozenset(comm)
            if min_size <= len(core) <= max_size and all(len(core & x) / len(core | x) < 0.8 for x in groups):
                groups.append(core)
    out = []
    for nodes in groups:
        covered = {}
        for t in LINK_TYPES:
            best = max(((len(es & nodes) / len(nodes), key, len(es)) for key, es in items.get(t, {}).items()
                        if len(es & nodes) >= 2), default=None)
            if best and best[0] >= min_cover:
                covered[t] = best
        # Single-type groups were tried (one rare identity item held by 3+ vendors): on the
        # starter data they recovered no further planted case and diluted the ranking.
        if len(covered) < min_types:
            continue
        rarity = sum(cov * idf(n, total) * (1.5 if t in IDENTITY_TYPES else 1.0) for t, (cov, _, n) in covered.items())
        identity = set(covered) & set(IDENTITY_TYPES)
        members = frozenset().union(*[ctx.records_of[e] for e in nodes])
        benign = []
        if not identity:
            benign.append("only process relations shared: consistent with a standardised local business process")
        grp = ctx.vendors.loc[sorted(members), "declared_parent_group"].dropna()
        if len(grp) and grp.nunique() == 1 and len(grp) >= 0.8 * len(members):
            benign.append(f"declared group {grp.iloc[0]}")
        out.append(Candidate(
            members=members, tags=(OPEN_SET_TAG,), score=float(rarity + np.log2(len(nodes))),
            emerged_at=onb[list(nodes)].max(), detector="rare_sharing_community",
            reason=(f"{len(nodes)} vendors form a tight group in the relation graph; shared by most members: "
                    + ", ".join(f"{t} {cov:.0%}" for t, (cov, _, _) in sorted(covered.items()))),
            evidence={"shared": {t: {"item": k, "coverage": c, "holders": n} for t, (c, k, n) in covered.items()}},
            benign=benign, features={"rarity": rarity, "n_types": len(covered), "n_identity": len(identity)},
            edges=tuple(shared_item_edges(ctx, {t: k for t, (_, k, _) in covered.items()}, members)),
        ))
    return out


def detect_change_points(ctx: Context) -> list[Candidate]:
    """Several independent changes on one vendor inside one window (ownership-agnostic)."""
    days = ctx.param("open_set", "change_window_days", 90)
    min_kinds = ctx.param("open_set", "min_change_kinds", 3)
    events = []
    a = ctx.addresses.sort_values("valid_from")
    first_addr = a.groupby("record_id")["valid_from"].transform("min")
    for r in a[a["valid_from"] > first_addr].itertuples():
        events.append((r.record_id, r.valid_from, "address"))
    e = ctx.engagements.dropna(subset=["sponsor_employee_id"]).sort_values("start_date")
    first_sp = e.groupby(["record_id", "sponsor_employee_id"])["start_date"].min().reset_index()
    first_any = first_sp.groupby("record_id")["start_date"].transform("min")
    for r in first_sp[first_sp["start_date"] > first_any].itertuples():
        events.append((r.record_id, r.start_date, "sponsor"))
    s = ctx.services
    s = s[s["taxonomy_version"] == "v2"]
    first_cat = s.groupby("record_id")["valid_from"].transform("min")
    for r in s[(s["valid_from"] > first_cat) & (s["valid_from"] > pd.Timestamp("2024-01-31"))].itertuples():
        events.append((r.record_id, r.valid_from, "category"))
    ms = ctx.monthly_spend.sort_values("month")
    for rec, g in ms.groupby("record_id"):
        v = g.set_index("month")["amount_usd"]
        if len(v) < 12:
            continue
        before = v.rolling(6, min_periods=4).median().shift(1)
        after = v[::-1].rolling(6, min_periods=4).median()[::-1]
        jump = (after >= 2.5 * before) & (before > 0)
        if jump.any():
            events.append((rec, jump.idxmax(), "spend_level"))
    ev = pd.DataFrame(events, columns=["record_id", "date", "kind"])
    out = []
    for rec, g in ev.groupby("record_id"):
        g = g.sort_values("date")
        for d in g["date"]:
            w = g[(g["date"] >= d) & (g["date"] <= d + pd.Timedelta(days=days))]
            kinds = set(w["kind"])
            if len(kinds) >= min_kinds:
                benign = ctx.benign_events("category", d, d + pd.Timedelta(days=days)) if "category" in kinds else []
                out.append(Candidate(
                    members=ctx.expand([rec]), tags=(OPEN_SET_TAG,), score=float(len(kinds) + 0.2 * len(w)),
                    emerged_at=d, active_until=d + pd.Timedelta(days=days), detector="entity_change_point",
                    reason=f"within {days} days from {d.date()} the vendor changed {', '.join(sorted(kinds))}",
                    evidence={"events": w.assign(date=w["date"].astype(str)).to_dict("records")},
                    benign=benign, features={"n_kinds": len(kinds)},
                    edges=tuple(address_edges(ctx, [rec], d, d + pd.Timedelta(days=days)) + engagement_edges(
                        ctx, ctx.engagements.loc[(ctx.engagements["record_id"] == rec)
                                                 & ctx.engagements["start_date"].between(d, d + pd.Timedelta(days=days)),
                                                 "engagement_id"])),
                ))
                break
    return out


def detect_duplicate_paid(ctx: Context) -> list[Candidate]:
    """Duplicate master records of one entity that were paid in the same months."""
    em = ctx.entity_map[ctx.entity_map["n_records"] > 1]
    ms = ctx.monthly_spend.merge(em, on="record_id")
    out = []
    for ent, g in ms.groupby("entity_id"):
        per_month = g.groupby("month")["record_id"].nunique()
        both = per_month[per_month >= 2]
        if len(both) == 0:
            continue
        recs = sorted(ctx.records_of[ent])
        out.append(Candidate(
            members=frozenset(recs), tags=(OPEN_SET_TAG,), score=float(np.log2(1 + len(both))),
            emerged_at=both.index.min(), active_until=both.index.max(), detector="duplicate_paid",
            reason=(f"{len(recs)} master records of one entity (merged on "
                    f"{', '.join(sorted(set(ctx.identity_pairs[ctx.identity_pairs['record_a'].isin(recs)]['reason'])))}) "
                    f"were paid in the same {len(both)} month(s), first {both.index.min().date()}"),
            evidence={"months": [str(m.date()) for m in both.index]}, features={"n_months": len(both)},
            edges=tuple(identity_edges(ctx, recs)),
        ))
    return out


def entity_feature_table(ctx: Context) -> pd.DataFrame:
    """Record features plus network position: shares per relation type, foreign
    institution links and (if enabled) ownership structure."""
    from ..features import record_features

    f = record_features(ctx)
    items = entity_items(ctx)
    ent = ctx.entity_of
    for t in LINK_TYPES:
        deg = defaultdict(int)
        for key, es in items.get(t, {}).items():
            if 2 <= len(es) <= 30:
                for e in es:
                    deg[e] += len(es) - 1
        f[f"deg_{t}"] = [deg.get(ent[r], 0) for r in f.index]
    li = ctx.links
    foreign = li[li["inst_country"] != li["record_id"].map(ctx.vendors["country"])].groupby("record_id").size()
    f["foreign_links"] = foreign.reindex(f.index).fillna(0)
    if ctx.use_ownership:
        from .ownership import _view
        ov = _view(ctx)
        snaps = {r: ov.snapshots(r) for r in f.index}
        f["own_layers"] = [max((m["layers"] for m in snaps[r]), default=0) for r in f.index]
        f["own_jurisdictions"] = [max((len(m["jurisdictions"]) for m in snaps[r]), default=0) for r in f.index]
        f["own_sink"] = [float(any(m["sink"] for m in snaps[r])) for r in f.index]
        f["own_pep"] = [float(any(m["peps"] for m in snaps[r])) for r in f.index]
        # Due diligence answered "no PEP" while the registry shows one.
        f["dd_pep_contradiction"] = f["own_pep"] * (1 - f["q_pep_connected"])
    return f


def tail_scores(f: pd.DataFrame, k: int = 2) -> tuple[pd.Series, pd.DataFrame]:
    """Sum of the k largest -log10 upper-tail probabilities per record.

    For each feature, p = share of records with a value at least as high;
    values at or below the median get p = 1. A record is unusual when it sits
    far in the tail of one or two features, which is what a reviewer can check.
    """
    ranks = f.rank(method="max", ascending=False) / len(f)
    ranks = ranks.where(f.gt(f.median()), 1.0)
    nlp = -np.log10(ranks.clip(lower=1.0 / len(f)))
    top = np.sort(nlp.to_numpy(), axis=1)[:, -k:].sum(axis=1)
    return pd.Series(top, index=f.index), nlp


def detect_entity_outlier(ctx: Context) -> list[Candidate]:
    share = ctx.param("open_set", "outlier_top_share", 0.02)
    f = entity_feature_table(ctx)
    score, nlp = tail_scores(f)
    top = score.sort_values(ascending=False).head(max(10, int(share * len(score))))
    out = []
    for rec, s in top.items():
        drivers = nlp.loc[rec].sort_values(ascending=False).head(2)
        out.append(Candidate(
            members=ctx.expand([rec]), tags=(OPEN_SET_TAG,), score=float(s),
            emerged_at=ctx.vendors.loc[rec, "onboarding_date"], detector="entity_outlier",
            reason="far in the tail of: " + ", ".join(
                f"{k}={f.at[rec, k]:.2f} (top {10 ** -v:.1%})" for k, v in drivers.items()),
            evidence={"tail_score": float(s), "drivers": {k: float(f.at[rec, k]) for k in drivers.index}},
            features={"tail": float(s)}, timeless=True,
        ))
    return out


OPEN_SET_DETECTORS = {
    "rare_sharing_community": detect_rare_sharing,
    "entity_change_point": detect_change_points,
    "duplicate_paid": detect_duplicate_paid,
    "entity_outlier": detect_entity_outlier,
}
