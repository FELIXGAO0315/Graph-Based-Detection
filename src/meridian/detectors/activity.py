"""ACTIVITY_TIMING_CONCENTRATION: synchronised spend spikes under one sponsor; payments after termination."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..context import Context
from .base import Candidate
from .evidence import engagement_edges

def spend_spikes(ctx: Context, min_ratio: float, half_window: int = 6, ramp_up: int = 3) -> pd.DataFrame:
    """Months where a vendor's spend is at least `min_ratio` x its own local median
    (the median of the surrounding +-half_window months, excluding the month itself),
    and is the local maximum of that neighbourhood. The first `ramp_up` months of
    a vendor's payment history are skipped: new vendors ramp up together."""
    ms = ctx.monthly_spend.sort_values(["record_id", "month"])
    rows = []
    for rec, g in ms.groupby("record_id"):
        if len(g) < 4:
            continue
        s = g.set_index("month")["amount_usd"]
        idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
        s = s.reindex(idx)  # gaps stay NaN: unknown, not zero
        vals = s.to_numpy()
        for k in range(ramp_up, len(vals)):
            v = vals[k]
            if np.isnan(v):
                continue
            nb = np.concatenate([vals[max(0, k - half_window):k], vals[k + 1:k + 1 + half_window]])
            nb = nb[~np.isnan(nb)]
            if len(nb) < 3:
                continue
            med = np.median(nb)
            if med > 0 and v >= min_ratio * med and v >= nb.max():
                rows.append((rec, idx[k], v, v / med))
    return pd.DataFrame(rows, columns=["record_id", "month", "amount", "ratio"])


def detect_sync_spikes(ctx: Context) -> list[Candidate]:
    min_ratio = ctx.param("activity", "min_spike_ratio", 2.5)
    min_vendors = ctx.param("activity", "min_vendors", 3)
    spikes = spend_spikes(ctx, min_ratio)
    if spikes.empty:
        return []
    eng = ctx.engagements.dropna(subset=["sponsor_employee_id"])
    # A spike belongs to a sponsor when the vendor had an open engagement with that sponsor that month.
    sp = spikes.merge(eng[["record_id", "sponsor_employee_id", "start_date", "end_date", "parent", "market"]],
                      on="record_id")
    sp = sp[(sp["start_date"] <= sp["month"] + pd.offsets.MonthEnd(0))
            & (sp["end_date"].isna() | (sp["end_date"] >= sp["month"]))]
    sp["entity"] = sp["record_id"].map(ctx.entity_of)
    found = []
    for s, g in sp.groupby("sponsor_employee_id"):
        n_vendors_sponsor = eng[eng["sponsor_employee_id"] == s]["record_id"].map(ctx.entity_of).nunique()
        for m in sorted(g["month"].unique()):
            w = g[(g["month"] >= m) & (g["month"] <= m + pd.offsets.MonthBegin(1))]
            ents = set(w["entity"])
            if len(ents) >= min_vendors:
                found.append((s, pd.Timestamp(m), frozenset(ents), w, n_vendors_sponsor))
    found.sort(key=lambda x: (-len(x[2]), x[1]))
    kept = []
    for f in found:
        if any(f[0] == k[0] and len(f[2] & k[2]) / len(f[2] | k[2]) >= 0.5 for k in kept):
            continue
        kept.append(f)
    out = []
    for s, m, ents, w, n_sp in kept:
        members = frozenset().union(*[ctx.records_of[e] for e in ents])
        share = len(ents) / max(n_sp, 1)
        med_ratio = float(w.groupby("entity")["ratio"].max().median())
        parents = set(w["parent"].dropna())
        mkt = w["market"].mode().iloc[0]
        e = ctx.engagements[ctx.engagements["record_id"].isin(members)]
        benign = ctx.benign_events("spend", m, m + pd.offsets.MonthEnd(1), mkt,
                                   cats=zip(e["parent"], e["cat_primary"]))
        if n_sp >= 30:
            benign.append(f"sponsor has {n_sp} vendors: spikes may be seasonal")
        out.append(Candidate(
            members=members, tags=("ACTIVITY_TIMING_CONCENTRATION",),
            score=float(np.log2(len(ents)) + np.log2(med_ratio) + share), emerged_at=m,
            active_until=m + pd.offsets.MonthEnd(2),
            detector="sync_spikes",
            reason=(f"{len(ents)} vendors under sponsor {s} ({ctx.role_of.get(s)}) in {mkt} spike together "
                    f"({med_ratio:.1f}x their own local median) within 2 months of {m.date()}; "
                    f"{share:.0%} of the sponsor's vendors"),
            evidence={"sponsor": s, "month": m, "spikes": w[["record_id", "month", "amount", "ratio"]].to_dict("records")},
            benign=benign, features={"n_vendors": len(ents), "median_ratio": med_ratio, "sponsor_share": share},
            edges=tuple(engagement_edges(ctx, e.loc[e["sponsor_employee_id"] == s, "engagement_id"])),
        ))
    return out


def detect_post_termination(ctx: Context) -> list[Candidate]:
    min_months = ctx.param("activity", "post_termination_min_months", 1)
    tp = ctx.vendors[ctx.vendors["terminated_date"].notna()]
    p = ctx.b.tp_payments_monthly
    p = p[p["record_id"].isin(tp.index)].copy()
    p["term"] = p["record_id"].map(tp["terminated_date"])
    after = p[p["month"] > p["term"] + pd.offsets.MonthEnd(0)]
    out = []
    for rec, g in after.groupby("record_id"):
        months = sorted(g["month"].unique())
        if len(months) < min_months:
            continue
        term = tp.loc[rec, "terminated_date"]
        eng = ctx.engagements[ctx.engagements["record_id"] == rec]
        benign = ctx.benign_events("termination", term - pd.Timedelta(days=180), term + pd.Timedelta(days=180),
                                   ctx.vendors.loc[rec, "country"], units=set(eng["business_unit"]))
        out.append(Candidate(
            members=ctx.expand([rec]), tags=("ACTIVITY_TIMING_CONCENTRATION",),
            score=float(1.5 + np.log2(1 + len(months)) + np.log10(1 + g["amount_usd"].sum()) / 3),
            emerged_at=pd.Timestamp(months[0]), active_until=pd.Timestamp(months[-1]), detector="post_termination",
            reason=(f"terminated on {term.date()} but paid in {len(months)} later month(s) "
                    f"({pd.Timestamp(months[0]).date()}–{pd.Timestamp(months[-1]).date()}), "
                    f"USD {g['amount_usd'].sum():,.0f}"),
            evidence={"terminated": term, "months": [str(pd.Timestamp(x).date()) for x in months],
                      "amount_usd": float(g["amount_usd"].sum())},
            benign=benign, features={"n_months": len(months)},
            edges=tuple(engagement_edges(ctx, sorted(set(g["engagement_id"])), ("CONTRACTED_UNDER",))),
        ))
    return out
