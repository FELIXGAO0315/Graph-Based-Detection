"""BRIDGE_ENTITY and GEOGRAPHY_LINKED_CONCENTRATION."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..context import Context
from .base import Candidate
from .evidence import engagement_edges, link_edges

BENIGN_BRIDGE_PARENTS = {"RND", "SCM"}  # CROs, logistics: cross-border by design


def detect_bridge(ctx: Context) -> list[Candidate]:
    min_foreign = ctx.param("bridge", "min_foreign_links", 2)
    li = ctx.links[ctx.links["link_type"].isin(["SERVES", "SITE_OF"])].copy()
    li["vendor_country"] = li["record_id"].map(ctx.vendors["country"])
    li["foreign"] = li["inst_country"] != li["vendor_country"]
    eng = ctx.engagements
    out = []
    for rec, g in li.groupby("record_id"):
        foreign = g[g["foreign"]]
        if len(foreign) < min_foreign:
            continue
        e = eng[eng["record_id"] == rec]
        sponsors = set(e["sponsor_employee_id"].dropna())
        home = ctx.vendors.loc[rec, "country"]
        local = [s for s in sponsors if ctx.roles.set_index("employee_id").at[s, "home_market"] == home] \
            if sponsors else []
        single_local = len(sponsors) == 1 and len(local) == 1
        flags = set().union(*e["cat_flags"]) if len(e) else set()
        parents = set(e["parent"].dropna())
        public = float(g["public_sector"].mean())
        countries = sorted(set(g["inst_country"]))
        score = np.log2(1 + len(foreign)) + public + 1.0 * single_local + 0.5 * bool(flags & {"int", "gov"})
        benign = []
        if len(sponsors) >= 3:
            benign.append(f"{len(sponsors)} sponsors: consistent with a national/regional distributor")
        if parents & BENIGN_BRIDGE_PARENTS:
            benign.append(f"category parent {','.join(sorted(parents & BENIGN_BRIDGE_PARENTS))}: cross-border by design")
        first_foreign = foreign["valid_from"].min()
        emerged = max(first_foreign, ctx.vendors.loc[rec, "onboarding_date"])
        out.append(Candidate(
            members=ctx.expand([rec]), tags=("BRIDGE_ENTITY",), score=float(score), emerged_at=emerged,
            detector="bridge",
            reason=(f"vendor in {home} links to institutions in {','.join(countries)} "
                    f"({len(foreign)} foreign links, {public:.0%} public sector); "
                    + (f"single local sponsor {next(iter(sponsors))}" if single_local else f"{len(sponsors)} sponsors")),
            evidence={"links": g[["link_id", "institution_id", "inst_country", "link_type", "public_sector",
                                  "valid_from", "valid_to"]].to_dict("records")},
            benign=benign, features={"n_foreign": len(foreign), "public": public, "single_local": single_local},
            edges=tuple(link_edges(ctx, g["link_id"]) + engagement_edges(ctx, e["engagement_id"])),
        ))
    return out


def detect_geography(ctx: Context) -> list[Candidate]:
    days = ctx.param("geography", "window_days", 365)
    min_new = ctx.param("geography", "min_new", 4)
    min_ratio = ctx.param("geography", "min_ratio", 3.0)
    e = ctx.engagements.copy()
    e = e[e["cat_flags_possible"].map(lambda f: "int" in f)]
    e["entity"] = e["record_id"].map(ctx.entity_of)
    e["vendor_country"] = e["record_id"].map(ctx.vendors["country"])
    first = e.sort_values("start_date").groupby("entity", as_index=False).first()
    found = []
    for mkt, g in first.groupby("vendor_country"):
        d = g.sort_values("start_date").reset_index(drop=True)
        dates = d["start_date"].to_numpy()
        for i in range(len(d)):
            j = np.searchsorted(dates, dates[i] + np.timedelta64(days, "D"), side="right")
            n_new = j - i
            if n_new < min_new:
                continue
            lo = d.loc[i, "start_date"]
            prior = d[d["start_date"] < lo]
            years = max((lo - prior["start_date"].min()).days / 365.0, 1.0) if len(prior) else 1.0
            baseline = len(prior) / years
            ratio = n_new / max(baseline, 1.0)
            if ratio < min_ratio:
                continue
            found.append((mkt, i, j, lo, n_new, len(prior), baseline, ratio, d))
    # Keep the strongest window per market.
    best = {}
    for f in found:
        if f[0] not in best or (f[4], f[7]) > (best[f[0]][4], best[f[0]][7]):
            best[f[0]] = f
    out = []
    for mkt, (_, i, j, lo, n_new, n_prior, baseline, ratio, d) in best.items():
        win = d.iloc[i:j]
        top_sp = win["sponsor_employee_id"].value_counts(normalize=True)
        sp_share = float(top_sp.iloc[0]) if len(top_sp) else 0.0
        cpi = ctx.country_index(mkt, "CPI", lo)
        cpi_tier = int(cpi) if cpi is not None and str(cpi).isdigit() else None
        cats = win["cat_primary"].dropna()
        certain = float(win["cat_flags"].map(lambda f: "int" in f).mean())
        score = np.log2(ratio) + np.log2(n_new) + (0.5 * (cpi_tier - 2) if cpi_tier else 0) + sp_share
        benign = ctx.benign_events("onboarding", lo, win["start_date"].max(), mkt,
                                   cats=zip(win["parent"], win["cat_primary"]))
        if n_prior == 0 and len(ctx.vendors[ctx.vendors["country"] == mkt]) <= n_new + 2:
            benign.append("new market entry: no prior vendors of any kind")
        out.append(Candidate(
            members=ctx.expand(win["record_id"]), tags=("GEOGRAPHY_LINKED_CONCENTRATION",), score=float(score),
            emerged_at=pd.Timestamp(d["start_date"].iloc[i + min_new - 1]), active_until=win["start_date"].max(),
            detector="geography",
            reason=(f"{n_new} new intermediaries ({', '.join(sorted(set(cats)))}) in {mkt} within {days} days from "
                    f"{lo.date()} on top of {n_prior} earlier (baseline {baseline:.1f}/yr, {ratio:.1f}x); "
                    f"CPI tier {cpi_tier}; {sp_share:.0%} under one sponsor; int flag certain for {certain:.0%}"),
            evidence={"market": mkt, "window_start": lo, "engagements": list(win["engagement_id"]),
                      "cpi_tier": cpi_tier, "baseline_per_year": baseline},
            benign=benign, features={"n_new": n_new, "ratio": ratio, "cpi_tier": cpi_tier, "sponsor_share": sp_share},
            edges=tuple(engagement_edges(ctx, win["engagement_id"],
                                         ("CONTRACTED_UNDER", "SPONSORED_BY", "FOR_SERVICE"))),
        ))
    return out
