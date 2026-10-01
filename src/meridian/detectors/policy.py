"""Single-record policy findings, scored separately from the typologies (Dossier §4).

The FMV card's `country_band` has no market mapping in the bundle, so the
mapping below is our own documented choice, overridable in
the config under params.fmv.band_map.
"""
from __future__ import annotations

import pandas as pd

from ..context import Context
from .base import POLICY_TAG, Candidate

# Service payments counted against the annual cap. Investigators are paid as
# clinical_services; excluding it finds no breach on the starter data, so it is
# included by default and the narrow set is kept as a sensitivity option.
FEE_CLASSIFICATIONS = ("consulting_fees", "honoraria", "professional_services", "clinical_services")
EU_JP_EXTRA = {"JP", "CA", "AU", "NZ"}


def country_band(ctx: Context, market: str) -> str:
    override = ctx.param("fmv", "band_map", {})
    if market in override:
        return override[market]
    if market == "US":
        return "US"
    if market in EU_JP_EXTRA:
        return "EU_JP"
    if market in ctx.markets.index and ctx.markets.at[market, "region"] == "EUROPE":
        return "EU_JP"
    return "EMERGING"


def detect_fmv(ctx: Context) -> list[Candidate]:
    fees = tuple(ctx.param("fmv", "fee_classifications", FEE_CLASSIFICATIONS))
    ind = ctx.vendors[ctx.vendors["party_type"] == "individual"]
    p = ctx.b.tp_payments_monthly
    p = p[p["record_id"].isin(ind.index) & p["expense_classification"].isin(fees)]
    p = p.merge(ctx.engagements[["engagement_id", "start_date"]], on="engagement_id")
    cards = ctx.b.fmv_rate_cards
    caps = cards.groupby(["card_version", "effective_from", "effective_to", "country_band", "expert_role"],
                         dropna=False)["annual_cap_usd"].first().reset_index()

    def cap_for(role, band, when):
        c = caps[(caps["expert_role"] == role) & (caps["country_band"] == band) & (caps["effective_from"] <= when)
                 & (caps["effective_to"].isna() | (caps["effective_to"] >= when))]
        return (float(c["annual_cap_usd"].iloc[0]), c["card_version"].iloc[0]) if len(c) else (None, None)

    out = []
    p["year"] = p["month"].dt.year
    for (rec, year), g in p.groupby(["record_id", "year"]):
        role = ind.loc[rec, "expert_type"]
        band = country_band(ctx, ind.loc[rec, "country"])
        # Card in force on each engagement's date (Dossier §5); the most lenient applies to the year.
        caps_found = [cap_for(role, band, d) for d in g["start_date"].unique()]
        caps_found = [c for c in caps_found if c[0] is not None]
        if not caps_found:
            continue
        cap, version = max(caps_found)
        total = g["amount_usd"].sum()
        if total <= cap:
            continue
        g = g.sort_values("month")
        crossed = g[g["amount_usd"].cumsum() > cap]["month"].iloc[0]
        out.append(Candidate(
            members=ctx.expand([rec]), tags=(POLICY_TAG,), score=float(1.0 + (total / cap - 1)),
            emerged_at=crossed, active_until=pd.Timestamp(f"{year}-12-31"), detector="fmv_cap",
            reason=(f"expert ({role}, {ind.loc[rec, 'country']}, band {band}) paid USD {total:,.0f} in fees in {year}, "
                    f"{total / cap:.2f}x the annual cap of USD {cap:,.0f} (card {version}); cap crossed {crossed.date()}"),
            evidence={"year": int(year), "total_fees": float(total), "annual_cap": cap, "card": version,
                      "band": band, "band_mapping": "own mapping, see policy.country_band"},
            features={"excess_ratio": total / cap},
        ))
    return out
