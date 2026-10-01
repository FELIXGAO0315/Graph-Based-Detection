"""Explicit features for the representation ladder.

Record features (level A) use only the record itself, its own engagements'
attributes and payments, and reference tables. Case features describe any
member set at four levels:

  A  aggregated record features of the members
  B  deterministic relation rules among the members (address, sponsor, approver, timing, market)
  C  graph structure of the members in the multi-relation vendor graph
  D  ownership (registry R paths)

Nothing here reads labels.
"""
from __future__ import annotations

from collections import Counter
from functools import cached_property

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from .context import Context
from .detectors.policy import FEE_CLASSIFICATIONS, country_band

LEVELS = ("A", "B", "C", "D")


# ---------------------------------------------------------------------------
# Record-level features (A)
# ---------------------------------------------------------------------------
def record_features(ctx: Context) -> pd.DataFrame:
    t = ctx.t
    tp = ctx.vendors
    f = pd.DataFrame(index=tp.index)
    dd = ctx.b.tp_due_diligence.set_index("record_id").reindex(tp.index)
    f["dd_enhanced"] = dd["dd_level"].eq("enhanced").astype(float)
    f["risk_rating"] = dd["risk_rating"].map({"low": 0, "medium": 1, "high": 2}).fillna(0)
    for q in ["q_gov_ownership", "q_pep_connected", "q_public_official_interaction", "q_former_public_official",
              "adverse_media"]:
        f[q] = dd[q].astype(float).fillna(0)
    f["screen_hit"] = dd["screening_result"].eq("potential_match_cleared").astype(float)
    f["individual"] = tp["party_type"].eq("individual").astype(float)
    f["remit_mismatch"] = (tp["remit_to_country"] != tp["country"]).astype(float)
    f["no_lei"] = tp["lei"].isna().astype(float)
    f["no_registry_no"] = tp["registry_no"].isna().astype(float)
    f["no_declared_group"] = tp["declared_parent_group"].isna().astype(float)
    f["tenure_years"] = (t - tp["onboarding_date"]).dt.days / 365.25
    f["terminated"] = tp["terminated_date"].notna().astype(float)
    cpi = {c: ctx.country_index(c, "CPI") for c in tp["country"].unique()}
    f["cpi_tier"] = tp["country"].map(lambda c: float(cpi[c]) if cpi[c] is not None and str(cpi[c]).isdigit() else 2.0)

    a = ctx.addresses.sort_values("valid_from")
    cur = a.groupby("record_id").last()
    f["n_addresses"] = a.groupby("record_id").size().reindex(tp.index).fillna(0)
    f["addr_residential"] = cur["address_type"].eq("residential").reindex(tp.index, fill_value=False).astype(float)
    f["addr_virtual"] = cur["address_type"].isin(["virtual_office", "po_box"]).reindex(tp.index, fill_value=False).astype(float)
    f["addr_agent"] = cur["address_type"].eq("registered_agent").reindex(tp.index, fill_value=False).astype(float)
    f["addr_foreign"] = (cur["address_country"] != tp["country"].reindex(cur.index)).reindex(
        tp.index, fill_value=False).astype(float)

    e = ctx.engagements.copy()
    e["vendor_country"] = e["record_id"].map(tp["country"])
    g = e.groupby("record_id")
    f["n_engagements"] = g.size().reindex(tp.index).fillna(0)
    f["max_value_band"] = g["value_band"].max().reindex(tp.index).fillna(0)
    for flag in ["int", "gov", "hcp", "flow"]:
        f[f"share_{flag}"] = g["cat_flags_possible"].apply(lambda s: np.mean([flag in x for x in s])).reindex(tp.index).fillna(0)
    f["n_categories"] = g["cat_primary"].nunique().reindex(tp.index).fillna(0)
    f["share_cross_border"] = g.apply(lambda d: (d["market"] != d["vendor_country"]).mean(),
                                      include_groups=False).reindex(tp.index).fillna(0)
    f["share_no_contract"] = g["contract_type"].apply(lambda s: (s == "none").mean()).reindex(tp.index).fillna(0)

    p = ctx.b.tp_payments_monthly.copy()
    p = p.merge(e[["engagement_id", "value_band", "end_date"]], on="engagement_id", how="left")
    pg = p.groupby("record_id")
    monthly = p.groupby(["record_id", "month"])["amount_usd"].sum()
    mg = monthly.groupby(level=0)
    f["log_total_spend"] = np.log1p(pg["amount_usd"].sum()).reindex(tp.index).fillna(0)
    f["n_paid_months"] = mg.size().reindex(tp.index).fillna(0)
    f["spend_spikiness"] = (mg.max() / mg.median().clip(lower=1)).reindex(tp.index).fillna(0)
    f["round_share"] = pg["round_amount_share"].mean().reindex(tp.index).fillna(0)
    f["manual_share"] = (pg["manual_payment_count"].sum() / pg["payment_count"].sum().clip(lower=1)).reindex(tp.index).fillna(0)
    nonpo = p.assign(np_amt=np.where(p["channel"] != "po_backed", p["amount_usd"], 0.0)).groupby("record_id")
    f["non_po_share"] = (nonpo["np_amt"].sum() / nonpo["amount_usd"].sum().clip(lower=1)).reindex(tp.index).fillna(0)
    f["invoice_over_band"] = (p["max_invoice_band"] > p["value_band"]).groupby(p["record_id"]).sum().reindex(tp.index).fillna(0)
    f["foreign_payer_share"] = (p["paying_market"] != p["vendor_country"]).groupby(p["record_id"]).mean().reindex(tp.index).fillna(0)
    term = tp["terminated_date"]
    p["term"] = p["record_id"].map(term)
    f["months_after_termination"] = (p["month"] > p["term"] + pd.offsets.MonthEnd(0)).groupby(
        p["record_id"]).sum().reindex(tp.index).fillna(0)
    f["months_after_contract_end"] = (p["end_date"].notna() & (p["month"] > p["end_date"] + pd.offsets.MonthEnd(1))).groupby(
        p["record_id"]).sum().reindex(tp.index).fillna(0)

    # FMV: worst annual fee total relative to the cap (individuals only).
    ind = tp[tp["party_type"] == "individual"]
    pf = p[p["record_id"].isin(ind.index) & p["expense_classification"].isin(FEE_CLASSIFICATIONS)].copy()
    pf["year"] = pf["month"].dt.year
    cards = ctx.b.fmv_rate_cards
    ratio = {}
    for (rec, year), s in pf.groupby(["record_id", "year"])["amount_usd"]:
        band = country_band(ctx, ind.loc[rec, "country"])
        version = "v2" if year >= 2024 else "v1"
        cap = cards[(cards["card_version"] == version) & (cards["country_band"] == band)
                    & (cards["expert_role"] == ind.loc[rec, "expert_type"])]["annual_cap_usd"]
        if len(cap):
            ratio[rec] = max(ratio.get(rec, 0.0), s.sum() / float(cap.iloc[0]))
    f["fmv_ratio"] = pd.Series(ratio).reindex(tp.index).fillna(0)
    return f.astype(float)


def record_outlier_scores(feat: pd.DataFrame, seed: int = 0) -> pd.Series:
    """Isolation-forest outlier score on record features (higher = more unusual)."""
    x = RobustScaler(quantile_range=(10, 90)).fit_transform(feat.to_numpy())
    x = np.clip(x, -10, 10)
    iso = IsolationForest(n_estimators=300, random_state=seed).fit(x)
    return pd.Series(-iso.score_samples(x), index=feat.index)


# ---------------------------------------------------------------------------
# Case-level features (A-D)
# ---------------------------------------------------------------------------
class CaseFeaturizer:
    """Features of arbitrary member sets, grouped by ladder level."""

    def __init__(self, ctx: Context, seed: int = 0):
        self.ctx = ctx
        self.seed = seed

    @cached_property
    def rec(self) -> pd.DataFrame:
        f = record_features(self.ctx)
        f["outlier"] = record_outlier_scores(f, self.seed)
        return f

    @cached_property
    def relations(self):
        from .detectors.relations import entity_items, pair_links
        items = entity_items(self.ctx, use_ownership=False)
        return items, pair_links(self.ctx, items)

    @cached_property
    def graph(self) -> nx.Graph:
        """Entity graph weighted by the number of shared (non-owner) relation types."""
        _, pairs = self.relations
        g = nx.Graph()
        for (a, b), types in pairs.items():
            g.add_edge(a, b, weight=len(types))
        return g

    @cached_property
    def communities(self) -> dict:
        comms = nx.community.louvain_communities(self.graph, weight="weight", seed=self.seed)
        return {n: i for i, c in enumerate(comms) for n in c}

    @cached_property
    def sponsor_burst(self) -> dict:
        from .detectors.sponsor import best_window, first_engagements
        fe = first_engagements(self.ctx)
        out = {}
        for sp, g in fe.groupby("sponsor_employee_id"):
            out[sp] = best_window(np.sort(g["start_date"].to_numpy()), 60)[1]
        role = self.ctx.role_of
        med = pd.Series(out).groupby(lambda s: role.get(s)).median()
        return {s: n / max(med.get(role.get(s), 1.0), 1.0) for s, n in out.items()}

    @cached_property
    def fin_approvals(self) -> pd.DataFrame:
        ap = self.ctx.approvals
        fin = ap[(ap["approval_type"] == "financial")].copy()
        fin["under"] = fin["ap_approver_level"] < fin["required_approver_level"]
        return fin

    @cached_property
    def spikes(self) -> pd.DataFrame:
        from .detectors.activity import spend_spikes
        return spend_spikes(self.ctx, 2.5)

    @cached_property
    def ownership(self):
        from .detectors.ownership import _view, person_holdings
        return _view(self.ctx), person_holdings(self.ctx)

    # ------------------------------------------------------------------
    def level_a(self, members: frozenset) -> dict:
        r = self.rec.loc[sorted(members)]
        return {
            "A_size": float(np.log2(1 + len(members))),
            "A_outlier_max": r["outlier"].max(), "A_outlier_mean": r["outlier"].mean(),
            "A_fmv_ratio": r["fmv_ratio"].max(), "A_after_term": r["months_after_termination"].max(),
            "A_after_end": r["months_after_contract_end"].max(), "A_risk": r["risk_rating"].mean(),
            "A_share_int": r["share_int"].mean(), "A_cpi": r["cpi_tier"].mean(),
            "A_residential": r["addr_residential"].mean(), "A_tenure": r["tenure_years"].min(),
            "A_spiky": r["spend_spikiness"].max(), "A_individual": r["individual"].mean(),
        }

    def level_b(self, members: frozenset) -> dict:
        ctx = self.ctx
        n = len(members)
        a = ctx.addresses[ctx.addresses["record_id"].isin(members)]
        addr_share = a.groupby("addr_key")["record_id"].nunique().max() / n if len(a) else 0.0
        e = ctx.engagements[ctx.engagements["record_id"].isin(members)]
        sp_counts = e.dropna(subset=["sponsor_employee_id"]).groupby("sponsor_employee_id")["record_id"].nunique()
        top_sp = sp_counts.idxmax() if len(sp_counts) else None
        fin = self.fin_approvals[self.fin_approvals["record_id"].isin(members)]
        pair = fin.dropna(subset=["approver_employee_id", "sponsor_employee_id"]).groupby(
            ["sponsor_employee_id", "approver_employee_id"])["record_id"].nunique()
        onb = ctx.vendors.loc[sorted(members), "onboarding_date"]
        span = (onb.max() - onb.min()).days
        sk = self.spikes[self.spikes["record_id"].isin(members)]
        sync = 0.0
        if len(sk):
            by_month = sk.groupby(sk["month"].dt.to_period("M"))["record_id"].nunique()
            roll = by_month.rolling(2, min_periods=1).sum() if len(by_month) > 1 else by_month
            sync = float(roll.max()) / n
        li = ctx.links[ctx.links["record_id"].isin(members)]
        foreign = (li["inst_country"] != li["record_id"].map(ctx.vendors["country"])).sum()
        cats = e["cat_primary"].dropna()
        return {
            "B_addr_share": addr_share,
            "B_sponsor_share": sp_counts.max() / n if len(sp_counts) else 0.0,
            "B_sponsor_burst": self.sponsor_burst.get(top_sp, 0.0) if top_sp else 0.0,
            "B_pair_share": pair.max() / n if len(pair) else 0.0,
            "B_under_authority": float(fin["under"].sum()),
            "B_onboard_span": float(np.log1p(span)),
            "B_market_share": ctx.vendors.loc[sorted(members), "country"].value_counts(normalize=True).iloc[0],
            "B_category_share": cats.value_counts(normalize=True).iloc[0] if len(cats) else 0.0,
            "B_sync_spikes": sync,
            "B_foreign_links": float(foreign),
        }

    def level_c(self, members: frozenset) -> dict:
        ctx = self.ctx
        ents = {ctx.entity_of[r] for r in members}
        g = self.graph
        inside = [e for e in ents if e in g]
        sub = g.subgraph(inside)
        k = len(ents)
        w_in = sum(d["weight"] for _, _, d in sub.edges(data=True))
        w_all = sum(d["weight"] for e in inside for _, _, d in g.edges(e, data=True))
        possible = k * (k - 1) / 2
        comm = Counter(self.communities.get(e) for e in inside)
        return {
            "C_density": sub.number_of_edges() / possible if possible else 0.0,
            "C_mean_types": w_in / possible if possible else 0.0,
            "C_conductance": 1 - (2 * w_in / w_all) if w_all else 1.0,
            "C_community_purity": comm.most_common(1)[0][1] / k if comm else 0.0,
            "C_max_degree": float(max((g.degree(e) for e in inside), default=0)),
        }

    def level_d(self, members: frozenset) -> dict:
        ctx = self.ctx
        ov, holdings = self.ownership
        ents = {ctx.entity_of[r] for r in members}
        best_share = 0.0
        for person, held in holdings.items():
            share = len(set(held) & ents) / len(ents)
            best_share = max(best_share, share if len(set(held) & ents) >= 1 else 0.0)
        layers = jur = sink = pep = undisclosed = 0.0
        for r in members:
            for m in ov.snapshots(r):
                layers = max(layers, m["layers"])
                jur = max(jur, len(m["jurisdictions"]))
                sink = max(sink, float(bool(m["sink"])))
                pep = max(pep, float(bool(m["peps"])))
                if m["peps"] and not bool(ov.dd.loc[r, "q_pep_connected"]):
                    undisclosed = 1.0
        linked = sum(r in ctx.reg_entity_of or r in ctx.reg_person_of for r in members) / len(members)
        return {"D_common_owner_share": best_share if len(ents) > 1 else 0.0, "D_layers": float(layers),
                "D_jurisdictions": float(jur), "D_sink": sink, "D_pep": pep, "D_pep_undisclosed": undisclosed,
                "D_registry_linked": linked}

    def features(self, members: frozenset, level: str = "D") -> dict:
        out = self.level_a(members)
        if level in ("B", "C", "D"):
            out.update(self.level_b(members))
        if level in ("C", "D"):
            out.update(self.level_c(members))
        if level == "D":
            out.update(self.level_d(members))
        return out

    def matrix(self, member_sets: list[frozenset], level: str) -> pd.DataFrame:
        return pd.DataFrame([self.features(m, level) for m in member_sets]).astype(float).fillna(0.0)
