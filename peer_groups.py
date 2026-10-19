"""Peer-group comparison with hierarchical backoff (dossier §2.3, §5, Appendix B).

A structural statistic is only informative relative to comparable entities: "37 active vendors" means
nothing without the peer median, percentile, peer definition and time window (§10.1). PeerScorer compares
each row with the most specific peer group that has at least `min_size` members and falls back level by
level otherwise (e.g. role × scope × market → role × scope × subregion → … → everyone).

Statistics: peer median, MAD, robust z = (x − median) / (1.4826 · MAD) and the mid-rank percentile.
When MAD is zero (common for counts) the scale falls back to the standard deviation; with no variation at
all the z-score is left empty and only the percentile is reported.

Also here: entity frames that carry the peer dimensions (employee_frame, vendor_frame, market_frame) and
the business-event calendar used to explain benign bursts (§1, §2.6).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_PEER_SIZE = 8  # team choice (dossier leaves it open); vary it in sensitivity runs

# Backoff hierarchies: most specific first, () = everyone. Each level is a tuple of column names.
EMPLOYEE_LEVELS = [("role", "scope_level", "home_market"), ("role", "scope_level", "subregion"),
                   ("role", "scope_level", "region"), ("role", "scope_level"), ("role",), ()]
VENDOR_LEVELS = [("party_type", "category_parent", "country"), ("party_type", "category_parent", "subregion"),
                 ("party_type", "category_parent", "region"), ("party_type", "category_parent"), ("party_type",), ()]
MARKET_LEVELS = [("subregion",), ("region",), ()]
ADDRESS_LEVELS = [("address_type", "country"), ("address_type", "subregion"), ("address_type",), ()]


@dataclass
class PeerScorer:
    levels: list
    min_size: int = MIN_PEER_SIZE

    def score(self, frame: pd.DataFrame, value: str) -> pd.DataFrame:
        """One row per input row: value, chosen peer level/key/size, median, MAD, robust z, percentile."""
        frame = frame.copy()
        out = pd.DataFrame(index=frame.index)
        out["value"] = frame[value].astype(float)
        out[["peer_level", "peer_key"]] = None
        out[["peer_n", "peer_median", "peer_mad", "peer_scale", "robust_z", "percentile"]] = np.nan
        remaining = frame[value].notna()
        for level in self.levels:
            if not remaining.any():
                break
            missing_dim = frame[list(level)].isna().any(axis=1) if level else pd.Series(False, index=frame.index)
            keys = frame[list(level)].astype(str).agg("|".join, axis=1) if level else pd.Series("all", index=frame.index)
            valid = frame[value].notna() & ~missing_dim
            sizes = keys[valid].map(keys[valid].value_counts())
            eligible = remaining & valid & sizes.reindex(frame.index).ge(self.min_size)
            if not eligible.any():
                continue
            for key, idx in keys[eligible].groupby(keys[eligible]).groups.items():
                peers = frame.loc[valid & keys.eq(key), value].astype(float)
                median = peers.median()
                mad = (peers - median).abs().median()
                scale = 1.4826 * mad if mad > 0 else peers.std(ddof=1)
                x = frame.loc[idx, value].astype(float)
                sorted_peers = np.sort(peers.to_numpy())
                below = np.searchsorted(sorted_peers, x.to_numpy(), side="left")
                equal = np.searchsorted(sorted_peers, x.to_numpy(), side="right") - below
                out.loc[idx, "peer_level"] = " × ".join(level) if level else "all"
                out.loc[idx, "peer_key"] = key
                out.loc[idx, "peer_n"] = len(peers)
                out.loc[idx, "peer_median"] = median
                out.loc[idx, "peer_mad"] = mad
                out.loc[idx, "peer_scale"] = scale if scale and scale > 0 else np.nan
                out.loc[idx, "robust_z"] = (x - median) / scale if scale and scale > 0 else np.nan
                out.loc[idx, "percentile"] = (below + 0.5 * equal) / len(peers)
            remaining &= ~eligible
        return out

    @staticmethod
    def level_report(scored: pd.DataFrame) -> pd.DataFrame:
        """How many rows were compared at each backoff level (report alongside any result)."""
        return scored.groupby("peer_level", dropna=False).agg(rows=("value", "size"), median_peer_n=("peer_n", "median"))


# ----------------------------------------------------------------------------- entity frames
def _strip(index):
    return index.str.split(":", n=1).str[1]


def scope_level(scope_markets):
    if pd.isna(scope_markets) or scope_markets == "":
        return "unknown"
    parts = str(scope_markets).split(";")
    if "*" in parts:
        return "global"
    if any(p.startswith("R:") for p in parts):
        return "region"
    if any(p.startswith("S:") for p in parts):
        return "subregion"
    return "multi_market" if len(parts) > 1 else "local"


def market_frame(graph) -> pd.DataFrame:
    m = graph.attributes["market"].copy()
    m.index = _strip(m.index)
    return m


def employee_frame(graph) -> pd.DataFrame:
    """Employees with their role at t and the peer dimensions (role, scope breadth, market hierarchy)."""
    e = graph.attributes["employee"].copy()
    e.index = _strip(e.index)
    markets = market_frame(graph)
    e["scope_level"] = e.scope_markets.map(scope_level)
    e["subregion"] = e.home_market.map(markets.subregion)
    e["region"] = e.home_market.map(markets.region)
    e["is_system_default"] = e.role.eq("system_default")
    return e


def engagement_frame(graph) -> pd.DataFrame:
    eng = graph.attributes["engagement"].copy()
    eng.index = _strip(eng.index)
    parents = graph.attributes["service_category"]["parent"]
    parents = parents.set_axis(_strip(parents.index))  # a new Series: never relabel the graph's own attribute table
    first_code = eng.category_v2.str.split(";").str[0].replace("", pd.NA)
    eng["category_parent"] = first_code.map(parents)  # v1 codes split across v2 codes share a parent
    return eng


def vendor_frame(graph) -> pd.DataFrame:
    """Vendor records with peer dimensions: party type, main category parent, country hierarchy."""
    v = graph.attributes["third_party"].copy()
    v.index = _strip(v.index)
    markets = market_frame(graph)
    eng = engagement_frame(graph)
    main = eng.dropna(subset=["category_parent"]).groupby("record_id").category_parent.agg(lambda s: s.value_counts().index[0])
    services = graph.edges_of("PROVIDES_SERVICE")
    parents = graph.attributes["service_category"].parent
    svc_parent = services.assign(record_id=_strip(pd.Index(services.source_node_id)),
                                 parent=services.target_node_id.map(parents)).dropna(subset=["parent"]) \
        .groupby("record_id").parent.agg(lambda s: s.value_counts().index[0])
    v["category_parent"] = main.reindex(v.index).fillna(svc_parent.reindex(v.index))
    v["subregion"] = v.country.map(markets.subregion)
    v["region"] = v.country.map(markets.region)
    resolved = graph.edges_of("RESOLVED_AS").set_index("source_node_id").target_node_id
    v["resolved_entity"] = resolved.reindex("third_party:" + v.index).to_numpy()
    return v


# ----------------------------------------------------------------------------- business events
# What each event can explain, read from its description in business_events.csv [data]:
# phenomena it produces and, where the description narrows it, the service-category parents affected.
EVENT_SCOPE = {
    "COVID-19 operating regime": ({"onboarding", "activity"}, {"SCM", "EVT"}),  # logistics surge, emergency sourcing, virtual events
    "ERP migration (legacy to new ERP)": ({"created_date", "system_sponsor"}, None),
    "Consumer healthcare demerger": ({"termination"}, None),
    "Biotech acquisition integration": ({"onboarding"}, None),
    "Adult respiratory vaccine launch": ({"onboarding", "activity"}, {"MED", "HCP", "EVT", "COM", "PAT"}),
    "Service taxonomy recode v1 to v2": ({"category_change"}, None),
    "IT / digital vendor rationalization": ({"onboarding", "termination"}, {"GNA"}),
}


def business_events(data_dir, t) -> pd.DataFrame:
    """Company events known by t (start <= t); open-ended events run to t."""
    ev = pd.read_csv(f"{data_dir}/business_events.csv", parse_dates=["start_date", "end_date"])
    ev = ev[ev.start_date.le(t)].copy()
    ev["end_date"] = ev.end_date.where(ev.end_date.notna() & ev.end_date.le(t), pd.Timestamp(t))
    ev["markets"] = ev.markets.fillna("ALL").str.split("|")
    return ev.reset_index(drop=True)


def events_covering(events: pd.DataFrame, date, market=None, margin_days=0, phenomenon=None, category_parent=None):
    """Business events whose window (± margin) covers `date`, include `market`, and — when given — can explain
    the phenomenon (e.g. "onboarding") for the category parent, per EVENT_SCOPE."""
    if pd.isna(date):
        return []
    margin = pd.Timedelta(days=margin_days)
    hits = events[(events.start_date - margin).le(date) & (events.end_date + margin).ge(date)]
    if market is not None:
        hits = hits.loc[hits.markets.map(lambda m: "ALL" in m or market in m).astype(bool)]
    names = []
    for name in hits.event_name:
        phenomena, parents = EVENT_SCOPE.get(name, (None, None))
        if phenomenon is not None and phenomena is not None and phenomenon not in phenomena:
            continue
        if category_parent is not None and parents is not None and category_parent not in parents:
            continue
        names.append(name)
    return names
