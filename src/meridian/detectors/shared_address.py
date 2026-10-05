"""SHARED_ADDRESS: several unrelated vendors registered at one address at the same time."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..context import Context
from .base import Candidate

# Address types where co-location is ordinary; they lower the score, never remove it.
BENIGN_TYPES = {"registered_agent", "coworking", "virtual_office", "hospital_campus", "corporate_hq"}
RARE_TYPES = {"residential"}


def _episodes(ctx: Context, g: pd.DataFrame, k: int) -> list[dict]:
    """Co-location episodes at one address.

    Sweeps the validity intervals. An episode starts on the day the number of
    concurrent vendor entities reaches k and ends on the day it falls below k
    again. Returns, per episode: the peak entity count, the records present at
    the peak, the day k was reached and the day it ended (None if ongoing).
    """
    events = []
    for r in g.itertuples():
        events.append((r.valid_from, 1, r.record_id))
        if pd.notna(r.valid_to):
            events.append((r.valid_to, 0, r.record_id))  # ends sort before starts on the same day
    events.sort(key=lambda x: (x[0], x[1]))
    cur: dict[str, int] = {}
    episodes, ep, below_since = [], None, None
    for d, is_start, rid in events:
        if below_since is not None and d > below_since:  # still below k at the end of that day
            ep["ended"] = below_since
            episodes.append(ep)
            ep = below_since = None
        if is_start:
            cur[rid] = cur.get(rid, 0) + 1
        else:
            cur[rid] -= 1
            if cur[rid] == 0:
                del cur[rid]
        n = ctx.n_entities(cur)
        if n >= k:
            below_since = None
            if ep is None:
                ep = {"peak": 0, "members": set(), "reached": d, "ended": None}
            if n > ep["peak"]:
                ep["peak"], ep["members"] = n, set(cur)
        elif ep is not None and below_since is None:
            below_since = d
    if ep is not None:
        ep["ended"] = below_since
        episodes.append(ep)
    return episodes


def address_instances(ctx: Context) -> pd.DataFrame:
    """Instances of the meta-path VendorRecord -LOCATED_AT-> Address, one row per edge."""
    from ..graph_ops import hetero
    mp = hetero(ctx).metapath("LOCATED_AT")
    a = pd.DataFrame({
        "record_id": mp["n0"].str.slice(len("vendor:")), "addr_key": mp["n1"].str.slice(len("address:")),
        "edge_id": mp["e1"], "valid_from": mp["valid_from1"], "valid_to": mp["valid_to1"],
        "observed_at": mp["observed_at1"], "raw_address": [p.get("raw_address") for p in mp["props1"]],
        "address_type": [p.get("address_type") for p in mp["props1"]],
    })
    return a.sort_values(["record_id", "valid_from", "observed_at", "raw_address"], kind="stable")


def detect(ctx: Context) -> list[Candidate]:
    """Address nodes reached by several vendor entities through overlapping LOCATED_AT edges.

    One candidate per co-location episode: a second group at the same address
    in a later period is a separate pattern.
    """
    k = ctx.param("shared_address", "min_entities", 3)
    a = address_instances(ctx)
    sizes = a.groupby("addr_key")["record_id"].transform("nunique")
    a = a[sizes >= 2]
    out = []
    for key, g in a.groupby("addr_key"):
        for ep in _episodes(ctx, g, k):
            out.append(_candidate(ctx, key, g, ep, k))
    return out


def _candidate(ctx: Context, key: str, g: pd.DataFrame, ep: dict, k: int) -> Candidate:
    peak, reached, ended = ep["peak"], ep["reached"], ep["ended"]
    members = ctx.expand(ep["members"])
    types = g["address_type"].dropna()
    atype = types.mode().iloc[0] if len(types) else None
    v = ctx.vendors.loc[sorted(members)]
    groups_declared = v["declared_parent_group"].dropna()
    benign = []
    if atype in BENIGN_TYPES:
        benign.append(f"address type {atype}")
    if len(groups_declared) and groups_declared.nunique() == 1 and len(groups_declared) >= 0.5 * len(v):
        benign.append(f"declared group {groups_declared.iloc[0]}")
    if atype is None:
        type_w = 1.0
    else:
        type_w = 1.3 if atype in RARE_TYPES else (0.5 if atype in BENIGN_TYPES else 1.0)
    score = float(np.log2(1 + peak) * type_w)
    raw = g["raw_address"].iloc[0]
    # LOCATED_AT edges of the members that were valid at some point during the episode
    during = (g["valid_from"] <= (ended if ended is not None else pd.Timestamp.max)) & (
        g["valid_to"].isna() | (g["valid_to"] >= reached))
    return Candidate(
        members=members, tags=("SHARED_ADDRESS",), score=score, emerged_at=reached, active_until=ended,
        detector="shared_address",
        reason=(f"{peak} vendors registered concurrently at one {atype or 'unclassified'} address "
                f"in {key.split(':')[0]} ('{raw}'); threshold {k} reached {reached.date()}"
                + (f", fell below it {ended.date()}" if ended is not None else "")),
        evidence={"address_key": key, "raw_addresses": sorted(set(g["raw_address"])), "address_type": atype,
                  "peak_entities": peak, "reached": reached, "ended": ended},
        benign=benign, features={"peak_entities": peak, "type_weight": type_w},
        edges=tuple(g.loc[g["record_id"].isin(members) & during, "edge_id"]),
    )


def detect_exact(ctx: Context) -> list[Candidate]:
    """Simplest baseline: identical raw address strings, ignoring time. Kept for the representation ladder."""
    a = ctx.b.tp_address_history
    out = []
    for addr, g in a.groupby("raw_address"):
        members = set(g["record_id"])
        if len(members) < 2:
            continue
        out.append(Candidate(members=members, tags=("SHARED_ADDRESS",), score=float(len(members)),
                             emerged_at=g["valid_from"].sort_values().iloc[1], detector="address_exact",
                             reason=f"{len(members)} records share the raw address '{addr}'"))
    return out
