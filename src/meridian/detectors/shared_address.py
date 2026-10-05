"""SHARED_ADDRESS: several unrelated vendors registered at one address at the same time."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..context import Context
from .base import Candidate

# Address types where co-location is ordinary; they lower the score, never remove it.
BENIGN_TYPES = {"registered_agent", "coworking", "virtual_office", "hospital_campus", "corporate_hq"}
RARE_TYPES = {"residential"}


def _peak_concurrency(ctx: Context, g: pd.DataFrame, k: int):
    """Sweep validity intervals; return (peak entity count, records at peak, date k was first reached)."""
    events = []
    for r in g.itertuples():
        events.append((r.valid_from, 1, r.record_id))
        if pd.notna(r.valid_to):
            events.append((r.valid_to, 0, r.record_id))  # ends sort before starts on the same day
    events.sort(key=lambda x: (x[0], x[1]))
    cur: dict[str, int] = {}
    best, best_set, reached = 0, set(), None
    for d, is_start, rid in events:
        if is_start:
            cur[rid] = cur.get(rid, 0) + 1
        else:
            cur[rid] -= 1
            if cur[rid] == 0:
                del cur[rid]
        n = ctx.n_entities(cur)
        if n >= k and reached is None:
            reached = d
        if n > best:
            best, best_set = n, set(cur)
    return best, best_set, reached


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
    """Address nodes reached by several vendor entities through overlapping LOCATED_AT edges."""
    k = ctx.param("shared_address", "min_entities", 3)
    a = address_instances(ctx)
    sizes = a.groupby("addr_key")["record_id"].transform("nunique")
    a = a[sizes >= 2]
    out = []
    for key, g in a.groupby("addr_key"):
        peak, members, reached = _peak_concurrency(ctx, g, k)
        if peak < k:
            continue
        members = ctx.expand(members)
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
        out.append(Candidate(
            members=members, tags=("SHARED_ADDRESS",), score=score, emerged_at=reached,
            detector="shared_address",
            reason=(f"{peak} vendors registered concurrently at one {atype or 'unclassified'} address "
                    f"in {key.split(':')[0]} ('{raw}'); threshold {k} reached {reached.date()}"),
            evidence={"address_key": key, "raw_addresses": sorted(set(g["raw_address"])), "address_type": atype,
                      "peak_entities": peak, "reached": reached},
            benign=benign, features={"peak_entities": peak, "type_weight": type_w},
            edges=tuple(g.loc[g["record_id"].isin(members), "edge_id"]),
        ))
    return out


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
