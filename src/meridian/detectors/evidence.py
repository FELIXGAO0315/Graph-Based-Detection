"""Look up the graph edges that support a detector's finding.

Each function returns edge ids from the canonical graph; detectors attach them
to their candidates so every case can be traced to the exact edges behind it.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..context import Context
from ..graph_ops import hetero
from .relations import METAPATHS

_END_PREFIX = {"address": "address:", "phone": "phone:", "domain": "domain:", "sponsor": "employee:",
               "approver": "employee:", "category": "category:", "institution": "institution:"}


def _index(ctx: Context, name: str, build):
    key = f"_evidence_{name}"
    if key not in ctx.__dict__:
        ctx.__dict__[key] = build()
    return ctx.__dict__[key]


def item_edges(ctx: Context, kind: str, item, records) -> list[str]:
    """Edges of the meta-path instances that link `records` to a shared item of type `kind`."""
    if kind == "owner":
        return owner_edges(ctx, records, item)
    steps = METAPATHS["contact" if kind in ("phone", "domain") else kind]
    mp = hetero(ctx).metapath(*steps)
    k = len(steps)
    end = _END_PREFIX[kind] + str(item)
    starts = {f"vendor:{r}" for r in records}
    hit = mp[(mp[f"n{k}"] == end) & mp["n0"].isin(starts)]
    if kind == "approver":
        hit = hit[[p.get("approval_type") == "financial" for p in hit["props2"]]]
    return [e for i in range(1, k + 1) for e in hit[f"e{i}"]]


def engagement_edges(ctx: Context, engagement_ids, relations=("CONTRACTED_UNDER", "SPONSORED_BY")) -> list[str]:
    """Edges that attach the given engagements to their vendor and sponsor (and other relations)."""
    def build():
        idx = defaultdict(list)
        for rel in ("CONTRACTED_UNDER", "SPONSORED_BY", "APPROVED_BY", "FOR_SERVICE"):
            e = hetero(ctx).edges(rel)
            eng = e["target_node_id"] if rel == "CONTRACTED_UNDER" else e["source_node_id"]
            for n, eid in zip(eng, e["edge_id"]):
                idx[(rel, n[len("engagement:"):])].append(eid)
        return idx
    idx = _index(ctx, "engagements", build)
    return [eid for en in engagement_ids for rel in relations for eid in idx.get((rel, en), [])]


def approval_edges(ctx: Context, engagement_ids, approver: str | None = None) -> list[str]:
    """Financial approval edges on the given engagements (optionally by one approver)."""
    e = hetero(ctx).edges("APPROVED_BY")
    ids = {f"engagement:{x}" for x in engagement_ids}
    keep = e["source_node_id"].isin(ids) & np.array([p.get("approval_type") == "financial" for p in e["props"]],
                                                     dtype=bool)
    if approver is not None:
        keep &= e["target_node_id"] == f"employee:{approver}"
    return list(e.loc[keep, "edge_id"])


def statement_edges(ctx: Context, statement_ids) -> list[str]:
    """Ownership/control edges published as the given registry statements."""
    def build():
        e = hetero(ctx).edges("OWNS", "CONTROLS")
        idx = defaultdict(list)
        for s, eid in zip(e["source_record_id"], e["edge_id"]):
            idx[s].append(eid)
        return idx
    idx = _index(ctx, "statements", build)
    return [eid for s in statement_ids for eid in idx.get(s, [])]


def registration_edges(ctx: Context, records) -> list[str]:
    """Accepted links from vendor records to registry entities or persons."""
    e = hetero(ctx).edges("REGISTERED_AS", "IS_PERSON")
    keep = e["source_node_id"].isin({f"vendor:{r}" for r in records}) & np.array(
        [p.get("status") == "accepted" for p in e["props"]], dtype=bool)
    return list(e.loc[keep, "edge_id"])


def path_edges(ctx: Context, record: str, paths) -> list[str]:
    """Registration edge plus the ownership edges along the given interest paths."""
    sids = [x["statement_id"] for p in paths for x in p if x.get("party_sid")]
    return registration_edges(ctx, [record]) + statement_edges(ctx, sids)


def owner_edges(ctx: Context, records, owner_sid: str) -> list[str]:
    """Edges by which each record reaches `owner_sid` (registration plus ownership chain)."""
    from .ownership import _view
    ov = _view(ctx)
    out = []
    for r in records:
        if ctx.reg_person_of.get(r) == owner_sid:
            out += registration_edges(ctx, [r])
            continue
        for m in ov.snapshots(r):
            hits = [p for p in m["paths"] if any(x.get("party_sid") == owner_sid for x in p)]
            if hits:
                out += path_edges(ctx, r, [p[:[x.get("party_sid") for x in p].index(owner_sid) + 1] for p in hits])
    return out


def shared_item_edges(ctx: Context, covered: dict, records) -> list[str]:
    """Edges linking `records` to the items they share: {type: item}. Window and category
    coverage without an end node (onboarding window) contribute no edge."""
    out = []
    for kind, item in covered.items():
        if kind in _END_PREFIX or kind == "owner":
            out += item_edges(ctx, kind, item, records)
    return out


def link_edges(ctx: Context, link_ids) -> list[str]:
    e = hetero(ctx).edges("SERVES", "SITE_OF", "AFFILIATED_WITH")
    return list(e.loc[e["source_record_id"].isin(set(link_ids)), "edge_id"])


def identity_edges(ctx: Context, records) -> list[str]:
    e = hetero(ctx).edges("SAME_AS")
    ids = {f"vendor:{r}" for r in records}
    return list(e.loc[e["source_node_id"].isin(ids) & e["target_node_id"].isin(ids), "edge_id"])


def address_edges(ctx: Context, records, since=None, until=None) -> list[str]:
    e = hetero(ctx).edges("LOCATED_AT")
    keep = e["source_node_id"].isin({f"vendor:{r}" for r in records})
    if since is not None:
        keep &= e["valid_from"] >= since
    if until is not None:
        keep &= e["valid_from"] <= until
    return list(e.loc[keep, "edge_id"])
