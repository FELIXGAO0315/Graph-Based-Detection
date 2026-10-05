"""Entity-level shared-relation items, weighted by rarity.

Each vendor entity carries items such as `sponsor:EMP-..` or `address:<key>`.
Two entities are related when they share an item. Items held by very many
entities (a central buyer, a registered agent, a generic mail domain) are
not expanded into all their pairs; their weight is also low.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from ..context import Context

# Relation types that define a structural link between vendors. Category and
# onboarding window are checked pairwise afterwards: they are too common to
# define links on their own.
LINK_TYPES = ("address", "phone", "domain", "sponsor", "approver", "owner", "institution")
# Relations that identify a party rather than reflect a shared business process.
IDENTITY_TYPES = ("address", "phone", "domain", "owner")


# The meta-path behind each relation type: two vendors share an item when their
# instances of the meta-path end on the same node.
METAPATHS = {
    "address": ("LOCATED_AT",),                                  # vendor -> address
    "contact": ("HAS_CONTACT",),                                 # vendor -> phone / email domain
    "sponsor": ("CONTRACTED_UNDER", "SPONSORED_BY"),             # vendor -> engagement -> employee
    "approver": ("CONTRACTED_UNDER", "APPROVED_BY"),             # vendor -> engagement -> employee (financial)
    "category": ("CONTRACTED_UNDER", "FOR_SERVICE"),             # vendor -> engagement -> category (unambiguous)
    "institution": (("SERVES", "SITE_OF", "AFFILIATED_WITH"),),  # vendor -> institution
    # owner: vendor -> registry entity <- (OWNS|CONTROLS)* <- owner, a variable-length
    # upward walk evaluated over each vendor's history (see ownership.OwnershipView)
}


def entity_items(ctx: Context, use_ownership: bool | None = None) -> dict[str, dict[str, set]]:
    """type -> item (the shared end node) -> set of vendor entities, from graph meta-paths."""
    from ..graph_ops import hetero, strip

    use_ownership = ctx.use_ownership if use_ownership is None else use_ownership
    hg = hetero(ctx)
    ent = ctx.entity_of
    items: dict[str, dict[str, set]] = {t: defaultdict(set) for t in LINK_TYPES + ("category",)}

    def fill(kind, mp, order, key=strip):
        mp = mp.sort_values(order, kind="stable")
        k = max(int(c[1:]) for c in mp.columns if c[0] == "n" and c[1:].isdigit())
        for start, end in zip(mp["n0"], mp[f"n{k}"]):
            items[kind][key(end)].add(ent[strip(start)])

    loc = hg.metapath(*METAPATHS["address"]).copy()
    loc["raw"] = [p.get("raw_address") for p in loc["props1"]]
    fill("address", loc, ["n0", "valid_from1", "observed_at1", "raw"],
         key=lambda n: n[len("address:"):])
    con = hg.metapath(*METAPATHS["contact"])
    fill("phone", con[con["n1"].str.startswith("phone:")], ["n0"])
    fill("domain", con[con["n1"].str.startswith("domain:")], ["n0"])
    fill("sponsor", hg.metapath(*METAPATHS["sponsor"]), ["n1"])
    certain = {2: lambda e: e["resolution_confidence"] == 1.0}
    cat = hg.metapath(*METAPATHS["category"], where=certain).sort_values(["n0", "n1"], kind="stable")
    for start, grp in cat.groupby("n0", sort=True):
        for c in pd.unique(grp["n2"]):
            items["category"][strip(c)].add(ent[strip(start)])
    financial = {2: lambda e: [p.get("approval_type") == "financial" for p in e["props"]]}
    fill("approver", hg.metapath(*METAPATHS["approver"], where=financial), ["source_record_id2"])
    fill("institution", hg.metapath(*METAPATHS["institution"]), ["source_record_id1"])
    if use_ownership:
        from .ownership import _view
        ov = _view(ctx)
        for rec in ctx.reg_entity_of:
            for m in ov.snapshots(rec):
                for p in m["persons"]:
                    items["owner"][p].add(ent[rec])
                for s in m["intermediates"]:
                    items["owner"][s].add(ent[rec])
        for rec, p in ctx.reg_person_of.items():
            items["owner"][p].add(ent[rec])
    return {t: dict(v) for t, v in items.items()}


def idf(n: int, total: int) -> float:
    return float(np.log(total / n) / np.log(total)) if n > 0 else 0.0


def pair_links(ctx: Context, items: dict, max_item_size: int = 30) -> dict[tuple, dict[str, list]]:
    """(entity_a, entity_b) -> {type: [shared items]} over LINK_TYPES, skipping hub items."""
    pairs: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for t in LINK_TYPES:
        for key, es in items.get(t, {}).items():
            if 2 <= len(es) <= max_item_size:
                for a, b in combinations(sorted(es), 2):
                    pairs[(a, b)][t].append(key)
    return pairs


def entity_onboarding(ctx: Context) -> pd.Series:
    v = ctx.vendors.assign(entity=ctx.vendors["record_id"].map(ctx.entity_of))
    return v.groupby("entity")["onboarding_date"].min()


def entity_country(ctx: Context) -> pd.Series:
    v = ctx.vendors.assign(entity=ctx.vendors["record_id"].map(ctx.entity_of))
    return v.groupby("entity")["country"].first()
