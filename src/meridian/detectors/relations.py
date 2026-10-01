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


def entity_items(ctx: Context, use_ownership: bool | None = None) -> dict[str, dict[str, set]]:
    """type -> item -> set of entities."""
    use_ownership = ctx.use_ownership if use_ownership is None else use_ownership
    ent = ctx.entity_of
    items: dict[str, dict[str, set]] = {t: defaultdict(set) for t in LINK_TYPES + ("category",)}

    for r in ctx.addresses.itertuples():
        items["address"][r.addr_key].add(ent[r.record_id])
    for r in ctx.contacts.itertuples():
        if isinstance(r.phone_key, str):
            items["phone"][r.phone_key].add(ent[r.record_id])
        if isinstance(r.domain_key, str):
            items["domain"][r.domain_key].add(ent[r.record_id])
    e = ctx.engagements
    for r in e.dropna(subset=["sponsor_employee_id"]).itertuples():
        items["sponsor"][r.sponsor_employee_id].add(ent[r.record_id])
    for rec, cats in e.groupby("record_id")["cat_primary"]:
        for c in cats.dropna().unique():
            items["category"][c].add(ent[rec])
    ap = ctx.approvals
    fin = ap[(ap["approval_type"] == "financial") & ap["approver_employee_id"].notna()]
    for r in fin.itertuples():
        items["approver"][r.approver_employee_id].add(ent[r.record_id])
    for r in ctx.links.itertuples():
        items["institution"][r.institution_id].add(ent[r.record_id])
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
