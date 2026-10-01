from dataclasses import replace

import pandas as pd

from meridian.context import Context
from meridian.data import as_of
from meridian.graph import build_graph, validate


def test_graph_is_valid_and_matches_the_source_tables(bundle):
    ctx = Context(as_of(bundle, "2026-06-30"))
    g = build_graph(ctx)
    assert validate(g) == []
    rel = g.edges["relation_type"].value_counts()
    assert rel["LOCATED_AT"] == len(ctx.addresses)
    assert rel["CONTRACTED_UNDER"] == len(ctx.engagements)
    assert rel["PART_OF"] == len(ctx.vendors)
    assert (g.nodes["node_type"] == "VendorRecord").sum() == len(ctx.vendors)


def test_same_inputs_give_the_same_hash_and_changed_inputs_do_not(bundle):
    h1 = build_graph(Context(as_of(bundle, "2026-06-30"))).hash
    h2 = build_graph(Context(as_of(bundle, "2026-06-30"))).hash
    assert h1 == h2
    a = bundle.tp_address_history.copy()
    a.loc[a.index[0], "raw_address"] = a.loc[a.index[0], "raw_address"] + " X"
    h3 = build_graph(Context(as_of(replace(bundle, tp_address_history=a), "2026-06-30"))).hash
    assert h3 != h1


def test_graph_as_of_an_earlier_date_holds_nothing_observed_later(bundle):
    t = pd.Timestamp("2022-06-30")
    g = build_graph(Context(as_of(bundle, t)))
    assert validate(g) == []
    assert (pd.to_datetime(g.edges["observed_at"]).dropna() <= t).all()
    assert (pd.to_datetime(g.edges["valid_from"]).dropna() <= t).all()
    full = build_graph(Context(as_of(bundle, "2026-06-30")))
    assert len(g.edges) < len(full.edges)


def test_ownership_switch_leaves_out_the_registries(bundle):
    g = build_graph(Context(as_of(bundle, "2026-06-30"), {"features": {"ownership": False}}))
    assert not g.nodes["node_type"].isin(["LegalEntity", "NaturalPerson", "AccountingParent"]).any()
