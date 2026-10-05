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


GRAPH_TABLES = ["third_parties", "tp_due_diligence", "tp_address_history", "tp_services", "tp_engagements",
                "tp_engagement_approvals", "tp_payments_monthly", "tp_institution_links", "healthcare_institutions",
                "employees", "employee_roles", "reg_entities", "reg_persons", "reg_interests"]
NOT_IN_GRAPH = {"reg_record_id", "reg_scheme", "person_type"}  # registry bookkeeping no detector reads


def test_every_table_the_detectors_read_round_trips_through_the_graph(bundle):
    from pandas.testing import assert_frame_equal
    from meridian.graph_views import bundle_from_graph
    view = as_of(bundle, "2026-06-30")
    etl = Context(view)
    rebuilt, pre = bundle_from_graph(build_graph(etl), view)
    for name in GRAPH_TABLES:
        src = getattr(view, name)
        cols = [c for c in src.columns if c not in NOT_IN_GRAPH]
        assert_frame_equal(getattr(rebuilt, name)[cols], src[cols], obj=name)
    assert_frame_equal(pre["addresses"], etl.addresses, obj="addresses")
    assert_frame_equal(pre["contacts"], etl.contacts, obj="contacts")
    assert_frame_equal(pre["entity_map"], etl.entity_map.sort_values("record_id").reset_index(drop=True))
    assert_frame_equal(pre["reg_matches"],
                       etl.reg_matches.sort_values(["record_id", "statement_id"]).reset_index(drop=True))


def test_detection_runs_on_the_graph_and_cites_its_edges(bundle):
    from meridian.detectors import run_detectors
    ctx = Context.for_detection(as_of(bundle, "2026-06-30"))
    cands = run_detectors(ctx)
    ids = set(ctx.graph.edges["edge_id"])
    assert all(e in ids for c in cands for e in c.edges)
    # Every relational finding cites edges; only record-level outlier scores may not.
    assert all(c.edges for c in cands if not c.timeless)


def test_removing_an_edge_type_removes_what_rests_on_it(bundle):
    from meridian.detectors.shared_address import detect
    view = as_of(bundle, "2026-06-30")
    assert detect(Context.for_detection(view))
    assert not detect(Context.for_detection(view, drop_relations=["LOCATED_AT"]))


def test_detection_survives_removing_whole_relation_families(bundle):
    from meridian.detectors import run_detectors
    view = as_of(bundle, "2026-06-30")
    for rels in (["APPROVED_BY"], ["REGISTERED_AS", "IS_PERSON", "OWNS", "CONTROLS", "CONSOLIDATED_BY"]):
        ctx = Context.for_detection(view, drop_relations=rels)
        cands = run_detectors(ctx)
        assert cands
        assert not ctx.graph.edges["relation_type"].isin(rels).any()
        ids = set(ctx.graph.edges["edge_id"])
        assert all(e in ids for c in cands for e in c.edges)
