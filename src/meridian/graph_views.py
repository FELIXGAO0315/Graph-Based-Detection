"""Relational views derived from the canonical graph.

Detection reads the graph, not the source files: `bundle_from_graph` rebuilds
every table the detectors use from node attributes, typed edges and the
activity table, and takes entity resolution, normalised addresses and
contact keys straight from the graph instead of recomputing them. Only
reference tables (markets, service taxonomy, approval policy, FMV rate cards,
country context, business events) come from outside the graph.

A round-trip test checks that these views equal the source tables, which is
the evidence that building the graph loses nothing the detectors need.
"""
from __future__ import annotations

import json

import pandas as pd

from .data import DATE_COLUMNS, Bundle, canonical
from .graph import Graph

REFERENCE_TABLES = ["markets", "paying_entities", "service_categories", "approval_policy_matrix", "fmv_rate_cards",
                    "country_risk_context", "business_events", "employee_delegations"]
DD_COLUMNS = ["dd_level", "dd_completed", "dd_refresh_due", "q_gov_ownership", "q_pep_connected",
              "q_public_official_interaction", "q_former_public_official", "screening_result", "adverse_media",
              "risk_rating"]
LINK_TYPES = ("SERVES", "SITE_OF", "AFFILIATED_WITH")


def _dd_attr(col: str) -> str:
    return col if col.startswith("dd_") else f"dd_{col}"


def _strip(ids: pd.Series, prefix: str) -> pd.Series:
    return ids.str.slice(len(prefix) + 1)


def _frame(rows: list[dict], columns: list[str], dates=(), ints=(), bools=()) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=columns)
    for c in dates:
        df[c] = pd.to_datetime(df[c]).astype("datetime64[ns]")
    for c in ints:
        df[c] = df[c].astype("int64")
    for c in bools:
        df[c] = df[c].astype(bool)
    return df


class _Index:
    def __init__(self, g: Graph):
        self.g = g
        self.attrs = {nid: json.loads(a) for nid, a in zip(g.nodes["node_id"], g.nodes["attributes"])}
        self.label = dict(zip(g.nodes["node_id"], g.nodes["label"]))
        self.type = dict(zip(g.nodes["node_id"], g.nodes["node_type"]))

    def nodes(self, node_type: str) -> pd.DataFrame:
        return self.g.nodes[self.g.nodes["node_type"] == node_type]

    def edges(self, *relations: str) -> pd.DataFrame:
        e = self.g.edges[self.g.edges["relation_type"].isin(relations)].copy()
        e["props"] = [json.loads(p) for p in e["relation_properties"]]
        return e


def bundle_from_graph(g: Graph, reference: Bundle) -> tuple[Bundle, dict]:
    """(Bundle of graph-derived tables plus reference tables, precomputed resolution views)."""
    ix = _Index(g)
    t = {}

    # Vendors and due diligence.
    vend = ix.nodes("VendorRecord")
    rows, dd_rows = [], []
    for nid, label in zip(vend["node_id"], vend["label"]):
        a = ix.attrs[nid]
        rid = nid[len("vendor:"):]
        rows.append(dict(record_id=rid, party_type=a.get("party_type"), legal_name=label, tax_id=a.get("tax_id"),
                         lei=a.get("lei"), registry_no=a.get("registry_no"), country=a.get("country"),
                         status=a.get("status"), onboarding_date=a.get("onboarding_date"),
                         created_date=a.get("created_date"), terminated_date=a.get("terminated_date"),
                         declared_parent_group=a.get("declared_parent_group"), phone=a.get("phone"),
                         email_domain=a.get("email_domain"), remit_to_country=a.get("remit_to_country"),
                         expert_type=a.get("expert_type")))
        if any(_dd_attr(c) in a for c in DD_COLUMNS):
            dd_rows.append({**{c: a.get(_dd_attr(c)) for c in DD_COLUMNS}, "record_id": rid})
    t["third_parties"] = _frame(rows, list(rows[0]) if rows else [], dates=DATE_COLUMNS["third_parties"])
    t["tp_due_diligence"] = _frame(dd_rows, DD_COLUMNS + ["record_id"], dates=DATE_COLUMNS["tp_due_diligence"],
                                   bools=[c for c in DD_COLUMNS if c.startswith("q_") or c == "adverse_media"])

    # Addresses (the address node id carries the normalised key).
    e = ix.edges("LOCATED_AT")
    addr = pd.DataFrame({
        "record_id": _strip(e["source_node_id"], "vendor"),
        "raw_address": [p.get("raw_address") for p in e["props"]],
        "address_country": [ix.attrs[n].get("country") for n in e["target_node_id"]],
        "address_type": [p.get("address_type") for p in e["props"]],
        "address_role": [p.get("address_role") for p in e["props"]],
        "valid_from": e["valid_from"], "valid_to": e["valid_to"], "observed_at": e["observed_at"],
        "addr_key": _strip(e["target_node_id"], "address"),
    })
    addr = _frame(addr.to_dict("records"), list(addr.columns), dates=DATE_COLUMNS["tp_address_history"])
    addr = canonical("tp_address_history", addr)
    t["tp_address_history"] = addr.drop(columns=["addr_key"])

    # Service registrations: one source row may map to several candidate v2 categories.
    e = ix.edges("PROVIDES_SERVICE")
    e["record_id"] = _strip(e["source_node_id"], "vendor")
    svc = e.groupby("source_record_id", sort=False).agg(
        record_id=("record_id", "first"), category_code=("props", lambda s: s.iloc[0].get("recorded_code")),
        taxonomy_version=("props", lambda s: s.iloc[0].get("taxonomy_version")),
        valid_from=("valid_from", "first"), valid_to=("valid_to", "first")).reset_index(drop=True)
    t["tp_services"] = _frame(svc.to_dict("records"), ["record_id", "category_code", "taxonomy_version",
                                                       "valid_from", "valid_to"], dates=DATE_COLUMNS["tp_services"])

    # Engagements, sponsors and approvals.
    con = ix.edges("CONTRACTED_UNDER").set_index("target_node_id")
    spo = ix.edges("SPONSORED_BY").set_index("source_node_id")["target_node_id"]
    rows, unatt = [], []
    for nid in ix.nodes("Engagement")["node_id"]:
        a = ix.attrs[nid]
        eid = nid[len("engagement:"):]
        rows.append(dict(engagement_id=eid, record_id=con.at[nid, "source_node_id"][len("vendor:"):],
                         service_category=a.get("service_category"), taxonomy_version=a.get("taxonomy_version"),
                         business_unit=a.get("business_unit"), market=a.get("market"),
                         contract_type=a.get("contract_type"), start_date=a.get("start_date"),
                         end_date=a.get("end_date"), value_band=a.get("value_band"),
                         sponsor_employee_id=spo[nid][len("employee:"):] if nid in spo.index else None,
                         observed_at=con.at[nid, "observed_at"]))
        for x in a.get("approvals_without_approver") or []:
            unatt.append(dict(approval_id=x["approval_id"], engagement_id=eid, approver_employee_id=None,
                              approval_type=x.get("approval_type"), approval_date=x.get("approval_date"),
                              observed_at=x.get("observed_at")))
    t["tp_engagements"] = _frame(rows, list(rows[0]) if rows else [], dates=DATE_COLUMNS["tp_engagements"],
                                 ints=["value_band"])
    e = ix.edges("APPROVED_BY")
    appr = [dict(approval_id=r.source_record_id, engagement_id=r.source_node_id[len("engagement:"):],
                 approver_employee_id=r.target_node_id[len("employee:"):], approval_type=r.props.get("approval_type"),
                 approval_date=r.valid_from, observed_at=r.observed_at) for r in e.itertuples()]
    t["tp_engagement_approvals"] = _frame(appr + unatt, ["approval_id", "engagement_id", "approver_employee_id",
                                                        "approval_type", "approval_date", "observed_at"],
                                          dates=DATE_COLUMNS["tp_engagement_approvals"])
    t["tp_payments_monthly"] = g.activity.drop(columns=["node_id"]).reset_index(drop=True)

    # Institutions and links.
    inst = ix.nodes("HealthcareInstitution")
    t["healthcare_institutions"] = _frame(
        [dict(institution_id=n[len("institution:"):], name=lab, country=ix.attrs[n].get("country"),
              inst_type=ix.attrs[n].get("inst_type"), public_sector=ix.attrs[n].get("public_sector"))
         for n, lab in zip(inst["node_id"], inst["label"])],
        ["institution_id", "name", "country", "inst_type", "public_sector"], bools=["public_sector"])
    e = ix.edges(*LINK_TYPES)
    t["tp_institution_links"] = _frame(
        [dict(link_id=r.source_record_id, record_id=r.source_node_id[len("vendor:"):],
              institution_id=r.target_node_id[len("institution:"):], link_type=r.relation_type,
              valid_from=r.valid_from, valid_to=r.valid_to) for r in e.itertuples()],
        ["link_id", "record_id", "institution_id", "link_type", "valid_from", "valid_to"],
        dates=DATE_COLUMNS["tp_institution_links"])

    # Employees and their dated roles.
    emp = ix.nodes("Employee")["node_id"]
    t["employees"] = _frame([dict(employee_id=n[len("employee:"):], name=ix.attrs[n].get("name")) for n in emp],
                            ["employee_id", "name"])
    role_cols = ["employee_id", "role", "franchise", "home_market", "scope_markets", "scope_categories",
                 "approver_level", "valid_from", "valid_to"]
    t["employee_roles"] = _frame(
        [dict(employee_id=n[len("employee:"):], role=a["role"], franchise=a.get("franchise"),
              home_market=a.get("home_market"), scope_markets=a.get("scope_markets"),
              scope_categories=a.get("scope_categories"), approver_level=a.get("approver_level"),
              valid_from=a.get("role_valid_from"), valid_to=a.get("role_valid_to"))
         for n in emp if "role" in (a := ix.attrs[n])],
        role_cols, dates=DATE_COLUMNS["employee_roles"], ints=["approver_level"])

    # Registries.
    ent = ix.nodes("LegalEntity")
    t["reg_entities"] = _frame(
        [dict(statement_id=n[len("legal_entity:"):], statement_date=a.get("published"), reg_record_id=None,
              entity_type=a.get("entity_type"), entity_subtype=a.get("entity_subtype"), name=lab,
              jurisdiction=a.get("jurisdiction"), reg_scheme=None, registry_no=a.get("registry_no"),
              lei=a.get("lei"), listed=a.get("listed"))
         for n, lab in zip(ent["node_id"], ent["label"]) if (a := ix.attrs[n]) is not None],
        ["statement_id", "statement_date", "reg_record_id", "entity_type", "entity_subtype", "name", "jurisdiction",
         "reg_scheme", "registry_no", "lei", "listed"], dates=DATE_COLUMNS["reg_entities"])
    per = ix.nodes("NaturalPerson")
    t["reg_persons"] = _frame(
        [dict(statement_id=n[len("person:"):], statement_date=ix.attrs[n].get("published"), reg_record_id=None,
              person_type=None, full_name=lab, nationality=ix.attrs[n].get("nationality"),
              pep_status=ix.attrs[n].get("pep_status")) for n, lab in zip(per["node_id"], per["label"])],
        ["statement_id", "statement_date", "reg_record_id", "person_type", "full_name", "nationality", "pep_status"],
        dates=DATE_COLUMNS["reg_persons"])
    e = ix.edges("OWNS", "CONTROLS")
    ints = [dict(statement_id=r.source_record_id, statement_date=r.observed_at,
                 subject_sid=r.target_node_id[len("legal_entity:"):],
                 party_sid=r.source_node_id.split(":", 1)[1],
                 party_kind="person" if r.source_node_id.startswith("person:") else "entity",
                 unspecified_reason=None, interest_type=r.props.get("interest_type"), details=r.props.get("details"),
                 share=r.props.get("share"), start_date=r.valid_from, end_date=r.valid_to) for r in e.itertuples()]
    for n in ent["node_id"]:
        for x in ix.attrs[n].get("chain_stops") or []:
            ints.append(dict(statement_id=x["statement_id"], statement_date=x.get("published"),
                             subject_sid=n[len("legal_entity:"):], party_sid=None, party_kind="unspecified",
                             unspecified_reason=x.get("reason"), interest_type=x.get("interest_type"),
                             details=x.get("details"), share=x.get("share"), start_date=x.get("start_date"),
                             end_date=x.get("end_date")))
    int_cols = ["statement_id", "statement_date", "subject_sid", "party_sid", "party_kind", "unspecified_reason",
                "interest_type", "details", "share", "start_date", "end_date"]
    t["reg_interests"] = _frame(ints, int_cols, dates=DATE_COLUMNS["reg_interests"])
    t["reg_interests"]["share"] = t["reg_interests"]["share"].astype("float64")
    e = ix.edges("CONSOLIDATED_BY")
    t["registry_l"] = _frame(
        [dict(child_lei=r.source_record_id, child_legal_name=ix.label[r.source_node_id],
              parent_lei=ix.attrs[r.target_node_id].get("lei"), parent_legal_name=ix.label[r.target_node_id],
              relationship_type="IS_DIRECTLY_CONSOLIDATED_BY", relationship_start=r.valid_from,
              relationship_status=r.props.get("status")) for r in e.itertuples()],
        ["child_lei", "child_legal_name", "parent_lei", "parent_legal_name", "relationship_type",
         "relationship_start", "relationship_status"], dates=DATE_COLUMNS["registry_l"])

    for name in REFERENCE_TABLES:
        t[name] = getattr(reference, name)
    t = {name: canonical(name, df) for name, df in t.items()}
    bundle = Bundle(**t, as_of=reference.as_of, source_dir=reference.source_dir, input_hashes=reference.input_hashes)

    # Entity resolution and normalisation, read from the graph rather than recomputed.
    e = ix.edges("PART_OF")
    n_records = {n: ix.attrs[n].get("n_records") for n in e["target_node_id"].unique()}
    entity_map = pd.DataFrame({"record_id": _strip(e["source_node_id"], "vendor"),
                               "entity_id": _strip(e["target_node_id"], "entity"),
                               "n_records": [n_records[n] for n in e["target_node_id"]]})
    entity_map = entity_map.sort_values("record_id").reset_index(drop=True)
    e = ix.edges("SAME_AS", "IDENTITY_CANDIDATE")
    identity_pairs = pd.DataFrame({
        "record_a": _strip(e["source_node_id"], "vendor"), "record_b": _strip(e["target_node_id"], "vendor"),
        "reason": [p.get("evidence") for p in e["props"]], "confidence": e["resolution_confidence"].astype(float),
        "relation": e["relation_type"]}).astype({"record_a": object, "record_b": object, "reason": object,
                                                 "relation": object})
    identity_pairs = identity_pairs.sort_values(["relation", "record_a", "record_b", "reason"]).reset_index(drop=True)
    e = ix.edges("REGISTERED_AS", "IS_PERSON")
    reg_matches = pd.DataFrame({
        "record_id": _strip(e["source_node_id"], "vendor"), "statement_id": e["source_record_id"].astype(object),
        "method": [p.get("method") for p in e["props"]], "confidence": e["resolution_confidence"].astype(float),
        "status": [p.get("status") for p in e["props"]],
    }).astype({"record_id": object, "statement_id": object, "method": object, "status": object})
    reg_matches = reg_matches.sort_values(["record_id", "statement_id"]).reset_index(drop=True)
    tp = bundle.third_parties
    keys = ix.edges("HAS_CONTACT")
    keys["record_id"] = _strip(keys["source_node_id"], "vendor")
    phone = keys[keys["target_node_id"].str.startswith("phone:")].set_index("record_id")["target_node_id"]
    domain = keys[keys["target_node_id"].str.startswith("domain:")].set_index("record_id")["target_node_id"]
    contacts = tp[["record_id", "phone", "email_domain"]].copy()
    for col, keys, prefix in (("phone_key", phone, "phone"), ("domain_key", domain, "domain")):
        mapped = contacts["record_id"].map(_strip(keys, prefix)).astype(object)
        contacts[col] = mapped.where(mapped.notna(), None)  # missing stays None, as in the source
    pre = {"entity_map": entity_map, "identity_pairs": identity_pairs, "reg_matches": reg_matches,
           "addresses": addr, "contacts": contacts}
    return bundle, pre
