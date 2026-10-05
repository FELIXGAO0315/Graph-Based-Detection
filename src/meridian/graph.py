"""Canonical heterogeneous graph: typed node and edge tables plus a content hash.

The graph is the single input of detection: normalisation and entity
resolution read the source tables to build it, and the detectors then read
only the graph (through Context.from_graph) plus reference tables such as
FMV rate cards and the approval policy matrix. It can be inspected,
exported to graph tools and fingerprinted (Dossier Appendix B: the same
inputs, configuration and seed give the same graph hash).

Monthly payments are a time series per engagement, not a relation: they are
kept as an activity table attached to Engagement nodes.

Every edge keeps its valid time (valid_from, valid_to), observation time
(observed_at), source table and source record, and the entity-resolution
confidence where a match is involved. Missing times stay empty: no
observation date or probability is invented. The graph is built from an
as_of(t) view, so it holds only what was known at t.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .context import Context

NODE_COLUMNS = ["node_id", "node_type", "label", "source_table", "attributes"]
EDGE_COLUMNS = ["edge_id", "source_node_id", "relation_type", "target_node_id", "valid_from", "valid_to",
                "observed_at", "source_table", "source_record_id", "dataset_version", "resolution_confidence",
                "relation_properties"]

# Permitted (source type, relation, target type) triples (Dossier Appendix B, type constraints).
ALLOWED = {
    ("VendorRecord", "PART_OF", "ResolvedEntity"),
    ("VendorRecord", "SAME_AS", "VendorRecord"),
    ("VendorRecord", "IDENTITY_CANDIDATE", "VendorRecord"),
    ("VendorRecord", "LOCATED_AT", "Address"),
    ("VendorRecord", "HAS_CONTACT", "ContactPoint"),
    ("VendorRecord", "DECLARED_MEMBER_OF", "DeclaredGroup"),
    ("VendorRecord", "CONTRACTED_UNDER", "Engagement"),
    ("Engagement", "SPONSORED_BY", "Employee"),
    ("Engagement", "APPROVED_BY", "Employee"),
    ("Engagement", "FOR_SERVICE", "ServiceCategory"),
    ("VendorRecord", "PROVIDES_SERVICE", "ServiceCategory"),
    ("VendorRecord", "SERVES", "HealthcareInstitution"),
    ("VendorRecord", "SITE_OF", "HealthcareInstitution"),
    ("VendorRecord", "AFFILIATED_WITH", "HealthcareInstitution"),
    ("VendorRecord", "REGISTERED_AS", "LegalEntity"),
    ("VendorRecord", "IS_PERSON", "NaturalPerson"),
    ("LegalEntity", "OWNS", "LegalEntity"),
    ("NaturalPerson", "OWNS", "LegalEntity"),
    ("LegalEntity", "CONTROLS", "LegalEntity"),
    ("NaturalPerson", "CONTROLS", "LegalEntity"),
    ("VendorRecord", "CONSOLIDATED_BY", "AccountingParent"),
}


ACTIVITY_KEY = ["node_id", "month", "expense_classification", "channel"]


@dataclass
class Graph:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    activity: pd.DataFrame          # monthly payments attached to Engagement nodes
    dataset_version: str
    snapshot: pd.Timestamp

    @property
    def hash(self) -> str:
        return graph_hash(self.nodes, self.edges, self.activity)

    def summary(self) -> dict:
        return {
            "graph_hash": self.hash, "dataset_version": self.dataset_version, "snapshot": str(self.snapshot.date()),
            "n_nodes": len(self.nodes), "n_edges": len(self.edges), "n_activity_rows": len(self.activity),
            "nodes_by_type": self.nodes["node_type"].value_counts().sort_index().to_dict(),
            "edges_by_relation": self.edges["relation_type"].value_counts().sort_index().to_dict(),
        }


def _json(d: dict) -> str:
    """Attributes as sorted JSON; missing values (None, NaN, NaT) are left out, not invented."""
    clean = {}
    for k, v in d.items():
        if v is None or (pd.api.types.is_scalar(v) and pd.isna(v)):
            continue
        if isinstance(v, pd.Timestamp):
            v = v.date().isoformat()
        elif hasattr(v, "item"):
            v = v.item()
        elif isinstance(v, list):
            v = [json.loads(_json(x)) if isinstance(x, dict) else x for x in v]
        clean[k] = v
    return json.dumps(clean, sort_keys=True, ensure_ascii=False, default=str)


def _date(v):
    return None if v is None or pd.isna(v) else pd.Timestamp(v).date().isoformat()


class _Builder:
    def __init__(self, dataset_version: str):
        self.version = dataset_version
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []

    def node(self, node_id: str, node_type: str, label, source_table: str, **attrs):
        if node_id not in self.nodes:
            self.nodes[node_id] = {"node_id": node_id, "node_type": node_type, "label": str(label),
                                   "source_table": source_table, "attributes": _json(attrs)}
        return node_id

    def edge(self, src: str, rel: str, dst: str, source_table: str, source_record_id, valid_from=None, valid_to=None,
             observed_at=None, confidence=None, **props):
        self.edges.append({
            "source_node_id": src, "relation_type": rel, "target_node_id": dst,
            "valid_from": _date(valid_from), "valid_to": _date(valid_to), "observed_at": _date(observed_at),
            "source_table": source_table, "source_record_id": str(source_record_id),
            "dataset_version": self.version,
            "resolution_confidence": None if confidence is None else float(confidence),
            "relation_properties": _json(props),
        })

    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        nodes = pd.DataFrame(list(self.nodes.values()), columns=NODE_COLUMNS).sort_values("node_id")
        edges = pd.DataFrame(self.edges, columns=EDGE_COLUMNS[1:])
        key = edges[["source_node_id", "relation_type", "target_node_id", "valid_from", "source_table",
                     "source_record_id", "relation_properties"]].astype(str).agg("|".join, axis=1)
        edges.insert(0, "edge_id", ["E" + hashlib.sha1(k.encode()).hexdigest()[:16] for k in key])
        edges = edges.drop_duplicates("edge_id").sort_values("edge_id")
        return nodes.reset_index(drop=True), edges.reset_index(drop=True)


def dataset_version(input_hashes: dict) -> str:
    return hashlib.sha256(json.dumps(input_hashes, sort_keys=True).encode()).hexdigest()[:12]


def build_graph(ctx: Context) -> Graph:
    """Materialise the typed relations of an as_of(t) context as node and edge tables."""
    b = _Builder(dataset_version(ctx.b.input_hashes))
    v = ctx.vendors

    # Vendor records (with their due-diligence answers) and resolved entities.
    dd = ctx.b.tp_due_diligence.set_index("record_id")
    for r in v.itertuples(index=False):
        d = dd.loc[r.record_id].to_dict() if r.record_id in dd.index else {}
        b.node(f"vendor:{r.record_id}", "VendorRecord", r.legal_name, "third_parties",
               party_type=r.party_type, country=r.country, status=r.status, onboarding_date=r.onboarding_date,
               created_date=r.created_date, terminated_date=r.terminated_date, expert_type=r.expert_type,
               remit_to_country=r.remit_to_country, tax_id=r.tax_id, lei=r.lei, registry_no=r.registry_no,
               declared_parent_group=r.declared_parent_group, phone=r.phone, email_domain=r.email_domain,
               **{f"dd_{k}" if not k.startswith("dd_") else k: val for k, val in d.items()})
    em = ctx.entity_map
    for r in em.itertuples(index=False):
        ent = b.node(f"entity:{r.entity_id}", "ResolvedEntity", r.entity_id, "resolution", n_records=r.n_records)
        b.edge(f"vendor:{r.record_id}", "PART_OF", ent, "resolution", r.record_id)
    for r in ctx.identity_pairs.itertuples(index=False):
        b.edge(f"vendor:{r.record_a}", r.relation, f"vendor:{r.record_b}", "resolution", f"{r.record_a}|{r.record_b}",
               confidence=r.confidence, evidence=r.reason)

    # Addresses, contacts, declared groups.
    a = ctx.addresses
    types = a.groupby("addr_key")["address_type"].agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None)
    for r in a.itertuples(index=False):
        node = b.node(f"address:{r.addr_key}", "Address", r.raw_address, "tp_address_history",
                      country=r.address_country, address_type=types.get(r.addr_key))
        b.edge(f"vendor:{r.record_id}", "LOCATED_AT", node, "tp_address_history", f"{r.record_id}@{_date(r.valid_from)}",
               valid_from=r.valid_from, valid_to=r.valid_to, observed_at=r.observed_at, raw_address=r.raw_address,
               address_type=r.address_type, address_role=r.address_role)
    for r in ctx.contacts.itertuples(index=False):
        if isinstance(r.phone_key, str):
            node = b.node(f"phone:{r.phone_key}", "ContactPoint", r.phone_key, "third_parties", kind="phone")
            b.edge(f"vendor:{r.record_id}", "HAS_CONTACT", node, "third_parties", r.record_id, raw=r.phone)
        if isinstance(r.domain_key, str):
            node = b.node(f"domain:{r.domain_key}", "ContactPoint", r.domain_key, "third_parties", kind="email_domain")
            b.edge(f"vendor:{r.record_id}", "HAS_CONTACT", node, "third_parties", r.record_id)
    for r in v.dropna(subset=["declared_parent_group"]).itertuples(index=False):
        node = b.node(f"group:{r.declared_parent_group}", "DeclaredGroup", r.declared_parent_group, "third_parties")
        b.edge(f"vendor:{r.record_id}", "DECLARED_MEMBER_OF", node, "third_parties", r.record_id)

    # Engagements, people and categories.
    # Employees exist independently of their roles. A role not yet recorded at t
    # leaves the attributes empty; in the starter data some engagements name a
    # sponsor or approver before that person's role starts.
    roles = ctx.roles.set_index("employee_id")
    names = ctx.b.employees.set_index("employee_id")["name"]
    for emp in ctx.b.employees["employee_id"]:
        r = roles.loc[emp] if emp in roles.index else None
        attrs = {} if r is None else dict(
            role=r["role"], franchise=r["franchise"], home_market=r["home_market"], scope_markets=r["scope_markets"],
            scope_categories=r["scope_categories"], approver_level=r["approver_level"],
            role_valid_from=r["valid_from"], role_valid_to=r["valid_to"])
        b.node(f"employee:{emp}", "Employee", emp, "employees", name=names.get(emp), **attrs)
    cats = ctx.categories
    for code, r in cats.iterrows():
        b.node(f"category:{code}", "ServiceCategory", r["name"], "service_categories", parent=r["parent"],
               flags=r["flags"], category_code_v1=r["category_code_v1"])
    # Approval events with no recorded approver cannot be an edge (there is no
    # endpoint); they stay on the engagement so nothing is dropped.
    ap = ctx.b.tp_engagement_approvals
    unattributed = {eid: [dict(approval_id=x.approval_id, approval_type=x.approval_type,
                               approval_date=x.approval_date, observed_at=x.observed_at) for x in g.itertuples()]
                    for eid, g in ap[ap["approver_employee_id"].isna()].groupby("engagement_id")}
    for r in ctx.engagements.itertuples(index=False):
        en = b.node(f"engagement:{r.engagement_id}", "Engagement", r.engagement_id, "tp_engagements",
                    service_category=r.service_category, taxonomy_version=r.taxonomy_version,
                    business_unit=r.business_unit, market=r.market, contract_type=r.contract_type,
                    value_band=r.value_band, start_date=r.start_date, end_date=r.end_date,
                    approvals_without_approver=unattributed.get(r.engagement_id))
        b.edge(f"vendor:{r.record_id}", "CONTRACTED_UNDER", en, "tp_engagements", r.engagement_id,
               valid_from=r.start_date, valid_to=r.end_date, observed_at=r.observed_at)
        if isinstance(r.sponsor_employee_id, str):
            b.edge(en, "SPONSORED_BY", f"employee:{r.sponsor_employee_id}", "tp_engagements", r.engagement_id,
                   valid_from=r.start_date, valid_to=r.end_date, observed_at=r.observed_at)
        # An ambiguous v1 code keeps one edge per candidate v2 code, with its share as confidence.
        for code in r.cat_v2:
            if code in cats.index:
                b.edge(en, "FOR_SERVICE", f"category:{code}", "tp_engagements", r.engagement_id,
                       confidence=1.0 if r.cat_certain else 1.0 / len(r.cat_v2), recorded_code=r.service_category)
    for r in ctx.b.tp_engagement_approvals.dropna(subset=["approver_employee_id"]).itertuples(index=False):
        b.edge(f"engagement:{r.engagement_id}", "APPROVED_BY", f"employee:{r.approver_employee_id}",
               "tp_engagement_approvals", r.approval_id, valid_from=r.approval_date, observed_at=r.observed_at,
               approval_type=r.approval_type)
    for r in ctx.services.itertuples(index=False):
        for code in r.cat_v2:
            if code in cats.index:
                b.edge(f"vendor:{r.record_id}", "PROVIDES_SERVICE", f"category:{code}", "tp_services",
                       f"{r.record_id}|{r.category_code}|{_date(r.valid_from)}", valid_from=r.valid_from,
                       valid_to=r.valid_to, confidence=1.0 if r.cat_certain else 1.0 / len(r.cat_v2),
                       recorded_code=r.category_code, taxonomy_version=r.taxonomy_version)

    # Healthcare institutions.
    for r in ctx.b.healthcare_institutions.itertuples(index=False):
        b.node(f"institution:{r.institution_id}", "HealthcareInstitution", r.name, "healthcare_institutions",
               country=r.country, inst_type=r.inst_type, public_sector=bool(r.public_sector))
    for r in ctx.b.tp_institution_links.itertuples(index=False):
        b.edge(f"vendor:{r.record_id}", r.link_type, f"institution:{r.institution_id}", "tp_institution_links",
               r.link_id, valid_from=r.valid_from, valid_to=r.valid_to)

    # Ownership (registry R) and accounting parents (registry L).
    if ctx.use_ownership:
        # Statements where the chain stops (no named owner) belong to the owned entity.
        unspecified = ctx.b.reg_interests[ctx.b.reg_interests["party_kind"] == "unspecified"]
        stops = {sid: [dict(statement_id=x.statement_id, reason=x.unspecified_reason, published=x.statement_date,
                            interest_type=x.interest_type, details=x.details, share=x.share,
                            start_date=x.start_date, end_date=x.end_date) for x in g.itertuples()]
                 for sid, g in unspecified.groupby("subject_sid")}
        for r in ctx.b.reg_entities.itertuples(index=False):
            b.node(f"legal_entity:{r.statement_id}", "LegalEntity", r.name, "registry_R", jurisdiction=r.jurisdiction,
                   entity_type=r.entity_type, entity_subtype=r.entity_subtype, registry_no=r.registry_no, lei=r.lei,
                   listed=r.listed, published=r.statement_date, chain_stops=stops.get(r.statement_id))
        for r in ctx.b.reg_persons.itertuples(index=False):
            b.node(f"person:{r.statement_id}", "NaturalPerson", r.full_name, "registry_R",
                   nationality=r.nationality, pep_status=r.pep_status, published=r.statement_date)
        for r in ctx.reg_matches.itertuples(index=False):
            if r.method.startswith("person"):
                b.edge(f"vendor:{r.record_id}", "IS_PERSON", f"person:{r.statement_id}", "registry_R",
                       r.statement_id, confidence=r.confidence, method=r.method, status=r.status)
            else:
                b.edge(f"vendor:{r.record_id}", "REGISTERED_AS", f"legal_entity:{r.statement_id}", "registry_R",
                       r.statement_id, confidence=r.confidence, method=r.method, status=r.status)
        ri = ctx.b.reg_interests.dropna(subset=["party_sid"])
        for r in ri.itertuples(index=False):
            owner = f"{'person' if r.party_kind == 'person' else 'legal_entity'}:{r.party_sid}"
            rel = "OWNS" if r.interest_type == "shareholding" else "CONTROLS"
            b.edge(owner, rel, f"legal_entity:{r.subject_sid}", "registry_R", r.statement_id,
                   valid_from=r.start_date, valid_to=r.end_date, observed_at=r.statement_date,
                   share=r.share, details=r.details, interest_type=r.interest_type)
        lei_to_records = v.dropna(subset=["lei"]).groupby("lei")["record_id"].apply(list).to_dict()
        for r in ctx.b.registry_l.itertuples(index=False):
            parent_key = r.parent_lei if isinstance(r.parent_lei, str) else r.parent_legal_name
            node = b.node(f"accounting_parent:{parent_key}", "AccountingParent", r.parent_legal_name, "registry_L",
                          lei=r.parent_lei if isinstance(r.parent_lei, str) else None)
            for rec in lei_to_records.get(r.child_lei, []):
                b.edge(f"vendor:{rec}", "CONSOLIDATED_BY", node, "registry_L", r.child_lei,
                       valid_from=r.relationship_start, status=r.relationship_status)

    nodes, edges = b.frames()
    activity = ctx.b.tp_payments_monthly.assign(node_id="engagement:" + ctx.b.tp_payments_monthly["engagement_id"])
    activity = activity[["node_id"] + [c for c in activity.columns if c != "node_id"]]
    activity = activity.sort_values(ACTIVITY_KEY).reset_index(drop=True)
    return Graph(nodes, edges, activity, b.version, ctx.t)


def without(g: Graph, relations=(), activity: bool = False) -> Graph:
    """A copy of the graph without the given edge types (and payments if `activity`)."""
    edges = g.edges[~g.edges["relation_type"].isin(set(relations))].reset_index(drop=True)
    act = g.activity.iloc[0:0] if activity else g.activity
    return Graph(g.nodes, edges, act, g.dataset_version, g.snapshot)


def graph_hash(nodes: pd.DataFrame, edges: pd.DataFrame, activity: pd.DataFrame | None = None) -> str:
    """SHA-256 over a canonical serialisation (rows sorted by key, fixed column order)."""
    h = hashlib.sha256()
    parts = [(nodes, ["node_id"], NODE_COLUMNS), (edges, ["edge_id"], EDGE_COLUMNS)]
    if activity is not None:
        parts.append((activity, ACTIVITY_KEY, list(activity.columns)))
    for df, key, cols in parts:
        buf = io.StringIO()
        df.sort_values(key)[cols].to_csv(buf, index=False, lineterminator="\n")
        h.update(buf.getvalue().encode())
    return h.hexdigest()[:16]


def validate(g: Graph) -> list[str]:
    """Appendix B checks on the graph itself. Returns a list of problems (empty when clean)."""
    problems = []
    if g.nodes["node_id"].duplicated().any():
        problems.append("duplicate node ids")
    types = g.nodes.set_index("node_id")["node_type"]
    missing = ~g.edges["source_node_id"].isin(types.index) | ~g.edges["target_node_id"].isin(types.index)
    if missing.any():
        problems.append(f"{int(missing.sum())} edges with an endpoint missing from the node table")
    ok = g.edges[~missing]
    triples = set(zip(types[ok["source_node_id"]], ok["relation_type"], types[ok["target_node_id"]]))
    for t in sorted(triples - ALLOWED):
        problems.append(f"relation not allowed: {t}")
    vf, vt = pd.to_datetime(g.edges["valid_from"]), pd.to_datetime(g.edges["valid_to"])
    if (vt.notna() & vf.notna() & (vt < vf)).any():
        problems.append("edges with valid_to before valid_from")
    if (pd.to_datetime(g.edges["observed_at"]) > g.snapshot).any():
        problems.append("edges observed after the snapshot")
    engagements = set(g.nodes.loc[g.nodes["node_type"] == "Engagement", "node_id"])
    if not g.activity["node_id"].isin(engagements).all():
        problems.append("activity rows attached to unknown engagements")
    return problems


def export(g: Graph, out_dir: str | Path) -> dict:
    """Write nodes, edges and activity (parquet) and graph.json (hash, version, counts)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g.nodes.to_parquet(out / "nodes.parquet", index=False)
    g.edges.to_parquet(out / "edges.parquet", index=False)
    g.activity.to_parquet(out / "activity.parquet", index=False)
    info = g.summary()
    (out / "graph.json").write_text(json.dumps(info, indent=2))
    return info
