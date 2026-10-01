"""Data checks run on every load (Dossier Appendix B).

Each check returns a row: area, check, status (ok / warn / fail), value, detail.
`warn` marks a known deviation we handle downstream; `fail` means the input
contract is broken and the run should not be trusted.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd

from .data import SNAPSHOT, Bundle

# Row counts and missingness stated in DATA_DICTIONARY.md, to compare with what we measure.
DICTIONARY_ROWS = {
    "third_parties": 1002, "tp_due_diligence": 1002, "tp_address_history": 1091,
    "tp_services": 1791, "tp_engagements": 2638, "tp_engagement_approvals": 4298,
    "tp_payments_monthly": 34467, "tp_institution_links": 403, "healthcare_institutions": 213,
    "employees": 1502, "employee_roles": 1502, "employee_delegations": 0, "markets": 74,
    "paying_entities": 74, "service_categories": 43, "approval_policy_matrix": 5,
    "fmv_rate_cards": 144, "country_risk_context": 1120, "business_events": 8,
}
DICTIONARY_NULL_RATE = {
    ("tp_engagements", "sponsor_employee_id"): 0.03,
    ("tp_engagement_approvals", "approver_employee_id"): 0.05,
    ("third_parties", "lei"): 0.75,
    ("tp_address_history", "address_type"): 0.40,
}
PRIMARY_KEYS = {
    "third_parties": ["record_id"], "tp_due_diligence": ["record_id"],
    "tp_engagements": ["engagement_id"], "tp_engagement_approvals": ["approval_id"],
    "tp_institution_links": ["link_id"], "healthcare_institutions": ["institution_id"],
    "employees": ["employee_id"], "markets": ["market_id"], "paying_entities": ["paying_entity_id"],
    "service_categories": ["category_code_v2"], "approval_policy_matrix": ["value_band"],
    "reg_entities": ["statement_id"], "reg_persons": ["statement_id"],
    "tp_payments_monthly": ["engagement_id", "month", "expense_classification", "channel"],
}
VALIDITY = {
    "tp_address_history": ("valid_from", "valid_to"), "tp_services": ("valid_from", "valid_to"),
    "tp_engagements": ("start_date", "end_date"), "tp_institution_links": ("valid_from", "valid_to"),
    "employee_roles": ("valid_from", "valid_to"), "third_parties": ("onboarding_date", "terminated_date"),
    "reg_interests": ("start_date", "end_date"), "fmv_rate_cards": ("effective_from", "effective_to"),
}
OBSERVED = {
    "tp_address_history": "observed_at", "tp_engagements": "observed_at",
    "tp_engagement_approvals": "observed_at", "reg_entities": "statement_date",
    "reg_persons": "statement_date", "reg_interests": "statement_date",
}


def _row(area, check, ok, value=None, detail="", warn=False):
    status = "ok" if ok else ("warn" if warn else "fail")
    return {"area": area, "check": check, "status": status, "value": value, "detail": detail}


def _fk(rows, child, ccol, parent, pcol, name, allow_null=True):
    vals = child[ccol].dropna() if allow_null else child[ccol]
    bad = ~vals.isin(set(parent[pcol].dropna()))
    rows.append(_row("referential", name, bad.sum() == 0, int(bad.sum()),
                     f"{bad.sum()} of {len(vals)} {ccol} values not in {pcol}"))


def run_checks(b: Bundle, snapshot=SNAPSHOT) -> pd.DataFrame:
    rows = []
    tables = b.tables()

    # Row counts against the dictionary.
    for name, n in DICTIONARY_ROWS.items():
        got = len(tables[name])
        rows.append(_row("contract", f"rows:{name}", got == n, got, f"dictionary says {n}", warn=True))

    # Missingness against the dictionary.
    for (tname, col), expected in DICTIONARY_NULL_RATE.items():
        rate = tables[tname][col].isna().mean()
        ok = abs(rate - expected) <= 0.02
        rows.append(_row("contract", f"null_rate:{tname}.{col}", ok, round(rate, 3),
                         f"dictionary implies ~{expected:.0%}", warn=True))

    # Primary keys.
    for tname, key in PRIMARY_KEYS.items():
        d = tables[tname]
        dup = d.duplicated(key).sum()
        blank = d[key].isna().any(axis=1).sum()
        rows.append(_row("identifiers", f"pk:{tname}", dup == 0 and blank == 0, int(dup + blank),
                         f"{dup} duplicate and {blank} blank keys on {key}"))

    # Foreign keys.
    tp, en, ap = b.third_parties, b.tp_engagements, b.tp_engagement_approvals
    for tname in ["tp_due_diligence", "tp_address_history", "tp_services", "tp_engagements",
                  "tp_payments_monthly", "tp_institution_links"]:
        _fk(rows, tables[tname], "record_id", tp, "record_id", f"fk:{tname}.record_id", allow_null=False)
    _fk(rows, en, "sponsor_employee_id", b.employees, "employee_id", "fk:engagements.sponsor")
    _fk(rows, en, "market", b.markets, "market_id", "fk:engagements.market")
    _fk(rows, ap, "engagement_id", en, "engagement_id", "fk:approvals.engagement_id", allow_null=False)
    _fk(rows, ap, "approver_employee_id", b.employees, "employee_id", "fk:approvals.approver")
    _fk(rows, b.tp_payments_monthly, "engagement_id", en, "engagement_id", "fk:payments.engagement_id")
    _fk(rows, b.tp_payments_monthly, "paying_entity_id", b.paying_entities, "paying_entity_id",
        "fk:payments.paying_entity")
    _fk(rows, b.tp_institution_links, "institution_id", b.healthcare_institutions, "institution_id",
        "fk:links.institution")
    _fk(rows, b.employee_roles, "employee_id", b.employees, "employee_id", "fk:roles.employee")
    _fk(rows, tp, "country", b.markets, "market_id", "fk:third_parties.country")
    all_codes = pd.DataFrame({"c": pd.concat([b.service_categories["category_code_v2"],
                                              b.service_categories["category_code_v1"]])})
    _fk(rows, b.tp_services, "category_code", all_codes, "c", "fk:services.category_code")
    _fk(rows, en, "service_category", all_codes, "c", "fk:engagements.service_category")
    reg_sids = pd.DataFrame({"s": pd.concat([b.reg_entities["statement_id"], b.reg_persons["statement_id"]])})
    _fk(rows, b.reg_interests, "subject_sid", reg_sids, "s", "fk:registry_R.subject")
    _fk(rows, b.reg_interests, "party_sid", reg_sids, "s", "fk:registry_R.interested_party")
    pay_en = b.tp_payments_monthly.merge(en[["engagement_id", "record_id"]], on="engagement_id",
                                         suffixes=("", "_en"))
    bad = (pay_en["record_id"] != pay_en["record_id_en"]).sum()
    rows.append(_row("referential", "payments.record_id matches engagement", bad == 0, int(bad)))
    unmatched_l = ~b.registry_l["child_lei"].isin(set(tp["lei"].dropna()))
    rows.append(_row("referential", "registry_L.child_lei in third_parties.lei", unmatched_l.sum() == 0,
                     int(unmatched_l.sum()), "children not in the vendor master", warn=True))

    # Temporal validity.
    for tname, (s, e) in VALIDITY.items():
        d = tables[tname]
        bad = (d[e].notna() & (d[e] < d[s])).sum()
        # Starter has one vendor terminated before onboarding (no engagements): a data glitch, not a broken contract.
        rows.append(_row("temporal", f"from<=to:{tname}", bad == 0, int(bad), f"{s} <= {e}",
                         warn=tname == "third_parties"))
    for tname, col in OBSERVED.items():
        late = (tables[tname][col] > snapshot).sum()
        rows.append(_row("temporal", f"observed<=snapshot:{tname}", late == 0, int(late)))
    st = (tp["status"] == "terminated") != tp["terminated_date"].notna()
    rows.append(_row("temporal", "status consistent with terminated_date", st.sum() == 0, int(st.sum())))
    ap_before = ap.merge(en[["engagement_id", "observed_at"]], on="engagement_id", suffixes=("", "_en"))
    rows.append(_row("temporal", "created_date spike (ERP migration)", True,
                     int((tp["created_date"] == pd.Timestamp("2022-05-01")).sum()),
                     "records with created_date = 2022-05-01; use onboarding_date"))
    early = (ap_before["approval_date"] > ap_before["observed_at_en"] + pd.Timedelta(days=365)).sum()
    rows.append(_row("temporal", "approval long after engagement recorded", True, int(early)))

    # Overlapping address validity for the same record.
    a = b.tp_address_history.sort_values(["record_id", "valid_from"])
    nxt = a.groupby("record_id")["valid_from"].shift(-1)
    ovl = (a["valid_to"].isna() & nxt.notna()) | (a["valid_to"] > nxt)
    rows.append(_row("overlap", "overlapping addresses per record", ovl.sum() == 0, int(ovl.sum()), warn=True))

    # Employees: one role per employee (succession = new employee id).
    per_emp = b.employee_roles.groupby("employee_id").size()
    rows.append(_row("contract", "roles per employee", per_emp.max() == 1, int(per_emp.max()),
                     "max role periods per employee", warn=True))

    # Service taxonomy: one-to-many v1 -> v2 mappings.
    sc = b.service_categories
    fan = sc.groupby("category_code_v1")["category_code_v2"].nunique()
    multi = fan[fan > 1]
    affected = (en["taxonomy_version"].eq("v1") & en["service_category"].isin(multi.index)).sum()
    rows.append(_row("contract", "v1 codes mapping to several v2 codes", True, int(len(multi)),
                     f"{len(multi)} of {len(fan)} v1 codes; {affected} v1 engagements affected"))

    # Country context: which indices are loaded.
    idx = sorted(b.country_risk_context["index"].unique())
    rows.append(_row("contract", "country_risk_context indices", {"FSI", "FATF"} <= set(idx), ",".join(idx),
                     "FSI and FATF expected when loaded", warn=True))

    # Ownership percentages and cycles.
    sh = b.reg_interests["share"].dropna()
    rows.append(_row("ownership", "share in [0,100]", sh.between(0, 100).all(), int((~sh.between(0, 100)).sum())))
    g = nx.DiGraph()
    ri = b.reg_interests[b.reg_interests["party_kind"] == "entity"]
    g.add_edges_from(zip(ri["party_sid"], ri["subject_sid"]))
    cycles = list(nx.simple_cycles(g))
    rows.append(_row("ownership", "ownership cycles", len(cycles) == 0, len(cycles), warn=True))
    reg_lag = (b.reg_interests["statement_date"] - b.reg_interests["start_date"]).dt.days.dropna()
    rows.append(_row("ownership", "registry R publication lag (days, median/max)", True,
                     f"{reg_lag.median():.0f}/{reg_lag.max():.0f}"))

    # Duplicates: shared tax_id.
    dup_tax = tp[tp["tax_id"].notna()].groupby("tax_id")["record_id"].nunique()
    rows.append(_row("duplicates", "tax_id groups with >1 record", True, int((dup_tax > 1).sum()),
                     f"{int(dup_tax[dup_tax > 1].sum())} records"))
    shared_dom = tp.groupby("email_domain")["record_id"].nunique()
    rows.append(_row("duplicates", "email domains shared by >1 record", True, int((shared_dom > 1).sum()),
                     f"{int(shared_dom[shared_dom > 1].sum())} records"))

    # Approval authority: approver level below the band requirement.
    under = approval_authority_gaps(b)
    rows.append(_row("policy", "financial approvals below required level", True,
                     int(under["engagement_id"].nunique()),
                     f"{under['record_id'].nunique()} records"))

    return pd.DataFrame(rows)


def approval_authority_gaps(b: Bundle) -> pd.DataFrame:
    """Financial approvals where the approver's level is below the band's requirement.

    Temporary delegations (when provided) excuse the gap.
    """
    ap = b.tp_engagement_approvals
    fin = ap[ap["approval_type"] == "financial"].merge(
        b.tp_engagements[["engagement_id", "record_id", "value_band", "sponsor_employee_id"]], on="engagement_id")
    fin = fin.merge(b.approval_policy_matrix[["value_band", "required_approver_level"]], on="value_band")
    fin = fin.merge(b.employee_roles[["employee_id", "approver_level", "valid_from", "valid_to", "role"]],
                    left_on="approver_employee_id", right_on="employee_id", how="left")
    under = fin[fin["approver_employee_id"].notna() & (fin["approver_level"] < fin["required_approver_level"])]
    dl = b.employee_delegations
    if len(dl):
        m = under.merge(dl, left_on="approver_employee_id", right_on="to_employee_id", how="left",
                        suffixes=("", "_dl"))
        covered = (m["approval_date"] >= m["valid_from_dl"]) & (m["valid_to_dl"].isna()
                                                                | (m["approval_date"] <= m["valid_to_dl"])) \
            & (m["approver_level_dl"] >= m["required_approver_level"])
        under = under[~under["approval_id"].isin(m.loc[covered, "approval_id"])]
    return under


def to_markdown(report: pd.DataFrame) -> str:
    lines = ["| area | check | status | value | detail |", "|---|---|---|---|---|"]
    for r in report.itertuples():
        lines.append(f"| {r.area} | {r.check} | {r.status} | {r.value} | {r.detail} |")
    return "\n".join(lines)
