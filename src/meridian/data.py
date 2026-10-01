"""Loading the Meridian bundle, parsing the registries and the as_of(t) view.

Labels are deliberately not loaded here: they belong to `scorer.py` only, so
nothing that builds features can reach them by accident.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import pandas as pd

SNAPSHOT = pd.Timestamp("2026-06-30")

# Business tables and reference tables, in the order of the data dictionary.
TABLES = [
    "third_parties",
    "tp_due_diligence",
    "tp_address_history",
    "tp_services",
    "tp_engagements",
    "tp_engagement_approvals",
    "tp_payments_monthly",
    "tp_institution_links",
    "healthcare_institutions",
    "employees",
    "employee_roles",
    "employee_delegations",
    "markets",
    "paying_entities",
    "service_categories",
    "approval_policy_matrix",
    "fmv_rate_cards",
    "country_risk_context",
    "business_events",
]

# Tables the loader tolerates as missing: employee_delegations has 0 rows and no
# file in the starter bundle.
OPTIONAL_TABLES = {"employee_delegations"}

DELEGATION_COLUMNS = [
    "delegation_id", "from_employee_id", "to_employee_id",
    "approver_level", "valid_from", "valid_to", "observed_at",
]

DATE_COLUMNS = {
    "third_parties": ["onboarding_date", "created_date", "terminated_date"],
    "tp_due_diligence": ["dd_completed", "dd_refresh_due"],
    "tp_address_history": ["valid_from", "valid_to", "observed_at"],
    "tp_services": ["valid_from", "valid_to"],
    "tp_engagements": ["start_date", "end_date", "observed_at"],
    "tp_engagement_approvals": ["approval_date", "observed_at"],
    "tp_payments_monthly": ["month"],
    "tp_institution_links": ["valid_from", "valid_to"],
    "employee_roles": ["valid_from", "valid_to"],
    "employee_delegations": ["valid_from", "valid_to", "observed_at"],
    "fmv_rate_cards": ["effective_from", "effective_to"],
    "country_risk_context": ["as_of_date", "valid_to"],
    "business_events": ["start_date", "end_date"],
    "registry_l": ["relationship_start"],
    "reg_entities": ["statement_date"],
    "reg_persons": ["statement_date"],
    "reg_interests": ["statement_date", "start_date", "end_date"],
}


@dataclass
class Bundle:
    """All input tables as DataFrames. `as_of` is the knowledge date of this view."""

    third_parties: pd.DataFrame
    tp_due_diligence: pd.DataFrame
    tp_address_history: pd.DataFrame
    tp_services: pd.DataFrame
    tp_engagements: pd.DataFrame
    tp_engagement_approvals: pd.DataFrame
    tp_payments_monthly: pd.DataFrame
    tp_institution_links: pd.DataFrame
    healthcare_institutions: pd.DataFrame
    employees: pd.DataFrame
    employee_roles: pd.DataFrame
    employee_delegations: pd.DataFrame
    markets: pd.DataFrame
    paying_entities: pd.DataFrame
    service_categories: pd.DataFrame
    approval_policy_matrix: pd.DataFrame
    fmv_rate_cards: pd.DataFrame
    country_risk_context: pd.DataFrame
    business_events: pd.DataFrame
    registry_l: pd.DataFrame
    reg_entities: pd.DataFrame
    reg_persons: pd.DataFrame
    reg_interests: pd.DataFrame
    as_of: pd.Timestamp = SNAPSHOT
    source_dir: str = ""
    input_hashes: dict = field(default_factory=dict)

    def tables(self) -> dict[str, pd.DataFrame]:
        return {f.name: getattr(self, f.name) for f in fields(self)
                if isinstance(getattr(self, f.name), pd.DataFrame)}


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _read_table(bundle_dir: Path, name: str, hashes: dict) -> pd.DataFrame | None:
    for ext in (".parquet", ".csv"):
        p = bundle_dir / f"{name}{ext}"
        if p.exists():
            hashes[p.name] = _file_hash(p)
            return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
    return None


def _to_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").astype("datetime64[ns]")
    return df


def parse_registry_r(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flatten the BODS-style statements into entity, person and interest tables.

    The file contains the non-standard JSON value NaN; Python's json
    module accepts it, and NaN identifiers become None here.
    """
    with open(path) as fh:
        statements = json.load(fh)

    def clean(v):
        return None if v is None or (isinstance(v, float) and v != v) else v

    ents, pers, ints = [], [], []
    for s in statements:
        rd = s.get("recordDetails", {})
        base = {"statement_id": s["statementId"], "statement_date": s.get("statementDate")}
        if s["recordType"] == "entity":
            ids = {i["scheme"]: clean(i.get("id")) for i in rd.get("identifiers", [])}
            reg_scheme = next((k for k in ids if k.endswith("-REG")), None)
            ents.append({
                **base,
                "reg_record_id": s.get("recordId"),
                "entity_type": rd.get("entityType", {}).get("type"),
                "entity_subtype": rd.get("entityType", {}).get("subtype"),
                "name": rd.get("name"),
                "jurisdiction": (rd.get("jurisdiction") or {}).get("code"),
                "reg_scheme": reg_scheme,
                "registry_no": ids.get(reg_scheme) if reg_scheme else None,
                "lei": ids.get("XI-LEI"),
                "listed": (rd.get("publicListing") or {}).get("hasPublicListing"),
            })
        elif s["recordType"] == "person":
            names = rd.get("names") or [{}]
            nats = rd.get("nationalities") or [{}]
            pers.append({
                **base,
                "reg_record_id": s.get("recordId"),
                "person_type": rd.get("personType"),
                "full_name": names[0].get("fullName"),
                "nationality": nats[0].get("code"),
                "pep_status": (rd.get("politicalExposure") or {}).get("status"),
            })
        elif s["recordType"] == "relationship":
            ip = rd.get("interestedParty", {})
            if "describedByEntityStatement" in ip:
                kind, party = "entity", ip["describedByEntityStatement"]
            elif "describedByPersonStatement" in ip:
                kind, party = "person", ip["describedByPersonStatement"]
            else:
                kind, party = "unspecified", None
            row = {
                **base,
                "subject_sid": rd.get("subject", {}).get("describedByEntityStatement"),
                "party_sid": party,
                "party_kind": kind,
                "unspecified_reason": ip.get("unspecifiedReason"),
            }
            interests = rd.get("interests") or [{}]
            for i in interests:
                share = i.get("share") or {}
                ints.append({
                    **row,
                    "interest_type": i.get("type"),
                    "details": i.get("details"),
                    "share": clean(share.get("exact")),
                    "start_date": i.get("startDate"),
                    "end_date": i.get("endDate"),
                })
    ents = _to_dates(pd.DataFrame(ents), DATE_COLUMNS["reg_entities"])
    pers = _to_dates(pd.DataFrame(pers), DATE_COLUMNS["reg_persons"])
    ints = _to_dates(pd.DataFrame(ints), DATE_COLUMNS["reg_interests"])
    return ents, pers, ints


def load_bundle(bundle_dir: str | Path) -> Bundle:
    bundle_dir = Path(bundle_dir)
    hashes: dict[str, str] = {}
    tables = {}
    for name in TABLES:
        df = _read_table(bundle_dir, name, hashes)
        if df is None:
            if name not in OPTIONAL_TABLES:
                raise FileNotFoundError(f"missing required table {name} in {bundle_dir}")
            df = pd.DataFrame(columns=DELEGATION_COLUMNS)
        tables[name] = _to_dates(df, DATE_COLUMNS.get(name, []))

    reg_l = bundle_dir / "registry_L_parents.csv"
    hashes[reg_l.name] = _file_hash(reg_l)
    tables["registry_l"] = _to_dates(pd.read_csv(reg_l), DATE_COLUMNS["registry_l"])

    reg_r = bundle_dir / "registry_R" / "statements.json"
    hashes["registry_R/statements.json"] = _file_hash(reg_r)
    ents, pers, ints = parse_registry_r(reg_r)
    return Bundle(**tables, reg_entities=ents, reg_persons=pers, reg_interests=ints,
                  as_of=SNAPSHOT, source_dir=str(bundle_dir), input_hashes=hashes)


# ---------------------------------------------------------------------------
# as_of(t)
# ---------------------------------------------------------------------------
# For each table: the column that says when the row became known, and the
# end-of-validity columns that must be blanked when they fall after t.
# Tables without an observation time use a documented proxy.
AS_OF_RULES: dict[str, tuple[str | None, list[str], str]] = {
    # table: (knowledge column, end columns to censor, assumption note)
    "third_parties": ("onboarding_date", ["terminated_date"], "record known from onboarding"),
    "tp_address_history": ("observed_at", ["valid_to"], "observed_at"),
    "tp_services": ("valid_from", ["valid_to"], "no observed_at: known from valid_from"),
    "tp_engagements": ("observed_at", ["end_date"], "observed_at"),
    "tp_engagement_approvals": ("observed_at", [], "observed_at"),
    "tp_institution_links": ("valid_from", ["valid_to"], "no observed_at: known from valid_from"),
    "employee_roles": ("valid_from", ["valid_to"], "no observed_at: known from valid_from"),
    "employee_delegations": ("observed_at", ["valid_to"], "observed_at"),
    "country_risk_context": ("as_of_date", ["valid_to"], "edition as_of_date"),
    "business_events": ("start_date", [], "event known from its start"),
    "registry_l": ("relationship_start", [], "no observed_at: known from relationship_start"),
    "reg_entities": ("statement_date", [], "statementDate = publication"),
    "reg_persons": ("statement_date", [], "statementDate = publication"),
    "reg_interests": ("statement_date", ["end_date"], "statementDate = publication"),
}


def month_available(month: pd.Series) -> pd.Series:
    """Monthly spend is treated as available the day after the month ends (our convention)."""
    return month + pd.offsets.MonthBegin(1)


def as_of(bundle: Bundle, t) -> Bundle:
    """The history as it was known at time t.

    Keeps rows recorded by t, and blanks end dates that fall after t (at time t
    those relationships were still open). Detectors decide for themselves
    whether they need rows valid *at* t (`valid_at`) or the full history up to t.
    """
    t = pd.Timestamp(t)
    out = {}
    for name, df in bundle.tables().items():
        rule = AS_OF_RULES.get(name)
        if rule is None:
            out[name] = df
            continue
        known_col, end_cols, _ = rule
        keep = df[known_col].isna() | (df[known_col] <= t)
        d = df.loc[keep].copy()
        for c in end_cols:
            d.loc[d[c] > t, c] = pd.NaT
        out[name] = d

    # Interests only count once they have started.
    ri = out["reg_interests"]
    out["reg_interests"] = ri[ri["start_date"].isna() | (ri["start_date"] <= t)]

    # Status is a snapshot field: recompute it from the censored termination date.
    tp = out["third_parties"].copy()
    tp["status"] = tp["terminated_date"].notna().map({True: "terminated", False: "active"})
    out["third_parties"] = tp
    live = set(tp["record_id"])
    for name in ["tp_due_diligence", "tp_address_history", "tp_services", "tp_engagements",
                 "tp_institution_links"]:
        out[name] = out[name][out[name]["record_id"].isin(live)]

    eng = set(out["tp_engagements"]["engagement_id"])
    ap = out["tp_engagement_approvals"]
    out["tp_engagement_approvals"] = ap[ap["engagement_id"].isin(eng)]

    pay = out["tp_payments_monthly"]
    out["tp_payments_monthly"] = pay[(month_available(pay["month"]) <= t + pd.Timedelta(days=1))
                                     & pay["engagement_id"].isin(eng)]

    ents = set(out["reg_entities"]["statement_id"]) | set(out["reg_persons"]["statement_id"])
    ri = out["reg_interests"]
    out["reg_interests"] = ri[ri["subject_sid"].isin(ents)
                              & (ri["party_sid"].isna() | ri["party_sid"].isin(ents))]

    return replace(bundle, **out, as_of=t)


def valid_at(df: pd.DataFrame, t, start: str = "valid_from", end: str = "valid_to") -> pd.DataFrame:
    """Rows whose validity interval contains t. End dates are treated as exclusive."""
    t = pd.Timestamp(t)
    return df[(df[start] <= t) & (df[end].isna() | (df[end] > t))]
