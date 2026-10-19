"""Canonical temporal graph for the Meridian third-party data (dossier §4).

build_graph(data_dir, t) returns node and edge tables as of time t: a relationship is included only if it
had started (valid time) and been recorded (observation time) by t. Anything that happened after t is cut:
end dates, termination dates and validity ends later than t become "open at t". Graph objects are derived
from these tables, which are the single source of truth (§4.5).

Run `python graph_build.py --t 2024-06-30` to write nodes / edges / attribute tables and a manifest.
Data-quality checks (Appendix B) live in check_graph(); tests are in tests/test_graph_build.py.
"""
from __future__ import annotations

import argparse, difflib, hashlib, json, math, re, unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "student_bundle"
SNAPSHOT = pd.Timestamp("2026-06-30")
BUILD_VERSION = "graph_build-2"

# Allowed (source type, relationship, target type) triples: the graph's type contract (Appendix B).
SCHEMA = {
    "CONTRACTED_UNDER": ("third_party", "engagement"),
    "SPONSORED_BY": ("engagement", "employee"),
    "APPROVED_BY": ("engagement", "employee"),
    "BOOKED_IN": ("engagement", "market"),
    "PAID_BY": ("engagement", "paying_entity"),
    "HOLDS_ROLE": ("employee", "market"),
    "LOCATED_AT": ("third_party", "address"),
    "PROVIDES_SERVICE": ("third_party", "service_category"),
    "SERVES": ("third_party", "institution"),
    "SITE_OF": ("third_party", "institution"),
    "AFFILIATED_WITH": ("third_party", "institution"),
    "DECLARED_MEMBER_OF": ("third_party", "declared_group"),
    "HAS_EMAIL_DOMAIN": ("third_party", "email_domain"),
    "HAS_PHONE": ("third_party", "phone"),
    "SAME_AS": ("third_party", ("third_party", "legal_entity", "person")),
    "IDENTITY_CANDIDATE": ("third_party", ("third_party", "legal_entity", "person")),
    "RESOLVED_AS": ("third_party", "resolved_entity"),
    "ULTIMATE_OWNER": ("third_party", "person"),  # derived from OWNS/CONTROLS paths current at t
    "OWNS": (("legal_entity", "person"), "legal_entity"),
    "CONTROLS": (("legal_entity", "person"), "legal_entity"),
    "CONSOLIDATED_BY": ("legal_entity", "legal_entity"),
}
EDGE_COLUMNS = ["edge_id", "source_node_id", "source_node_type", "relationship_type", "target_node_id", "target_node_type",
                "valid_from", "valid_to", "observed_at", "weight_name", "weight_value", "source_id", "source_type",
                "resolution_confidence", "properties", "dataset_version"]
NODE_COLUMNS = ["node_id", "node_type", "key", "label", "source_type", "dataset_version"]
GENERIC_EMAIL = {"gmail.com", "outlook.com", "yahoo.com", "hotmail.com", "icloud.com"}  # same list as Layer1
NAME_MATCH_THRESHOLD, NAME_MATCH_MARGIN, CANDIDATE_THRESHOLD = 0.85, 0.05, 0.70
# Resolution and traversal parameters (§7.5: the merge threshold is a parameter worth varying).
DEFAULT_CONFIG = dict(
    er_merge_threshold=0.90,           # SAME_AS between vendor records at or above this score
    er_candidate_threshold=0.75,       # IDENTITY_CANDIDATE between this and the merge threshold
    er_uncorroborated_discount=0.85,   # names alone (no shared phone/domain/address) never reach the merge threshold
    er_conflict_override_evidence=None,  # None: conflicting tax id / LEI always blocks a name merge;
                                         # k: allow it when >= k of phone / email domain / address also match
    ownership_max_depth=10,            # traversal cap; deeper paths end with terminal reason "depth_cap"
)
PLANNED_DATE_COLUMNS = {"dd_refresh_due"}  # scheduled dates, legitimately in the future of the record


# ----------------------------------------------------------------------------- reading and time filters
def read_table(data_dir, name, date_columns=()):
    frame = pd.read_csv(Path(data_dir) / f"{name}.csv")
    for c in frame.select_dtypes("object"):
        frame[c] = frame[c].astype("string").str.strip().replace("", pd.NA)
    for c in date_columns:
        frame[c] = pd.to_datetime(frame[c], errors="raise")
    return frame


def known_by(frame, t, start, observed=None):
    """Rows whose valid time had started by t and, where the table records it, were observed by t."""
    mask = frame[start].notna() & frame[start].le(t)
    if observed is not None:
        mask &= frame[observed].notna() & frame[observed].le(t)
    return frame.loc[mask].copy()


def clip_future(series, t):
    """An end date after t was not knowable at t: the relationship is open at t."""
    return series.where(series.le(t))


def input_hashes(data_dir):
    files = sorted(p for p in Path(data_dir).rglob("*") if p.is_file() and p.suffix in {".csv", ".json"})
    return {str(p.relative_to(data_dir)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


# ----------------------------------------------------------------------------- normalisation
ADDRESS_ABBREVIATIONS = {"st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
                         "blvd": "boulevard", "ste": "suite", "apt": "apartment", "fl": "floor", "bldg": "building"}


def normalize_address(raw, country):
    """NFKC + casefold + limited abbreviations + token sort; numbers kept; no fuzzy merging."""
    if pd.isna(raw) or pd.isna(country):
        return None  # unknown country/address must not form a shared 'missing' node
    tokens = re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", str(raw)).casefold(), flags=re.UNICODE)
    tokens = [ADDRESS_ABBREVIATIONS.get(t, t) for t in tokens]
    return str(country).upper() + "|" + " ".join(sorted(tokens)) if tokens else None


def phone_key(raw):
    if pd.isna(raw):
        return None
    digits = re.sub(r"\D", "", re.split(r"x|ext", str(raw).lower())[0])
    return digits[-9:] if len(digits) >= 7 else None


GREEK = dict(zip("αβγδεζηθικλμνξοπρσςτυφχψωάέήίόύώϊϋΐΰ",
                 "a v g d e z i th i k l m n x o p r s s t y f ch ps o a e i i o y o i y i y".split()))
CYRILLIC = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюяієїґ",
                    "a b v g d e e zh z i y k l m n o p r s t u f kh ts ch sh shch _ y _ e yu ya i ye yi g".split()))
DIGRAPHS = {"ου": "ou", "αι": "ai", "ει": "ei", "οι": "oi", "μπ": "b", "ντ": "d"}
LEGAL_FORMS = {"ltd", "limited", "llc", "inc", "corp", "corporation", "co", "plc", "gmbh", "kg", "ag", "sa", "sas", "sarl",
               "srl", "spa", "bv", "nv", "pty", "sl", "sp", "z", "o", "oo", "ooo", "tov", "too", "kk", "as", "ab", "oy", "lp",
               "llp", "pte", "pvt", "bhd", "sdn", "de", "cv", "ltda", "jsc", "pjsc", "the", "and"}


def transliterate(text):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    for k, v in DIGRAPHS.items():
        text = text.replace(k, v)
    out = []
    for ch in text:
        if ch in GREEK:
            out.append(GREEK[ch])
        elif ch in CYRILLIC:
            out.append(CYRILLIC[ch].strip("_"))
        else:
            base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
            out.append(base if base.isascii() else " ")  # CJK/Hebrew/Hangul cannot be romanised without a library
    return "".join(out)


def name_tokens(name, drop_legal=True):
    if pd.isna(name):
        return ()
    tokens = re.findall(r"[a-z0-9]+", transliterate(re.sub(r"\([^)]*\)", " ", str(name))))
    return tuple(dict.fromkeys(t for t in tokens if not (drop_legal and t in LEGAL_FORMS)))


class NameMatcher:
    """IDF-weighted fuzzy token overlap; zero unless the rarest token on each side has a match."""

    def __init__(self, token_sets):
        self.df = Counter(t for ts in token_sets for t in set(ts))
        self.n = len(token_sets)

    def idf(self, token):
        return math.log((self.n + 1) / (self.df.get(token, 0) + 1)) + 1

    @staticmethod
    def token_similarity(a, b):
        if a == b:
            return 1.
        if min(len(a), len(b)) < 4:
            return 0.
        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        return ratio if ratio >= .8 else 0.

    def _directed(self, a, b):
        weights = [self.idf(t) for t in a]
        matched = [max((self.token_similarity(t, u) for u in b), default=0.) for t in a]
        return sum(w * m for w, m in zip(weights, matched)) / sum(weights), matched[int(np.argmax(weights))]

    def similarity(self, a, b):
        if not a or not b:
            return 0.
        (s1, k1), (s2, k2) = self._directed(a, b), self._directed(b, a)
        return min(s1, s2) if k1 > 0 and k2 > 0 else 0.


def _present(value):
    if isinstance(value, (list, tuple, set, dict)):
        return True
    try:
        return not pd.isna(value)
    except (TypeError, ValueError):  # list-like values
        return True


# ----------------------------------------------------------------------------- graph container
@dataclass
class Graph:
    t: pd.Timestamp
    nodes: pd.DataFrame
    edges: pd.DataFrame
    attributes: dict = field(default_factory=dict)  # node_type -> DataFrame indexed by node_id
    manifest: dict = field(default_factory=dict)

    def graph_hash(self):
        h = hashlib.sha256()
        h.update(self.nodes.sort_values("node_id").to_csv(index=False).encode())
        h.update(self.edges.sort_values("edge_id").to_csv(index=False).encode())
        for node_type in sorted(self.attributes):
            h.update(node_type.encode())
            h.update(self.attributes[node_type].sort_index().to_csv().encode())
        return h.hexdigest()

    def edges_of(self, relationship_type, state=False):
        """Edges of one type; state=True keeps only relationships still valid at t (history view otherwise)."""
        e = self.edges[self.edges.relationship_type == relationship_type]
        return e[e.valid_to.isna() | e.valid_to.ge(self.t)] if state else e

    def write(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.nodes.to_csv(out / "nodes.csv", index=False)
        self.edges.to_csv(out / "edges.csv", index=False)
        for node_type, frame in self.attributes.items():
            frame.to_csv(out / f"attributes_{node_type}.csv")
        for name in ["vendor_pairs", "ownership_paths", "matches"]:  # review tables behind the derived edges
            if isinstance(getattr(self, name, None), pd.DataFrame):
                getattr(self, name).to_csv(out / f"{name}.csv", index=False)
        (out / "manifest.json").write_text(json.dumps({**self.manifest, "graph_hash": self.graph_hash()}, indent=2, default=str))
        return out


# ----------------------------------------------------------------------------- builder
class _Builder:
    def __init__(self, data_dir, t, config=None):
        self.data_dir, self.t = Path(data_dir), pd.Timestamp(t)
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        unknown = set(self.config) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        hashes = input_hashes(self.data_dir)
        self.dataset_version = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
        self.input_hashes = hashes
        self.node_rows, self.edge_rows, self.attributes = [], [], {}

    # -- helpers
    def node(self, node_type, key, label=None, source_type=None):
        node_id = f"{node_type}:{key}"
        self.node_rows.append(dict(node_id=node_id, node_type=node_type, key=str(key), label=label,
                                   source_type=source_type, dataset_version=self.dataset_version))
        return node_id

    def edge(self, rel, src, dst, source_type, source_id, valid_from=pd.NaT, valid_to=pd.NaT, observed_at=pd.NaT,
             weight=(None, None), confidence=None, **properties):
        src_type, dst_type = src.split(":", 1)[0], dst.split(":", 1)[0]
        edge_id = hashlib.sha1(f"{rel}|{src}|{dst}|{source_type}|{source_id}".encode()).hexdigest()[:16]
        self.edge_rows.append(dict(edge_id=edge_id, source_node_id=src, source_node_type=src_type, relationship_type=rel,
                                   target_node_id=dst, target_node_type=dst_type, valid_from=valid_from,
                                   valid_to=valid_to if pd.notna(valid_to) and valid_to <= self.t else pd.NaT,
                                   observed_at=observed_at,
                                   weight_name=weight[0], weight_value=weight[1], source_id=str(source_id),
                                   source_type=source_type, resolution_confidence=confidence,
                                   properties=json.dumps({k: v for k, v in properties.items() if _present(v)},
                                                         sort_keys=True, default=str),
                                   dataset_version=self.dataset_version))

    def attrs(self, node_type, frame):
        frame = frame.copy()
        frame.index = [f"{node_type}:{k}" for k in frame.index]
        frame.index.name = "node_id"
        self.attributes[node_type] = frame

    # -- reference data
    def reference(self):
        t, d = self.t, self.data_dir
        self.markets = read_table(d, "markets").set_index("market_id")
        risk = read_table(d, "country_risk_context", ["as_of_date"])
        risk = risk[risk.as_of_date.le(t)].sort_values("as_of_date")  # use the edition current at t
        latest = risk.groupby(["iso2", "index"]).last()
        pick = lambda idx, col: latest.xs(idx, level="index")[col] if idx in latest.index.get_level_values("index") else pd.Series(dtype=object)
        sanctions = risk[risk["index"] == "SANCTIONS"]
        sanctions = sanctions[sanctions.valid_to.isna() | pd.to_datetime(sanctions.valid_to).ge(t)]
        market_attrs = self.markets[["name", "region", "subregion"]].copy()
        market_attrs["cpi_value"] = pd.to_numeric(pick("CPI", "value"), errors="coerce").reindex(market_attrs.index)
        market_attrs["cpi_tier"] = pd.to_numeric(pick("CPI", "tier"), errors="coerce").reindex(market_attrs.index)
        market_attrs["bo_register_access"] = pick("BO_REGISTER_ACCESS", "tier").reindex(market_attrs.index)
        market_attrs["sanctioned"] = market_attrs.index.isin(set(sanctions.iso2))
        self.country_context = dict(cpi_tier=pd.to_numeric(pick("CPI", "tier"), errors="coerce").to_dict(),
                                    bo_access=pick("BO_REGISTER_ACCESS", "tier").to_dict(), sanctioned=set(sanctions.iso2))
        for m in market_attrs.index:
            self.node("market", m, market_attrs.at[m, "name"], "markets")
        self.attrs("market", market_attrs)

        pe = read_table(d, "paying_entities").set_index("paying_entity_id")
        for k in pe.index:
            self.node("paying_entity", k, pe.at[k, "name"], "paying_entities")
        self.attrs("paying_entity", pe)

        sc = read_table(d, "service_categories").set_index("category_code_v2")
        sc["compliance_flags"] = sc["flags"].fillna("")
        for k in sc.index:
            self.node("service_category", k, sc.at[k, "name"], "service_categories")
        self.attrs("service_category", sc)
        self.v1_to_v2 = sc.reset_index().groupby("category_code_v1").category_code_v2.agg(list).to_dict()
        self.category_flags = {k: {f for f in str(v).split(";") if f} for k, v in sc["flags"].fillna("").items()}

        policy = read_table(d, "approval_policy_matrix").set_index("value_band")
        self.required_level = policy.required_approver_level.to_dict()

    def v2_codes(self, code, version):
        if pd.isna(code):
            return []
        return [code] if version == "v2" else self.v1_to_v2.get(code, [])

    # -- third parties and their direct attributes
    def third_parties(self):
        t, d = self.t, self.data_dir
        tp = read_table(d, "third_parties", ["onboarding_date", "created_date", "terminated_date"])
        tp = tp[tp.onboarding_date.le(t)].copy()  # a record that had not been onboarded by t does not exist yet
        tp["terminated_date"] = clip_future(tp.terminated_date, t)
        tp["created_date"] = clip_future(tp.created_date, t)  # an ERP re-entry after t was not visible at t
        tp["terminated_at_t"] = tp.terminated_date.notna()
        tp = tp.drop(columns=["status"])  # status is as of the snapshot; terminated_at_t replaces it
        dd = read_table(d, "tp_due_diligence", ["dd_completed", "dd_refresh_due"])
        dd = dd[dd.dd_completed.isna() | dd.dd_completed.le(t)]
        tp = tp.merge(dd, on="record_id", how="left").set_index("record_id")
        tp["email_domain"] = tp.email_domain.str.lower()
        tp["generic_email"] = tp.email_domain.isin(GENERIC_EMAIL)
        tp["phone_key"] = tp.phone.map(phone_key)
        self.tp = tp
        for k in tp.index:
            self.node("third_party", k, tp.at[k, "legal_name"], "third_parties")
        self.attrs("third_party", tp)

        # contact infrastructure and declared groups (no observed_at in the source: taken as of the extract)
        for rid, dom in tp.email_domain[~tp.generic_email].dropna().items():
            self.edge("HAS_EMAIL_DOMAIN", f"third_party:{rid}", f"email_domain:{dom}", "third_parties", rid)
        for rid, ph in tp.phone_key.dropna().items():
            self.edge("HAS_PHONE", f"third_party:{rid}", f"phone:{ph}", "third_parties", rid)
        groups = tp.declared_parent_group.dropna().map(lambda s: " ".join(name_tokens(s)) or None).dropna()
        for rid, g in groups.items():
            self.edge("DECLARED_MEMBER_OF", f"third_party:{rid}", f"declared_group:{g}", "third_parties", rid,
                      declared_name=tp.at[rid, "declared_parent_group"])
        for kind, keys in [("email_domain", tp.email_domain[~tp.generic_email].dropna()), ("phone", tp.phone_key.dropna()),
                           ("declared_group", groups)]:
            for k in sorted(set(keys)):
                self.node(kind, k, k, "third_parties")

        addresses = known_by(read_table(d, "tp_address_history", ["valid_from", "valid_to", "observed_at"]), t, "valid_from", "observed_at")
        addresses = addresses[addresses.record_id.isin(tp.index)]
        addresses["address_key"] = [normalize_address(a, c) for a, c in zip(addresses.raw_address, addresses.address_country)]
        addresses = addresses.dropna(subset=["address_key"])
        self.vendor_addresses = addresses[["record_id", "address_key"]]
        ids = {k: hashlib.sha1(k.encode()).hexdigest()[:12] for k in addresses.address_key.unique()}
        for idx, r in addresses.iterrows():
            self.edge("LOCATED_AT", f"third_party:{r.record_id}", f"address:{ids[r.address_key]}", "tp_address_history", f"row:{idx}",
                      r.valid_from, r.valid_to, r.observed_at, address_role=r.address_role, address_type=r.address_type,
                      raw_address=r.raw_address)
        addr_attrs = addresses.groupby("address_key").agg(country=("address_country", "first"),
                                                          address_types=("address_type", lambda s: ";".join(sorted(set(s.dropna())))))
        addr_attrs.index = [ids[k] for k in addr_attrs.index]
        addr_attrs["normalized"] = [k for k in ids if ids[k] in addr_attrs.index]
        for k in addr_attrs.index:
            self.node("address", k, addr_attrs.at[k, "normalized"], "tp_address_history")
        self.attrs("address", addr_attrs)

        services = known_by(read_table(d, "tp_services", ["valid_from", "valid_to"]), t, "valid_from")
        for idx, r in services[services.record_id.isin(tp.index)].iterrows():
            codes = self.v2_codes(r.category_code, r.taxonomy_version)
            for code in codes:  # a v1 code that maps to several v2 codes is split evenly and flagged
                self.edge("PROVIDES_SERVICE", f"third_party:{r.record_id}", f"service_category:{code}", "tp_services", f"row:{idx}",
                          r.valid_from, r.valid_to, weight=("crosswalk_share", 1 / len(codes)),
                          source_code=r.category_code, taxonomy_version=r.taxonomy_version, crosswalk_ambiguous=len(codes) > 1)

        institutions = read_table(d, "healthcare_institutions").set_index("institution_id")
        links = known_by(read_table(d, "tp_institution_links", ["valid_from", "valid_to"]), t, "valid_from")
        links = links[links.record_id.isin(tp.index)]
        for r in links.itertuples():
            self.edge(r.link_type, f"third_party:{r.record_id}", f"institution:{r.institution_id}", "tp_institution_links",
                      r.link_id, r.valid_from, r.valid_to)
        for k in institutions.index:
            self.node("institution", k, institutions.at[k, "name"], "healthcare_institutions")
        self.attrs("institution", institutions)

    # -- engagements, employees, approvals, payments
    def engagements(self):
        t, d = self.t, self.data_dir
        roles = known_by(read_table(d, "employee_roles", ["valid_from", "valid_to"]), t, "valid_from")
        roles["valid_to"] = clip_future(roles.valid_to, t)
        roles["approver_level"] = pd.to_numeric(roles.approver_level, errors="coerce")
        employees = read_table(d, "employees").set_index("employee_id")
        # role at t: the one in force at t, otherwise the latest that started by t
        roles["active_at_t"] = roles.valid_to.isna() | roles.valid_to.ge(t)
        role_at_t = roles.sort_values(["employee_id", "active_at_t", "valid_from"]).drop_duplicates("employee_id", keep="last").set_index("employee_id")
        emp_attrs = role_at_t.reindex(employees.index)
        emp_attrs["has_role_by_t"] = emp_attrs.role.notna()
        for k in employees.index:
            self.node("employee", k, employees.at[k, "name"], "employees")
        self.attrs("employee", emp_attrs)
        for r in roles.itertuples():
            if pd.notna(r.home_market):
                self.edge("HOLDS_ROLE", f"employee:{r.employee_id}", f"market:{r.home_market}", "employee_roles",
                          f"{r.employee_id}|{r.valid_from.date()}", r.valid_from, r.valid_to, role=r.role,
                          approver_level=r.approver_level, scope_markets=r.scope_markets, scope_categories=r.scope_categories)

        eng = known_by(read_table(d, "tp_engagements", ["start_date", "end_date", "observed_at"]), t, "start_date", "observed_at")
        eng = eng[eng.record_id.isin(self.tp.index)].set_index("engagement_id")
        eng["end_date"] = clip_future(eng.end_date, t)
        eng["category_v2"] = [";".join(self.v2_codes(c, v)) for c, v in zip(eng.service_category, eng.taxonomy_version)]
        eng["compliance_flags"] = [";".join(sorted(set().union(*[self.category_flags.get(c, set()) for c in codes.split(";") if c])))
                                   for codes in eng.category_v2]
        eng["required_approver_level"] = eng.value_band.map(self.required_level)
        for k, r in eng.iterrows():
            self.node("engagement", k, None, "tp_engagements")
            self.edge("CONTRACTED_UNDER", f"third_party:{r.record_id}", f"engagement:{k}", "tp_engagements", k,
                      r.start_date, r.end_date, r.observed_at)
            if pd.notna(r.sponsor_employee_id):
                self.edge("SPONSORED_BY", f"engagement:{k}", f"employee:{r.sponsor_employee_id}", "tp_engagements", k,
                          r.start_date, r.end_date, r.observed_at)
            if pd.notna(r.market):
                self.edge("BOOKED_IN", f"engagement:{k}", f"market:{r.market}", "tp_engagements", k, r.start_date, r.end_date, r.observed_at)
        self.attrs("engagement", eng)

        approvals = known_by(read_table(d, "tp_engagement_approvals", ["approval_date", "observed_at"]), t, "approval_date", "observed_at")
        approvals = approvals[approvals.engagement_id.isin(eng.index) & approvals.approver_employee_id.notna()]
        all_roles = roles[["employee_id", "approver_level", "valid_from", "valid_to"]]
        approvals = approvals.merge(all_roles, left_on="approver_employee_id", right_on="employee_id", how="left")
        approvals["role_active"] = (approvals.valid_from.le(approvals.approval_date)
                                    & (approvals.valid_to.isna() | approvals.valid_to.ge(approvals.approval_date))).fillna(False)
        approvals = approvals.sort_values(["approval_id", "role_active", "valid_from"], na_position="first").drop_duplicates("approval_id", keep="last")
        for r in approvals.itertuples():
            self.edge("APPROVED_BY", f"engagement:{r.engagement_id}", f"employee:{r.approver_employee_id}", "tp_engagement_approvals",
                      r.approval_id, r.approval_date, pd.NaT, r.observed_at, approval_type=r.approval_type,
                      approver_level_at_approval=None if pd.isna(r.approver_level) else float(r.approver_level),
                      role_active_at_approval=bool(r.role_active),
                      required_approver_level=eng.at[r.engagement_id, "required_approver_level"])

        pay = read_table(d, "tp_payments_monthly", ["month"])
        pay = pay[(pay.month + pd.offsets.MonthEnd(0)).le(t) & pay.engagement_id.isin(eng.index)]  # a month is known once it has closed
        agg = pay.groupby(["engagement_id", "paying_entity_id"]).agg(first_month=("month", "min"), last_month=("month", "max"),
                                                                    n_months=("month", "nunique"), amount_usd=("amount_usd", "sum"))
        for (e, p), r in agg.iterrows():
            self.edge("PAID_BY", f"engagement:{e}", f"paying_entity:{p}", "tp_payments_monthly", f"{e}|{p}",
                      r.first_month, r.last_month + pd.offsets.MonthEnd(0), weight=("amount_usd", float(r.amount_usd)),
                      n_months=int(r.n_months))
        self.payments = pay  # monthly series stays tabular; detectors read it by engagement

    # -- ownership registers and their link to vendor records
    def ownership(self):
        t, d = self.t, self.data_dir
        statements = [s for s in json.loads((d / "registry_R" / "statements.json").read_text())
                      if pd.Timestamp(s["statementDate"]) <= t]
        entities = {s["statementId"]: s for s in statements if s["recordType"] == "entity"}
        persons = {s["statementId"]: s for s in statements if s["recordType"] == "person"}
        cc = self.country_context
        ent_rows = {}
        for k, s in entities.items():
            r = s["recordDetails"]
            jur = r.get("jurisdiction", {}).get("code")
            ent_rows[k] = dict(name=r.get("name"), source="R", jurisdiction=jur, entity_type=r.get("entityType", {}).get("type"),
                               subtype=r.get("entityType", {}).get("subtype"), public_listing=r.get("publicListing", {}).get("hasPublicListing"),
                               statement_date=s["statementDate"], jurisdiction_cpi_tier=cc["cpi_tier"].get(jur),
                               jurisdiction_bo_access=cc["bo_access"].get(jur), jurisdiction_sanctioned=jur in cc["sanctioned"],
                               jurisdiction_in_markets=jur in self.markets.index)
        unspecified = Counter()
        for s in statements:
            if s["recordType"] != "relationship":
                continue
            rd = s["recordDetails"]
            subject = rd["subject"].get("describedByEntityStatement")
            party = rd["interestedParty"]
            owner = party.get("describedByEntityStatement") or party.get("describedByPersonStatement")
            if subject not in entities:
                continue
            if party.get("unspecifiedReason"):
                ent_rows[subject]["unspecified_owner_reason"] = party["unspecifiedReason"]
                unspecified[party["unspecifiedReason"]] += 1
                continue
            if owner not in entities and owner not in persons:
                continue
            for i, interest in enumerate(rd["interests"]):
                start, end = pd.to_datetime(interest.get("startDate")), pd.to_datetime(interest.get("endDate"))
                if pd.notna(start) and start > t:
                    continue
                rel = "OWNS" if interest.get("type") == "shareholding" else "CONTROLS"
                share = (interest.get("share") or {}).get("exact")
                owner_node = f"{'legal_entity' if owner in entities else 'person'}:{owner}"
                self.edge(rel, owner_node, f"legal_entity:{subject}", "registry_R", f"{s['statementId']}#{i}", start, end,
                          pd.Timestamp(s["statementDate"]), weight=("share_pct", share), confidence=1.0,
                          control_basis=interest.get("details") or interest.get("type"), share_known=share is not None)

        # Source L: accounting parents (corporate only, never a natural person)
        parents = read_table(d, "registry_L_parents", ["relationship_start"])
        parents = parents[parents.relationship_start.isna() | parents.relationship_start.le(t)]
        lei_to_entity = {i["id"]: k for k, s in entities.items() for i in s["recordDetails"].get("identifiers", [])
                         if isinstance(i.get("id"), str) and i["scheme"].endswith("LEI")}
        def L_node(lei, name):
            if pd.notna(lei) and lei in lei_to_entity:
                return lei_to_entity[lei]
            key = f"L-{lei}" if pd.notna(lei) else "L-" + "-".join(name_tokens(name))
            ent_rows.setdefault(key, dict(name=name, source="L", lei=None if pd.isna(lei) else lei))
            return key
        for idx, r in parents.iterrows():
            child, parent = L_node(r.child_lei, r.child_legal_name), L_node(r.parent_lei, r.parent_legal_name)
            self.edge("CONSOLIDATED_BY", f"legal_entity:{child}", f"legal_entity:{parent}", "registry_L_parents", f"row:{idx}",
                      r.relationship_start, confidence=1.0, relationship_status=r.relationship_status)
        L_child_by_lei = {r.child_lei: L_node(r.child_lei, r.child_legal_name) for r in parents.itertuples() if pd.notna(r.child_lei)}

        ent = pd.DataFrame.from_dict(ent_rows, orient="index")
        for k in ent.index:
            self.node("legal_entity", k, ent.at[k, "name"], "registry_R" if ent.at[k, "source"] == "R" else "registry_L_parents")
        self.attrs("legal_entity", ent)
        per = pd.DataFrame.from_dict({k: dict(name=s["recordDetails"]["names"][0]["fullName"],
                                              nationalities=";".join(n.get("code", "") for n in s["recordDetails"].get("nationalities", [])),
                                              pep_status=s["recordDetails"].get("politicalExposure", {}).get("status"),
                                              statement_date=s["statementDate"]) for k, s in persons.items()}, orient="index")
        for k in per.index:
            self.node("person", k, per.at[k, "name"], "registry_R")
        self.attrs("person", per)

        # vendor record -> registry entity: identifiers first, then names within the jurisdiction
        id_lookup = {}
        for k, s in entities.items():
            for i in s["recordDetails"].get("identifiers", []):
                if isinstance(i.get("id"), str):
                    id_lookup[("LEI" if i["scheme"].endswith("LEI") else "REG", i["id"])] = k
        tp = self.tp
        tokens_tp = {rid: name_tokens(n) for rid, n in tp.legal_name.items()}
        tokens_ent = {k: name_tokens(s["recordDetails"].get("name")) for k, s in entities.items()}
        matcher = NameMatcher(list(tokens_tp.values()) + list(tokens_ent.values()))
        by_jur = {}
        for k, s in entities.items():
            by_jur.setdefault(s["recordDetails"].get("jurisdiction", {}).get("code"), []).append(k)

        def candidates(rid, country):
            scored = sorted(((matcher.similarity(tokens_tp[rid], tokens_ent[k]), k) for k in by_jur.get(country, [])), reverse=True)
            if not scored or scored[0][0] < CANDIDATE_THRESHOLD:
                return [], 0., None
            best, best_tokens = scored[0][0], set(tokens_ent[scored[0][1]])
            rivals = [s for s, k in scored if set(tokens_ent[k]) != best_tokens]  # identical registry duplicates are not rivals
            margin = best - rivals[0] if rivals else 1.
            return [k for s, k in scored if s == best and set(tokens_ent[k]) == best_tokens], best, margin

        self.match_rows = []
        self.name_check = Counter(identifier_linked=0, name_matched=0, agrees_with_identifier=0)
        for rid, r in tp.iterrows():
            by_id = {}
            if pd.notna(r.lei) and ("LEI", r.lei) in id_lookup:
                by_id[id_lookup[("LEI", r.lei)]] = "lei"
            elif pd.notna(r.lei) and r.lei in L_child_by_lei:
                by_id[L_child_by_lei[r.lei]] = "lei_L"
            if pd.notna(r.registry_no) and ("REG", r.registry_no) in id_lookup:
                by_id.setdefault(id_lookup[("REG", r.registry_no)], "registry_no")
            self.name_check["identifier_linked"] += int(bool(by_id))
            for k, method in by_id.items():
                self.edge("SAME_AS", f"third_party:{rid}", f"legal_entity:{k}", "identifier", f"{rid}|{k}", confidence=1.0, method=method)
                self.match_rows.append(dict(record_id=rid, target=k, method=method, score=1.0))
            if r.party_type != "organization":
                continue
            ks, score, margin = candidates(rid, r.country)
            if by_id and ks and score >= NAME_MATCH_THRESHOLD and margin >= NAME_MATCH_MARGIN:
                # identifiers act as ground truth for the name matcher (no labels involved)
                agree = any(set(tokens_ent.get(j, ())) == set(tokens_ent[k]) for j in by_id for k in ks)
                self.name_check["name_matched"] += 1
                self.name_check["agrees_with_identifier"] += int(agree)
            for k in set(ks) - set(by_id):
                accepted = score >= NAME_MATCH_THRESHOLD and margin >= NAME_MATCH_MARGIN
                self.edge("SAME_AS" if accepted else "IDENTITY_CANDIDATE", f"third_party:{rid}", f"legal_entity:{k}", "name_match",
                          f"{rid}|{k}", confidence=round(score, 4), method="name", margin=round(margin, 4))
                self.match_rows.append(dict(record_id=rid, target=k, method="name" if accepted else "name_candidate", score=score))

        # individual vendor -> registry person: same nationality as operating country, near-exact name
        tokens_per = {k: " ".join(name_tokens(s["recordDetails"]["names"][0]["fullName"], drop_legal=False)) for k, s in persons.items()}
        nat = {k: {n.get("code") for n in s["recordDetails"].get("nationalities", [])} for k, s in persons.items()}
        for rid, r in tp[tp.party_type == "individual"].iterrows():
            name = " ".join(name_tokens(r.legal_name, drop_legal=False))
            scored = sorted(((difflib.SequenceMatcher(None, name, tokens_per[k]).ratio(), k) for k in persons if r.country in nat[k]), reverse=True)
            if scored and scored[0][0] >= .92 and (len(scored) == 1 or scored[0][0] - scored[1][0] >= .05):
                self.edge("SAME_AS", f"third_party:{rid}", f"person:{scored[0][1]}", "name_match", f"{rid}|{scored[0][1]}",
                          confidence=round(scored[0][0], 4), method="person_name")
                self.match_rows.append(dict(record_id=rid, target=scored[0][1], method="person_name", score=scored[0][0]))
        self.unspecified_owner_reasons = dict(unspecified)

    # -- vendor <-> vendor entity resolution (§7.5)
    def resolve_vendors(self):
        """Duplicate vendor-master records -> SAME_AS / IDENTITY_CANDIDATE edges and ResolvedEntity nodes.

        1. deterministic: a shared tax_id, LEI or register number (same party type) is SAME_AS with confidence 1,
           even when another identifier conflicts (recorded as evidence for review);
        2. scored: within a country, records whose names score >= candidate threshold (IDF-weighted fuzzy tokens)
           get a score = name similarity, discounted when no contact/address evidence corroborates it;
           conflicting identifiers (two different non-null tax ids / LEIs) cap the pair at IDENTITY_CANDIDATE;
        3. merge: SAME_AS pairs at or above the merge threshold form ResolvedEntity components; the rest stay visible.
        """
        cfg, tp = self.config, self.tp
        tokens = {rid: name_tokens(n) for rid, n in tp.legal_name.items()}
        matcher = NameMatcher(list(tokens.values()))
        addr_keys = self.vendor_addresses.groupby("record_id").address_key.agg(set).to_dict()
        domain = tp.email_domain.where(~tp.generic_email)
        pairs = {}

        def conflict(a, b, col):
            x, y = tp.at[a, col], tp.at[b, col]
            return pd.notna(x) and pd.notna(y) and x != y

        for col, method in [("tax_id", "tax_id"), ("lei", "lei"), ("registry_no", "registry_no")]:
            for _, ids in tp.dropna(subset=[col]).groupby([col, "party_type"]).groups.items():
                ids = sorted(ids)
                for i, a in enumerate(ids):
                    for b in ids[i + 1:]:
                        pairs.setdefault((a, b), dict(score=1.0, methods=set(), evidence=set()))["methods"].add(method)
        for _, grp in tp.groupby(["country", "party_type"]):
            ids = sorted(grp.index)
            by_token = {}
            for rid in ids:
                for tok in tokens[rid]:
                    by_token.setdefault(tok, []).append(rid)
            seen = set()
            for tok, members in by_token.items():  # blocking key: country + party type + shared name token
                if len(members) > 50:
                    continue  # generic token: too unspecific to block on
                for i, a in enumerate(members):
                    for b in members[i + 1:]:
                        if (a, b) in seen:
                            continue
                        seen.add((a, b))
                        sim = matcher.similarity(tokens[a], tokens[b])
                        if sim < cfg["er_candidate_threshold"]:
                            continue
                        evidence = set()
                        if pd.notna(tp.at[a, "phone_key"]) and tp.at[a, "phone_key"] == tp.at[b, "phone_key"]:
                            evidence.add("phone")
                        if pd.notna(domain[a]) and domain[a] == domain[b]:
                            evidence.add("email_domain")
                        if addr_keys.get(a, set()) & addr_keys.get(b, set()):
                            evidence.add("address")
                        score = sim if evidence else sim * cfg["er_uncorroborated_discount"]
                        entry = pairs.setdefault((a, b), dict(score=0., methods=set(), evidence=set()))
                        if conflict(a, b, "tax_id") or conflict(a, b, "lei"):
                            # e.g. LLC vs Inc. of one group: different legal entities, never merged on names
                            entry["blocked"] = True
                            evidence.add("conflicting_identifier")
                        entry["score"] = max(entry["score"], score)
                        entry["methods"].add("name")
                        entry["evidence"] |= evidence
                        entry["name_similarity"] = round(sim, 4)

        parent = {rid: rid for rid in tp.index}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        self.vendor_pairs = []
        for (a, b), p in sorted(pairs.items()):
            deterministic = bool(p["methods"] & {"tax_id", "lei", "registry_no"})
            k = cfg["er_conflict_override_evidence"]
            corroboration = len(p["evidence"] & {"phone", "email_domain", "address"})
            blocked = p.get("blocked") and (k is None or corroboration < k)
            merged = deterministic or (p["score"] >= cfg["er_merge_threshold"] and not blocked)
            rel = "SAME_AS" if merged else "IDENTITY_CANDIDATE"
            self.edge(rel, f"third_party:{a}", f"third_party:{b}", "entity_resolution", f"{a}|{b}", confidence=round(p["score"], 4),
                      methods=sorted(p["methods"]), evidence=sorted(p["evidence"]), name_similarity=p.get("name_similarity"))
            self.vendor_pairs.append(dict(a=a, b=b, score=p["score"], merged=merged, methods=";".join(sorted(p["methods"])),
                                          evidence=";".join(sorted(p["evidence"]))))
            if merged:
                parent[find(a)] = find(b)
        components = {}
        for rid in tp.index:
            components.setdefault(find(rid), []).append(rid)
        rows = {}
        for members in components.values():
            members = sorted(members)
            key = "RE-" + hashlib.sha1("|".join(members).encode()).hexdigest()[:12]
            rows[key] = dict(n_records=len(members), member_record_ids="|".join(members),
                             countries=";".join(sorted(set(tp.loc[members, "country"].dropna()))),
                             first_onboarding=tp.loc[members, "onboarding_date"].min(),
                             distinct_tax_ids=tp.loc[members, "tax_id"].dropna().nunique())
            self.node("resolved_entity", key, tp.at[members[0], "legal_name"], "entity_resolution")
            for rid in members:
                self.edge("RESOLVED_AS", f"third_party:{rid}", f"resolved_entity:{key}", "entity_resolution", rid, confidence=1.0)
        self.attrs("resolved_entity", pd.DataFrame.from_dict(rows, orient="index"))
        self.resolved_of = {rid: "RE-" + hashlib.sha1("|".join(sorted(m)).encode()).hexdigest()[:12]
                            for m in components.values() for rid in m}

    # -- ownership completion (§7.3-7.4): ultimate owners, path shares, terminal reasons
    def complete_ownership(self):
        cfg = self.config
        edges = pd.DataFrame(self.edge_rows)
        own = edges[edges.relationship_type.isin(["OWNS", "CONTROLS"]) & edges.valid_to.isna()]  # current at t
        cons = edges[edges.relationship_type == "CONSOLIDATED_BY"]
        owners_of = {}
        for r in own.itertuples():
            owners_of.setdefault(r.target_node_id, []).append(
                (r.source_node_id, r.relationship_type, r.weight_value, json.loads(r.properties).get("control_basis"), r.source_id))
        accounting_parent = {}
        for r in cons.itertuples():
            accounting_parent.setdefault(r.source_node_id, []).append(r.target_node_id)
        ent = self.attributes["legal_entity"]
        persons = self.attributes["person"]
        stated_reason = ent.get("unspecified_owner_reason", pd.Series(dtype=object))

        def walk(start):
            """All upward paths from one entity. Each path ends in a person or a terminal reason."""
            paths, stack = [], [(start, [start], 1.0, False, [])]
            while stack:
                node, path, share, control_only, statements = stack.pop()
                ups = owners_of.get(node, [])
                if not ups:
                    if node.startswith("person:"):
                        reason = "natural_person"
                    elif pd.notna(stated_reason.get(node)):
                        reason = f"stated:{stated_reason.get(node)}"
                    elif node in accounting_parent:
                        reason = "accounting_parent_only"  # Source L reaches no person by construction
                    else:
                        reason = "no_owner_recorded"
                    paths.append(dict(path=path, share=None if control_only else share, end=node, reason=reason, statements=statements))
                    continue
                for owner, rel, pct, basis, sid in ups:
                    if owner in path:
                        paths.append(dict(path=path + [owner], share=None, end=owner, reason="cycle", statements=statements + [sid]))
                        continue
                    if len(path) > cfg["ownership_max_depth"]:
                        paths.append(dict(path=path, share=None, end=node, reason="depth_cap", statements=statements))
                        continue
                    step = pct / 100 if rel == "OWNS" and pct is not None and pct == pct else None
                    stack.append((owner, path + [owner], share * step if step is not None else share,
                                  control_only or step is None, statements + [sid]))
            return paths

        cc = self.country_context
        summary_rows, path_rows = {}, []
        linked = edges[(edges.relationship_type == "SAME_AS") & (edges.target_node_type == "legal_entity")]
        for r in linked.itertuples():
            rid = r.source_node_id.split(":", 1)[1]
            for p in walk(r.target_node_id):
                entities = [n for n in p["path"] if n.startswith("legal_entity:")]
                jurisdictions = [ent.at[n, "jurisdiction"] for n in entities if n in ent.index and pd.notna(ent.at[n, "jurisdiction"])]
                bases = [b for n in p["path"] for (o, _, _, b, _) in owners_of.get(n, []) if o in p["path"]]
                path_rows.append(dict(record_id=rid, entry_entity=r.target_node_id, match_method=json.loads(r.properties).get("method"),
                                      match_confidence=r.resolution_confidence, depth=len(p["path"]) - 1, end=p["end"],
                                      terminal_reason=p["reason"], path_share=p["share"], path=" > ".join(p["path"]),
                                      statements=";".join(p["statements"]), jurisdictions=">".join(jurisdictions),
                                      n_jurisdictions=len(set(jurisdictions)),
                                      jurisdiction_not_in_markets=any(j not in self.markets.index for j in jurisdictions),
                                      bo_access_low_on_path=any(cc["bo_access"].get(j) == "low" for j in jurisdictions),
                                      sanctioned_on_path=any(j in cc["sanctioned"] for j in jurisdictions),
                                      trust_or_nominee=bool(set(bases) & {"trust", "nominee"})
                                      or any(ent.at[n, "subtype"] in {"trust", "fund"} for n in entities if n in ent.index),
                                      end_is_pep=p["end"] in persons.index and persons.at[p["end"], "pep_status"] == "isPep"))
        paths = pd.DataFrame(path_rows)
        self.ownership_paths = paths
        if paths.empty:
            return
        for rid, grp in paths.groupby("record_id"):
            people = grp[grp.terminal_reason == "natural_person"]
            summary_rows[rid] = dict(
                own_linked_entities=grp.entry_entity.nunique(), own_min_match_confidence=grp.match_confidence.min(),
                own_n_paths=len(grp), own_max_depth=grp.depth.max(), own_n_jurisdictions=len(set(">".join(grp.jurisdictions).split(">")) - {""}),
                own_n_ultimate_persons=people.end.nunique(), own_max_person_share=people.path_share.max(),
                own_terminal_reasons=";".join(sorted(set(grp.terminal_reason))), own_cycle=(grp.terminal_reason == "cycle").any(),
                own_pep_owner=grp.end_is_pep.any(), own_trust_or_nominee=grp.trust_or_nominee.any(),
                own_jurisdiction_not_in_markets=grp.jurisdiction_not_in_markets.any(), own_bo_access_low_on_path=grp.bo_access_low_on_path.any(),
                own_sanctioned_on_path=grp.sanctioned_on_path.any())
            for person, pp in people.groupby("end"):
                shares = pp.path_share.dropna()
                self.edge("ULTIMATE_OWNER", f"third_party:{rid}", person, "derived", f"{rid}|{person}",
                          weight=("indirect_share", float(shares.sum()) if len(shares) else None),
                          confidence=float(pp.match_confidence.min()), n_paths=len(pp), max_depth=int(pp.depth.max()),
                          control_only=bool(shares.empty), statements=sorted(set(";".join(pp.statements).split(";")) - {""}))
        summary = pd.DataFrame.from_dict(summary_rows, orient="index")
        summary.index = [f"third_party:{k}" for k in summary.index]
        self.attributes["third_party"] = self.attributes["third_party"].join(summary)

    def finish(self):
        nodes = pd.DataFrame(self.node_rows, columns=NODE_COLUMNS).drop_duplicates("node_id").reset_index(drop=True)
        edges = pd.DataFrame(self.edge_rows, columns=EDGE_COLUMNS)
        for c in ["valid_from", "valid_to", "observed_at"]:
            edges[c] = pd.to_datetime(edges[c])
        edges = edges.sort_values("edge_id").reset_index(drop=True)
        resolved = self.attributes["resolved_entity"]
        paths = self.ownership_paths
        manifest = dict(build_version=BUILD_VERSION, t=str(self.t.date()), dataset_version=self.dataset_version, config=self.config,
                        entity_resolution=dict(records=int(resolved.n_records.sum()), resolved_entities=len(resolved),
                                               merged_entities=int((resolved.n_records > 1).sum()), largest=int(resolved.n_records.max()),
                                               same_as_pairs=sum(p["merged"] for p in self.vendor_pairs),
                                               candidate_pairs=sum(not p["merged"] for p in self.vendor_pairs)),
                        ownership=dict(vendors_with_paths=int(paths.record_id.nunique()) if len(paths) else 0,
                                       terminal_reasons=paths.terminal_reason.value_counts().to_dict() if len(paths) else {}),
                        input_sha256=self.input_hashes, name_match=dict(threshold=NAME_MATCH_THRESHOLD, margin=NAME_MATCH_MARGIN,
                                                                         candidate_threshold=CANDIDATE_THRESHOLD,
                                                                         check_against_identifiers=dict(self.name_check)),
                        generic_email_domains=sorted(GENERIC_EMAIL), unspecified_owner_reasons=self.unspecified_owner_reasons,
                        node_counts=nodes.node_type.value_counts().to_dict(),
                        edge_counts=edges.relationship_type.value_counts().to_dict(),
                        limitations=["Contact fields, institution links and employee roles have no observed_at; taken as recorded.",
                                     "registry_L relationship_status is as of the extract, not as of t.",
                                     "CJK/Hebrew/Hangul names cannot be romanised; such entities link only by identifier.",
                                     "Ownership paths start only from accepted SAME_AS links; IDENTITY_CANDIDATE links are not traversed."])
        return Graph(self.t, nodes, edges, self.attributes, manifest)


def build_graph(data_dir=DEFAULT_DATA_DIR, t=SNAPSHOT, config=None):
    b = _Builder(data_dir, t, config)
    b.reference()
    b.third_parties()
    b.resolve_vendors()
    b.engagements()
    b.ownership()
    b.complete_ownership()
    graph = b.finish()
    graph.payments = b.payments
    graph.matches = pd.DataFrame(b.match_rows, columns=["record_id", "target", "method", "score"])
    graph.vendor_pairs = pd.DataFrame(b.vendor_pairs, columns=["a", "b", "score", "merged", "methods", "evidence"])
    graph.ownership_paths = b.ownership_paths
    return graph


def resolution_sensitivity(data_dir=DEFAULT_DATA_DIR, t=SNAPSHOT, thresholds=(0.80, 0.85, 0.90, 0.95), overrides=(None, 2, 3)):
    """§7.5 step 7: how the resolved population changes with the merge threshold and the conflict policy."""
    rows = []
    for th in thresholds:
        for k in overrides:
            g = build_graph(data_dir, t, dict(er_merge_threshold=th, er_conflict_override_evidence=k))
            r = g.attributes["resolved_entity"]
            rows.append(dict(merge_threshold=th, conflict_override_evidence=k, resolved_entities=len(r),
                             merged_entities=int((r.n_records > 1).sum()), largest_entity=int(r.n_records.max()),
                             same_as_pairs=int(g.vendor_pairs.merged.sum()), candidate_pairs=int((~g.vendor_pairs.merged).sum())))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Appendix B checks
def ownership_cycles(graph):
    own = graph.edges[graph.edges.relationship_type.isin(["OWNS", "CONTROLS", "CONSOLIDATED_BY"])]
    parents = {}
    for r in own.itertuples():  # edge direction for cycles: subject -> owner
        child, parent = (r.target_node_id, r.source_node_id) if r.relationship_type != "CONSOLIDATED_BY" else (r.source_node_id, r.target_node_id)
        parents.setdefault(child, set()).add(parent)
    cycles, state = [], {}
    def visit(node, path):
        state[node] = 1
        for p in parents.get(node, ()):
            if state.get(p) == 1:
                cycles.append(path[path.index(p):] + [p] if p in path else [node, p])
            elif state.get(p) is None:
                visit(p, path + [p])
        state[node] = 2
    for n in list(parents):
        if n not in state:
            visit(n, [n])
    return cycles


def check_graph(graph, cases=None, min_support=5):
    """Appendix B data-quality checks. Returns a DataFrame of (check, status, detail); status is pass/fail/warn/info."""
    n, e, t = graph.nodes, graph.edges, graph.t
    results = []
    def add(check, ok, detail="", info=False, warn=False):
        status = "info" if info else "pass" if ok else "warn" if warn else "fail"
        results.append(dict(check=check, status=status, detail=str(detail)))

    add("identifiers: node ids unique", n.node_id.is_unique, n.node_id[n.node_id.duplicated()].head(3).tolist())
    add("identifiers: no blank keys", n.key.notna().all() and n.key.str.len().gt(0).all())
    add("identifiers: edge ids unique", e.edge_id.is_unique, e.edge_id[e.edge_id.duplicated()].head(3).tolist())
    known = set(n.node_id)
    dangling = e[~e.source_node_id.isin(known) | ~e.target_node_id.isin(known)]
    add("referential integrity", dangling.empty, dangling.edge_id.head(3).tolist())
    as_tuple = lambda x: x if isinstance(x, tuple) else (x,)
    bad = [r.edge_id for r in e.itertuples() if r.relationship_type not in SCHEMA
           or r.source_node_type not in as_tuple(SCHEMA[r.relationship_type][0])
           or r.target_node_type not in as_tuple(SCHEMA[r.relationship_type][1])]
    add("type constraints: allowed triples only", not bad, bad[:3])
    order = e[e.valid_from.notna() & e.valid_to.notna() & e.valid_from.gt(e.valid_to)]
    add("temporal: valid_from <= valid_to", order.empty, order[["relationship_type", "source_id"]].head(3).values.tolist())
    future = e[e.valid_from.gt(t) | e.valid_to.gt(t) | e.observed_at.gt(t)]
    add("temporal: nothing after t on edges", future.empty, future[["relationship_type", "source_id"]].head(3).values.tolist())
    late_attr = []
    for node_type, frame in graph.attributes.items():
        for c in frame.select_dtypes(include=["datetime64[ns]"]).columns:
            if c not in PLANNED_DATE_COLUMNS and frame[c].gt(t).any():
                late_attr.append(f"{node_type}.{c}")
    add("feature leakage: no attribute date after t", not late_attr, late_attr)

    loc = graph.edges_of("LOCATED_AT", state=True)
    roles = loc.properties.map(lambda p: json.loads(p).get("address_role"))
    multi = loc.assign(role=roles).groupby(["source_node_id", "role"]).target_node_id.nunique()
    add("overlapping assertions: >1 current address per vendor and role", True, f"{int((multi > 1).sum())} vendor-role pairs", info=True)
    own = e[e.relationship_type == "OWNS"]
    current_own = own[own.valid_to.isna()]
    share_sum = current_own.groupby("target_node_id").weight_value.sum(min_count=1)
    add("overlapping assertions: current shares sum > 100%", True, f"{int((share_sum > 100.0001).sum())} entities", info=True)
    shares = own.weight_value.dropna()
    add("ownership percentages within [0, 100]", shares.between(0, 100).all(), shares[~shares.between(0, 100)].head(3).tolist())
    add("ownership: unknown share kept as null, not zero", True, f"{int(own.weight_value.isna().sum())} OWNS edges without a share", info=True)
    cycles = ownership_cycles(graph)
    add("ownership cycles detected and reported", True, f"{len(cycles)} cycles" + (f", e.g. {cycles[0]}" if cycles else ""), info=True)
    tp = graph.attributes["third_party"]
    dup_tax = tp.dropna(subset=["tax_id"]).groupby("tax_id").size()
    add("duplicates: records sharing a tax_id (before resolution)", True, f"{int((dup_tax > 1).sum())} tax ids, {int(dup_tax[dup_tax > 1].sum())} records", info=True)
    exact = e.duplicated(["relationship_type", "source_node_id", "target_node_id", "source_id"])
    add("duplicates: no exact duplicate edges", not exact.any(), int(exact.sum()))
    ext = e[e.source_type.isin(["registry_R", "registry_L_parents", "identifier", "name_match"])]
    add("provenance: external edges carry source id and confidence", ext.source_id.notna().all() and ext.resolution_confidence.notna().all())
    reg = e[e.source_type == "registry_R"]
    add("provenance: registry edges carry observation time", reg.observed_at.notna().all())
    ent = graph.attributes["legal_entity"]
    owned = set(e.loc[e.relationship_type.isin(["OWNS", "CONTROLS"]), "target_node_id"])
    no_owner = ent.index[~ent.index.isin(owned)]
    reasons = ent.loc[no_owner, "unspecified_owner_reason"].value_counts(dropna=False).to_dict() if "unspecified_owner_reason" in ent else {}
    add("missingness: no-owner entities split by stated reason vs not recorded", True, {str(k): v for k, v in reasons.items()}, info=True)
    resolved = graph.attributes.get("resolved_entity")
    if resolved is not None:
        records = e[e.relationship_type == "RESOLVED_AS"]
        add("entity resolution: every record resolves to exactly one entity",
            records.source_node_id.is_unique and set(records.source_node_id) == set(n.node_id[n.node_type == "third_party"]))
        add("entity resolution: merged entities and largest component", True,
            f"{int((resolved.n_records > 1).sum())} merged, largest {int(resolved.n_records.max())} records", info=True)
        mixed = resolved[resolved.distinct_tax_ids > 1]
        add("entity resolution: merged entities with conflicting tax ids (review)", True, f"{len(mixed)} entities", info=True)
        cand = e[(e.relationship_type == "IDENTITY_CANDIDATE") & (e.target_node_type == "third_party")]
        add("entity resolution: unmerged candidate pairs kept visible", True, f"{len(cand)} pairs", info=True)
    uo = e[e.relationship_type == "ULTIMATE_OWNER"]
    add("ownership: derived ultimate-owner edges cite their statements",
        uo.properties.map(lambda p: bool(json.loads(p).get("statements"))).all() if len(uo) else True)
    if "own_terminal_reasons" in tp:
        reasons = Counter(r for v in tp.own_terminal_reasons.dropna() for r in v.split(";"))
        add("ownership: terminal reasons of vendor paths", True, dict(reasons), info=True)
    add("snapshot reproducibility: graph hash", True, graph.graph_hash()[:16], info=True)

    if cases is not None:  # case-dependent checks (labels used for checking only)
        members = cases.member_record_ids.str.split("|").explode()
        tp = tp.set_axis(tp.index.str.split(":", n=1).str[1])  # record ids without the node-type prefix
        present = members.isin(tp.index)
        add("case members present at t", True, f"{int(present.sum())}/{len(present)} (members onboarding after t are excluded)", info=True)
        if "period" in cases:
            add("case partition: one period per case", cases.groupby("case_id").period.nunique().le(1).all())
        positive = tp.index.isin(set(members))
        leaks = []
        for c in ["party_type", "country", "expert_type", "risk_rating", "dd_level", "screening_result", "remit_to_country"]:
            if c not in tp:
                continue
            for value, grp in pd.Series(positive, index=tp.index).groupby(tp[c].astype(str)):
                if len(grp) >= min_support and grp.all():
                    leaks.append(f"{c}={value} ({len(grp)} records, all positive)")
        rank = pd.Series(tp.index, index=tp.index).rank()
        auc = (rank[positive].mean() - (positive.sum() + 1) / 2) / max((~positive).sum(), 1) if positive.any() else np.nan
        # a property of the labelled set, not a build error: surfaced as a warning for whoever uses these attributes
        add("shortcut check: no single attribute value isolates cases", not leaks, leaks[:5], warn=True)
        # ids are random tokens; their ordering must not separate case members
        add("shortcut check: record_id ordering vs labels (AUC ~ 0.5)", abs(auc - .5) < .1 if auc == auc else True, round(float(auc), 3))
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--t", default=str(SNAPSHOT.date()), help="as-of date, e.g. 2024-06-30")
    parser.add_argument("--out", default=None, help="output directory (default results/graph/<t>)")
    args = parser.parse_args()
    graph = build_graph(args.data_dir, args.t)
    cases = pd.read_csv(Path(args.data_dir) / "labelled_cases_TRAINING_SET.csv")
    periods = ROOT / "results" / "case_periods.csv"
    if periods.exists():
        cases = cases.merge(pd.read_csv(periods)[["case_id", "period"]], on="case_id", how="left")
    checks = check_graph(graph, cases)
    out = graph.write(args.out or ROOT / "results" / "graph" / args.t)
    checks.to_csv(out / "quality_checks.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 90):
        print(checks.to_string(index=False))
    print(f"\nnodes={len(graph.nodes)} edges={len(graph.edges)} -> {out}")
    if (checks.status == "fail").any():
        raise SystemExit("Data-quality checks failed")


if __name__ == "__main__":
    main()
