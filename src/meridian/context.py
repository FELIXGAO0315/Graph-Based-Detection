"""Per-time-point context shared by all detectors.

Built once from an as_of(t) bundle. Derived views are computed lazily and
cached, so detectors only pay for what they use.
"""
from __future__ import annotations

from collections import defaultdict
from functools import cached_property

import numpy as np
import pandas as pd

from . import resolution as res
from .data import Bundle


# Which business events can explain which pattern kinds (from the event descriptions in
# business_events). Kinds: onboarding, sponsor, spend, termination, category.
EVENT_RULES = {
    "covid": {"kinds": {"onboarding", "sponsor", "spend"}, "parents": {"SCM"}, "categories": {"RND.CSL"}},
    "erp migration": {"kinds": {"sponsor"}, "roles": {"system_default"}},
    "demerger": {"kinds": {"termination"}, "units": {"CONSUMER"}},
    "acquisition integration": {"kinds": {"onboarding", "sponsor"}},
    "vaccine launch": {"kinds": {"onboarding", "sponsor", "spend"}, "parents": {"HCP", "EVT", "MED", "PAT"},
                       "categories": {"COM.MACC"}},
    "taxonomy recode": {"kinds": {"category"}},
    "rationalization": {"kinds": {"onboarding", "sponsor", "termination"}, "categories": {"GNA.IT"}},
}


class Context:
    def __init__(self, bundle: Bundle, cfg: dict | None = None):
        self.b = bundle
        self.t = bundle.as_of
        self.cfg = cfg or {}

    @property
    def use_ownership(self) -> bool:
        return bool(self.cfg.get("features", {}).get("ownership", True))

    def param(self, detector: str, key: str, default):
        return self.cfg.get("params", {}).get(detector, {}).get(key, default)

    # ------------------------------------------------------------------
    # Vendors and entity resolution
    # ------------------------------------------------------------------
    @cached_property
    def vendors(self) -> pd.DataFrame:
        return self.b.third_parties.set_index("record_id", drop=False)

    @property
    def rcfg(self) -> dict:
        """Entity-resolution settings (strictness and perturbation for sensitivity runs)."""
        return self.cfg.get("resolution", {})

    @cached_property
    def _resolution(self):
        return res.resolve_duplicates(self.b, self.rcfg.get("name_merge", "corroborated"))

    @cached_property
    def entity_map(self) -> pd.DataFrame:
        return self._resolution[0]

    @cached_property
    def identity_pairs(self) -> pd.DataFrame:
        return self._resolution[1]

    @cached_property
    def entity_of(self) -> dict:
        return dict(zip(self.entity_map["record_id"], self.entity_map["entity_id"]))

    @cached_property
    def records_of(self) -> dict:
        out = defaultdict(set)
        for r, e in self.entity_of.items():
            out[e].add(r)
        return out

    def expand(self, records) -> frozenset:
        """All vendor-master records of the entities behind `records`; cases list every record of an entity."""
        out = set()
        for r in records:
            out |= self.records_of.get(self.entity_of.get(r, r), {r})
        return frozenset(out)

    def n_entities(self, records) -> int:
        return len({self.entity_of.get(r, r) for r in records})

    # ------------------------------------------------------------------
    # Addresses and contacts
    # ------------------------------------------------------------------
    @cached_property
    def addresses(self) -> pd.DataFrame:
        return res.normalise_addresses(self.b)

    @cached_property
    def contacts(self) -> pd.DataFrame:
        return res.normalise_contacts(self.b)

    # ------------------------------------------------------------------
    # Service categories (v1 -> v2 disambiguation: 11 of 25 v1 codes map to several v2 codes)
    # ------------------------------------------------------------------
    @cached_property
    def categories(self) -> pd.DataFrame:
        sc = self.b.service_categories.copy()
        sc["flag_set"] = sc["flags"].fillna("").map(lambda s: frozenset(f for f in s.split(";") if f))
        return sc.set_index("category_code_v2")

    @cached_property
    def _v1_to_v2(self) -> dict:
        return self.b.service_categories.groupby("category_code_v1")["category_code_v2"].apply(tuple).to_dict()

    @cached_property
    def _vendor_v2(self) -> dict:
        s = self.b.tp_services
        return s[s["taxonomy_version"] == "v2"].groupby("record_id")["category_code"].apply(set).to_dict()

    def to_v2(self, record_id: str, code: str, version: str) -> tuple[tuple, bool]:
        """Map a category to v2. Returns (candidate v2 codes, certain)."""
        if version == "v2":
            return (code,), True
        cands = self._v1_to_v2.get(code, (code,))
        if len(cands) == 1:
            return cands, True
        own = tuple(c for c in cands if c in self._vendor_v2.get(record_id, ()))
        if len(own) == 1:
            return own, True
        return (own or cands), False

    def flags_of(self, v2_codes) -> tuple[frozenset, frozenset]:
        """(certain flags, possible flags) over candidate v2 codes."""
        sets = [self.categories.at[c, "flag_set"] for c in v2_codes if c in self.categories.index]
        if not sets:
            return frozenset(), frozenset()
        return frozenset.intersection(*sets), frozenset.union(*sets)

    def parent_of(self, v2_codes) -> str | None:
        parents = {self.categories.at[c, "parent"] for c in v2_codes if c in self.categories.index}
        return parents.pop() if len(parents) == 1 else None

    @cached_property
    def engagements(self) -> pd.DataFrame:
        """Engagements with v2 category, flags, parent and sponsor role attached."""
        e = self.b.tp_engagements.copy()
        mapped = [self.to_v2(r, c, v) for r, c, v in zip(e["record_id"], e["service_category"], e["taxonomy_version"])]
        e["cat_v2"] = [m[0] for m in mapped]
        e["cat_certain"] = [m[1] for m in mapped]
        e["cat_primary"] = [m[0][0] if m[1] else None for m in mapped]
        fl = [self.flags_of(m[0]) for m in mapped]
        e["cat_flags"] = [f[0] for f in fl]
        e["cat_flags_possible"] = [f[1] for f in fl]
        e["parent"] = [self.parent_of(m[0]) for m in mapped]
        roles = self.roles[["employee_id", "role", "home_market", "scope_markets", "scope_categories",
                            "approver_level", "franchise"]].add_prefix("sp_")
        e = e.merge(roles, left_on="sponsor_employee_id", right_on="sp_employee_id", how="left")
        return e.drop(columns=["sp_employee_id"])

    @cached_property
    def services(self) -> pd.DataFrame:
        s = self.b.tp_services.copy()
        mapped = [self.to_v2(r, c, v) for r, c, v in zip(s["record_id"], s["category_code"], s["taxonomy_version"])]
        s["cat_v2"] = [m[0] for m in mapped]
        s["cat_certain"] = [m[1] for m in mapped]
        s["cat_flags"] = [self.flags_of(m[0])[0] for m in mapped]
        s["parent"] = [self.parent_of(m[0]) for m in mapped]
        return s

    # ------------------------------------------------------------------
    # People and approvals
    # ------------------------------------------------------------------
    @cached_property
    def roles(self) -> pd.DataFrame:
        return self.b.employee_roles

    @cached_property
    def role_of(self) -> dict:
        return self.roles.set_index("employee_id")["role"].to_dict()

    @cached_property
    def approvals(self) -> pd.DataFrame:
        """Approvals joined to engagement, band requirement and approver role."""
        ap = self.b.tp_engagement_approvals.merge(
            self.engagements[["engagement_id", "record_id", "value_band", "sponsor_employee_id", "market",
                              "start_date", "cat_flags", "parent", "sp_role"]], on="engagement_id")
        ap = ap.merge(self.b.approval_policy_matrix[["value_band", "required_approver_level"]], on="value_band",
                      how="left")
        r = self.roles[["employee_id", "role", "approver_level", "home_market", "scope_markets", "valid_from",
                        "valid_to"]].add_prefix("ap_")
        return ap.merge(r, left_on="approver_employee_id", right_on="ap_employee_id", how="left").drop(
            columns=["ap_employee_id"])

    @cached_property
    def markets(self) -> pd.DataFrame:
        return self.b.markets.set_index("market_id")

    def in_scope(self, scope: str | None, market: str) -> bool:
        """Does an employee scope string ('*', 'R:EUROPE', 'S:EU_WEST', 'US;CA') cover market?"""
        if not isinstance(scope, str):
            return False
        m = self.markets.loc[market] if market in self.markets.index else None
        for s in scope.split(";"):
            s = s.strip()
            if s == "*" or s == market:
                return True
            if m is not None and s.startswith("R:") and s[2:] == m["region"]:
                return True
            if m is not None and s.startswith("S:") and s[2:] == m["subregion"]:
                return True
        return False

    # ------------------------------------------------------------------
    # Business events (benign explanations)
    # ------------------------------------------------------------------
    def events_overlapping(self, lo, hi, market: str | None = None) -> list[str]:
        ev = self.b.business_events
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        end = ev["end_date"].fillna(ev["start_date"] + pd.Timedelta(days=90))
        hit = ev[(ev["start_date"] <= hi) & (end >= lo)]
        if market and len(hit):
            hit = hit[hit["markets"].map(lambda s: s == "ALL" or market in str(s).split("|")).astype(bool)]
        return list(hit["event_name"])

    def benign_events(self, kind: str, lo, hi, market: str | None = None, cats=(), roles=(), units=()) -> list[str]:
        """Business events that plausibly explain a pattern of this kind.

        `cats` is a list of (parent, v2 category) pairs for the vendors involved.
        An event only counts when its rule covers the pattern kind and, where
        the rule names categories, business units or roles, most of the
        involved vendors fall under them. Without that, a two-year event such
        as COVID would excuse every burst in the period.
        """
        out = []
        cats = list(cats)
        for name in self.events_overlapping(lo, hi, market):
            rule = next((r for key, r in EVENT_RULES.items() if key in name.lower()), None)
            if rule is None or kind not in rule["kinds"]:
                continue
            ok = True
            if "parents" in rule or "categories" in rule:
                hits = [p in rule.get("parents", ()) or c in rule.get("categories", ()) for p, c in cats]
                ok = bool(hits) and sum(hits) >= 0.5 * len(hits)
            if ok and "roles" in rule:
                ok = any(r in rule["roles"] for r in roles)
            if ok and "units" in rule:
                ok = any(u in rule["units"] for u in units)
            if ok:
                out.append(f"business event: {name}")
        return out

    # ------------------------------------------------------------------
    # Country context
    # ------------------------------------------------------------------
    def country_index(self, iso2: str, index: str, at=None):
        """Tier of the edition current at `at` (default t)."""
        at = pd.Timestamp(at) if at is not None else self.t
        c = self.b.country_risk_context
        c = c[(c["iso2"] == iso2) & (c["index"] == index) & (c["as_of_date"] <= at)]
        if len(c) == 0:
            return None
        return c.sort_values("as_of_date").iloc[-1]["tier"]

    # ------------------------------------------------------------------
    # Registry and ownership
    # ------------------------------------------------------------------
    @cached_property
    def reg_matches(self) -> pd.DataFrame:
        m = pd.concat([res.match_registry(self.b), res.match_persons(self.b)], ignore_index=True)
        rc = self.rcfg
        by_name = ~m["method"].isin(["registry_no", "lei"])
        if not rc.get("registry_name_match", True):
            m = m[~by_name]
            by_name = by_name[~by_name]
        if rc.get("accept_candidates", False):
            m = m.assign(status="accepted")
        drop = rc.get("drop_name_matches", 0.0)
        if drop:
            rng = np.random.default_rng(rc.get("perturb_seed", 0))
            m = m[~(by_name & (m["status"] == "accepted") & (rng.random(len(m)) < drop))]
        return m

    @cached_property
    def reg_entity_of(self) -> dict:
        """record_id -> accepted registry entity statement id (organisations)."""
        m = self.reg_matches
        m = m[(m["status"] == "accepted") & ~m["method"].str.startswith("person")]
        return m.drop_duplicates("record_id").set_index("record_id")["statement_id"].to_dict()

    @cached_property
    def reg_person_of(self) -> dict:
        m = self.reg_matches
        m = m[(m["status"] == "accepted") & m["method"].str.startswith("person")]
        return m.drop_duplicates("record_id").set_index("record_id")["statement_id"].to_dict()

    @cached_property
    def records_of_reg(self) -> dict:
        out = defaultdict(set)
        for r, s in {**self.reg_entity_of, **self.reg_person_of}.items():
            out[s].add(r)
        return out

    @cached_property
    def reg_nodes(self) -> pd.DataFrame:
        e = self.b.reg_entities.assign(kind="entity")
        p = self.b.reg_persons.assign(kind="person", name=self.b.reg_persons["full_name"],
                                      jurisdiction=self.b.reg_persons["nationality"])
        cols = ["statement_id", "kind", "name", "jurisdiction", "entity_type", "entity_subtype", "pep_status",
                "statement_date", "listed"]
        return pd.concat([e.reindex(columns=cols), p.reindex(columns=cols)]).set_index("statement_id")

    @cached_property
    def _interest_rows(self) -> list[dict]:
        return self.b.reg_interests.to_dict("records")

    @cached_property
    def _interests_by_subject(self) -> dict:
        out = defaultdict(list)
        for r in self._interest_rows:
            out[r["subject_sid"]].append(r)
        return out

    def owners_at(self, at=None) -> dict:
        """subject statement -> interest rows valid at `at` (valid time; default t).

        Knowledge is already cut at t by as_of, so an earlier `at` shows the
        structure as it stood then, as far as it was published by t.
        """
        at = self.t if at is None else pd.Timestamp(at)
        cache = self.__dict__.setdefault("_owners_cache", {})
        if at not in cache:
            out = defaultdict(list)
            for r in self._interest_rows:
                s, e = r["start_date"], r["end_date"]
                if (pd.isna(s) or s <= at) and (pd.isna(e) or e > at):
                    out[r["subject_sid"]].append(r)
            cache[at] = out
        return cache[at]

    def ownership_paths(self, sid: str, at=None, max_depth: int = 12) -> list[list[dict]]:
        """All upward chains from an entity to a person, a chain stop, or a dead end.

        Each path is a list of interest rows, bottom-up. Cycles are cut and
        marked. A path ending on an entity with no recorded owner is a gap
        (missing data), not evidence of concealment.
        """
        owners = self.owners_at(at)
        paths = []

        def walk(node, path, seen):
            ups = owners.get(node, [])
            if not ups or len(path) >= max_depth:
                paths.append(path)
                return
            for u in ups:
                nxt = u["party_sid"]
                if nxt is None or u["party_kind"] != "entity":
                    paths.append(path + [u])
                elif nxt in seen:
                    paths.append(path + [dict(u, cycle=True)])
                else:
                    walk(nxt, path + [u], seen | {nxt})

        walk(sid, [], {sid})
        return [p for p in paths if p]

    def ultimate_persons(self, sid: str, at=None) -> dict:
        """person statement -> shortest path length from entity `sid`."""
        out = {}
        for p in self.ownership_paths(sid, at):
            last = p[-1]
            if last["party_kind"] == "person":
                out[last["party_sid"]] = min(out.get(last["party_sid"], 99), len(p))
        return out

    def ownership_dates(self, sid: str) -> list[pd.Timestamp]:
        """Dates at which the upward structure of `sid` may change (interest starts and ends)."""
        seen, frontier, dates = {sid}, [sid], set()
        while frontier:
            node = frontier.pop()
            for r in self._interests_by_subject.get(node, []):
                for d in (r["start_date"], r["end_date"]):
                    if pd.notna(d):
                        dates.add(pd.Timestamp(d))
                nxt = r["party_sid"]
                if nxt and r["party_kind"] == "entity" and nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        return sorted(dates)

    # ------------------------------------------------------------------
    # Activity
    # ------------------------------------------------------------------
    @cached_property
    def monthly_spend(self) -> pd.DataFrame:
        """record x month total spend."""
        p = self.b.tp_payments_monthly
        return p.groupby(["record_id", "month"], as_index=False)["amount_usd"].sum()

    @cached_property
    def links(self) -> pd.DataFrame:
        li = self.b.tp_institution_links.merge(
            self.b.healthcare_institutions.rename(columns={"country": "inst_country", "name": "inst_name"}),
            on="institution_id", how="left")
        return li
