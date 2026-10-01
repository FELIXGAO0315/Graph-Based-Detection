"""The five ownership typologies, built on registry R paths.

Ownership structures change over time (on the starter data one vendor was
layered through a trust until 2019 and then acquired through a BVI holdco),
so every vendor is evaluated at each date its upward structure changed while
it was our vendor, not only at t.

Rules followed throughout:
  - a missing path is unknown, never evidence of concealment;
  - registry L accounting parents are not beneficial owners and are not used here;
  - evidence records both valid time and publication date (statementDate).
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd

from ..context import Context
from ..resolution import name_key, person_key
from .base import Candidate

# Jurisdiction lists are a documented modelling choice, not an index value from the
# bundle (FSI and FATF are not loaded in the bundle). "Sink" follows the classic
# offshore financial centres (IMF 2000 OFC list; Garcia-Bernardo et al. 2017,
# "Uncovering Offshore Financial Centers", sink OFCs); "conduit" follows the
# conduit OFCs of the same paper. Conduits carry less weight.
OFFSHORE_SINK = frozenset({
    "JE", "GG", "IM", "BM", "KY", "VG", "BS", "BZ", "SC", "MU", "LI", "MC", "AD", "GI", "TC", "AI", "VC",
    "KN", "AG", "LC", "WS", "VU", "MH", "CW", "BB", "PA",
})
OFFSHORE_CONDUIT = frozenset({"NL", "LU", "IE", "CY", "MT", "CH", "SG", "HK", "AE"})


def _jurisdiction_note(j: str) -> str:
    if j in OFFSHORE_SINK:
        return "sink OFC"
    if j in OFFSHORE_CONDUIT:
        return "conduit OFC"
    return ""


class OwnershipView:
    """Per-vendor ownership snapshots over the vendor's active life."""

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.nodes = ctx.reg_nodes
        self.dd = ctx.b.tp_due_diligence.set_index("record_id")

    def node(self, sid):
        return self.nodes.loc[sid] if sid in self.nodes.index else None

    def eval_dates(self, rec: str, sid: str) -> list[pd.Timestamp]:
        v = self.ctx.vendors.loc[rec]
        lo = v["onboarding_date"]
        hi = v["terminated_date"] if pd.notna(v["terminated_date"]) else self.ctx.t
        dates = {lo, hi - pd.Timedelta(days=1) if hi > lo else lo}
        dates |= {d for d in self.ctx.ownership_dates(sid) if lo <= d < hi}
        return sorted(dates)

    def metrics(self, rec: str, sid: str, at) -> dict:
        paths = self.ctx.ownership_paths(sid, at)
        home = self.node(sid)["jurisdiction"]
        inter, persons, jur = set(), {}, {home}
        trust = nominee = cycle = False
        gaps = 0
        for p in paths:
            for x in p:
                if x.get("details") == "trust":
                    trust = True
                if x.get("details") == "nominee":
                    nominee = True
                if x.get("cycle"):
                    cycle = True
                ps = x["party_sid"]
                if ps is None:
                    gaps += 1
                    continue
                n = self.node(ps)
                if n is None:
                    continue
                if n["kind"] == "entity":
                    inter.add(ps)
                    jur.add(n["jurisdiction"])
                    if n["entity_subtype"] == "trust":
                        trust = True
                else:
                    persons[ps] = min(persons.get(ps, 99), len(p))
            last = p[-1]
            if last["party_kind"] == "entity" and not self.ctx.owners_at(at).get(last["party_sid"]):
                gaps += 1
        sink = {s for s in inter if self.node(s)["jurisdiction"] in OFFSHORE_SINK
                and self.node(s)["jurisdiction"] != home}
        conduit = {s for s in inter if self.node(s)["jurisdiction"] in OFFSHORE_CONDUIT
                   and self.node(s)["jurisdiction"] != home}
        peps = {p for p in persons if self.node(p)["pep_status"] == "isPep"}
        layers = max((len(p) for p in paths), default=0)
        return {"at": pd.Timestamp(at), "paths": paths, "layers": layers, "n_paths": len(paths),
                "jurisdictions": jur, "trust": trust, "nominee": nominee, "cycle": cycle, "gaps": gaps,
                "intermediates": inter, "persons": persons, "peps": peps, "sink": sink, "conduit": conduit}

    @lru_cache(maxsize=None)
    def snapshots(self, rec: str) -> tuple:
        sid = self.ctx.reg_entity_of.get(rec)
        if sid is None:
            return ()
        return tuple(self.metrics(rec, sid, d) for d in self.eval_dates(rec, sid))

    def path_text(self, rec: str, path: list[dict]) -> str:
        sid = self.ctx.reg_entity_of[rec]
        parts = [f"{self.node(sid)['name']} [{self.node(sid)['jurisdiction']}]"]
        for x in path:
            ps = x["party_sid"]
            n = self.node(ps) if ps else None
            share = f"{x['share']:g}%" if pd.notna(x.get("share")) else (x.get("details") or "?")
            if n is None:
                parts.append(f"<-{share}- (chain stops: {x.get('unspecified_reason')})")
            else:
                tag = n["entity_subtype"] if n["kind"] == "entity" else ("PEP" if n["pep_status"] == "isPep" else "person")
                parts.append(f"<-{share} from {pd.Timestamp(x['start_date']).date() if pd.notna(x['start_date']) else '?'}"
                             f" (published {pd.Timestamp(x['statement_date']).date()})- {n['name']} [{n['jurisdiction']},{tag}]")
        return " ".join(parts)


def _view(ctx: Context) -> OwnershipView:
    if "_ownership_view" not in ctx.__dict__:
        ctx.__dict__["_ownership_view"] = OwnershipView(ctx)
    return ctx.__dict__["_ownership_view"]


def _org_records(ctx: Context):
    return [r for r in ctx.reg_entity_of]


# ---------------------------------------------------------------------------
# LAYERED_COMPLEX_OWNERSHIP
# ---------------------------------------------------------------------------
def complexity(m: dict) -> float:
    return ((m["layers"] - 1) + (len(m["jurisdictions"]) - 1) + 1.5 * m["trust"] + 1.5 * m["nominee"]
            + 0.5 * (m["n_paths"] > 1) + 0.5 * bool(m["sink"]))


def detect_layered(ctx: Context) -> list[Candidate]:
    min_layers = ctx.param("layered", "min_layers", 4)
    ov = _view(ctx)
    out = []
    for rec in _org_records(ctx):
        snaps = ov.snapshots(rec)
        hits = [m for m in snaps if m["layers"] >= min_layers
                or (m["layers"] >= 3 and len(m["jurisdictions"]) >= 3)]
        if not hits:
            continue
        best = max(hits, key=complexity)
        later = [m for m in snaps if m["at"] > hits[-1]["at"]]
        ended = later[0]["at"] if later else None  # the structure changed at the next snapshot
        benign = []
        listed = [s for s in best["intermediates"] if ov.node(s)["listed"]]
        funds = [s for s in best["intermediates"] if ov.node(s)["entity_subtype"] == "fund"]
        if listed:
            benign.append("listed entity in the chain")
        if funds:
            benign.append("fund / private-equity structure")
        if best["gaps"]:
            benign.append(f"{best['gaps']} chain gap(s): structure partly unknown")
        longest = max(best["paths"], key=len)
        out.append(Candidate(
            members=ctx.expand([rec]), tags=("LAYERED_COMPLEX_OWNERSHIP",), score=complexity(best),
            emerged_at=hits[0]["at"], active_until=ended, detector="layered_ownership",
            reason=(f"owned through {best['layers']} layers across {len(best['jurisdictions'])} jurisdictions "
                    f"({'>'.join(sorted(best['jurisdictions']))}), {best['n_paths']} branch(es)"
                    + (", trust at the top" if best["trust"] else "") + (", nominee" if best["nominee"] else "")
                    + f"; structure as of {best['at'].date()}"),
            evidence={"path": ov.path_text(rec, longest), "as_of": best["at"], "layers": best["layers"],
                      "jurisdictions": sorted(best["jurisdictions"]),
                      "offshore": {s: _jurisdiction_note(ov.node(s)["jurisdiction"])
                                   for s in best["sink"] | best["conduit"]}},
            benign=benign, uses_ownership=True,
            features={"layers": best["layers"], "n_jur": len(best["jurisdictions"]), "trust": best["trust"]},
        ))
    return out


# ---------------------------------------------------------------------------
# COMMON_BENEFICIAL_OWNER and OWNERSHIP_OPERATIONAL_OVERLAP
# ---------------------------------------------------------------------------
def person_holdings(ctx: Context) -> dict:
    """person -> {entity: (earliest date the path existed while a vendor, record, path length)}."""
    ov = _view(ctx)
    out: dict = defaultdict(dict)
    for rec in _org_records(ctx):
        ent = ctx.entity_of[rec]
        for m in ov.snapshots(rec):
            for p, dist in m["persons"].items():
                prev = out[p].get(ent)
                if prev is None or m["at"] < prev[0]:
                    out[p][ent] = (m["at"], rec, dist)
    return out


def _name_similarity(names: list[str]) -> float:
    toks = [set(name_key(n).split()) for n in names]
    sims = []
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            u = toks[i] | toks[j]
            sims.append(len(toks[i] & toks[j]) / len(u) if u else 0.0)
    return float(np.mean(sims)) if sims else 0.0


def _operational_links(ctx: Context, records: set) -> dict:
    """Shared addresses, sponsors and contacts among member entities.

    Returns kind -> list of (key, entities, formed) where `formed` is the date
    the second entity joined the shared item (valid time), so the overlap's
    emergence can be dated (both overlaps must hold).
    """
    recs = set(records)
    ent = ctx.entity_of
    rel = defaultdict(list)

    def add(kind, frame, key_col, date_col):
        for key, g in frame.groupby(key_col):
            first = g.assign(e=g["record_id"].map(ent)).groupby("e")[date_col].min().sort_values()
            if len(first) >= 2:
                rel[kind].append((key, frozenset(first.index), first.iloc[1]))

    a = ctx.addresses[ctx.addresses["record_id"].isin(recs)]
    add("address", a, "addr_key", "valid_from")
    e = ctx.engagements[ctx.engagements["record_id"].isin(recs)].dropna(subset=["sponsor_employee_id"])
    add("sponsor", e, "sponsor_employee_id", "start_date")
    c = ctx.contacts[ctx.contacts["record_id"].isin(recs)].merge(
        ctx.vendors[["record_id", "onboarding_date"]].reset_index(drop=True), on="record_id")
    add("phone", c.dropna(subset=["phone_key"]), "phone_key", "onboarding_date")
    add("domain", c.dropna(subset=["domain_key"]), "domain_key", "onboarding_date")
    return rel


def _personal_vehicle(ctx: Context, expert_records: set, members) -> bool:
    """Is every organisation in the group named after the expert (a personal service company)?

    An expert's own consultancy carrying their name is a disclosed vehicle; an
    unrelated name is what makes the ownership undeclared in practice.
    """
    if not expert_records:
        return False
    toks = {t for r in expert_records for t in person_key(ctx.vendors.loc[r, "legal_name"]).split() if len(t) >= 3}
    orgs = [r for r in members if ctx.vendors.loc[r, "party_type"] == "organization"]
    return bool(orgs) and all(toks & set(name_key(ctx.vendors.loc[r, "legal_name"]).split()) for r in orgs)


_GROUP_WORDS = {"group", "holdings", "holding", "international", "global", "partners"}


def _group_explains(group: str, v: pd.DataFrame) -> bool:
    """Does a declared group plausibly cover the members? Only ~70% of group members
    declare it, so members carrying the group's brand stem also count."""
    stem = [t for t in name_key(group).split() if t not in _GROUP_WORDS][:1]
    covered = v["declared_parent_group"].eq(group) | (
        v["legal_name"].map(lambda n: bool(stem) and stem[0] in name_key(n).split()))
    return bool(covered.mean() >= 0.99)


def detect_common_owner(ctx: Context) -> list[Candidate]:
    min_entities = ctx.param("common_owner", "min_entities", 2)
    ov = _view(ctx)
    holdings = person_holdings(ctx)
    out = []
    for person, ents in holdings.items():
        # An individual expert who is the person also belongs to the group.
        own_records = {r for r, s in ctx.reg_person_of.items() if s == person}
        n_ent = len(ents) + len({ctx.entity_of[r] for r in own_records} - set(ents))
        if n_ent < min_entities:
            continue
        recs = {v[1] for v in ents.values()} | own_records
        members = ctx.expand(recs)
        v = ctx.vendors.loc[sorted(members)]
        groups = v["declared_parent_group"].dropna()
        declared_share = len(groups) / len(v)
        names = list(v.drop_duplicates("legal_name")["legal_name"])
        sim = _name_similarity(names)
        same_group = groups.nunique() == 1 and _group_explains(groups.iloc[0], v)
        pn = ov.node(person)
        dates = sorted(d[0] for d in ents.values())
        k = min(3, len(dates))
        emerged = max(dates[k - 1], *(ctx.vendors.loc[r, "onboarding_date"] for r in own_records)) if own_records \
            else dates[k - 1]
        benign = []
        if same_group:
            benign.append(f"declared group {groups.iloc[0]} (declared or brand-named members)")
        if any(ov.node(s)["entity_subtype"] == "fund" or ov.node(s)["listed"]
               for m_ in (ov.snapshots(r) for r in recs if r in ctx.reg_entity_of) for snap in m_
               for s in snap["intermediates"]):
            benign.append("fund or listed entity in the chain: possible portfolio holding")
        if sim > 0.5:
            benign.append(f"similar names (token overlap {sim:.2f}): possibly one brand")
        if _personal_vehicle(ctx, own_records, members):
            benign.append("expert's personal service company, named after the expert")
        score = 2 * np.log2(n_ent) + (1 - declared_share) + (1 - sim)
        dist = sorted(d[2] for d in ents.values())
        out.append(Candidate(
            members=members, tags=("COMMON_BENEFICIAL_OWNER",), score=float(score), emerged_at=emerged,
            detector="common_owner",
            reason=(f"{n_ent} vendor entities converge on natural person {pn['name']} ({pn['jurisdiction']}"
                    + (", PEP" if pn["pep_status"] == "isPep" else "") + f") via {min(dist)}–{max(dist)}-hop paths; "
                    f"{1 - declared_share:.0%} undeclared, name overlap {sim:.2f}"
                    + (f"; the person is also engaged as expert {sorted(own_records)}" if own_records else "")),
            evidence={"person": person, "person_name": pn["name"],
                      "paths": {rec: ov.path_text(rec, next(p for m in ov.snapshots(rec) for p in m["paths"]
                                                            if p and p[-1]["party_sid"] == person))
                                for _, rec, _ in ents.values()},
                      "expert_records": sorted(own_records)},
            benign=benign, uses_ownership=True,
            features={"n_entities": n_ent, "declared_share": declared_share, "name_sim": sim,
                      "expert_owner": bool(own_records)},
        ))

        # OWNERSHIP_OPERATIONAL_OVERLAP: the owner group also overlaps operationally.
        rel = _operational_links(ctx, set(members))
        overlap_ents = set()
        kinds = []
        formed = []
        for kind, items in rel.items():
            for key, es, when in items:
                overlap_ents |= es
                kinds.append(kind)
                formed.append(when)
        if len(overlap_ents) >= 2:
            omembers = ctx.expand({r for r in members if ctx.entity_of.get(r, r) in overlap_ents})
            n_kinds = len(set(kinds))
            # An expert who owns a vendor that the same sponsor engages is a direct conflict of interest.
            expert_conflict = bool(own_records) and "sponsor" in kinds
            # The overlap case emerges once both the common owner and an operational overlap hold.
            o_emerged = max(emerged, min(formed))
            out.append(Candidate(
                members=omembers, tags=("OWNERSHIP_OPERATIONAL_OVERLAP", "COMMON_BENEFICIAL_OWNER"),
                score=float(score + n_kinds + 2 * expert_conflict), emerged_at=o_emerged, detector="ownership_overlap",
                reason=(f"{len(overlap_ents)} entities owned by {pn['name']} also share "
                        f"{', '.join(sorted(set(kinds)))} (overlap since {min(formed).date()})"
                        + ("; the owner is an expert engaged by the same sponsor" if expert_conflict else "")),
                evidence={"person": person, "shared": {k: [(key, sorted(es), when) for key, es, when in rel[k]]
                                                       for k in rel}},
                benign=benign, uses_ownership=True,
                features={"n_entities": len(overlap_ents), "n_kinds": n_kinds},
            ))
    return out


# ---------------------------------------------------------------------------
# OPAQUE_JURISDICTION_ROUTING
# ---------------------------------------------------------------------------
def detect_opaque(ctx: Context) -> list[Candidate]:
    """One candidate per offshore route and owner set, with the period it held.

    A vendor whose route changes (sold from one offshore structure into
    another) yields one candidate per route, so the two periods can become
    separate cases.
    """
    ov = _view(ctx)
    by_route = defaultdict(list)
    for rec in _org_records(ctx):
        snaps = ov.snapshots(rec)
        spans: dict = {}
        for idx, m in enumerate(snaps):
            if not (m["sink"] or m["conduit"]):
                continue
            if not (m["sink"] or m["peps"]):
                continue  # conduit-only holding without a PEP is an ordinary structure
            dd = ov.dd.loc[rec] if rec in ov.dd.index else None
            undisclosed = bool(m["peps"]) and dd is not None and not bool(dd["q_pep_connected"])
            s = 2.0 * bool(m["sink"]) + 1.0 * bool(m["conduit"]) + 2.0 * bool(m["peps"]) + 2.0 * undisclosed
            key = (tuple(sorted(m["sink"] | m["conduit"])), tuple(sorted(m["peps"])))
            span = spans.setdefault(key, {"first": m["at"], "last_idx": idx, "best": None, "score": -1.0})
            span["last_idx"] = idx
            if s > span["score"]:
                span.update(best=dict(m, undisclosed=undisclosed), score=s)
        for key, span in spans.items():
            nxt = span["last_idx"] + 1
            until = snaps[nxt]["at"] if nxt < len(snaps) else None
            by_route[key].append((rec, span["best"], span["score"], span["first"], until))

    out = []
    for (route, peps), items in by_route.items():
        recs = [r for r, *_ in items]
        _, m, s, _, _ = max(items, key=lambda x: x[2])
        juris = sorted({ov.node(x)["jurisdiction"] for x in route})
        benign = []
        if not m["peps"]:
            benign.append("offshore holding without PEP: may be an ordinary holding location")
        if all(ov.node(x)["jurisdiction"] in OFFSHORE_CONDUIT for x in route):
            benign.append("conduit jurisdiction only (common for legitimate holdcos)")
        pep_names = [ov.node(p)["name"] for p in m["peps"]]
        untils = [x[4] for x in items]
        out.append(Candidate(
            members=ctx.expand(recs), tags=("OPAQUE_JURISDICTION_ROUTING",), score=float(s),
            emerged_at=min(x[3] for x in items), active_until=None if None in untils else max(untils),
            detector="opaque_routing",
            reason=(f"{len(recs)} vendor(s) owned via holdco(s) in {','.join(juris)} "
                    f"({', '.join(_jurisdiction_note(j) for j in juris)})"
                    + (f"; ultimate owner is a PEP ({', '.join(pep_names)})" if pep_names else "")
                    + ("; PEP connection not disclosed in due diligence" if m["undisclosed"] else "")),
            evidence={"route": list(route),
                      "paths": {r: ov.path_text(r, max(b["paths"], key=len)) for r, b, *_ in items},
                      "jurisdiction_basis": "IMF 2000 OFC list; Garcia-Bernardo et al. 2017 sink/conduit OFCs"},
            benign=benign, uses_ownership=True,
            features={"sink": bool(m["sink"]), "pep": bool(m["peps"]), "undisclosed": m["undisclosed"]},
        ))
    return out


# ---------------------------------------------------------------------------
# OWNERSHIP_CHANGE_STRUCTURAL_SHIFT
# ---------------------------------------------------------------------------
def detect_change(ctx: Context) -> list[Candidate]:
    days = ctx.param("ownership_change", "window_days", 90)
    min_signals = ctx.param("ownership_change", "min_signals", 2)
    ov = _view(ctx)
    ri = ctx.b.reg_interests
    addr = ctx.addresses
    eng = ctx.engagements
    svc = ctx.b.tp_services
    spend = ctx.monthly_spend
    owned_by = defaultdict(set)  # party -> records whose chain contains it (at any snapshot)
    for rec in _org_records(ctx):
        for m in ov.snapshots(rec):
            for s in m["intermediates"] | set(m["persons"]):
                owned_by[s].add(rec)
    out = []
    for rec, sid in ctx.reg_entity_of.items():
        v = ctx.vendors.loc[rec]
        end = v["terminated_date"] if pd.notna(v["terminated_date"]) else ctx.t
        direct = ri[(ri["subject_sid"] == sid) & ri["party_sid"].notna()]
        for x in direct.itertuples():
            d = x.start_date
            if pd.isna(d) or not (v["onboarding_date"] + pd.Timedelta(days=30) < d < end):
                continue
            w_hi = d + pd.Timedelta(days=days)
            signals = []
            a = addr[addr["record_id"] == rec]
            if ((a["valid_from"] >= d) & (a["valid_from"] <= w_hi)).any():
                signals.append("address")
            e = eng[eng["record_id"] == rec]
            prior_sp = set(e[e["start_date"] < d]["sponsor_employee_id"].dropna())
            new_sp = set(e[(e["start_date"] >= d) & (e["start_date"] <= w_hi)]["sponsor_employee_id"].dropna())
            if new_sp - prior_sp:
                signals.append("sponsor")
            sv = svc[svc["record_id"] == rec]
            prior_cat = set(sv[sv["valid_from"] < d]["category_code"])
            new_cat = set(sv[(sv["valid_from"] >= d) & (sv["valid_from"] <= w_hi)
                             & (sv["taxonomy_version"] == sv["taxonomy_version"].max())]["category_code"])
            if new_cat - prior_cat and not (d <= pd.Timestamp("2024-01-01") <= w_hi):
                signals.append("service_expansion")
            sp = spend[spend["record_id"] == rec]
            before = sp[(sp["month"] >= d - pd.Timedelta(days=180)) & (sp["month"] < d)]["amount_usd"]
            after = sp[(sp["month"] >= d) & (sp["month"] <= d + pd.Timedelta(days=180))]["amount_usd"]
            if len(before) and len(after) and after.mean() >= 2 * before.mean():
                signals.append("spend_jump")
            related = set()
            for r2 in owned_by.get(x.party_sid, set()) - {rec}:
                onb = ctx.vendors.loc[r2, "onboarding_date"]
                if d - pd.Timedelta(days=30) <= onb <= d + pd.Timedelta(days=days + 30):
                    related.add(r2)
            if related:
                signals.append("related_vendors")
            if len(signals) < min_signals:
                continue
            party = ov.node(x.party_sid)
            lag = (x.statement_date - d).days if pd.notna(x.statement_date) else None
            ve = eng[eng["record_id"] == rec]
            benign = ctx.benign_events("onboarding", d, w_hi, v["country"], cats=zip(ve["parent"], ve["cat_primary"])) \
                if related else []
            if party is not None and v["declared_parent_group"] and isinstance(v["declared_parent_group"], str) \
                    and name_key(v["declared_parent_group"]).split()[:1] == name_key(party["name"]).split()[:1]:
                benign.append("new owner matches declared group")
            out.append(Candidate(
                members=ctx.expand({rec} | related), tags=("OWNERSHIP_CHANGE_STRUCTURAL_SHIFT",),
                score=float(len(signals) + 0.5 * len(related)), emerged_at=d, active_until=w_hi,
                detector="ownership_change",
                reason=(f"acquired by {party['name'] if party is not None else x.party_sid} "
                        f"[{party['jurisdiction'] if party is not None else '?'}] on {d.date()} "
                        f"(registry lag {lag}d); within {days} days: {', '.join(signals)}"
                        + (f"; {len(related)} related vendor(s) onboarded" if related else "")),
                evidence={"interest": x.statement_id, "valid_from": d, "published": x.statement_date,
                          "signals": signals, "related": sorted(related)},
                benign=benign, uses_ownership=True,
                features={"n_signals": len(signals), "n_related": len(related)},
            ))
    return out
