"""Typology detectors over the canonical graph (dossier §2, §6).

Each detector reads a Graph built as of t and returns candidate groups ("review subgraphs"): member record
ids, a score, the main statistic with its peer comparison, corroborating and weakening evidence, and a
plain-language reason. Detectors never read labels.

Score convention (team choice, varied later): start from the peer percentile of the detector's main
statistic, add `corroboration_step` per independent corroborating relation (capped), subtract
`benign_penalty` when a documented benign explanation applies, and clip to [0, 1]. Scores are ranks for
review, not probabilities (§9.2).

Parameter sources, as agreed: [data] business rule in the extract; [dossier] method or example from the
dossier; [team] our choice, to be varied in sensitivity runs. Case why_flagged texts are never used.
"""
from __future__ import annotations

import itertools, json, sys
from pathlib import Path
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "layer2"))  # shared with Layer 2
import graph_build as gb
import peer_groups as pg

CONFIG = dict(
    corroboration_step=0.10,          # [team] score added per independent corroborating relation
    max_corroborations=3,             # [team]
    benign_penalty=0.30,              # [team] subtracted when a documented benign explanation applies
    min_peer_size=pg.MIN_PEER_SIZE,   # [team]
    # T1
    address_min_vendors=2,            # [team] candidate generation; dossier example A uses >=3 as a baseline rule
    benign_address_types=("registered_agent", "coworking", "corporate_hq", "hospital_campus"),  # [dossier §2.2]
    onboarding_window_days=90,        # [team] "synchronized onboarding" corroboration window
    # T2
    sponsor_window_days=90,           # [team] rolling window for new-vendor bursts
    # T3/T4
    coupling_min_engagements=5,       # [team] sponsors with fewer financially approved engagements are not scored
    # T5
    cluster_min_relation_types=3,     # [team] distinct independent relation types a pair must share
    cluster_max_hub_degree=30,        # [team] relation nodes above this degree are hubs, not rare links
    cluster_min_size=3,               # [team]
    cluster_split_size=12,            # [team] larger components are chained; split them with Louvain into dense cores
    # T6
    rapid_window_days=30,             # [team]
    rapid_min_vendors=3,              # [team]
    # T7
    bridge_recent_days=365,           # [team] "abrupt appearance": onboarded within this many days before t
    louvain_resolution=1.0, louvain_seed=0,  # [team] vary resolution and seed for stability (§8.2)
    # T8
    geo_window_months=12,             # [team]
    # T9
    spike_ratio=3.0,                  # [team] month amount vs the vendor's trailing median
    spike_sync_months=1,              # [team] spikes within this many months count as synchronized
    spike_min_vendors=3,              # [team]
    # T14
    ownership_change_window_days=90,  # [dossier example D] structural delta window after an ownership change
    # open set
    open_set_quantile=0.97,           # [team] entities above this anomaly percentile become open-set candidates
    open_set_weight=0.8,              # [team] open-set scores are discounted against typology evidence
    open_set_seed=0,                  # [team] vary for stability
)


# ----------------------------------------------------------------------------- shared context
def _key(node_id):
    return node_id.split(":", 1)[1]


def _props(frame):
    return frame.properties.map(json.loads)


class Context:
    """Tables the detectors share, derived once from a Graph."""

    def __init__(self, graph, data_dir=gb.DEFAULT_DATA_DIR, config=None):
        self.g, self.t = graph, graph.t
        self.cfg = {**CONFIG, **(config or {})}
        self.vendors = pg.vendor_frame(graph)
        self.employees = pg.employee_frame(graph)
        self.markets = pg.market_frame(graph)
        self.engagements = pg.engagement_frame(graph)
        self.events = pg.business_events(data_dir, self.t)
        self.re = self.vendors.resolved_entity  # record -> resolved entity
        e = graph.edges
        eng_vendor = e[e.relationship_type == "CONTRACTED_UNDER"]
        self.engagements["vendor"] = eng_vendor.set_index(eng_vendor.target_node_id.map(_key)).source_node_id.map(_key)
        sp = e[e.relationship_type == "SPONSORED_BY"]
        self.engagements["sponsor"] = sp.set_index(sp.source_node_id.map(_key)).target_node_id.map(_key)
        ap = e[e.relationship_type == "APPROVED_BY"].copy()
        p = _props(ap)
        self.approvals = pd.DataFrame(dict(engagement=ap.source_node_id.map(_key).to_numpy(), approver=ap.target_node_id.map(_key).to_numpy(),
                                           date=ap.valid_from.to_numpy(), approval_type=p.map(lambda x: x.get("approval_type")).to_numpy(),
                                           level=p.map(lambda x: x.get("approver_level_at_approval")).to_numpy(),
                                           required=p.map(lambda x: x.get("required_approver_level")).to_numpy(),
                                           role_active=p.map(lambda x: x.get("role_active_at_approval")).to_numpy()))
        for c in ["level", "required"]:  # stored as JSON properties; may arrive as strings
            self.approvals[c] = pd.to_numeric(self.approvals[c], errors="coerce")
        self.approvals["vendor"] = self.approvals.engagement.map(self.engagements.vendor)
        self.approvals["sponsor"] = self.approvals.engagement.map(self.engagements.sponsor)
        loc = e[e.relationship_type == "LOCATED_AT"]
        lp = _props(loc)
        self.addresses = pd.DataFrame(dict(vendor=loc.source_node_id.map(_key).to_numpy(), address=loc.target_node_id.map(_key).to_numpy(),
                                           valid_from=loc.valid_from.to_numpy(), valid_to=loc.valid_to.to_numpy(),
                                           address_type=lp.map(lambda x: x.get("address_type")).to_numpy()))
        uo = e[e.relationship_type == "ULTIMATE_OWNER"]
        self.owners = pd.DataFrame(dict(vendor=uo.source_node_id.map(_key).to_numpy(), person=uo.target_node_id.to_numpy(),
                                        confidence=uo.resolution_confidence.to_numpy(), share=uo.weight_value.to_numpy()))
        self.contacts = e[e.relationship_type.isin(["HAS_PHONE", "HAS_EMAIL_DOMAIN"])].assign(
            vendor=lambda d: d.source_node_id.map(_key))[["vendor", "target_node_id", "relationship_type"]]
        dg = e[e.relationship_type == "DECLARED_MEMBER_OF"]
        self.declared = dg.set_index(dg.source_node_id.map(_key)).target_node_id
        self.payments = graph.payments.copy()
        self.paths = graph.ownership_paths
        self.tokens = {rid: gb.name_tokens(n) for rid, n in self.vendors.legal_name.items()}
        self.matcher = gb.NameMatcher(list(self.tokens.values()))
        self._relations = None

    # -- helpers
    def score(self, percentile, n_corroborations=0, benign=False):
        c = self.cfg
        s = (0 if pd.isna(percentile) else percentile) + c["corroboration_step"] * min(n_corroborations, c["max_corroborations"])
        return float(np.clip(s - (c["benign_penalty"] if benign else 0), 0, 1))

    def relations(self):
        """Per independent relation type: relation node -> set of resolved entities (history view, at t)."""
        if self._relations is not None:
            return self._relations
        re = self.re
        rel = defaultdict(lambda: defaultdict(set))
        for r in self.addresses.itertuples():
            rel["address"][r.address].add(re[r.vendor])
        for r in self.engagements.dropna(subset=["sponsor", "vendor"]).itertuples():
            if not self.employees.is_system_default.get(r.sponsor, False):
                rel["sponsor"][r.sponsor].add(re[r.vendor])
        fin = self.approvals[self.approvals.approval_type == "financial"].dropna(subset=["vendor"])
        for r in fin.itertuples():
            rel["approver"][r.approver].add(re[r.vendor])
        for r in self.owners.itertuples():
            rel["owner"][r.person].add(re[r.vendor])
        for r in self.contacts.itertuples():  # phone and domain come from one source table: one relation type
            rel["contact"][r.target_node_id].add(re[r.vendor])
        for r in self.vendors.dropna(subset=["category_parent", "country"]).itertuples():
            rel["category"][(r.category_parent, r.country)].add(re[r.Index])
        self._relations = {k: dict(v) for k, v in rel.items()}
        return self._relations

    def shared_relations(self, members):
        """Relation types (and nodes) shared by at least two of the given records' resolved entities."""
        ents = {self.re[m] for m in members}
        out = {}
        for kind, groups in self.relations().items():
            nodes = [n for n, s in groups.items() if len(s & ents) >= 2]
            if nodes:
                out[kind] = nodes
        return out

    def onboarding_synchronized(self, members):
        d = self.vendors.loc[list(members), "onboarding_date"].dropna().sort_values()
        return len(d) >= 2 and (d.diff().dt.days.dropna() <= self.cfg["onboarding_window_days"]).any()

    def declared_together(self, members):
        groups = self.declared.reindex(list(members)).dropna()
        return len(groups) >= 2 and groups.value_counts().iloc[0] >= 2

    def event_names(self, date, market, phenomenon=None, category_parent=None):
        return pg.events_covering(self.events, date, market, phenomenon=phenomenon, category_parent=category_parent)

    def resolved_members(self, members):
        """Expand to all records of the members' resolved entities (duplicates belong in the same case)."""
        ents = {self.re[m] for m in members}
        return sorted(self.vendors.index[self.vendors.resolved_entity.isin(ents)])


def finding(detector, typology, members, score, main_stat, peer=None, corroboration=(), benign=(), reason="", evidence=None):
    peer = peer or {}
    return dict(detector=detector, typology=typology, members="|".join(sorted(set(members))), n_members=len(set(members)),
                score=round(float(score), 4), main_stat=main_stat.get("name"), main_value=main_stat.get("value"),
                peer_level=peer.get("peer_level"), peer_n=peer.get("peer_n"), peer_median=peer.get("peer_median"),
                percentile=peer.get("percentile"), robust_z=peer.get("robust_z"),
                corroboration=";".join(corroboration), benign=";".join(benign), reason=reason,
                evidence=json.dumps(evidence or {}, default=str, sort_keys=True))


def _peer_dict(row):
    return {k: (None if pd.isna(row[k]) else row[k]) for k in ["peer_level", "peer_n", "peer_median", "percentile", "robust_z"]}


# ----------------------------------------------------------------------------- T1 shared address (§2.2)
def shared_address(ctx):
    cfg, out = ctx.cfg, []
    a = ctx.addresses.assign(entity=lambda d: d.vendor.map(ctx.re))
    attrs = ctx.g.attributes["address"]
    stats = []
    for addr, grp in a.groupby("address"):
        if grp.entity.nunique() < cfg["address_min_vendors"]:
            continue
        # maximum number of distinct resolved entities valid at the same time (sweep over interval starts)
        ends = grp.valid_to.fillna(ctx.t)
        concurrent = max(grp[(grp.valid_from <= s) & (ends >= s)].entity.nunique() for s in grp.valid_from)
        types = set(grp.address_type.dropna())
        stats.append(dict(address=addr, n_entities=grp.entity.nunique(), concurrent=concurrent,
                          address_type=";".join(sorted(types)) or "unclassified", members=sorted(set(grp.vendor)),
                          country=attrs.at[f"address:{addr}", "country"]))
    if not stats:
        return out
    s = pd.DataFrame(stats)
    s["subregion"] = s.country.map(ctx.markets.subregion)
    peers = pg.PeerScorer(pg.ADDRESS_LEVELS, cfg["min_peer_size"]).score(s.assign(address_type=s.address_type), "concurrent")
    for (i, r), (_, p) in zip(s.iterrows(), peers.iterrows()):
        if r.concurrent < cfg["address_min_vendors"]:
            continue  # shared only historically, never at the same time
        shared = ctx.shared_relations(r.members)
        corr = [k for k in ["owner", "sponsor", "contact", "approver"] if k in shared]
        if ctx.onboarding_synchronized(r.members):
            corr.append("onboarding_window")
        if "residential" in r.address_type:
            corr.append("residential_address")
        benign = [t for t in cfg["benign_address_types"] if t in r.address_type]
        if ctx.declared_together(r.members):
            benign.append("declared_group")
        out.append(finding("T1", "SHARED_ADDRESS", ctx.resolved_members(r.members), ctx.score(p.percentile, len(corr), bool(benign)),
                           dict(name="concurrent_entities_at_address", value=r.concurrent), _peer_dict(p), corr, benign,
                           f"{r.concurrent} resolved third parties used one {r.address_type} address in {r.country} at the same time"
                           + (f"; also share {', '.join(corr)}" if corr else ""),
                           dict(address=r.address, n_entities_ever=r.n_entities)))
    return out


# ----------------------------------------------------------------------------- T2 sponsor concentration (§2.3)
def sponsor_concentration(ctx):
    cfg, out = ctx.cfg, []
    eng = ctx.engagements.dropna(subset=["sponsor", "vendor"]).copy()
    eng = eng[~eng.sponsor.map(ctx.employees.is_system_default).fillna(False).astype(bool)]
    eng["entity"] = eng.vendor.map(ctx.re)
    first = eng.sort_values("start_date").drop_duplicates(["sponsor", "entity"])  # when each sponsor first took on each vendor
    w = pd.Timedelta(days=cfg["sponsor_window_days"])
    rows = []
    for sponsor, grp in first.groupby("sponsor"):
        dates = grp.start_date.sort_values()
        counts = [(d, ((dates >= d) & (dates < d + w)).sum()) for d in dates]
        start, burst = max(counts, key=lambda x: (x[1], x[0]))
        in_burst = grp[(grp.start_date >= start) & (grp.start_date < start + w)]
        cats = in_burst.category_parent.value_counts(normalize=True)
        rows.append(dict(sponsor=sponsor, burst=burst, window_start=start, members=sorted(in_burst.vendor),
                         portfolio=grp.entity.nunique(), category_hhi=float((cats ** 2).sum()),
                         main_category=cats.index[0] if len(cats) else None,
                         market=in_burst.market.mode().iloc[0] if len(in_burst) else None))
    s = pd.DataFrame(rows).set_index("sponsor").join(ctx.employees)
    peers = pg.PeerScorer(pg.EMPLOYEE_LEVELS, cfg["min_peer_size"]).score(s, "burst")
    for sponsor, r in s.iterrows():
        p = peers.loc[sponsor]
        if r.burst < 2:
            continue
        shared = ctx.shared_relations(r.members)
        corr = [k for k in ["owner", "address", "contact"] if k in shared]
        if r.category_hhi >= 0.99 and r.burst >= 3:
            corr.append("single_category")
        benign = []
        if r.scope_level in ("global", "region"):
            benign.append(f"{r.scope_level}_scope_role")
        if r.role in ("category_manager", "procurement_ss"):
            benign.append("central_role")
        events = ctx.event_names(r.window_start, r.market, "onboarding", r.main_category)
        if events:
            benign.append("business_event:" + "|".join(events))
        out.append(finding("T2", "SPONSOR_CONCENTRATION", ctx.resolved_members(r.members), ctx.score(p.percentile, len(corr), bool(benign)),
                           dict(name=f"new_vendors_in_{cfg['sponsor_window_days']}d", value=r.burst), _peer_dict(p), corr, benign,
                           f"{r.role} ({r.scope_level}, {r.home_market}) took on {r.burst} new third parties within "
                           f"{cfg['sponsor_window_days']} days from {r.window_start.date()}; peer median {p.peer_median} ({p.peer_level})",
                           dict(sponsor=sponsor, portfolio=r.portfolio, category_hhi=r.category_hhi, window_start=r.window_start)))
    return out


# ----------------------------------------------------------------------------- T3 approver concentration, T4 coupling (§2.4)
def _covers(scope, market, markets):
    if pd.isna(scope):
        return False
    parts = str(scope).split(";")
    if "*" in parts or market in parts:
        return True
    region, subregion = markets.region.get(market), markets.subregion.get(market)
    return f"R:{region}" in parts or f"S:{subregion}" in parts


def _eligible_approvers(ctx):
    """For each financial approval: how many employees could have approved it (level and market scope at t)."""
    emp = ctx.employees[ctx.employees.approver_level.fillna(0) > 0]
    fin = ctx.approvals[ctx.approvals.approval_type == "financial"].dropna(subset=["vendor"]).copy()
    fin["market"] = fin.engagement.map(ctx.engagements.market)
    cache = {}
    def n_eligible(market, required):
        key = (market, required)
        if key not in cache:
            need = 0 if pd.isna(required) else required
            ok = emp[(emp.approver_level >= need) & emp.scope_markets.map(lambda s: _covers(s, market, ctx.markets))]
            cache[key] = max(len(ok), 1)
        return cache[key]
    fin["n_eligible"] = [n_eligible(m, r) for m, r in zip(fin.market, fin.required)]
    fin["exceeds_authority"] = fin.level < fin.required
    fin["out_of_scope"] = [not _covers(ctx.employees.scope_markets.get(a), m, ctx.markets) for a, m in zip(fin.approver, fin.market)]
    return fin


def approver_concentration(ctx):
    cfg, out = ctx.cfg, []
    fin = _eligible_approvers(ctx)
    fin["category_parent"] = fin.engagement.map(ctx.engagements.category_parent)
    pools = fin.groupby(["market", "category_parent"])
    rows = []
    for (market, cat), grp in pools:
        if len(grp) < cfg["coupling_min_engagements"]:
            continue
        share = grp.approver.value_counts(normalize=True)
        top = share.index[0]
        expected = 1 / grp.n_eligible.median()
        rows.append(dict(market=market, category=cat, approver=top, share=share.iloc[0], lift=share.iloc[0] / expected,
                         n=len(grp), n_eligible=grp.n_eligible.median(),
                         members=sorted(set(grp[grp.approver == top].vendor)),
                         exceed=int(grp[grp.approver == top].exceeds_authority.sum()),
                         out_of_scope=int(grp[grp.approver == top].out_of_scope.sum())))
    if not rows:
        return out
    s = pd.DataFrame(rows)
    s["subregion"] = s.market.map(ctx.markets.subregion)
    s["region"] = s.market.map(ctx.markets.region)
    peers = pg.PeerScorer([("category", "subregion"), ("category", "region"), ("category",), ()], cfg["min_peer_size"]).score(s, "lift")
    for (i, r), (_, p) in zip(s.iterrows(), peers.iterrows()):
        corr = []
        if r.exceed:
            corr.append("approval_exceeds_authority")
        if r.out_of_scope:
            corr.append("approver_out_of_scope")
        shared = ctx.shared_relations(r.members)
        corr += [k for k in ["owner", "address", "contact"] if k in shared]
        benign = ["single_eligible_approver"] if r.n_eligible <= 1 else []
        out.append(finding("T3", "APPROVER_CONCENTRATION", ctx.resolved_members(r.members), ctx.score(p.percentile, len(corr), bool(benign)),
                           dict(name="approver_share_lift_vs_eligible", value=round(r.lift, 3)), _peer_dict(p), corr, benign,
                           f"one approver handled {r.share:.0%} of {r.n} financial approvals for {r.category} in {r.market} "
                           f"while ~{r.n_eligible:.0f} approvers were eligible"
                           + (f"; {r.exceed} approvals above the approver's authority" if r.exceed else ""),
                           dict(approver=r.approver, market=r.market, category=r.category, exceed=r.exceed, out_of_scope=r.out_of_scope)))
    return out


def sponsor_approver_coupling(ctx):
    cfg, out = ctx.cfg, []
    fin = _eligible_approvers(ctx).dropna(subset=["sponsor"])
    fin = fin[~fin.sponsor.map(ctx.employees.is_system_default).fillna(False).astype(bool)]
    rows = []
    for sponsor, grp in fin.groupby("sponsor"):
        if grp.engagement.nunique() < cfg["coupling_min_engagements"]:
            continue
        per = grp.groupby("approver").engagement.nunique().sort_values(ascending=False)
        approver, n_pair = per.index[0], per.iloc[0]
        share = n_pair / grp.engagement.nunique()
        expected = 1 / grp.n_eligible.median()
        pair = grp[grp.approver == approver]
        rows.append(dict(sponsor=sponsor, approver=approver, n_pair=n_pair, share=share, lift=share / expected,
                         n_eligible=grp.n_eligible.median(), members=sorted(set(pair.vendor)),
                         exceed=int(pair.exceeds_authority.sum()), out_of_scope=int(pair.out_of_scope.sum())))
    if not rows:
        return out
    s = pd.DataFrame(rows).set_index("sponsor").join(ctx.employees)
    peers = pg.PeerScorer(pg.EMPLOYEE_LEVELS, cfg["min_peer_size"]).score(s, "lift")
    for sponsor, r in s.iterrows():
        p = peers.loc[sponsor]
        corr = []
        if r.exceed:
            corr.append("approval_exceeds_authority")
        if r.out_of_scope:
            corr.append("approver_out_of_scope")
        shared = ctx.shared_relations(r.members)
        corr += [k for k in ["owner", "address", "contact"] if k in shared]
        benign = ["single_eligible_approver"] if r.n_eligible <= 1 else []
        out.append(finding("T4", "SPONSOR_APPROVER_COUPLING", ctx.resolved_members(r.members), ctx.score(p.percentile, len(corr), bool(benign)),
                           dict(name="pair_share_lift_vs_eligible", value=round(r.lift, 3)), _peer_dict(p), corr, benign,
                           f"sponsor ({r.role}, {r.home_market}) and one approver share {r.n_pair} engagements "
                           f"({r.share:.0%} of the sponsor's) while ~{r.n_eligible:.0f} approvers were eligible"
                           + (f"; {r.exceed} approvals above the approver's authority" if r.exceed else ""),
                           dict(sponsor=sponsor, approver=r.approver, exceed=r.exceed, out_of_scope=r.out_of_scope)))
    return out


# ----------------------------------------------------------------------------- T5 dense multi-relation cluster (§2.5)
def _entity_records(ctx, entities):
    return sorted(ctx.vendors.index[ctx.vendors.resolved_entity.isin(set(entities))])


def dense_cluster(ctx):
    """Pairs sharing >= k independent relation types (rare relation nodes only); components of such pairs."""
    cfg, out = ctx.cfg, []
    kinds = defaultdict(set)
    weight = Counter()
    for kind, groups in ctx.relations().items():
        for node, ents in groups.items():
            if not 2 <= len(ents) <= cfg["cluster_max_hub_degree"]:
                continue
            for a, b in itertools.combinations(sorted(ents), 2):
                kinds[(a, b)].add(kind)
                weight[(a, b)] += 1 / np.log(len(ents) + 1)  # Adamic–Adar: rarer shared nodes weigh more
    onboard = ctx.vendors.groupby("resolved_entity").onboarding_date.min()
    window = pd.Timedelta(days=cfg["onboarding_window_days"])
    for (a, b) in list(kinds):
        if abs(onboard[a] - onboard[b]) <= window:
            kinds[(a, b)].add("onboarding_window")
    strong = [(a, b) for (a, b), k in kinds.items() if len(k) >= cfg["cluster_min_relation_types"]]
    G = nx.Graph()
    G.add_weighted_edges_from((a, b, len(kinds[(a, b)])) for a, b in strong)
    clusters = []
    for comp in nx.connected_components(G):
        if len(comp) <= cfg["cluster_split_size"]:
            clusters.append(comp)
        else:  # single-linkage chains several groups together; keep the dense cores
            clusters += nx.community.louvain_communities(G.subgraph(comp), weight="weight",
                                                         resolution=cfg["louvain_resolution"], seed=cfg["louvain_seed"])
    rows = []
    for comp in clusters:
        if len(comp) < cfg["cluster_min_size"]:
            continue
        pairs = [tuple(sorted(p)) for p in itertools.combinations(comp, 2)]
        present = [p for p in pairs if p in kinds]
        kind_counts = Counter(k for p in present for k in kinds[p])
        probs = np.array(list(kind_counts.values()), float) / sum(kind_counts.values())
        rows.append(dict(entities=sorted(comp), size=len(comp), mean_types=np.mean([len(kinds[p]) for p in present]),
                         density=sum(weight[p] for p in present) / len(pairs), entropy=float(-(probs * np.log(probs)).sum()),
                         kinds=dict(kind_counts)))
    if not rows:
        return out
    s = pd.DataFrame(rows)
    s["strength"] = s.mean_types * s.density
    peers = pg.PeerScorer([()], 1).score(s, "strength")  # few clusters: compare with all candidate clusters
    for (i, r), (_, p) in zip(s.iterrows(), peers.iterrows()):
        members = _entity_records(ctx, r.entities)
        corr = [k for k in ["owner", "address", "contact"] if k in r.kinds]
        benign = ["declared_group"] if ctx.declared_together(members) else []
        out.append(finding("T5", "DENSE_MULTI_RELATION_CLUSTER", members, ctx.score(p.percentile, len(corr), bool(benign)),
                           dict(name="mean_shared_relation_types", value=round(r.mean_types, 2)), _peer_dict(p), corr, benign,
                           f"{r.size} third parties overlap on {len(r.kinds)} relation types "
                           f"({', '.join(sorted(r.kinds))}); on average {r.mean_types:.1f} types per linked pair",
                           dict(kinds=r.kinds, rarity_weighted_density=round(r.density, 3), typed_edge_entropy=round(r.entropy, 3))))
    return out


# ----------------------------------------------------------------------------- T6 rapid onboarding of linked vendors (§2.6)
def rapid_onboarding(ctx):
    cfg, out = ctx.cfg, []
    w = pd.Timedelta(days=cfg["rapid_window_days"])
    first = ctx.vendors.groupby("resolved_entity").agg(onboarding=("onboarding_date", "min"), country=("country", "first"),
                                                       category=("category_parent", "first"))
    candidates = {}
    for kind in ["sponsor", "owner", "address", "contact"]:  # strong links only; category/approver are too common
        for node, ents in ctx.relations().get(kind, {}).items():
            if len(ents) < cfg["rapid_min_vendors"] or len(ents) > cfg["cluster_max_hub_degree"] * 3:
                continue
            d = first.loc[list(ents), "onboarding"].sort_values()
            for start in d:
                inside = d[(d >= start) & (d < start + w)]
                if len(inside) >= cfg["rapid_min_vendors"]:
                    key = frozenset(inside.index)
                    if key not in candidates or len(candidates[key]["kinds"]) < 1:
                        candidates[key] = dict(start=start, kinds={kind}, node=node)
                    else:
                        candidates[key]["kinds"].add(kind)
    # keep maximal groups only
    keys = sorted(candidates, key=len, reverse=True)
    kept = [k for i, k in enumerate(keys) if not any(k < other for other in keys[:i])]
    if not kept:
        return out
    rows = []
    for k in kept:
        c = candidates[k]
        sub = first.loc[list(k)]
        country = sub.country.mode().iloc[0]
        market_all = first[(first.country == country) & (first.onboarding >= c["start"]) & (first.onboarding < c["start"] + w)]
        prior = first[(first.country == country) & (first.onboarding < c["start"]) & (first.onboarding >= c["start"] - pd.Timedelta(days=730))]
        expected = len(prior) / 730 * cfg["rapid_window_days"]
        rows.append(dict(entities=sorted(k), n=len(k), start=c["start"], kinds=c["kinds"], country=country,
                         category=sub.category.mode().iloc[0] if sub.category.notna().any() else None,
                         market_onboardings=len(market_all), expected_market=expected,
                         excess=len(k) - expected))
    s = pd.DataFrame(rows)
    peers = pg.PeerScorer([()], 1).score(s, "excess")
    for (i, r), (_, p) in zip(s.iterrows(), peers.iterrows()):
        members = _entity_records(ctx, r.entities)
        shared = ctx.shared_relations(members)
        corr = [k for k in ["owner", "address", "contact", "sponsor"] if k in shared and k not in r.kinds]
        if r.expected_market < r.n:
            corr.append("quiet_period")
        events = ctx.event_names(r.start, r.country, "onboarding", r.category)
        benign = ["business_event:" + "|".join(events)] if events else []
        if ctx.declared_together(members):
            benign.append("declared_group")
        out.append(finding("T6", "RAPID_ONBOARDING_LINKED", members, ctx.score(p.percentile, len(corr), bool(benign)),
                           dict(name=f"linked_onboardings_in_{cfg['rapid_window_days']}d", value=r.n), _peer_dict(p), corr, benign,
                           f"{r.n} third parties linked by {', '.join(sorted(r.kinds))} were onboarded in {r.country} within "
                           f"{cfg['rapid_window_days']} days from {r.start.date()} (market expectation ~{r.expected_market:.1f})",
                           dict(start=r.start, market_onboardings=r.market_onboardings, expected_market=round(r.expected_market, 2))))
    return out


# ----------------------------------------------------------------------------- T7 bridge entity (§2.7, §3.8)
def bridge_entity(ctx):
    cfg, out = ctx.cfg, []
    links = ctx.g.edges[ctx.g.edges.relationship_type.isin(["SERVES", "SITE_OF", "AFFILIATED_WITH"])]
    links = links.assign(vendor=links.source_node_id.map(_key), inst=links.target_node_id)
    inst = ctx.g.attributes["institution"]
    by_vendor = links.groupby("vendor").inst.agg(set)
    candidates = by_vendor[by_vendor.map(len) >= 2]

    def communities(exclude):
        G = nx.Graph()
        G.add_nodes_from(inst.index)
        for v, insts in by_vendor.items():
            if v == exclude:
                continue
            for a, b in itertools.combinations(sorted(insts), 2):
                G.add_edge(a, b, weight=G.get_edge_data(a, b, {"weight": 0})["weight"] + 1)
        comms = nx.community.louvain_communities(G, weight="weight", resolution=cfg["louvain_resolution"], seed=cfg["louvain_seed"])
        return {n: i for i, c in enumerate(comms) for n in c}

    rows = []
    for v, insts in candidates.items():
        comm = communities(v)  # communities without the candidate's own links, so it cannot merge them
        counts = Counter(comm[i] for i in insts)
        participation = 1 - sum((c / len(insts)) ** 2 for c in counts.values())
        rows.append(dict(vendor=v, n_links=len(insts), n_communities=len(counts), participation=participation,
                         countries=len({inst.at[i, "country"] for i in insts}),
                         foreign=sum(inst.at[i, "country"] != ctx.vendors.at[v, "country"] for i in insts),
                         public=sum(str(inst.at[i, "public_sector"]).lower() == "true" for i in insts), comm=comm))
    if not rows:
        return out
    s = pd.DataFrame(rows).set_index("vendor").join(ctx.vendors[["party_type", "category_parent", "country", "subregion", "region", "onboarding_date"]])
    peers = pg.PeerScorer(pg.VENDOR_LEVELS, ctx.cfg["min_peer_size"]).score(s, "participation")
    sponsors = ctx.engagements.dropna(subset=["sponsor"]).groupby("vendor").sponsor.agg(set)
    for v, r in s.iterrows():
        p = peers.loc[v]
        if r.n_communities < 2:
            continue
        corr = []
        if (ctx.t - r.onboarding_date).days <= cfg["bridge_recent_days"]:
            corr.append("recently_onboarded")
        if r.foreign:
            corr.append("cross_border_links")
        if r.public:
            corr.append("public_institutions")
        for sp in sponsors.get(v, set()):  # dossier example C: the sponsor's other vendors serve one community only
            others = [o for o in ctx.engagements[ctx.engagements.sponsor == sp].vendor.unique() if o != v and o in by_vendor]
            if others and len({r.comm[i] for o in others for i in by_vendor[o]}) == 1:
                corr.append("sponsor_otherwise_single_community")
                break
        benign = ["long_standing"] if (ctx.t - r.onboarding_date).days > 5 * 365 else []
        out.append(finding("T7", "BRIDGE_ENTITY", ctx.resolved_members([v]), ctx.score(p.percentile, len(corr), bool(benign)),
                           dict(name="participation_coefficient", value=round(r.participation, 3)), _peer_dict(p), corr, benign,
                           f"third party links {r.n_links} institutions across {r.n_communities} institution communities "
                           f"and {r.countries} countries ({r.foreign} outside its own country)",
                           dict(n_links=r.n_links, n_communities=r.n_communities, public_links=r.public)))
    return out


# ----------------------------------------------------------------------------- T8 geography-linked concentration (§2.8)
def geography_concentration(ctx):
    cfg, out = ctx.cfg, []
    eng = ctx.engagements.dropna(subset=["vendor"])
    intermediaries = set(eng[eng.compliance_flags.fillna("").str.contains("int")].vendor)
    v = ctx.vendors[ctx.vendors.index.isin(intermediaries)]
    first = v.groupby("resolved_entity").agg(onboarding=("onboarding_date", "min"), country=("country", "first"))
    months = cfg["geo_window_months"]
    rows = []
    for country, grp in first.groupby("country"):
        ends = pd.date_range(grp.onboarding.min() + pd.DateOffset(months=months), ctx.t, freq="MS")
        best = None
        for end in ends:
            start = end - pd.DateOffset(months=months)
            count = ((grp.onboarding >= start) & (grp.onboarding < end)).sum()
            prior = ((grp.onboarding >= start - pd.DateOffset(months=2 * months)) & (grp.onboarding < start)).sum() / 2
            if best is None or count - prior > best[0]:
                best = (count - prior, start, end, count, prior)
        if best is None or best[3] < 2:
            continue
        excess, start, end, count, prior = best
        ents = grp[(grp.onboarding >= start) & (grp.onboarding < end)].index
        rows.append(dict(market=country, excess=excess, count=count, prior=prior, start=start, end=end, entities=list(ents)))
    if not rows:
        return out
    s = pd.DataFrame(rows).set_index("market").join(ctx.markets[["subregion", "region", "cpi_tier"]])
    peers = pg.PeerScorer(pg.MARKET_LEVELS, cfg["min_peer_size"]).score(s, "excess")
    for market, r in s.iterrows():
        p = peers.loc[market]
        members = _entity_records(ctx, r.entities)
        corr = []
        sp = ctx.engagements[ctx.engagements.vendor.isin(members)].dropna(subset=["sponsor"]).groupby("sponsor").vendor.nunique()
        if len(sp) and sp.max() / len(members) >= 0.5:
            corr.append("one_sponsor_majority")
        if pd.notna(r.cpi_tier) and r.cpi_tier >= 3:
            corr.append("cpi_tier_3plus_dated")
        shared = ctx.shared_relations(members)
        corr += [k for k in ["owner", "address", "contact"] if k in shared]
        events = ctx.event_names(r.start + (r.end - r.start) / 2, market, "onboarding")
        benign = ["business_event:" + "|".join(events)] if events else []
        out.append(finding("T8", "GEOGRAPHY_LINKED_CONCENTRATION", members, ctx.score(p.percentile, len(corr), bool(benign)),
                           dict(name=f"new_intermediaries_{months}m_excess", value=round(r.excess, 2)), _peer_dict(p), corr, benign,
                           f"{r['count']} new intermediaries in {market} between {r.start.date()} and {r.end.date()} "
                           f"against ~{r.prior:.1f} per {months} months before",
                           dict(window_start=r.start, window_end=r.end, prior_rate=r.prior,
                                cpi_tier=None if pd.isna(r.cpi_tier) else r.cpi_tier)))
    return out


# ----------------------------------------------------------------------------- T9 activity / timing concentration (§2.6)
def activity_timing(ctx):
    cfg, out = ctx.cfg, []
    pay = ctx.payments.groupby(["record_id", "month"]).amount_usd.sum().reset_index()
    spikes = []
    for rid, grp in pay.groupby("record_id"):
        series = grp.set_index("month").amount_usd.sort_index()
        trailing = series.shift(1).rolling(6, min_periods=3).median()
        hit = series[(trailing > 0) & (series >= cfg["spike_ratio"] * trailing)]
        for month, amount in hit.items():
            spikes.append(dict(record_id=rid, month=month, ratio=amount / trailing[month]))
    if not spikes:
        return out
    sp = pd.DataFrame(spikes)
    sp["entity"] = sp.record_id.map(ctx.re)
    sp["category"] = sp.record_id.map(ctx.vendors.category_parent)
    active = pay.assign(category=pay.record_id.map(ctx.vendors.category_parent)).groupby(["category", "month"]).record_id.nunique()
    rows, seen = [], set()
    for kind in ["sponsor", "owner", "address", "contact"]:
        for node, ents in ctx.relations().get(kind, {}).items():
            sub = sp[sp.entity.isin(ents)]
            if sub.entity.nunique() < cfg["spike_min_vendors"]:
                continue
            for month in sorted(sub.month.unique()):
                near = sub[(sub.month - month).abs() <= pd.Timedelta(days=31 * cfg["spike_sync_months"])]
                if near.entity.nunique() >= cfg["spike_min_vendors"]:
                    key = frozenset(near.entity)
                    if key in seen:
                        continue
                    seen.add(key)
                    cat = near.category.mode().iloc[0] if near.category.notna().any() else None
                    wide = sp[(sp.category == cat) & ((sp.month - month).abs() <= pd.Timedelta(days=31))].record_id.nunique()
                    rows.append(dict(entities=sorted(key), n=len(key), month=month, kind=kind, category=cat,
                                     median_ratio=near.ratio.median(),
                                     category_spike_share=wide / max(active.get((cat, month), 1), 1)))
    if not rows:
        return out
    s = pd.DataFrame(rows)
    peers = pg.PeerScorer([()], 1).score(s, "n")
    for (i, r), (_, p) in zip(s.iterrows(), peers.iterrows()):
        members = _entity_records(ctx, r.entities)
        shared = ctx.shared_relations(members)
        corr = [k for k in ["owner", "address", "contact", "sponsor"] if k in shared and k != r.kind]
        if r.median_ratio >= 2 * cfg["spike_ratio"]:
            corr.append("large_spikes")
        benign = []
        if r.category_spike_share >= 0.2:  # [team] much of the category spiked at once: seasonal or programme-wide
            benign.append("category_wide_spike")
        events = ctx.event_names(r.month, ctx.vendors.at[members[0], "country"], "activity", r.category)
        if events:
            benign.append("business_event:" + "|".join(events))
        out.append(finding("T9", "ACTIVITY_TIMING_CONCENTRATION", members, ctx.score(p.percentile, len(corr), bool(benign)),
                           dict(name="synchronized_spiking_vendors", value=r.n), _peer_dict(p), corr, benign,
                           f"{r.n} third parties sharing a {r.kind} had spend spikes (median {r.median_ratio:.1f}x their trailing "
                           f"median) within {cfg['spike_sync_months']} month(s) of {r.month.date()}",
                           dict(month=r.month, linked_by=r.kind, category_spike_share=round(r.category_spike_share, 3))))
    return out


# ----------------------------------------------------------------------------- T10 common beneficial ownership (§2.9)
def _path_entities(path):
    return [n for n in path.split(" > ") if n.startswith("legal_entity:")]


def common_beneficial_owner(ctx):
    cfg, out = ctx.cfg, []
    if ctx.paths is None or ctx.paths.empty:
        return out
    ent = ctx.g.attributes["legal_entity"]
    paths = ctx.paths.assign(entity=lambda d: d.record_id.map(ctx.re))
    groups = []
    for person, grp in paths[paths.terminal_reason == "natural_person"].groupby("end"):
        if grp.entity.nunique() >= 2:
            groups.append(("person", person, grp))
    # convergence on a controlling legal entity above the vendors' own registered entities
    rows = [(e, r.record_id, r.entity, r.match_confidence, r.path) for r in paths.itertuples() for e in _path_entities(r.path)[1:]]
    via = pd.DataFrame(rows, columns=["node", "record_id", "entity", "match_confidence", "path"])
    for node, grp in via.groupby("node"):
        if grp.entity.nunique() >= 2:
            groups.append(("entity", node, grp))
    if not groups:
        return out
    stats = pd.DataFrame([dict(kind=k, node=n, n=g.entity.nunique()) for k, n, g in groups])
    peers = pg.PeerScorer([("kind",), ()], cfg["min_peer_size"]).score(stats, "n")
    for (kind, node, grp), (_, p) in zip(groups, peers.iterrows()):
        members = ctx.resolved_members(grp.record_id)
        corr, benign = [], []
        if not ctx.declared_together(members):
            corr.append("undeclared")
        else:
            benign.append("declared_group")
        sims = [ctx.matcher.similarity(ctx.tokens[a], ctx.tokens[b]) for a, b in itertools.combinations(sorted(set(grp.record_id)), 2)]
        if sims and max(sims) < 0.5:  # [team]
            corr.append("dissimilar_names")
        subtypes = {ent.at[e, "subtype"] for pth in grp.path for e in _path_entities(pth) if e in ent.index}
        if "fund" in subtypes:
            benign.append("fund_portfolio")
        if kind == "entity" and ent.get("public_listing", pd.Series(dtype=object)).get(node):
            benign.append("listed_parent")
        low_conf = grp.match_confidence.min() < 1
        out.append(finding("T10", "COMMON_BENEFICIAL_OWNER", members, ctx.score(p.percentile, len(corr), bool(benign)),
                           dict(name=f"vendors_converging_on_{kind}", value=grp.entity.nunique()), _peer_dict(p), corr, benign,
                           f"{grp.entity.nunique()} nominally separate third parties converge on one "
                           f"{'natural person' if kind == 'person' else 'controlling entity'} through current ownership paths"
                           + (" (one or more name-matched links: check matching)" if low_conf else ""),
                           dict(owner=node, min_match_confidence=float(grp.match_confidence.min()),
                                paths=sorted(set(grp.path))[:10])))
    return out


# ----------------------------------------------------------------------------- T11 layered / complex ownership (§2.10)
def layered_ownership(ctx):
    cfg, out = ctx.cfg, []
    if ctx.paths is None or ctx.paths.empty:
        return out
    rows = []
    for rid, grp in ctx.paths.groupby("record_id"):
        observed = grp[~grp.terminal_reason.isin(["no_owner_recorded"])]
        transitions = max((sum(a != b for a, b in zip(j.split(">"), j.split(">")[1:])) for j in grp.jurisdictions if j), default=0)
        rows.append(dict(record_id=rid, depth=int(observed.depth.max()) if len(observed) else 0, branches=len(grp),
                         n_jurisdictions=int(grp.n_jurisdictions.max()), transitions=transitions,
                         trust_or_fund=bool(grp.trust_or_nominee.any()), cycle=bool((grp.terminal_reason == "cycle").any()),
                         missing=sorted(set(grp.terminal_reason) - {"natural_person"}),
                         listed=bool(grp.terminal_reason.str.contains("exempt_listed").any())))
    s = pd.DataFrame(rows).set_index("record_id")
    s["complexity"] = s.depth + s.transitions + (s.branches - 1)  # [team] observed complexity only; gaps reported apart
    s = s.join(ctx.vendors[["party_type", "category_parent", "country", "subregion", "region"]])
    peers = pg.PeerScorer(pg.VENDOR_LEVELS, cfg["min_peer_size"]).score(s, "complexity")
    for rid, r in s.iterrows():
        p = peers.loc[rid]
        if r.depth < 2:
            continue
        corr = [c for c, ok in [("trust_or_fund_on_path", r.trust_or_fund), ("ownership_cycle", r.cycle),
                                ("multi_jurisdiction", r.n_jurisdictions >= 3)] if ok]
        benign = ["listed_parent"] if r.listed else []
        out.append(finding("T11", "LAYERED_COMPLEX_OWNERSHIP", ctx.resolved_members([rid]), ctx.score(p.percentile, len(corr), bool(benign)),
                           dict(name="observed_ownership_complexity", value=int(r.complexity)), _peer_dict(p), corr, benign,
                           f"ownership traced through {r.depth} layers, {r.n_jurisdictions} jurisdictions ({r.transitions} changes) "
                           f"and {r.branches} branch(es)" + (f"; gaps reported separately: {', '.join(r.missing)}" if r.missing else ""),
                           dict(depth=r.depth, branches=r.branches, transitions=r.transitions, missing=r.missing)))
    return out


# ----------------------------------------------------------------------------- T12 ownership plus operational overlap (§2.11)
def ownership_operational_overlap(ctx):
    cfg, out = ctx.cfg, []
    owned = ctx.owners.assign(entity=lambda d: d.vendor.map(ctx.re))
    links = set()
    for person, grp in owned.groupby("person"):
        for a, b in itertools.combinations(sorted(set(grp.entity)), 2):
            links.add((a, b, "common_owner"))
    same_person = ctx.g.edges[(ctx.g.edges.relationship_type == "SAME_AS") & (ctx.g.edges.target_node_type == "person")]
    for r in same_person.itertuples():  # an individual vendor who owns another vendor
        expert = ctx.re[_key(r.source_node_id)]
        for owned_entity in set(owned[owned.person == r.target_node_id].entity) - {expert}:
            a, b = sorted([expert, owned_entity])
            links.add((a, b, "expert_owns_vendor"))
    operational = ["sponsor", "address", "contact"]
    rel = ctx.relations()
    G = nx.Graph()
    overlap = {}
    for a, b, kind in links:
        shared = [k for k in operational if any({a, b} <= ents for ents in rel.get(k, {}).values())]
        cat = rel.get("category", {})
        if any({a, b} <= ents for ents in cat.values()):
            shared.append("category")
        if [k for k in shared if k != "category"]:
            G.add_edge(a, b)
            overlap[(a, b)] = (kind, shared)
    rows = []
    for comp in nx.connected_components(G):
        pairs = [overlap[p] for p in itertools.combinations(sorted(comp), 2) if p in overlap]
        kinds = sorted({k for _, sh in pairs for k in sh})
        rows.append(dict(entities=sorted(comp), n_types=len(kinds), kinds=kinds, link=sorted({k for k, _ in pairs})))
    if not rows:
        return out
    s = pd.DataFrame(rows)
    peers = pg.PeerScorer([()], 1).score(s, "n_types")
    for (i, r), (_, p) in zip(s.iterrows(), peers.iterrows()):
        members = _entity_records(ctx, r.entities)
        benign = ["declared_group"] if ctx.declared_together(members) else []
        corr = (["undeclared"] if not benign else []) + (["expert_owns_vendor"] if "expert_owns_vendor" in r.link else [])
        out.append(finding("T12", "OWNERSHIP_OPERATIONAL_OVERLAP", members, ctx.score(p.percentile, len(corr), bool(benign)),
                           dict(name="operational_overlap_types", value=r.n_types), _peer_dict(p), corr, benign,
                           f"{len(members)} third parties linked by {', '.join(r.link).replace('_', ' ')} also share "
                           f"{', '.join(r.kinds)}",
                           dict(ownership_link=r.link, operational=r.kinds)))
    return out


# ----------------------------------------------------------------------------- T13 routing through opaque jurisdictions (§2.12)
def opaque_routing(ctx):
    cfg, out = ctx.cfg, []
    if ctx.paths is None or ctx.paths.empty:
        return out
    paths = ctx.paths
    rows = []
    for rid, grp in paths.groupby("record_id"):
        flags = dict(low_bo_access=bool(grp.bo_access_low_on_path.any()), not_in_markets=bool(grp.jurisdiction_not_in_markets.any()),
                     sanctioned=bool(grp.sanctioned_on_path.any()))
        own_country = ctx.vendors.at[rid, "country"]
        foreign = any(j and j != own_country for js in grp.jurisdictions for j in js.split(">")[1:])
        rows.append(dict(record_id=rid, n_context=sum(flags.values()), foreign_holding=foreign, pep=bool(grp.end_is_pep.any()),
                         trust=bool(grp.trust_or_nominee.any()), gaps=sorted(set(grp.terminal_reason) - {"natural_person"}),
                         jurisdictions=sorted({j for js in grp.jurisdictions for j in js.split(">") if j}), **flags))
    s = pd.DataFrame(rows).set_index("record_id")
    peers = pg.PeerScorer([()], 1).score(s, "n_context")
    dd = ctx.vendors.q_pep_connected.astype(str).str.lower()
    for rid, r in s.iterrows():
        p = peers.loc[rid]
        if r.n_context == 0 or not r.foreign_holding:
            continue
        corr = []
        if r.pep:
            corr.append("pep_owner")
            if dd.get(rid) == "false":
                corr.append("pep_not_disclosed_in_due_diligence")
        if r.trust:
            corr.append("trust_or_fund_on_path")
        context = [k for k in ["low_bo_access", "not_in_markets", "sanctioned"] if r[k]]
        out.append(finding("T13", "OPAQUE_JURISDICTION_ROUTING", ctx.resolved_members([rid]), ctx.score(p.percentile, len(corr), False),
                           dict(name="context_flags_on_path", value=int(r.n_context)), _peer_dict(p), corr, [],
                           f"ownership path includes a jurisdiction with {', '.join(context).replace('_', ' ')} according to "
                           f"country_risk_context as of {ctx.t.date()} ({' > '.join(r.jurisdictions)})"
                           + (f"; path gaps: {', '.join(r.gaps)}" if r.gaps else ""),
                           dict(jurisdictions=r.jurisdictions, context=context, gaps=r.gaps)))
    return out


# ----------------------------------------------------------------------------- T14 ownership change then structural shift (§2.13)
def ownership_change(ctx):
    cfg, out = ctx.cfg, []
    if ctx.paths is None or ctx.paths.empty:
        return out
    e = ctx.g.edges
    own = e[e.relationship_type.isin(["OWNS", "CONTROLS"])]
    w = pd.Timedelta(days=cfg["ownership_change_window_days"])
    monthly = ctx.payments.groupby(["record_id", "month"]).amount_usd.sum()
    services = e[e.relationship_type == "PROVIDES_SERVICE"].assign(vendor=lambda d: d.source_node_id.map(_key))
    owner_vendors = ctx.owners.groupby("person").vendor.agg(set)
    rows = []
    for rid, grp in ctx.paths.groupby("record_id"):
        onboard = ctx.vendors.at[rid, "onboarding_date"]
        entities = {n for pth in grp.path for n in _path_entities(pth)}
        changes = own[own.target_node_id.isin(entities) & own.valid_from.gt(onboard)]
        for c in changes.itertuples():
            t0 = c.valid_from
            if t0 > ctx.t:
                continue
            after = lambda d: (d > t0) & (d <= t0 + w)
            shifts = []
            if after(ctx.addresses[ctx.addresses.vendor == rid].valid_from).any():
                shifts.append("new_address")
            eng = ctx.engagements[ctx.engagements.vendor == rid]
            before_sponsors = set(eng[eng.start_date <= t0].sponsor.dropna())
            if (set(eng[after(eng.start_date)].sponsor.dropna()) - before_sponsors):
                shifts.append("new_sponsor")
            sv = services[services.vendor == rid]
            if after(sv.valid_from).any() and len(sv[sv.valid_from <= t0]):
                shifts.append("service_expansion")
            related = set()
            for person in [c.source_node_id] if c.source_node_id.startswith("person:") else []:
                related = {v for v in owner_vendors.get(person, set()) if v != rid and after(pd.Series([ctx.vendors.at[v, "onboarding_date"]])).iloc[0]}
            if related:
                shifts.append("new_owner_linked_vendors")
            if rid in monthly.index.get_level_values(0):
                m = monthly.loc[rid]
                pre, post = m[(m.index > t0 - w) & (m.index <= t0)].mean(), m[after(m.index.to_series()).to_numpy()].mean()
                if pd.notna(pre) and pd.notna(post) and pre > 0 and post >= 2 * pre:  # [team]
                    shifts.append("spend_jump")
            if shifts:
                rows.append(dict(record_id=rid, t0=t0, owner=c.source_node_id, lag_days=(c.observed_at - t0).days if pd.notna(c.observed_at) else None,
                                 shifts=shifts, n_shifts=len(shifts), related=sorted(related), complete=t0 + w <= ctx.t))
    if not rows:
        return out
    s = pd.DataFrame(rows).sort_values("n_shifts", ascending=False).drop_duplicates("record_id")
    peers = pg.PeerScorer([()], 1).score(s, "n_shifts")
    for (i, r), (_, p) in zip(s.iterrows(), peers.iterrows()):
        events = ctx.event_names(r.t0, ctx.vendors.at[r.record_id, "country"], "onboarding")
        benign = ["business_event:" + "|".join(events)] if events else []
        corr = ["registry_lag_over_60d"] if r.lag_days and r.lag_days > 60 else []  # [team] effective vs observed time
        out.append(finding("T14", "OWNERSHIP_CHANGE_STRUCTURAL_SHIFT", ctx.resolved_members([r.record_id] + r.related),
                           ctx.score(p.percentile, len(corr), bool(benign)),
                           dict(name=f"structural_changes_within_{cfg['ownership_change_window_days']}d", value=r.n_shifts),
                           _peer_dict(p), corr, benign,
                           f"new owner effective {r.t0.date()} (registered {r.lag_days} days later); within "
                           f"{cfg['ownership_change_window_days']} days: {', '.join(r.shifts).replace('_', ' ')}"
                           + ("" if r.complete else " (window not yet complete at t)"),
                           dict(effective=r.t0, owner=r.owner, lag_days=r.lag_days, shifts=r.shifts, related=r.related)))
    return out


# ----------------------------------------------------------------------------- record-level findings (§4: scored separately)
def record_level(ctx):
    out = []
    fin = _eligible_approvers(ctx)
    for rid, grp in fin[fin.exceeds_authority].groupby("vendor"):
        out.append(dict(rid=rid, check="APPROVAL_AUTHORITY_EXCEEDED", value=len(grp),
                        reason=f"{len(grp)} financial approval(s) by an approver below the level required for the value band"))
    eng = ctx.engagements.dropna(subset=["vendor"])
    reviewed = set(ctx.approvals[ctx.approvals.approval_type == "compliance_review"].engagement)
    need = eng[eng.compliance_flags.fillna("").map(lambda f: bool(set(f.split(";")) & {"hcp", "gov", "int"}))]
    missing = need[~need.index.isin(reviewed) & (need.start_date <= ctx.t - pd.Timedelta(days=30))]  # [team] grace period
    for rid, grp in missing.groupby("vendor"):
        out.append(dict(rid=rid, check="COMPLIANCE_REVIEW_MISSING", value=len(grp),
                        reason=f"{len(grp)} engagement(s) in a compliance-flagged category without a compliance review"))
    term = ctx.vendors.terminated_date.dropna()
    pay = ctx.payments[ctx.payments.record_id.isin(term.index)]
    late = pay[pay.month > pay.record_id.map(term) + pd.offsets.MonthEnd(0)]
    for rid, grp in late.groupby("record_id"):
        out.append(dict(rid=rid, check="PAYMENT_AFTER_TERMINATION", value=grp.month.nunique(),
                        reason=f"payments in {grp.month.nunique()} month(s) after termination on {term[rid].date()}"))
    out += _fmv_annual_cap(ctx)
    rows = []
    if not out:
        return rows
    f = pd.DataFrame(out)
    for check, grp in f.groupby("check"):
        peers = pg.PeerScorer([()], 1).score(grp, "value")
        for (i, r), (_, p) in zip(grp.iterrows(), peers.iterrows()):
            rows.append(finding("R", check, ctx.resolved_members([r.rid]), ctx.score(p.percentile), dict(name=check.lower(), value=r.value),
                                _peer_dict(p), [], [], r.reason))
    return rows


def _fmv_annual_cap(ctx, data_dir=gb.DEFAULT_DATA_DIR):
    cards = pd.read_csv(f"{data_dir}/fmv_rate_cards.csv", parse_dates=["effective_from", "effective_to"])
    experts = ctx.vendors[ctx.vendors.party_type == "individual"]
    band = lambda c: "US" if c == "US" else ("EU_JP" if ctx.markets.region.get(c) in ("EUROPE",) or c == "JP" else "EMERGING")
    pay = ctx.payments[ctx.payments.record_id.isin(experts.index)].assign(year=lambda d: d.month.dt.year)
    out = []
    for (rid, year), grp in pay.groupby(["record_id", "year"]):
        if pd.Timestamp(f"{year}-12-31") > ctx.t:
            continue  # an annual cap can only be checked on a closed year
        role, country = experts.at[rid, "expert_type"], experts.at[rid, "country"]
        mid = pd.Timestamp(f"{year}-07-01")
        card = cards[(cards.expert_role == role) & (cards.country_band == band(country)) & (cards.effective_from <= mid)
                     & (cards.effective_to.isna() | (cards.effective_to >= mid))]
        if card.empty:
            continue
        cap = card.annual_cap_usd.max()
        total = grp.amount_usd.sum()
        if total > cap:
            out.append(dict(rid=rid, check="FMV_ANNUAL_CAP_EXCEEDED", value=round(total / cap, 3),
                            reason=f"paid {total:,.0f} USD in {year}, {total / cap:.1f}x the {card.card_version.iloc[0]} annual cap "
                                   f"for {role} ({band(country)})"))
    return out


# ----------------------------------------------------------------------------- open set (§8.3)
def structural_features(ctx):
    """One row per resolved entity: relational, ownership, approval, timing and activity structure (no labels)."""
    cfg = ctx.cfg
    rel = ctx.relations()
    ents = sorted(set(ctx.re))
    feats = pd.DataFrame(index=ents)
    pair_kinds = defaultdict(set)
    for kind, groups in rel.items():
        co = Counter()
        rare = Counter()
        for node, members in groups.items():
            for e in members:
                co[e] += len(members) - 1
                if 2 <= len(members) <= cfg["cluster_max_hub_degree"]:
                    rare[e] += len(members) - 1
            if 2 <= len(members) <= cfg["cluster_max_hub_degree"] and kind != "category":
                for a, b in itertools.combinations(sorted(members), 2):
                    pair_kinds[a].add((b, kind)); pair_kinds[b].add((a, kind))
        feats[f"log_co_{kind}"] = np.log1p(pd.Series(co).reindex(ents).fillna(0))
        feats[f"log_rare_{kind}"] = np.log1p(pd.Series(rare).reindex(ents).fillna(0))
    per_partner = {e: Counter(b for b, _ in v) for e, v in pair_kinds.items()}
    feats["max_kinds_with_one_partner"] = [max(per_partner.get(e, Counter()).values(), default=0) for e in ents]
    feats["n_shared_kinds"] = [len({k for _, k in pair_kinds.get(e, set())}) for e in ents]
    v = ctx.vendors
    by_ent = v.groupby("resolved_entity")
    onboard = by_ent.onboarding_date.min()
    country = by_ent.country.first()
    same_window = [((onboard[country == country[e]] - onboard[e]).abs() <= pd.Timedelta(days=cfg["rapid_window_days"])).sum() - 1 for e in ents]
    feats["log_country_onboardings_30d"] = np.log1p(same_window)
    fin = _eligible_approvers(ctx).assign(entity=lambda d: d.vendor.map(ctx.re))
    feats["n_exceeds_authority"] = fin.groupby("entity").exceeds_authority.sum().reindex(ents).fillna(0)
    feats["n_out_of_scope"] = fin.groupby("entity").out_of_scope.sum().reindex(ents).fillna(0)
    feats["top_approver_share"] = fin.groupby("entity").approver.agg(lambda s: s.value_counts(normalize=True).iloc[0]).reindex(ents).fillna(0)
    own = ctx.vendors.filter(like="own_").assign(entity=ctx.re)
    for c in ["own_max_depth", "own_n_jurisdictions", "own_n_paths"]:
        feats[c] = own.groupby("entity")[c].max().reindex(ents).fillna(0) if c in own else 0
    for c in ["own_pep_owner", "own_bo_access_low_on_path", "own_trust_or_nominee", "own_jurisdiction_not_in_markets"]:
        feats[c] = (own[c] == True).groupby(own.entity).any().reindex(ents).fillna(False).astype(float) if c in own else 0
    pay = ctx.payments.groupby(["record_id", "month"]).amount_usd.sum()
    burst = pay.groupby(level=0).agg(lambda s: (s.std(ddof=0) - s.mean()) / (s.std(ddof=0) + s.mean()) if s.mean() > 0 else 0)
    feats["burstiness"] = burst.groupby(ctx.re).max().reindex(ents).fillna(0)
    term = v.terminated_date.dropna()
    late = ctx.payments[ctx.payments.record_id.isin(term.index)]
    late = late[late.month > late.record_id.map(term) + pd.offsets.MonthEnd(0)]
    feats["months_paid_after_termination"] = late.groupby(late.record_id.map(ctx.re)).month.nunique().reindex(ents).fillna(0)
    return feats.astype(float)


def open_set(ctx):
    """Isolation Forest on structural features, ranks only (no contamination tuning, §8.2), then grouping of the
    most anomalous entities through rare shared relations. Catches structure no typology detector encodes."""
    from sklearn.ensemble import IsolationForest
    cfg, out = ctx.cfg, []
    feats = structural_features(ctx)
    ranks = feats.rank(pct=True)  # scale-free; heavy tails cannot dominate
    model = IsolationForest(n_estimators=500, random_state=cfg["open_set_seed"]).fit(ranks)
    anomaly = pd.Series(-model.score_samples(ranks), index=feats.index).rank(pct=True)
    top = set(anomaly[anomaly >= cfg["open_set_quantile"]].index)
    G = nx.Graph()
    G.add_nodes_from(top)
    for kind in ["address", "owner", "contact", "sponsor"]:
        for node, members in ctx.relations().get(kind, {}).items():
            inside = sorted(members & top)
            if 2 <= len(members) <= cfg["cluster_max_hub_degree"] and len(inside) >= 2:
                G.add_edges_from(itertools.combinations(inside, 2))
    median = feats.median()
    for comp in nx.connected_components(G):
        comp = sorted(comp)
        members = _entity_records(ctx, comp)
        lead = max(comp, key=lambda e: anomaly[e])
        dev = (ranks.loc[lead] - 0.5).abs().sort_values(ascending=False).head(3).index
        drivers = [f"{c}={feats.at[lead, c]:.2f} (median {median[c]:.2f})" for c in dev]
        out.append(finding("O", "STRUCTURAL_ANOMALY", members, cfg["open_set_weight"] * float(anomaly[comp].max()),
                           dict(name="isolation_forest_percentile", value=round(float(anomaly[comp].max()), 4)),
                           dict(peer_level="all", peer_n=len(feats), percentile=float(anomaly[comp].max())), [], [],
                           f"structurally unusual relative to all third parties; most different on {', '.join(drivers)}",
                           dict(drivers=list(dev), n_entities=len(comp))))
    return out


DETECTORS = {"T1": shared_address, "T2": sponsor_concentration, "T3": approver_concentration, "T4": sponsor_approver_coupling,
             "T5": dense_cluster, "T6": rapid_onboarding, "T7": bridge_entity, "T8": geography_concentration, "T9": activity_timing,
             "T10": common_beneficial_owner, "T11": layered_ownership, "T12": ownership_operational_overlap,
             "T13": opaque_routing, "T14": ownership_change, "R": record_level, "O": open_set}


def run_detectors(graph, data_dir=gb.DEFAULT_DATA_DIR, config=None, only=None):
    """All detectors on one graph; returns one row per candidate group, best first within each detector."""
    ctx = Context(graph, data_dir, config)
    rows = []
    for name, fn in DETECTORS.items():
        if only is None or name in only:
            rows += fn(ctx)
    findings = pd.DataFrame(rows)
    if findings.empty:
        return findings
    findings = findings.sort_values(["detector", "score", "robust_z"], ascending=[True, False, False], na_position="last")
    findings.insert(0, "finding_id", [f"{d}-{i:04d}" for d, i in zip(findings.detector, findings.groupby("detector").cumcount() + 1)])
    findings["t"] = str(graph.t.date())
    return findings.reset_index(drop=True)


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Run the typology detectors on the graph as of t")
    parser.add_argument("--t", default=str(gb.SNAPSHOT.date()))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    findings = run_detectors(gb.build_graph(t=args.t))
    out = Path(args.out or gb.ROOT / "results" / "detectors" / args.t)
    out.mkdir(parents=True, exist_ok=True)
    findings.to_csv(out / "findings.csv", index=False)
    (out / "config.json").write_text(json.dumps(CONFIG, indent=2, default=str))
    print(findings.groupby("detector").agg(findings=("score", "size"), median_members=("n_members", "median")).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
