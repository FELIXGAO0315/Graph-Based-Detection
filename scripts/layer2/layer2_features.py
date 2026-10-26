"""Layer 2 variables: deterministic relationship rules on the as-of-t data of graph_build, one function per typology.
Each typology function returns (raw statistics, typology signal s_k in [0, 1])."""
import json
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

import graph_build as gb
import peer_groups as pg

# parameters (sources in the table above)
BENIGN_ADDRESS_TYPES = {"registered_agent", "coworking", "hospital_campus", "virtual_office"}  # dossier 2.2, 6.2
DENSE_KINDS = 3      # dossier 3.6 worked example A: address + sponsor + owner = three shared relation kinds
MIN_GROUP = 3        # dossier 3.6 baseline: "at least three active vendors" -> a pattern needs >= 3 entities
MIN_COUNT = 5        # Cochran (1954): >= 5 observations before a frequency/concentration is read
Z_OUT = 3.5          # Iglewicz & Hoaglin (1993): modified z-score (median/MAD) outlier cut-off
HHI_HIGH = 0.25      # US DOJ/FTC Horizontal Merger Guidelines (2010): HHI > 2,500 = highly concentrated
MIN_POOL = 5         # derived: with <= 4 approvers HHI >= 1/4 holds trivially, so >= 5 are needed
W_ONBOARD = 28       # dossier App. C: "five entities onboarded within 28 days"
W_SPONSOR = 60       # dossier 3.9: "within 60 days ... two new related vendors onboarded under the same sponsor"
W_SHIFT = 90         # dossier 3.9: "an ownership change with a large 90-day structural delta"
W_SYNC = 1           # data grain: activity exists only as monthly aggregates (dossier 1) -> adjacent month
W_MARKET = 365       # one annual cycle (CPI editions and DD refresh cycles in the data are annual)
DEPTH_MIN = 3        # dossier 2.10: control "invisible at the immediate-parent level" -> >= 2 intermediate entities
NAME_SIM = gb.CANDIDATE_THRESHOLD  # team entity resolution: names below 0.70 similarity are not candidates
TUKEY_K = 3.0        # Tukey (1977) "far out" fence Q3 + 3 IQR: relation values beyond it are hubs, not links


def strip(s):
    return s.str.split(":", n=1).str[1]


def props(frame, key):
    return frame.properties.map(lambda p: json.loads(p).get(key))


def rarity(x, mask):
    """Typology signal in [0, 1]: rarity of a value at least this extreme among ALL vendors at t.
    s = -log10(share of vendors that meet the pattern condition with a value >= x) / log10(N); 0 if the condition fails.
    A pattern shared by many vendors (a hub approver touching 80 vendors) is therefore weak by construction."""
    x = pd.Series(x, dtype=float)
    mask = pd.Series(mask, index=x.index).fillna(False).astype(bool)
    n = len(x)
    out = pd.Series(0.0, index=x.index)
    if mask.sum() == 0 or n < 2:
        return out
    v = np.sort(x[mask].to_numpy())
    tail = len(v) - np.searchsorted(v, x[mask].to_numpy(), side="left")       # eligible vendors with value >= x
    out[mask] = -np.log10(tail / n) / np.log10(n)
    return out


def rarity_within(x, mask, group):
    """rarity computed inside each class of `group` (dossier 2.2: separate peer distributions, e.g. address types)."""
    x = pd.Series(x, dtype=float); mask = pd.Series(mask, index=x.index).fillna(False).astype(bool)
    group = pd.Series(group, index=x.index)
    out = pd.Series(0.0, index=x.index)
    n_all = len(x)
    for g_, idx in group.groupby(group).groups.items():
        xs, ms = x.loc[idx], mask.loc[idx]
        if ms.sum() == 0:
            continue
        v = np.sort(xs[ms].to_numpy())
        tail = len(v) - np.searchsorted(v, xs[ms].to_numpy(), side="left")
        out.loc[xs[ms].index] = -np.log10(tail / len(idx)) / np.log10(n_all)
    return out.clip(0, 1)


def rarity_cases(x, mask, size, group=None):
    """rarity counted in pattern occurrences instead of vendors. A vendor in a group of m vendors that share the pattern
    counts 1/m, so a group of m members is one occurrence (dossier 6.1: what is reviewed is a case, a set of member records).
    Used where the group size is itself the evidence; elsewhere rarity stays per vendor (see 2.2)."""
    x = pd.Series(x, dtype=float); mask = pd.Series(mask, index=x.index).fillna(False).astype(bool)
    w = 1.0 / pd.Series(size, index=x.index, dtype=float).where(mask, 1).clip(lower=1)
    group = pd.Series(0 if group is None else group, index=x.index)
    out = pd.Series(0.0, index=x.index)
    n_all = w.sum()
    for g_, idx in group.groupby(group).groups.items():
        xs, ms, ws = x.loc[idx], mask.loc[idx], w.loc[idx]
        if ms.sum() == 0:
            continue
        xm, wm = xs[ms].to_numpy(), ws[ms].to_numpy()
        order = np.argsort(xm)
        tail_w = np.cumsum(wm[order][::-1])[::-1]                    # weighted count of eligible vendors with value >= x
        tail = tail_w[np.searchsorted(xm[order], xm, side="left")]
        out.loc[xs[ms].index] = -np.log10(tail / ws.sum()) / np.log10(n_all)
    return out.clip(0, 1)


def hub_values(by_key):
    """relation values whose entity degree is beyond Tukey's far-out fence, per relation kind."""
    deg = pd.Series({k: len(v) for k, v in by_key.items()})
    kind = deg.index.str.split(":").str[0]
    hubs = set()
    for k_, d in deg.groupby(kind):
        d = d[d >= 2]                       # the fence is read on values that are actually shared
        if d.empty:
            continue
        q1, q3 = d.quantile([0.25, 0.75])
        fence = max(q3 + TUKEY_K * (q3 - q1), MIN_GROUP)
        hubs |= set(d[d > fence].index)
    return hubs


def layer2_tables(g, data_dir):
    """Vendor-level relational inputs from a graph built as of g.t (graph_build semantics)."""
    t = g.t
    tp = g.attributes["third_party"].copy()
    tp.index = strip(tp.index.to_series())
    re_of = strip(g.edges_of("RESOLVED_AS").set_index(strip(g.edges_of("RESOLVED_AS").source_node_id)).target_node_id)
    tp["re"] = re_of.reindex(tp.index)
    grp = strip(g.edges_of("DECLARED_MEMBER_OF").set_index(strip(g.edges_of("DECLARED_MEMBER_OF").source_node_id)).target_node_id)
    tp["group"] = grp.reindex(tp.index)
    eng = pg.engagement_frame(g)
    eng["sponsor"] = eng.sponsor_employee_id.where(~eng.sponsor_employee_id.isin(
        set(pg.employee_frame(g).query("is_system_default").index)))
    ap = g.edges_of("APPROVED_BY").copy()
    ap["engagement_id"], ap["approver"] = strip(ap.source_node_id), strip(ap.target_node_id)
    ap["approval_type"] = props(ap, "approval_type")
    fin = ap[ap.approval_type == "financial"].merge(eng[["record_id", "market", "sponsor", "start_date", "value_band"]],
                                                    left_on="engagement_id", right_index=True)
    loc = g.edges_of("LOCATED_AT").copy()
    loc["record_id"], loc["address"] = strip(loc.source_node_id), strip(loc.target_node_id)
    loc["address_type"] = props(loc, "address_type")
    loc["current"] = loc.valid_to.isna()
    inst_links = pd.concat([g.edges_of(r) for r in ["SERVES", "SITE_OF", "AFFILIATED_WITH"]])
    inst_links = inst_links[inst_links.valid_to.isna()].copy()          # links valid at t
    inst_links["record_id"], inst_links["institution"] = strip(inst_links.source_node_id), strip(inst_links.target_node_id)
    inst_links["link_type"] = inst_links.relationship_type
    uo = g.edges_of("ULTIMATE_OWNER").copy()
    uo["record_id"], uo["person"] = strip(uo.source_node_id), strip(uo.target_node_id)
    contact = pd.concat([g.edges_of("HAS_PHONE").assign(kind="phone"), g.edges_of("HAS_EMAIL_DOMAIN").assign(kind="domain")])
    contact["record_id"], contact["value"] = strip(contact.source_node_id), contact.kind + ":" + strip(contact.target_node_id)
    svc = g.edges_of("PROVIDES_SERVICE").copy()
    svc["record_id"], svc["category"] = strip(svc.source_node_id), strip(svc.target_node_id)
    flags = g.attributes["service_category"].compliance_flags
    svc["int_flag"] = svc.target_node_id.map(flags).fillna("").str.contains("int")
    pay = g.payments.copy()
    return dict(t=t, tp=tp, eng=eng, fin=fin, loc=loc, inst=inst_links, uo=uo, contact=contact, svc=svc, pay=pay,
                employees=pg.employee_frame(g), events=pg.business_events(data_dir, t),
                paths=g.ownership_paths, owns=g.edges_of("OWNS"), same_as=g.edges_of("SAME_AS"),
                institutions=g.attributes["institution"],
                split_limit=float(pd.read_csv(f"{data_dir}/approval_policy_matrix.csv").set_index("value_band").max_usd[1]))


def t1_shared_address(T):
    tp, loc = T["tp"], T["loc"]
    cur = loc[loc.current].assign(re=lambda d: d.record_id.map(tp.re), group=lambda d: d.record_id.map(tp.group))
    occ = cur.groupby("address").re.agg(lambda s: set(s))
    rows = []
    for r, d in cur.groupby("record_id"):
        best = dict(n=0, n_undeclared=0, benign=0)
        for a, at in zip(d.address, d.address_type):
            others = occ[a] - {tp.at[r, "re"]}
            g_ = tp.at[r, "group"]
            und = {o for o in others if pd.isna(g_) or o not in set(tp.loc[(tp.re == o) & (tp.group == g_)].re)}
            if len(und) > best["n_undeclared"] or (len(und) == best["n_undeclared"] and len(others) > best["n"]):
                best = dict(n=len(others), n_undeclared=len(und), benign=int(at in BENIGN_ADDRESS_TYPES))
        rows.append(dict(record_id=r, **best))
    f = pd.DataFrame(rows).set_index("record_id").reindex(tp.index).fillna(0)
    out = pd.DataFrame(index=tp.index)
    out["sa_n_cohabitants"] = f.n
    out["sa_n_cohab_undeclared"] = f.n_undeclared
    out["sa_benign_hub"] = f.benign
    s = rarity_within(f.n_undeclared, f.n_undeclared >= MIN_GROUP - 1, f.benign)
    return out, s.rename("s_SHARED_ADDRESS")


def first_engagement(T):
    e = T["eng"].dropna(subset=["sponsor"])
    return e.groupby(["sponsor", "record_id"]).start_date.min().reset_index().assign(
        re=lambda d: d.record_id.map(T["tp"].re))


def t2_sponsor_concentration(T):
    tp, emp = T["tp"], T["employees"]
    fe = first_engagement(T)
    burst = {}
    for s, d in fe.groupby("sponsor"):
        d = d.sort_values("start_date")
        days = d.start_date.values.astype("datetime64[D]").astype(int)
        for r, x in zip(d.record_id, days):
            win = d.re[(np.abs(days - x) <= W_SPONSOR)].nunique()
            burst[(s, r)] = win
    fe["burst"] = [burst[(s, r)] for s, r in zip(fe.sponsor, fe.record_id)]
    portfolio = fe.groupby("sponsor").re.nunique()
    # peer-relative: sponsor's max burst against role x scope x market peers (backoff)
    sp = emp.reindex(portfolio.index).assign(max_burst=fe.groupby("sponsor").burst.max(), portfolio=portfolio)
    z = pg.PeerScorer(pg.EMPLOYEE_LEVELS).score(sp, "max_burst")
    fe["sponsor_z"] = fe.sponsor.map(z.robust_z.fillna(0))
    fe["portfolio"] = fe.sponsor.map(portfolio)
    idx = fe.groupby("record_id").burst.idxmax()
    best = fe.loc[idx].set_index("record_id")
    out = pd.DataFrame(index=tp.index)
    out["sc_max_portfolio"] = fe.groupby("record_id").portfolio.max().reindex(tp.index).fillna(0)
    out["sc_burst"] = best.burst.reindex(tp.index).fillna(0)
    out["sc_burst_peer_z"] = best.sponsor_z.reindex(tp.index).fillna(0).clip(-5, 20)
    s = rarity(out.sc_burst, (out.sc_burst >= MIN_GROUP) & (out.sc_burst_peer_z >= Z_OUT)).rename("s_SPONSOR_CONCENTRATION")
    return out, s


def t3_approver_concentration(T):
    tp, fin, emp = T["tp"], T["fin"], T["employees"]
    fin = fin.assign(re=fin.record_id.map(tp.re))
    port = fin.groupby("approver").re.nunique()
    # workload relative to role x scope x market peers (backoff)
    z = pg.PeerScorer(pg.EMPLOYEE_LEVELS).score(emp.reindex(port.index).assign(portfolio=port), "portfolio")
    # routes: sponsors whose approvals are highly concentrated (HHI >= HHI_HIGH) with this approver on top, in a market
    # where >= MIN_POOL approvers exist (other approvers were available)
    routes = sponsor_routes(fin)
    n_routes = routes[routes.concentrated].groupby("top_approver").sponsor.nunique()
    exceeds = props(fin, "approver_level_at_approval").astype(float) < props(fin, "required_approver_level").astype(float)
    exceed_rate = exceeds.groupby(fin.approver).mean()
    a_stat = (n_routes.reindex(port.index).fillna(0) + exceed_rate.reindex(port.index).fillna(0))
    star = (z.robust_z.fillna(0) >= Z_OUT) & (a_stat > 0)
    a_stat = a_stat.where(star, 0)
    fin["a_routes"] = fin.approver.map(n_routes).fillna(0)
    fin["a_z"] = fin.approver.map(z.robust_z.fillna(0))
    fin["a_exceed"] = fin.approver.map(exceed_rate).fillna(0)
    fin["a_stat"] = fin.approver.map(a_stat).fillna(0)
    g_ = fin.groupby("record_id")
    out = pd.DataFrame(index=tp.index)
    out["ac_max_approver_portfolio"] = fin.approver.map(port).groupby(fin.record_id).max().reindex(tp.index).fillna(0)
    out["ac_approver_peer_z"] = g_.a_z.max().reindex(tp.index).fillna(0).clip(-5, 20)
    out["ac_approver_exclusive_routes"] = g_.a_routes.max().reindex(tp.index).fillna(0)
    out["ac_approver_exceed_rate"] = g_.a_exceed.max().reindex(tp.index).fillna(0)
    stat = g_.a_stat.max().reindex(tp.index).fillna(0)
    s = rarity(stat, stat > 0).rename("s_APPROVER_CONCENTRATION")
    return out, s


def sponsor_routes(fin):
    """per sponsor: approver HHI over its approved engagements, top approver and its pair count."""
    fs = fin.dropna(subset=["sponsor"])
    n_pair = fs.groupby(["sponsor", "approver"]).engagement_id.nunique()
    n_sp = n_pair.groupby(level=0).sum()
    share = n_pair / n_pair.index.get_level_values(0).map(n_sp).values
    pool = fs.groupby("market").approver.nunique()
    sp_market = fs.groupby("sponsor").market.agg(lambda m: m.value_counts().index[0])
    r = pd.DataFrame({"hhi": (share ** 2).groupby(level=0).sum(), "n": n_sp,
                      "top_approver": share.groupby(level=0).idxmax().str[1],
                      "top_share": share.groupby(level=0).max()})
    r["top_n"] = [n_pair[(s, a)] for s, a in zip(r.index, r.top_approver)]
    r["pool"] = r.index.map(sp_market).map(pool).fillna(0)
    r["concentrated"] = (r.hhi >= HHI_HIGH) & (r.top_n >= MIN_COUNT) & (r.pool >= MIN_POOL)
    return r.rename_axis("sponsor").reset_index()


def t4_sponsor_approver_coupling(T):
    tp, fin = T["tp"], T["fin"].dropna(subset=["sponsor"])
    r = sponsor_routes(fin).set_index("sponsor")
    on_route = fin[fin.approver.values == fin.sponsor.map(r.top_approver).values]
    on_route = on_route.assign(hhi=on_route.sponsor.map(r.hhi), n=on_route.sponsor.map(r.top_n),
                               ok=on_route.sponsor.map(r.concentrated).fillna(False))
    on_route["coupling"] = np.where(on_route.ok, on_route.n * on_route.hhi, 0)
    out = pd.DataFrame(index=tp.index)
    out["pc_max_pair_n"] = on_route.groupby("record_id").n.max().reindex(tp.index).fillna(0)
    out["pc_max_sponsor_hhi"] = on_route.groupby("record_id").hhi.max().reindex(tp.index).fillna(0)
    out["pc_coupling"] = on_route.groupby("record_id").coupling.max().reindex(tp.index).fillna(0)
    s = rarity(out.pc_coupling, out.pc_coupling > 0).rename("s_SPONSOR_APPROVER_COUPLING")
    return out, s


def relation_sets(T):
    """record -> set of 'kind:value' relation keys used by the dense-cluster and linking logic."""
    tp = T["tp"]
    rel = defaultdict(set)
    for r, a in T["loc"][T["loc"].current][["record_id", "address"]].values:
        rel[r].add("address:" + a)
    for r, s in T["eng"].dropna(subset=["sponsor"])[["record_id", "sponsor"]].values:
        rel[r].add("sponsor:" + s)
    for r, a in T["fin"][["record_id", "approver"]].values:
        rel[r].add("approver:" + a)
    for r, v in T["contact"][["record_id", "value"]].values:
        rel[r].add(v)
    for r, p in T["uo"][["record_id", "person"]].values:
        rel[r].add("owner:" + p)
    for r, i in T["inst"][["record_id", "institution"]].values:
        rel[r].add("institution:" + i)
    return rel


def pair_overlaps(T, rel=None):
    """pairs of resolved entities sharing >= 1 relation key; returns dict (re_a, re_b) -> set of kinds."""
    tp = T["tp"]
    rel = rel or relation_sets(T)
    by_key = defaultdict(set)
    for r, keys in rel.items():
        for k in keys:
            by_key[k].add(tp.at[r, "re"])
    pairs = defaultdict(set)
    hubs = hub_values(by_key)
    for k, res in by_key.items():
        if len(res) < 2 or k in hubs:
            continue
        kind = k.split(":", 1)[0]
        for a, b in combinations(sorted(res), 2):
            pairs[(a, b)].add(kind)
    # onboarding window, same country: a separate relation kind
    on = tp.groupby("re").agg(d=("onboarding_date", "min"), c=("country", "first"))
    for c, d in on.groupby("c"):
        d = d.sort_values("d")
        days = d.d.values.astype("datetime64[D]").astype(int)
        ids = d.index.to_numpy()
        for i in range(len(ids)):
            j = i + 1
            while j < len(ids) and days[j] - days[i] <= W_ONBOARD:
                pairs[tuple(sorted((ids[i], ids[j])))].add("window")
                j += 1
    return pairs


def t5_dense_cluster(T, pairs):
    tp = T["tp"]
    strong = {p: k for p, k in pairs.items() if len(k - {"window"}) >= 1 and len(k) >= DENSE_KINDS}
    n_types = defaultdict(int)
    n_partners = defaultdict(int)
    for (a, b), k in pairs.items():
        n_types[a] = max(n_types[a], len(k)); n_types[b] = max(n_types[b], len(k))
    for (a, b) in strong:
        n_partners[a] += 1; n_partners[b] += 1
    # component of strong pairs
    parent = {}
    def f(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in strong:
        parent[f(a)] = f(b)
    comp = defaultdict(set)
    for x in list(parent):
        comp[f(x)].add(x)
    size = {x: len(comp[f(x)]) for x in parent}
    out = pd.DataFrame(index=tp.index)
    out["dc_max_shared_types"] = tp.re.map(n_types).fillna(0)
    out["dc_n_partners_3types"] = tp.re.map(n_partners).fillna(0)
    out["dc_component_size"] = tp.re.map(size).fillna(0)
    stat = out.dc_max_shared_types + 0.1 * out.dc_n_partners_3types
    s = rarity_cases(stat, (out.dc_max_shared_types >= DENSE_KINDS) & (out.dc_component_size >= MIN_GROUP),
                     out.dc_component_size).rename("s_DENSE_MULTI_RELATION_CLUSTER")
    return out, s


def t6_rapid_onboarding(T, pairs):
    tp, ev = T["tp"], T["events"]
    linked = defaultdict(set)
    for (a, b), k in pairs.items():
        if "window" in k and len(k - {"window"}) >= 1:
            linked[a].add(b); linked[b].add(a)
    on = tp.groupby("re").agg(d=("onboarding_date", "min"), c=("country", "first"))
    # quiet period: onboardings in the country over the previous 365 days, per 30 days
    rate = {}
    for re_, (d, c) in on.iterrows():
        same = on[(on.c == c)]
        prior = same[(same.d < d - pd.Timedelta(days=W_ONBOARD)) & (same.d >= d - pd.Timedelta(days=365))]
        rate[re_] = len(prior) * W_ONBOARD / 365
    out = pd.DataFrame(index=tp.index)
    out["ro_n_linked_in_window"] = tp.re.map(lambda r: len(linked.get(r, ()))).astype(float)
    out["ro_country_prior_rate"] = tp.re.map(rate).astype(float)
    out["ro_burst_ratio"] = (out.ro_n_linked_in_window + 1) / (out.ro_country_prior_rate + 1)
    benign = [bool(pg.events_covering(ev, d, c, margin_days=0, phenomenon="onboarding"))
              for d, c in zip(tp.onboarding_date, tp.country)]
    out["ro_event_explained"] = np.array(benign, dtype=float)
    s = rarity_cases(out.ro_burst_ratio, out.ro_n_linked_in_window >= MIN_GROUP - 1, out.ro_n_linked_in_window + 1,
                     out.ro_event_explained)
    return out, s.rename("s_RAPID_ONBOARDING_LINKED")


def t7_bridge(T):
    tp, il, inst = T["tp"], T["inst"], T["institutions"]
    inst = inst.set_axis(strip(inst.index.to_series()))
    by_vendor = il.groupby("record_id").institution.agg(set)
    rows = {}
    for r, mine in by_vendor.items():
        if len(mine) < 2:
            continue
        # institution communities formed by the OTHER vendors' links
        parent = {i: i for i in mine}
        def f(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for r2, theirs in by_vendor.items():
            if r2 == r:
                continue
            common = sorted(theirs & mine)
            for x, y in zip(common, common[1:]):
                parent[f(x)] = f(y)
        comms = {f(i) for i in mine}
        countries = inst.reindex(list(mine)).country.nunique()
        rows[r] = dict(n_comm=len(comms), n_countries=countries,
                       public=inst.reindex(list(mine)).public_sector.astype(str).str.lower().eq("true").mean())
    f_ = pd.DataFrame.from_dict(rows, orient="index").reindex(tp.index).fillna(0)
    out = pd.DataFrame(index=tp.index)
    out["br_n_communities"] = f_.n_comm
    out["br_n_inst_countries"] = f_.n_countries
    out["br_public_share"] = f_.public
    stat = f_.n_comm + f_.n_countries
    s = rarity(stat, (f_.n_comm >= 2) & (f_.n_countries >= 2)).rename("s_BRIDGE_ENTITY")
    return out, s


def t8_geography(T):
    tp, svc = T["tp"], T["svc"]
    # intermediaries (category flag "int") onboarded in the same country within +-6 months vs those already present
    intm = set(svc[svc.int_flag].record_id)
    fe = first_engagement(T)
    sp_of = fe.sort_values("start_date").drop_duplicates("record_id").set_index("record_id").sponsor
    d = tp.loc[tp.index.isin(intm), ["onboarding_date", "country", "re"]].assign(sponsor=lambda x: x.index.map(sp_of))
    rows = {}
    for r, (od, c) in d[["onboarding_date", "country"]].iterrows():
        same = d[d.country == c]
        h = pd.Timedelta(days=W_MARKET // 2)
        win = same[(same.onboarding_date >= od - h) & (same.onboarding_date <= od + h)]
        existing = same[same.onboarding_date < od - h]
        top = win.sponsor.value_counts(normalize=True).max() if win.sponsor.notna().any() else 0
        rows[r] = dict(n_new=win.re.nunique(), n_existing=existing.re.nunique(), top=top)
    f_ = pd.DataFrame.from_dict(rows, orient="index").reindex(tp.index).fillna(0)
    out = pd.DataFrame(index=tp.index)
    out["gc_new_intermediaries_12m"] = f_.n_new
    out["gc_existing_intermediaries"] = f_.n_existing
    out["gc_new_ratio"] = f_.n_new / (f_.n_existing + 1)
    out["gc_top_sponsor_share"] = f_.top
    stat = out.gc_new_ratio * (0.5 + out.gc_top_sponsor_share)
    s = rarity(stat, (f_.n_new >= MIN_GROUP) & (out.gc_new_ratio >= 1)).rename("s_GEOGRAPHY_LINKED_CONCENTRATION")
    return out, s


def t9_activity_timing(T):
    tp, pay = T["tp"], T["pay"]
    m = pay.assign(re=pay.record_id.map(tp.re), p=pay.month.dt.to_period("M")).groupby(["re", "p"]).amount_usd.sum()
    m = m[m > 0]
    lvl = m.index.get_level_values(0)
    med = m.groupby(level=0).median()
    mad = (m - lvl.map(med).values).abs().groupby(level=0).median()
    nmon = m.groupby(level=0).size()
    z = 0.6745 * (m - lvl.map(med).values) / lvl.map(mad).values           # modified z (Iglewicz & Hoaglin)
    spikes = m[(z >= Z_OUT) & (lvl.map(mad).values > 0) & (lvl.map(nmon).values >= MIN_COUNT)]
    spike_months = spikes.reset_index().groupby("re").p.agg(set)
    fe = first_engagement(T)
    by_sp = fe.groupby("sponsor").re.agg(set)
    sync = defaultdict(int)
    for s, res in by_sp.items():
        res = [r for r in res if r in spike_months.index]
        for a in res:
            near = 0
            for b in res:
                if b == a:
                    continue
                if any(abs((pa - pb).n) <= W_SYNC for pa in spike_months[a] for pb in spike_months[b]):
                    near += 1
            sync[a] = max(sync[a], near)
    out = pd.DataFrame(index=tp.index)
    out["at_n_spike_months"] = tp.re.map(spike_months.map(len)).fillna(0)
    out["at_sync_spike_peers"] = tp.re.map(sync).fillna(0)
    s = rarity(out.at_sync_spike_peers, out.at_sync_spike_peers >= MIN_GROUP - 1).rename("s_ACTIVITY_TIMING_CONCENTRATION")
    return out, s


def name_similarity(matcher, a, b):
    return matcher.similarity(gb.name_tokens(a), gb.name_tokens(b))


def t10_common_owner(T):
    tp, uo = T["tp"], T["uo"]
    uo = uo.assign(re=uo.record_id.map(tp.re))
    by_p = uo.groupby("person").re.agg(set)
    matcher = gb.NameMatcher([gb.name_tokens(n) for n in tp.legal_name])
    rows = {}
    for r, d in uo.groupby("record_id"):
        me = tp.at[r, "re"]; g_ = tp.at[r, "group"]
        others = set().union(*[by_p[p] for p in d.person]) - {me}
        und = [o for o in others if pd.isna(g_) or not (tp[tp.re == o].group == g_).any()]
        diss = [o for o in und if name_similarity(matcher, tp.at[r, "legal_name"], tp[tp.re == o].legal_name.iloc[0]) < NAME_SIM]
        rows[r] = dict(n=len(others), und=len(und), diss=len(diss))
    f_ = pd.DataFrame.from_dict(rows, orient="index").reindex(tp.index).fillna(0)
    out = pd.DataFrame(index=tp.index)
    out["co_n_coowned"] = f_.n
    out["co_n_coowned_undeclared"] = f_.und
    out["co_n_coowned_dissimilar"] = f_.diss
    s = rarity_cases(f_.und + f_.diss, f_.und >= 1, f_.und + 1).rename("s_COMMON_BENEFICIAL_OWNER")
    return out, s


def t11_layered(T):
    tp = T["tp"]
    out = pd.DataFrame(index=tp.index)
    out["lo_max_depth"] = tp.own_max_depth.fillna(0).astype(float)
    out["lo_n_jurisdictions"] = tp.own_n_jurisdictions.fillna(0).astype(float)
    out["lo_n_paths"] = tp.own_n_paths.fillna(0).astype(float)
    out["lo_trust_or_nominee"] = tp.own_trust_or_nominee.fillna(False).astype(float)
    out["lo_cycle"] = tp.own_cycle.fillna(False).astype(float)
    stat = out.lo_max_depth + out.lo_n_jurisdictions + 0.5 * out.lo_n_paths + out.lo_trust_or_nominee + out.lo_cycle
    s = rarity(stat, (out.lo_max_depth >= DEPTH_MIN) & (out.lo_n_jurisdictions >= 2)).rename("s_LAYERED_COMPLEX_OWNERSHIP")
    return out, s


def t12_ownership_operational(T, rel):
    tp, uo, sa = T["tp"], T["uo"], T["same_as"]
    uo = uo.assign(re=uo.record_id.map(tp.re))
    by_p = uo.groupby("person").record_id.agg(set)
    ops = {r: {k for k in v if k.split(":")[0] in {"sponsor", "address", "phone", "domain"}} for r, v in rel.items()}
    n_overlap = defaultdict(int)
    for p, recs in by_p.items():
        for a, b in combinations(sorted(recs), 2):
            if tp.at[a, "re"] != tp.at[b, "re"] and ops.get(a, set()) & ops.get(b, set()):
                n_overlap[a] += 1; n_overlap[b] += 1
    # individual expert who is the ultimate owner of an organisation vendor engaged by the same sponsor
    person_of = sa[(sa.target_node_type == "person")]
    person_of = dict(zip(strip(person_of.source_node_id), strip(person_of.target_node_id)))
    spons = {r: {k for k in v if k.startswith("sponsor:")} for r, v in rel.items()}
    expert_agency = defaultdict(int)
    for ind, p in person_of.items():
        for org in by_p.get(p, set()):
            if org != ind and spons.get(ind, set()) & spons.get(org, set()):
                expert_agency[ind] = 1; expert_agency[org] = 1
    out = pd.DataFrame(index=tp.index)
    out["oo_n_owner_partners_ops_overlap"] = pd.Series(n_overlap).reindex(tp.index).fillna(0)
    out["oo_expert_owns_agency_same_sponsor"] = pd.Series(expert_agency).reindex(tp.index).fillna(0)
    stat = out.oo_n_owner_partners_ops_overlap + 2 * out.oo_expert_owns_agency_same_sponsor
    s = rarity_cases(stat, stat >= 1, out.oo_n_owner_partners_ops_overlap + 1).rename("s_OWNERSHIP_OPERATIONAL_OVERLAP")
    return out, s


def t13_opaque(T):
    tp = T["tp"]
    pep_q = tp.q_pep_connected.astype(str).str.lower().eq("true")
    out = pd.DataFrame(index=tp.index)
    out["oj_not_in_markets"] = tp.own_jurisdiction_not_in_markets.fillna(False).astype(float)
    out["oj_bo_access_low"] = tp.own_bo_access_low_on_path.fillna(False).astype(float)
    out["oj_sanctioned"] = tp.own_sanctioned_on_path.fillna(False).astype(float)
    out["oj_pep_owner"] = tp.own_pep_owner.fillna(False).astype(float)
    out["oj_pep_undisclosed"] = ((out.oj_pep_owner == 1) & ~pep_q).astype(float)
    stat = out.oj_not_in_markets + out.oj_bo_access_low + out.oj_sanctioned + out.oj_pep_owner + 2 * out.oj_pep_undisclosed
    s = rarity(stat, (out.oj_pep_owner + out.oj_not_in_markets >= 1) & (stat >= 2)).rename("s_OPAQUE_JURISDICTION_ROUTING")
    return out, s


def t14_ownership_change(T, rel):
    tp, owns, paths = T["tp"], T["owns"], T["paths"]
    t = T["t"]
    out = pd.DataFrame(index=tp.index, data=0.0, columns=["oc_change_observed", "oc_registry_lag_days", "oc_n_shift_types",
                                                         "oc_ego_jaccard_distance", "oc_new_owner_linked"])
    if paths is None or len(paths) == 0:
        return out, pd.Series(0.0, index=tp.index, name="s_OWNERSHIP_CHANGE_STRUCTURAL_SHIFT")
    entry = paths.groupby("record_id").entry_entity.first()
    loc, eng, svc, pay, uo = T["loc"], T["eng"], T["svc"], T["pay"], T["uo"]
    owner_res = uo.assign(re=uo.record_id.map(tp.re)).groupby("person").re.agg(set)
    persons_of = uo.groupby("record_id").person.agg(set)
    for r, ent in entry.items():
        if r not in tp.index:
            continue
        inc = owns[(owns.target_node_id == ent) & owns.valid_from.notna() & (owns.observed_at <= t)]
        new = inc[inc.valid_from > tp.at[r, "onboarding_date"]]
        if new.empty:
            continue
        c = new.valid_from.max()
        lag = (new.loc[new.valid_from.idxmax(), "observed_at"] - c).days
        w0, w1 = c, min(c + pd.Timedelta(days=W_SHIFT), t)
        shifts = 0
        shifts += int(((loc.record_id == r) & loc.valid_from.between(w0, w1)).any())                       # address
        prev_sp = set(eng[(eng.record_id == r) & (eng.start_date < w0)].sponsor.dropna())
        new_sp = set(eng[(eng.record_id == r) & eng.start_date.between(w0, w1)].sponsor.dropna())
        shifts += int(bool(new_sp - prev_sp))                                                                # sponsor
        shifts += int(((svc.record_id == r) & svc.valid_from.between(w0, w1)).any())                        # service expansion
        mp = pay[pay.record_id == r].groupby("month").amount_usd.sum()
        pre = mp[(mp.index < w0) & (mp.index >= w0 - pd.Timedelta(days=W_SHIFT))].sum()
        post = mp[(mp.index >= w0) & (mp.index <= w1)].sum()
        shifts += int(post > 2 * max(pre, 1.0))                                                             # spend jump
        rel_res = set().union(*[owner_res.get(p, set()) for p in persons_of.get(r, set())]) - {tp.at[r, "re"]}
        rel_on = tp[tp.re.isin(rel_res) & tp.onboarding_date.between(w0, w1)]
        shifts += int(len(rel_on) > 0)                                                                       # related vendors
        # ego network before vs after (addresses, sponsors, services): Jaccard distance (dossier 3.9)
        def ego(lo, hi):
            return ({"a:" + x for x in loc[(loc.record_id == r) & loc.valid_from.between(lo, hi)].address} |
                    {"s:" + str(x) for x in eng[(eng.record_id == r) & eng.start_date.between(lo, hi)].sponsor.dropna()} |
                    {"c:" + x for x in svc[(svc.record_id == r) & svc.valid_from.between(lo, hi)].category})
        before, after = ego(pd.Timestamp("1900-01-01"), w0 - pd.Timedelta(days=1)), ego(pd.Timestamp("1900-01-01"), w1)
        jd = 1 - len(before & after) / len(before | after) if (before | after) else 0.0
        out.loc[r] = [1.0, float(lag), float(shifts), jd, float(len(rel_on))]
    s = rarity(out.oc_n_shift_types, (out.oc_change_observed == 1) & (out.oc_n_shift_types >= 2))
    return out, s.rename("s_OWNERSHIP_CHANGE_STRUCTURAL_SHIFT")


# additional dossier candidate features (sec. 2.x / App. A)
def entropy(p):
    p = np.asarray([x for x in p if x > 0], float)
    p = p / p.sum() if p.sum() else p
    return float(-(p * np.log(p)).sum()) if len(p) else 0.0


def burstiness(x):
    """Appendix A: (sigma - mu) / (sigma + mu); -1 periodic, 0 Poisson-like, -> 1 bursty."""
    x = np.asarray(x, float)
    if len(x) < 2 or (x.std() + x.mean()) == 0:
        return 0.0
    return float((x.std() - x.mean()) / (x.std() + x.mean()))


def link_metrics(T, rel):
    """Appendix A link metrics on the entity-relation bipartite graph: Jaccard and Adamic-Adar (1/log degree of each
    shared relation value). Hub values (Tukey far-out) are dropped first, as for the pair index."""
    tp = T["tp"]
    by_key = defaultdict(set)
    for r, keys in rel.items():
        for k in keys:
            by_key[k].add(tp.at[r, "re"])
    hubs = hub_values(by_key)
    keys_of = defaultdict(set)
    for k, res in by_key.items():
        if k in hubs:
            continue
        for e in res:
            keys_of[e].add(k)
    kinds = ["address", "sponsor", "approver", "owner", "phone", "domain", "institution"]
    best = defaultdict(lambda: defaultdict(float))
    rows = []
    for k, res in by_key.items():
        if k in hubs or len(res) < 2:
            continue
        w = 1 / np.log(len(res))
        kind = k.split(":", 1)[0]
        for a, b in combinations(sorted(res), 2):
            rows.append((a, b, kind, w))
    pair = pd.DataFrame(rows, columns=["a", "b", "kind", "w"])
    if pair.empty:
        return pd.DataFrame(index=tp.index), pair
    aa = pair.groupby(["a", "b"]).w.sum()
    aa_kind = pair.groupby(["a", "b", "kind"]).w.sum().unstack("kind").reindex(columns=kinds).fillna(0)
    jac = pd.Series([len(keys_of[a] & keys_of[b]) / len(keys_of[a] | keys_of[b]) for a, b in aa.index], index=aa.index)
    pm = pd.concat([aa.rename("aa"), jac.rename("jac"), aa_kind], axis=1).reset_index()
    both = pd.concat([pm.assign(e=pm.a), pm.assign(e=pm.b)], ignore_index=True)
    g = both.groupby("e")
    out = pd.DataFrame(index=tp.index)
    out["lk_jaccard_max"] = tp.re.map(g.jac.max()).fillna(0)
    out["lk_adamic_adar_max"] = tp.re.map(g.aa.max()).fillna(0)
    out["lk_adamic_adar_sum"] = tp.re.map(g.aa.sum()).fillna(0)
    for kd in kinds:
        out[f"lk_aa_{kd}_max"] = tp.re.map(g[kd].max()).fillna(0)
    return out, pm


def dossier_extras(T, rel, pairs):
    tp, loc, eng, fin, pay, paths = T["tp"], T["loc"], T["eng"], T["fin"], T["pay"], T["paths"]
    t = T["t"]
    out = pd.DataFrame(index=tp.index)
    # T1 (2.2): temporal-overlap duration and sponsor overlap with the cohabitants of the current address
    cur = loc[loc.current].assign(re=lambda d: d.record_id.map(tp.re))
    spons = {r: {k for k in v if k.startswith("sponsor:")} for r, v in rel.items()}
    ov, sh = defaultdict(float), defaultdict(int)
    for a, d in cur.groupby("address"):
        if d.re.nunique() < 2:
            continue
        for x in d.itertuples():
            others = d[d.re != x.re]
            if len(others):
                ov[x.record_id] = max(ov[x.record_id], float((t - others.valid_from.clip(lower=x.valid_from)).dt.days.max()))
                sh[x.record_id] = max(sh[x.record_id], sum(bool(spons.get(x.record_id, set()) & spons.get(o, set()))
                                                           for o in others.record_id))
    out["sa_overlap_days"] = pd.Series(ov, dtype=float).reindex(tp.index).fillna(0)
    out["sa_cohab_shared_sponsor"] = pd.Series(sh, dtype=float).reindex(tp.index).fillna(0)
    # T2 (2.3): portfolio HHI and entropy over category parents for the vendor's sponsors
    e = eng.dropna(subset=["sponsor", "category_parent"])
    port = e.drop_duplicates(["sponsor", "record_id"]).groupby("sponsor").category_parent.value_counts(normalize=True)
    hhi = (port ** 2).groupby(level=0).sum()
    ent = port.groupby(level=0).apply(lambda p: entropy(p.values))
    out["sc_portfolio_hhi"] = e.sponsor.map(hhi).groupby(e.record_id).max().reindex(tp.index).fillna(0)
    out["sc_portfolio_entropy"] = e.sponsor.map(ent).groupby(e.record_id).min().reindex(tp.index).fillna(0)
    # T3/T4 (2.4): sponsor-approver pointwise mutual information log2 P(a|s) / P(a|market)
    fs = fin.dropna(subset=["sponsor"])
    p_as = fs.groupby(["sponsor", "approver"]).size() / fs.groupby("sponsor").size()
    p_am = fs.groupby(["market", "approver"]).size() / fs.groupby("market").size()
    pmi = np.log2([p_as[(s, a)] / p_am[(m, a)] for s, a, m in zip(fs.sponsor, fs.approver, fs.market)])
    out["pc_pair_pmi"] = pd.Series(pmi, index=fs.index).groupby(fs.record_id).max().reindex(tp.index).fillna(0)
    # T5 (2.5): typed-edge entropy of the relation kinds a vendor shares
    kc = defaultdict(lambda: defaultdict(int))
    for (a, b), k in pairs.items():
        for kd in k:
            kc[a][kd] += 1; kc[b][kd] += 1
    out["dc_kind_entropy"] = tp.re.map({e_: entropy(list(v.values())) for e_, v in kc.items()}).fillna(0)
    # T6 (2.6): inter-arrival time to the nearest onboarding in the country, burstiness of onboarding in the prior year
    on = tp.groupby("re").agg(d=("onboarding_date", "min"), c=("country", "first"))
    ia, bu = {}, {}
    for c, d in on.groupby("c"):
        dd = d.d.sort_values()
        for e_, x in dd.items():
            gaps = (dd - x).dt.days.abs()
            ia[e_] = float(gaps[gaps > 0].min()) if (gaps > 0).any() else 9999.0
            prior = dd[(dd <= x) & (dd >= x - pd.Timedelta(days=365))]
            bu[e_] = burstiness(np.diff(prior.values.astype("datetime64[D]").astype(int)))
    out["ro_min_interarrival_days"] = tp.re.map(ia).fillna(9999).clip(upper=3650)
    out["ro_country_burstiness"] = tp.re.map(bu).fillna(0)
    # T7 (2.7): participation coefficient 1 - sum_c (k_c / k)^2 over the institution communities of the vendor
    il = T["inst"]
    by_vendor = il.groupby("record_id").institution.agg(set)
    part = {}
    for r, mine in by_vendor.items():
        parent = {i: i for i in mine}
        def f(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for r2, theirs in by_vendor.items():
            if r2 != r:
                common = sorted(theirs & mine)
                for x, y in zip(common, common[1:]):
                    parent[f(x)] = f(y)
        k = pd.Series([f(i) for i in mine]).value_counts(normalize=True)
        part[r] = float(1 - (k ** 2).sum())
    out["br_participation"] = pd.Series(part, dtype=float).reindex(tp.index).fillna(0)
    # T8 (2.8): owner domicile versus operating market
    if paths is not None and len(paths):
        dom = paths.assign(j=paths.jurisdictions.str.split(">")).explode("j").dropna(subset=["j"])
        dom = dom[dom.j != ""].assign(c=lambda d: d.record_id.map(tp.country))
        out["gc_owner_domicile_differs"] = (dom.j != dom.c).groupby(dom.record_id).any().reindex(tp.index).fillna(False).astype(float)
    else:
        out["gc_owner_domicile_differs"] = 0.0
    # T9 (App. A): burstiness of the vendor's monthly spend
    m = pay.groupby(["record_id", pay.month.dt.to_period("M")]).amount_usd.sum()
    out["at_spend_burstiness"] = m.groupby(level=0).apply(lambda s: burstiness(s.values)).reindex(tp.index).fillna(0)
    # T10 (2.9) path confidence; T11 (2.10) missing intermediates; T13 (2.12) missing-owner segments
    out["co_min_match_confidence"] = tp.own_min_match_confidence.fillna(0).astype(float)
    reasons = tp.own_terminal_reasons.fillna("")
    out["lo_missing_intermediate"] = reasons.str.contains("no_owner_recorded|stated:unknown").astype(float)
    out["oj_missing_owner_segment"] = reasons.str.contains("stated:").astype(float)
    return out


def new_candidates(T, rel):
    tp, inst, eng, pay = T["tp"], T["inst"], T["eng"], T["pay"]
    out = pd.DataFrame(index=tp.index)
    # N1 experts affiliated with the same institution and engaged by the same sponsor
    aff = inst[inst.link_type == "AFFILIATED_WITH"].merge(tp[["party_type", "re"]], left_on="record_id", right_index=True)
    aff = aff[aff.party_type == "individual"]
    sp = eng.dropna(subset=["sponsor"]).groupby("record_id").sponsor.agg(set)
    pub = T["institutions"].set_axis(strip(T["institutions"].index.to_series())).public_sector.astype(str).str.lower().eq("true")
    n1 = defaultdict(int)
    for i, d in aff.groupby("institution"):
        recs = list(d.record_id)
        for a in recs:
            n1[a] = max(n1[a], sum(1 for b in recs if b != a and sp.get(a, set()) & sp.get(b, set())))
    out["nx_inst_expert_cluster"] = pd.Series(n1, dtype=float).reindex(tp.index).fillna(0)
    out["nx_inst_expert_cluster_public"] = out.nx_inst_expert_cluster * tp.index.map(
        lambda r: float(pub.reindex(list(aff[aff.record_id == r].institution)).any())).to_numpy()
    # N2 contact infrastructure (phone / own e-mail domain) shared with another entity outside its declared group
    c = T["contact"].assign(re=lambda d: d.record_id.map(tp.re), group=lambda d: d.record_id.map(tp.group))
    n2 = defaultdict(int)
    hubs = hub_values({v: set(d.re) for v, d in c.groupby("value")})
    for v, d in c.groupby("value"):
        if d.re.nunique() < 2 or v in hubs:
            continue
        for r, me, g_ in zip(d.record_id, d.re, d.group):
            n2[r] = max(n2[r], int(((d.re != me) & (pd.isna(g_) | (d.group != g_))).sum()))
    out["nx_contact_shared_undeclared"] = pd.Series(n2, dtype=float).reindex(tp.index).fillna(0)
    # N3 re-onboarded after termination: same tax id or phone as a record terminated before this one was onboarded
    keyed = pd.concat([tp[["tax_id"]].dropna().rename(columns={"tax_id": "k"}).assign(kind="tax"),
                       tp[["phone_key"]].dropna().rename(columns={"phone_key": "k"}).assign(kind="phone")])
    n3 = defaultdict(int)
    for k, d in keyed.groupby(["kind", "k"]):
        recs = d.index.unique()
        for b in recs:
            for a in recs:
                if a != b and pd.notna(tp.at[a, "terminated_date"]) and tp.at[b, "onboarding_date"] > tp.at[a, "terminated_date"]:
                    n3[b] = 1; n3[a] = 1
    out["nx_reonboarded_after_termination"] = pd.Series(n3, dtype=float).reindex(tp.index).fillna(0)
    # N4 split engagements: same vendor and sponsor, >= 2 band <= 1 engagements within 28 days, paid together above the
    # band-1 limit of approval_policy_matrix (above it a level-2 approver is required)
    spend = pay.groupby("engagement_id").amount_usd.sum()
    e = eng.assign(paid=eng.index.map(spend).fillna(0)).dropna(subset=["sponsor"])
    e = e[e.value_band <= 1].sort_values("start_date")
    n4 = defaultdict(int)
    for (r, s_), d in e.groupby(["record_id", "sponsor"]):
        for d0 in d.start_date:
            w = d[(d.start_date >= d0) & (d.start_date <= d0 + pd.Timedelta(days=W_ONBOARD))]
            if len(w) >= 2 and w.paid.sum() > T["split_limit"]:
                n4[r] = max(n4[r], len(w))
    out["nx_split_engagements"] = pd.Series(n4, dtype=float).reindex(tp.index).fillna(0)
    return out


TYPOLOGIES = ["SHARED_ADDRESS", "SPONSOR_CONCENTRATION", "APPROVER_CONCENTRATION", "SPONSOR_APPROVER_COUPLING",
              "DENSE_MULTI_RELATION_CLUSTER", "RAPID_ONBOARDING_LINKED", "BRIDGE_ENTITY", "GEOGRAPHY_LINKED_CONCENTRATION",
              "ACTIVITY_TIMING_CONCENTRATION", "COMMON_BENEFICIAL_OWNER", "LAYERED_COMPLEX_OWNERSHIP",
              "OWNERSHIP_OPERATIONAL_OVERLAP", "OPAQUE_JURISDICTION_ROUTING", "OWNERSHIP_CHANGE_STRUCTURAL_SHIFT"]
OWNERSHIP_TYPOLOGIES = TYPOLOGIES[9:]


def risk_score(S, typologies=TYPOLOGIES):
    """noisy-OR (Pearl 1988): R = 1 - prod_k (1 - s_k). Independent signals corroborate each other without a tuned bonus."""
    S = S[[f"s_{k}" for k in typologies]].clip(0, 1)
    R = 1 - (1 - S).prod(axis=1)
    top = S.idxmax(axis=1).str[2:].where(S.max(axis=1) > 0, "")
    return R, top, (S >= 0.5).sum(axis=1)


def build_layer2(t, data_dir):
    g = gb.build_graph(data_dir, t)
    T = layer2_tables(g, data_dir)
    rel = relation_sets(T)
    pairs = pair_overlaps(T, rel)
    parts = [t1_shared_address(T), t2_sponsor_concentration(T), t3_approver_concentration(T),
             t4_sponsor_approver_coupling(T), t5_dense_cluster(T, pairs), t6_rapid_onboarding(T, pairs), t7_bridge(T),
             t8_geography(T), t9_activity_timing(T), t10_common_owner(T), t11_layered(T),
             t12_ownership_operational(T, rel), t13_opaque(T), t14_ownership_change(T, rel)]
    F = pd.concat([f for f, _ in parts], axis=1)
    S = pd.concat([s for _, s in parts], axis=1)
    N = new_candidates(T, rel)
    LK, pair_metrics = link_metrics(T, rel)
    X = dossier_extras(T, rel, pairs)
    F = pd.concat([F, LK, X], axis=1)
    R, top, n_strong = risk_score(S)
    F = pd.concat([F, N, S], axis=1)
    F["r_closed"] = R
    F.index.name = "record_id"
    return F, dict(graph=g, T=T, rel=rel, pairs=pairs, top=top, pair_metrics=pair_metrics)


PREFIX = {"sa_": "SHARED_ADDRESS", "sc_": "SPONSOR_CONCENTRATION", "ac_": "APPROVER_CONCENTRATION",
          "pc_": "SPONSOR_APPROVER_COUPLING", "dc_": "DENSE_MULTI_RELATION_CLUSTER", "ro_": "RAPID_ONBOARDING_LINKED",
          "br_": "BRIDGE_ENTITY", "gc_": "GEOGRAPHY_LINKED_CONCENTRATION", "at_": "ACTIVITY_TIMING_CONCENTRATION",
          "co_": "COMMON_BENEFICIAL_OWNER", "lo_": "LAYERED_COMPLEX_OWNERSHIP", "oo_": "OWNERSHIP_OPERATIONAL_OVERLAP",
          "oj_": "OPAQUE_JURISDICTION_ROUTING", "oc_": "OWNERSHIP_CHANGE_STRUCTURAL_SHIFT",
          "lk_aa_address": "SHARED_ADDRESS", "lk_aa_sponsor": "SPONSOR_CONCENTRATION", "lk_aa_approver": "APPROVER_CONCENTRATION",
          "lk_aa_owner": "COMMON_BENEFICIAL_OWNER", "lk_aa_institution": "BRIDGE_ENTITY", "lk_": "DENSE_MULTI_RELATION_CLUSTER"}


def typology_of(v):
    """typology group of a Layer 2 variable (None for candidate variables and the risk score)"""
    if v.startswith("s_"):
        return v[2:]
    return next((k for p, k in PREFIX.items() if v.startswith(p)), None)


def groups_of(columns):
    return {k: [v for v in columns if typology_of(v) == k] for k in TYPOLOGIES}


# new typologies found in the label-free search; their variables and signal conditions
NEW_TYPOLOGIES = ["PUBLIC_INSTITUTION_EXPERT_CLUSTER", "FMV_CAP_BREACH", "PAYMENT_AFTER_TERMINATION"]
NEW_GROUPS = {"PUBLIC_INSTITUTION_EXPERT_CLUSTER": ["nx_inst_expert_cluster", "nx_inst_expert_cluster_public",
                                                    "s_PUBLIC_INSTITUTION_EXPERT_CLUSTER"],
              "FMV_CAP_BREACH": ["max_fmv_cap_ratio", "s_FMV_CAP_BREACH"],
              "PAYMENT_AFTER_TERMINATION": ["n_months_paid_after_term", "s_PAYMENT_AFTER_TERMINATION"]}


def new_signals(L2t, L1t):
    """signals of the three new typologies, same rarity transform as the 14"""
    x = L2t.nx_inst_expert_cluster_public
    return pd.DataFrame({
        "s_PUBLIC_INSTITUTION_EXPERT_CLUSTER": rarity(x, x >= MIN_GROUP - 1),
        "s_FMV_CAP_BREACH": rarity(L1t.max_fmv_cap_ratio, L1t.max_fmv_cap_ratio > 1),
        "s_PAYMENT_AFTER_TERMINATION": rarity(L1t.n_months_paid_after_term, L1t.n_months_paid_after_term >= 1),
    }).reindex(L2t.index).fillna(0)
