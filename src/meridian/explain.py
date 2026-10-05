"""Analyst case packages (Dossier §10.1 and Appendix C).

One markdown file per case plus an index. Each package lets a reviewer
reproduce the alert from the source tables: members, why flagged, evidence
with peer comparison, a timeline that keeps valid time and observation time
apart, ownership paths with match confidence, a relationship sketch, the
benign explanations to check first, and provenance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .cases import Case
from .context import Context
from .detectors.base import OPEN_SET_TAG, POLICY_TAG
from .ranking import select_tags

# What a reviewer should check for each kind of benign explanation.
REVIEW_QUESTIONS = [
    ("address type", "Confirm the address type in the address source. If it is a registered agent, coworking space "
                     "or campus, check whether the vendors share anything besides the address."),
    ("declared group", "Check the onboarding group declarations: are all members declared under that group, and is "
                       "the group's structure consistent with the registry?"),
    ("business event", "Were these vendors part of that event (integration list, launch plan, migration batch)? "
                       "A vendor outside the event's scope keeps the concern."),
    ("central role", "Check the sponsor's documented scope; a central buyer is expected to add many vendors."),
    ("regional/global scope", "Compare against peers with the same regional or global scope, not local roles."),
    ("batch entry", "Check whether the vendors were loaded together from a migration or integration file."),
    ("small approver pool", "Confirm how many approvers held the required level in that market and period."),
    ("personal service company", "Confirm the company was declared as the expert's own vehicle and that the "
                                 "payments to both are not for the same services."),
    ("process relations", "Shared sponsor, approver and category can reflect a standard local process. Look for "
                          "shared address, contacts or owners before escalating."),
    ("fund", "Check whether the common owner is a fund or listed holder with a diversified portfolio."),
    ("conduit", "Holding companies in conduit jurisdictions are common; check the business rationale and whether "
                "the ultimate owner is known."),
    ("offshore holding without PEP", "Check the business rationale for the holding location and the ultimate owner."),
    ("distributor", "Check whether the vendor is a national or regional distributor serving many institutions."),
    ("cross-border by design", "CROs and logistics providers work across borders; check the contract scope."),
    ("chain gap", "Part of the ownership chain is missing from the register: treat as unknown, request it."),
    ("similar names", "Similar names may mean one brand: check whether the members are one declared business."),
    ("seasonal", "Compare with the sponsor's spend in the same months of other years."),
    ("demerger", "Check the transitional services agreement for this vendor."),
]

TAG_QUESTIONS = {
    "SHARED_ADDRESS": "Are the vendors at the address independent businesses, and why do they share it?",
    "SPONSOR_CONCENTRATION": "Why did one sponsor add these vendors in so short a window? Were they competitively selected?",
    "APPROVER_CONCENTRATION": "Did the approver hold the required authority for each approval?",
    "SPONSOR_APPROVER_COUPLING": "Why were the other eligible approvers not used?",
    "DENSE_MULTI_RELATION_CLUSTER": "Are these vendors one business presented as several?",
    "RAPID_ONBOARDING_LINKED": "What business need explains onboarding these linked vendors together?",
    "BRIDGE_ENTITY": "Why does a local vendor with one sponsor serve institutions abroad?",
    "GEOGRAPHY_LINKED_CONCENTRATION": "Why did intermediaries in this market increase so fast?",
    "ACTIVITY_TIMING_CONCENTRATION": "What activity explains the simultaneous spend, or the payments after termination?",
    "COMMON_BENEFICIAL_OWNER": "Were the common owner and the relationship between the vendors disclosed?",
    "LAYERED_COMPLEX_OWNERSHIP": "What is the business reason for the layered structure; who is the ultimate owner?",
    "OWNERSHIP_OPERATIONAL_OVERLAP": "Do the commonly owned vendors bill for distinct services?",
    "OPAQUE_JURISDICTION_ROUTING": "Why is ownership routed offshore, and was the PEP connection disclosed?",
    "OWNERSHIP_CHANGE_STRUCTURAL_SHIFT": "Was the change of owner notified, and does it explain the new activity?",
    POLICY_TAG: "Is there an approved exception for this payment above policy?",
    OPEN_SET_TAG: "Which relationship or record attribute makes this group unusual, and is there a business reason?",
}


def _fmt_date(d) -> str:
    return "" if d is None or pd.isna(d) else str(pd.Timestamp(d).date())


def members_table(ctx: Context, case: Case) -> str:
    rows = ["| record_id | legal name | country | type | onboarded | status | declared group | entity | registry link |",
            "|---|---|---|---|---|---|---|---|---|"]
    rm = ctx.reg_matches[ctx.reg_matches["status"] == "accepted"].drop_duplicates("record_id").set_index("record_id")
    for r in sorted(case.members):
        v = ctx.vendors.loc[r]
        link = (f"{rm.at[r, 'method']} ({rm.at[r, 'confidence']:.1f})" if r in rm.index else "not linked")
        ent = ctx.entity_of.get(r, r)
        n = len(ctx.records_of.get(ent, {r}))
        rows.append(f"| {r} | {v['legal_name']} | {v['country']} | {v['party_type']} | {_fmt_date(v['onboarding_date'])} "
                    f"| {v['status']} | {v['declared_parent_group'] or ''} | {ent}{f' ({n} records)' if n > 1 else ''} "
                    f"| {link} |")
    return "\n".join(rows)


def timeline(ctx: Context, case: Case) -> pd.DataFrame:
    """Dated events for the members. `valid` is when it was true, `observed` when it was recorded."""
    m = set(case.members)
    ev = []
    for r in m:
        v = ctx.vendors.loc[r]
        ev.append((v["onboarding_date"], None, r, "onboarded"))
        if pd.notna(v["terminated_date"]):
            ev.append((v["terminated_date"], None, r, "terminated"))
    a = ctx.addresses[ctx.addresses["record_id"].isin(m)]
    for x in a.itertuples():
        ev.append((x.valid_from, x.observed_at, x.record_id, f"address from: {x.raw_address} ({x.address_type or 'unclassified'})"))
    e = ctx.engagements[ctx.engagements["record_id"].isin(m)].dropna(subset=["sponsor_employee_id"])
    first = e.sort_values("start_date").groupby(["record_id", "sponsor_employee_id"]).first().reset_index()
    for x in first.itertuples():
        ev.append((x.start_date, x.observed_at, x.record_id,
                   f"first engagement with sponsor {x.sponsor_employee_id} ({x.sp_role}), {x.cat_primary or x.service_category}"))
    ri = ctx.b.reg_interests
    for r in m:
        sid = ctx.reg_entity_of.get(r)
        if sid is None:
            continue
        for x in ri[ri["subject_sid"] == sid].itertuples():
            n = ctx.reg_nodes.loc[x.party_sid] if x.party_sid in ctx.reg_nodes.index else None
            who = n["name"] if n is not None else f"(chain stops: {x.unspecified_reason})"
            ev.append((x.start_date, x.statement_date, r, f"owner interest starts: {who} {x.share if pd.notna(x.share) else ''}"
                       f"{'%' if pd.notna(x.share) else ''} {x.details or ''}".rstrip()))
            if pd.notna(x.end_date):
                ev.append((x.end_date, x.statement_date, r, f"owner interest ends: {who}"))
    df = pd.DataFrame(ev, columns=["valid", "observed", "record_id", "event"]).sort_values(["valid", "record_id"])
    return df


def ownership_section(ctx: Context, case: Case) -> str:
    from .detectors.ownership import _view
    if not ctx.use_ownership:
        return "_Ownership layer switched off for this run._"
    ov = _view(ctx)
    lines = []
    for r in sorted(case.members):
        if r not in ctx.reg_entity_of:
            if r in ctx.reg_person_of:
                lines.append(f"- {r}: the expert is registry person {ctx.reg_person_of[r]}")
            continue
        m = _snapshot_for_case(ov, r, case)
        if m is None:
            continue
        for p in m["paths"]:
            lines.append(f"- {r} (as of {m['at'].date()}): {ov.path_text(r, p)}")
        latest = ov.snapshots(r)[-1]
        if latest["at"] != m["at"] and latest["intermediates"] != m["intermediates"]:
            lines.append(f"  - the structure changed afterwards; as of {latest['at'].date()}: "
                         + "; ".join(ov.path_text(r, p) for p in latest["paths"]))
        if m["gaps"]:
            lines.append(f"  - {m['gaps']} gap(s) in the chain: unknown, not evidence of concealment")
    unmatched = [r for r in case.members if r not in ctx.reg_entity_of and r not in ctx.reg_person_of]
    if unmatched:
        lines.append(f"- not linked to the register: {', '.join(sorted(unmatched))}")
    cand = ctx.reg_matches[(ctx.reg_matches["record_id"].isin(case.members)) & (ctx.reg_matches["status"] != "accepted")]
    for x in cand.itertuples():
        lines.append(f"- uncertain registry match kept as candidate: {x.record_id} -> {x.statement_id} ({x.method}, {x.confidence})")
    return "\n".join(lines) if lines else "_No member is linked to an ownership path._"


def _snapshot_for_case(ov, rec: str, case: Case):
    """The ownership snapshot in force when the case emerged (else the earliest one)."""
    snaps = ov.snapshots(rec)
    if not snaps:
        return None
    at = case.emerged_at or snaps[-1]["at"]
    before = [m for m in snaps if m["at"] <= at]
    return before[-1] if before else snaps[0]


_NODE_LABEL = {"VendorRecord": "vendor", "Address": "address", "ContactPoint": "contact", "Employee": "employee",
               "Engagement": "engagement", "ServiceCategory": "category", "HealthcareInstitution": "institution",
               "LegalEntity": "entity", "NaturalPerson": "person", "ResolvedEntity": "entity",
               "DeclaredGroup": "group", "AccountingParent": "parent"}
_SHAPE = {"VendorRecord": ('["', '"]'), "NaturalPerson": ('{{"', '"}}'), "LegalEntity": ('[["', '"]]')}


# Identity and ownership edges first, process edges last, so a truncated drawing keeps them.
_RELATION_ORDER = ["LOCATED_AT", "HAS_CONTACT", "SAME_AS", "REGISTERED_AS", "IS_PERSON", "OWNS", "CONTROLS",
                   "CONSOLIDATED_BY", "SERVES", "SITE_OF", "AFFILIATED_WITH", "SPONSORED_BY", "APPROVED_BY",
                   "CONTRACTED_UNDER", "FOR_SERVICE", "PROVIDES_SERVICE"]


def _evidence_frame(ctx: Context, case: Case) -> pd.DataFrame:
    if ctx.graph is None or not case.edges:
        return pd.DataFrame()
    e = ctx.graph.edges.set_index("edge_id")
    ev = e.loc[[x for x in case.edges if x in e.index]].reset_index()
    rank = {r: i for i, r in enumerate(_RELATION_ORDER)}
    ev["_o"] = ev["relation_type"].map(lambda r: rank.get(r, len(rank)))
    return ev.sort_values(["_o", "source_node_id", "valid_from"], kind="stable").drop(columns="_o")


def evidence_edges_table(ctx: Context, case: Case, limit: int = 60) -> str:
    """The graph edges behind the case, with their times and sources."""
    ev = _evidence_frame(ctx, case)
    if ev.empty:
        return "_This case rests on record attributes only; no relation edges._"
    rows = ["| edge | source | relation | target | valid from | valid to | observed | source record | confidence |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in ev.head(limit).itertuples():
        conf = "" if pd.isna(r.resolution_confidence) else f"{r.resolution_confidence:.2f}"
        rows.append(f"| {r.edge_id} | {r.source_node_id} | {r.relation_type} | {r.target_node_id} | "
                    f"{r.valid_from or ''} | {r.valid_to or ''} | {r.observed_at or ''} | "
                    f"{r.source_table}:{r.source_record_id} | {conf} |")
    more = f"\n\n{len(ev) - limit} more edges in `cases.jsonl`." if len(ev) > limit else ""
    return f"{len(ev)} edges in graph `{ctx.graph.hash}`.\n\n" + "\n".join(rows) + more


def sketch(ctx: Context, case: Case, max_edges: int = 40) -> str:
    """Mermaid drawing of the evidence subgraph (the case's own edges in the graph)."""
    ev = _evidence_frame(ctx, case)
    if ev.empty:
        return "_No relation edges to draw._"
    ev = ev.head(max_edges)
    nodes = ctx.graph.nodes.set_index("node_id")
    ids = {n: f"N{i}" for i, n in enumerate(dict.fromkeys(list(ev["source_node_id"]) + list(ev["target_node_id"])))}
    lines = ["```mermaid", "graph LR"]
    for n, short in ids.items():
        ntype = nodes.at[n, "node_type"] if n in nodes.index else "?"
        label = str(nodes.at[n, "label"] if n in nodes.index else n).replace('"', "'")[:30]
        left, right = _SHAPE.get(ntype, ("(", ")"))
        if left == "(":
            left, right = '("', '")'
        lines.append(f"  {short}{left}{_NODE_LABEL.get(ntype, ntype)}<br/>{label}{right}")
    for r in ev.itertuples():
        lines.append(f"  {ids[r.source_node_id]} -->|{r.relation_type}| {ids[r.target_node_id]}")
    lines.append("```")
    note = f"\nShowing {max_edges} of {len(case.edges)} evidence edges." if len(case.edges) > max_edges else ""
    return "\n".join(lines) + note


def benign_section(case: Case) -> str:
    lines = []
    for b in case.benign:
        q = next((q for key, q in REVIEW_QUESTIONS if key in b), "Check whether this explains the pattern.")
        lines.append(f"- **{b}**. {q}")
    for t in case.tags:
        if t in TAG_QUESTIONS:
            lines.append(f"- Review question ({t}): {TAG_QUESTIONS[t]}")
    return "\n".join(lines)


def evidence_section(case: Case) -> str:
    out = []
    for c in case.candidates + case.support:
        role = "supporting" if c in case.support else "detector"
        out.append(f"### {c.detector} ({role}; tags {', '.join(c.tags)})")
        out.append(f"{c.reason}")
        out.append(f"- emerged {_fmt_date(c.emerged_at)}" + (f", held until {_fmt_date(c.active_until)}" if c.active_until else ""))
        for k, v in c.evidence.items():
            if k in ("edges",):
                continue
            text = v if isinstance(v, str) else json.dumps(v, default=str, ensure_ascii=False)
            out.append(f"- {k}: {text[:600]}{' ...' if len(text) > 600 else ''}")
        if c.benign:
            out.append("- benign explanations recorded: " + "; ".join(c.benign))
    return "\n\n".join(out)


def case_package(ctx: Context, case: Case, rank: int, meta: dict) -> str:
    tl = timeline(ctx, case)
    tl_rows = ["| valid | observed | record | event |", "|---|---|---|---|"]
    for x in tl.itertuples():
        tl_rows.append(f"| {_fmt_date(x.valid)} | {_fmt_date(x.observed)} | {x.record_id} | {x.event} |")
    return "\n\n".join([
        f"# Case {rank}: {' | '.join(select_tags(case))}",
        (f"Snapshot {meta['snapshot']} · run `{meta['run']}` · graph `{meta.get('graph_hash', 'n/a')}` · "
         f"config `{meta['config_hash']}` · commit `{meta['git_commit']}`  \n"
         f"Ranking score {case.effective:.3f} (case score {case.calibrated:.3f} before type interleaving; a review "
         f"priority, not a probability) · emerged {_fmt_date(case.emerged_at)} · "
         f"detectors: {', '.join(case.detectors)}"),
        "## Members", members_table(ctx, case),
        f"All tags raised by the evidence: {', '.join(case.tags)}",
        "## Why flagged", "\n".join(f"- {c.reason} (`{c.detector}`)" for c in case.candidates + case.support),
        "## Benign explanations to check first", benign_section(case) or "_None recorded._",
        "## Evidence", evidence_section(case),
        "## Timeline", "Valid time is when a fact was true; observed time is when it was recorded or published.",
        "\n".join(tl_rows),
        "## Ownership", ownership_section(ctx, case),
        "## Evidence edges", evidence_edges_table(ctx, case),
        "## Evidence subgraph", sketch(ctx, case),
        "## Provenance",
        (f"- inputs: {', '.join(f'{k}={v}' for k, v in sorted(meta['input_hashes'].items()))}\n"
         f"- graph: `{meta.get('graph_hash', 'n/a')}` (node and edge tables in the run's `graph/` folder)\n"
         f"- detector parameters: see `run.json` of run `{meta['run']}`\n"
         f"- reproduce: `meridian --config <config> --out <dir>` with the same inputs gives the same case"),
    ])


def write_packages(ctx: Context, ordered: list[Case], out_dir: Path, meta: dict, top: int = 10) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    index = ["# Case packages", f"Snapshot {meta['snapshot']} · run `{meta['run']}`", "",
             "| rank | tags | members | score | emerged | detectors | benign to check |", "|---|---|---|---|---|---|---|"]
    for rank, case in enumerate(ordered[:top], start=1):
        name = f"case_{rank:03d}.md"
        (out_dir / name).write_text(case_package(ctx, case, rank, meta))
        index.append(f"| [{rank}]({name}) | {', '.join(select_tags(case))} | {len(case.members)} | {case.effective:.3f} | "
                     f"{_fmt_date(case.emerged_at)} | {', '.join(case.detectors)} | {len(case.benign)} |")
    (out_dir / "README.md").write_text("\n".join(index) + "\n")
    return out_dir


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
