"""From detector findings to ranked review cases, the submission file, and §9.1-style self-evaluation.

1. Period: the sponsor scores cases that emerge in a period (§8.1). Detectors run on the graph at the period
   end t1; for a period with a start t0, a finding counts as emerged when no finding of the same detector
   existed at t0 with overlapping members (Jaccard >= emerge_overlap), or when it existed but has grown since
   with members onboarded after t0 and a larger main statistic (snapshot difference, no labels).
2. Cases (§6.1): findings are taken best first; a finding joins an existing case when its member set overlaps
   the case's core (Jaccard >= merge_overlap), otherwise it seeds a new case. A case's members are its seed's
   members plus any record named by at least two of its findings. Tags are the typologies of its findings
   scoring >= tag_min_score (the seed's always). Score = seed score + multi_typology_bonus per additional
   typology, so corroborated multi-typology structures rank first; ties break on the sum of finding scores.
3. Evaluation (§9.1): one-to-one matching down the ranking (a submitted case matches the unmatched planted
   case with the highest IoU if IoU >= match_iou), then P@k, R@k, AP, median first-hit rank, mean IoU,
   member recall, exact recovery, recall by typology and difficulty, tag accuracy, false positives in top 50.
   Labels are read only by `evaluate`, never by the case builder.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repository root (this file is in scripts/layer3/)
sys.path[:0] = [str(ROOT / "scripts" / "layer2"), str(ROOT / "scripts" / "layer3")]  # graph_build is shared with Layer 2
import graph_build as gb
import layer3_detectors as dt
PERIODS = {  # dossier §8.1
    "TRAINING": (None, pd.Timestamp("2024-06-30")),
    "VALIDATION": (pd.Timestamp("2024-06-30"), pd.Timestamp("2025-06-30")),
    "TEST": (pd.Timestamp("2025-06-30"), pd.Timestamp("2026-06-30")),
}
CASE_CONFIG = dict(
    emerge_overlap=0.5,        # [team]
    merge_overlap=0.5,         # [team] Jaccard between a finding and a case core to merge them
    tag_min_score=0.6,         # [team] a finding's typology is added to the case tags at or above this score
    multi_typology_bonus=0.05, # [team]
    max_cases=500,             # [team] submission length
)
# Record-level checks are scored separately (§4); this maps them to the closest typology hypothesis.
# An assumption to confirm with the sponsor; FMV has no typology and keeps its own tag.
RECORD_TAGS = {"PAYMENT_AFTER_TERMINATION": "ACTIVITY_TIMING_CONCENTRATION",   # §2.6 activity timing
               "APPROVAL_AUTHORITY_EXCEEDED": "APPROVER_CONCENTRATION",        # §2.4 outside documented authority
               "COMPLIANCE_REVIEW_MISSING": "APPROVER_CONCENTRATION",          # §2.4 approval pathway / control
               "FMV_ANNUAL_CAP_EXCEEDED": "FMV_ANNUAL_CAP_EXCEEDED"}
REASON_CODES = {"SHARED_ADDRESS": "SHARED_ADDRESS_RARE", "SPONSOR_CONCENTRATION": "SPONSOR_CONCENTRATION_HIGH",
                "APPROVER_CONCENTRATION": "APPROVER_CONCENTRATION_HIGH", "SPONSOR_APPROVER_COUPLING": "APPROVER_PAIR_RECURRENT",
                "DENSE_MULTI_RELATION_CLUSTER": "DENSE_MULTI_RELATION_CLUSTER", "RAPID_ONBOARDING_LINKED": "RAPID_ONBOARDING_COMPONENT",
                "BRIDGE_ENTITY": "BRIDGE_ENTITY_HIGH_PARTICIPATION", "GEOGRAPHY_LINKED_CONCENTRATION": "GEOGRAPHY_CONCENTRATION",
                "ACTIVITY_TIMING_CONCENTRATION": "ACTIVITY_SYNCHRONIZED", "COMMON_BENEFICIAL_OWNER": "COMMON_BENEFICIAL_OWNER",
                "LAYERED_COMPLEX_OWNERSHIP": "OWNERSHIP_DEPTH_OUTLIER", "OWNERSHIP_OPERATIONAL_OVERLAP": "OWNERSHIP_OPERATIONAL_OVERLAP",
                "OPAQUE_JURISDICTION_ROUTING": "OWNERSHIP_PATH_LIMITED_TRANSPARENCY",
                "OWNERSHIP_CHANGE_STRUCTURAL_SHIFT": "OWNERSHIP_CHANGE_STRUCTURAL_SHIFT"}  # §10.2 examples where given


def _members(s):
    return frozenset(s.split("|"))


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 0.


# ----------------------------------------------------------------------------- 1. period
def emerged(findings_t1, findings_t0, onboarding, t0, cfg=CASE_CONFIG):
    """Findings at t1 that were absent at t0, or present but grown with members onboarded after t0."""
    if findings_t0 is None:
        return findings_t1.assign(emergence="present_at_t1")
    keep, how = [], []
    old = {d: [(_members(r.members), r.main_value) for r in g.itertuples()] for d, g in findings_t0.groupby("detector")}
    for r in findings_t1.itertuples():
        m = _members(r.members)
        matches = [(jaccard(m, om), ov) for om, ov in old.get(r.detector, []) if jaccard(m, om) >= cfg["emerge_overlap"]]
        if not matches:
            keep.append(True); how.append("new_since_t0")
            continue
        new_members = [x for x in m if onboarding.get(x, pd.NaT) > t0]
        grown = new_members and all(pd.notna(r.main_value) and pd.notna(ov) and float(r.main_value) > float(ov) for _, ov in matches)
        keep.append(bool(grown)); how.append("grown_since_t0" if grown else "existed_at_t0")
    return findings_t1.assign(emergence=how)[keep]


# ----------------------------------------------------------------------------- 2. cases
def assemble_cases(findings, cfg=CASE_CONFIG):
    f = findings.copy()
    f["tag"] = [RECORD_TAGS.get(t, t) if d == "R" else t for d, t in zip(f.detector, f.typology)]
    f = f.sort_values(["score", "robust_z", "n_members"], ascending=[False, False, False], na_position="last")
    cases = []
    for r in f.itertuples():
        m = _members(r.members)
        best, best_j = None, 0.
        for c in cases:
            j = jaccard(m, c["core"])
            if j > best_j:
                best, best_j = c, j
        if best is not None and best_j >= cfg["merge_overlap"]:
            best["findings"].append(r)
        else:
            cases.append(dict(core=m, findings=[r]))
    rows = []
    for c in cases:
        seed = c["findings"][0]
        counts = pd.Series([x for fnd in c["findings"] for x in _members(fnd.members)]).value_counts()
        members = set(c["core"]) | set(counts[counts >= 2].index)
        tags = [seed.tag] + [fnd.tag for fnd in c["findings"][1:] if fnd.score >= cfg["tag_min_score"]]
        tags = list(dict.fromkeys(tags))
        score = min(1.0, seed.score + cfg["multi_typology_bonus"] * (len(tags) - 1))
        reasons = []
        for fnd in c["findings"]:
            code = REASON_CODES.get(fnd.typology, fnd.typology)
            if f"{code}:" not in " ".join(reasons):
                extra = f" [also: {fnd.corroboration}]" if fnd.corroboration else ""
                caution = f" [benign to check: {fnd.benign}]" if fnd.benign else ""
                reasons.append(f"{code}: {fnd.reason}{extra}{caution}")
        rows.append(dict(member_record_ids="|".join(sorted(members)), typology_tags="|".join(tags), score=round(score, 4),
                         tiebreak=sum(fnd.score for fnd in c["findings"]), n_findings=len(c["findings"]),
                         finding_ids="|".join(fnd.finding_id for fnd in c["findings"]), reason_codes=" || ".join(reasons),
                         emergence=getattr(seed, "emergence", None)))
    out = pd.DataFrame(rows).sort_values(["score", "tiebreak"], ascending=False).head(cfg["max_cases"]).reset_index(drop=True)
    out.insert(0, "case_rank", np.arange(1, len(out) + 1))
    return out


def submission(cases):
    """The template columns only: case_rank, member_record_ids, typology_tags, score, reason_codes."""
    return cases[["case_rank", "member_record_ids", "typology_tags", "score", "reason_codes"]]


def build_period(period, data_dir=gb.DEFAULT_DATA_DIR, detector_config=None, case_config=None):
    """Graph(s) -> findings -> emerged findings -> ranked cases, for one sponsor period. No labels involved."""
    cfg = {**CASE_CONFIG, **(case_config or {})}
    t0, t1 = PERIODS[period]
    g1 = gb.build_graph(data_dir, t1)
    f1 = dt.run_detectors(g1, data_dir, detector_config)
    f0 = dt.run_detectors(gb.build_graph(data_dir, t0), data_dir, detector_config) if t0 is not None else None
    onboarding = g1.attributes["third_party"].onboarding_date.rename(lambda i: i.split(":", 1)[1])
    em = emerged(f1, f0, onboarding, t0, cfg)
    if t0 is not None:
        em = em[~context_only(em, g1, onboarding, t0)]
    return assemble_cases(em, cfg), em, g1


OWNERSHIP_STATE_DETECTORS = {"T10", "T11", "T13"}


def context_only(findings, graph, onboarding, t0):
    """Ownership-state findings that are 'new' only because dated context changed (e.g. a new index edition):
    no member onboarded after t0 and none of their ownership statements was observed after t0."""
    e = graph.edges
    observed = e.set_index(e.source_id).observed_at
    latest = {}
    for r in graph.ownership_paths.itertuples():
        dates = [observed.get(sid) for sid in r.statements.split(";") if sid]
        dates = [d for d in dates if isinstance(d, pd.Timestamp)]
        if dates:
            latest[r.record_id] = max(latest.get(r.record_id, pd.Timestamp.min), max(dates))
    flags = []
    for r in findings.itertuples():
        members = r.members.split("|")
        fresh = any(onboarding.get(m, pd.NaT) > t0 or latest.get(m, pd.Timestamp.min) > t0 for m in members)
        flags.append(r.detector in OWNERSHIP_STATE_DETECTORS and not fresh)
    return pd.Series(flags, index=findings.index)


# ----------------------------------------------------------------------------- 3. evaluation (labels read here only)
def planted_cases(period, t1, present, data_dir=gb.DEFAULT_DATA_DIR, periods_file=ROOT / "results" / "case_periods.csv"):
    cases = pd.read_csv(Path(data_dir) / "labelled_cases_TRAINING_SET.csv").merge(pd.read_csv(periods_file)[["case_id", "period"]], on="case_id")
    cases = cases[cases.period == period].copy()
    cases["members"] = cases.member_record_ids.map(lambda s: frozenset(s.split("|")) & present)  # onboarded after t1: excluded
    return cases[cases.members.map(len) > 0]


def evaluate(cases, planted, match_iou=0.5, ks=(10, 25, 50, 100)):
    planted = planted.reset_index(drop=True)
    unmatched = set(planted.index)
    hits = []
    for r in cases.itertuples():
        m = _members(r.member_record_ids)
        cand = [(jaccard(m, planted.members[i]), i) for i in unmatched]
        iou, i = max(cand, default=(0., None))
        if i is not None and iou >= match_iou:
            unmatched.discard(i)
            p = planted.loc[i]
            tags = set(r.typology_tags.split("|"))
            hits.append(dict(rank=r.case_rank, case_id=p.case_id, iou=iou, member_recall=len(m & p.members) / len(p.members),
                             exact=m == p.members, primary_tag_correct=r.typology_tags.split("|")[0] in set(p.typology_tags.split("|")),
                             tag_jaccard=jaccard(tags, set(p.typology_tags.split("|")))))
    hits = pd.DataFrame(hits, columns=["rank", "case_id", "iou", "member_recall", "exact", "primary_tag_correct", "tag_jaccard"])
    n = len(planted)
    is_hit = cases.case_rank.isin(hits["rank"]).to_numpy()
    metrics = dict(planted_cases=n, submitted_cases=len(cases), matched=len(hits), match_iou=match_iou)
    for k in ks:
        metrics[f"P@{k}"] = round(is_hit[:k].sum() / k, 4)
        metrics[f"R@{k}"] = round(is_hit[:k].sum() / n, 4) if n else np.nan
    precision_at_hits = np.cumsum(is_hit)[is_hit] / (np.flatnonzero(is_hit) + 1)
    metrics["AP"] = round(precision_at_hits.sum() / n, 4) if n else np.nan
    metrics["first_hit_rank"] = int(hits["rank"].min()) if len(hits) else None
    metrics["mean_IoU"] = round(hits.iou.mean(), 4) if len(hits) else None
    metrics["member_recall"] = round(hits.member_recall.mean(), 4) if len(hits) else None
    metrics["exact_recovery"] = round(hits.exact.mean(), 4) if len(hits) else None
    metrics["tag_accuracy_primary"] = round(hits.primary_tag_correct.mean(), 4) if len(hits) else None
    metrics["tag_jaccard"] = round(hits.tag_jaccard.mean(), 4) if len(hits) else None
    metrics["false_positives_top50"] = int((~is_hit[:50]).sum())
    matched_ids = set(hits.case_id)
    by_tag = {}
    for tag in sorted({t for ts in planted.typology_tags for t in ts.split("|")}):
        sub = planted[planted.typology_tags.str.split("|").map(lambda ts: tag in ts)]
        by_tag[tag] = round(sub.case_id.isin(matched_ids).mean(), 4)
    metrics["recall_by_typology"] = by_tag
    metrics["recall_macro"] = round(np.mean(list(by_tag.values())), 4) if by_tag else None
    metrics["recall_worst_typology"] = min(by_tag.values()) if by_tag else None
    metrics["recall_by_difficulty"] = planted.assign(hit=planted.case_id.isin(matched_ids)).groupby("difficulty_tier").hit.mean().round(4).to_dict()
    per_case = planted[["case_id", "typology_tags", "difficulty_tier"]].merge(hits, on="case_id", how="left")
    return metrics, per_case


def main():
    parser = argparse.ArgumentParser(description="Build ranked review cases for a sponsor period; optionally self-evaluate")
    parser.add_argument("--period", choices=list(PERIODS), required=True)
    parser.add_argument("--evaluate", action="store_true", help="score against the starter labels for this period")
    args = parser.parse_args()
    cases, findings, g1 = build_period(args.period)
    out = ROOT / "results" / "submissions" / args.period
    out.mkdir(parents=True, exist_ok=True)
    submission(cases).to_csv(out / "submission.csv", index=False)
    cases.to_csv(out / "cases_with_trace.csv", index=False)
    findings.to_csv(out / "findings_emerged.csv", index=False)
    (out / "config.json").write_text(json.dumps(dict(period=args.period, t=[str(x) for x in PERIODS[args.period]], case=CASE_CONFIG,
                                                      detectors=dt.CONFIG, record_tags=RECORD_TAGS, graph_hash=g1.graph_hash()),
                                                 indent=2, default=str))
    print(f"{len(cases)} cases from {len(findings)} emerged findings -> {out}")
    if args.evaluate:
        present = set(g1.attributes["third_party"].index.str.split(":", n=1).str[1])
        planted = planted_cases(args.period, PERIODS[args.period][1], present)
        results = {}
        for iou in (0.5, 0.3):
            metrics, per_case = evaluate(cases, planted, iou)
            results[f"IoU>={iou}"] = metrics
            per_case.to_csv(out / f"evaluation_cases_iou{iou}.csv", index=False)
        (out / "evaluation.json").write_text(json.dumps(results, indent=2, default=str))
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
