"""Robustness of the pipeline (dossier §8.2, §8.3, §9.2): one-at-a-time parameter variation and
leave-one-typology-out evaluation, on the TRAINING and VALIDATION periods only (TEST stays untouched).

  python scripts/layer3/layer3_stability.py --what sensitivity    -> results/stability/sensitivity.csv
  python scripts/layer3/layer3_stability.py --what loto           -> results/stability/leave_one_typology_out.csv

Leave-one-typology-out: switch off the detector written for typology X and ask whether X's planted cases are
still recovered (top 100, IoU >= 0.5) by the other detectors, with and without the open-set detector. That is
the closest available estimate of how the pipeline does on a pattern it has no rule for.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repository root (this file is in scripts/layer3/)
sys.path[:0] = [str(ROOT / "scripts" / "layer2"), str(ROOT / "scripts" / "layer3")]  # graph_build is shared with Layer 2
import graph_build as gb
import layer3_detectors as dt
import layer3_cases as cs

OUT = ROOT / "results" / "stability"
EVAL_PERIODS = ["TRAINING", "VALIDATION"]
TYPOLOGY_DETECTOR = {"SHARED_ADDRESS": "T1", "SPONSOR_CONCENTRATION": "T2", "APPROVER_CONCENTRATION": "T3",
                     "SPONSOR_APPROVER_COUPLING": "T4", "DENSE_MULTI_RELATION_CLUSTER": "T5", "RAPID_ONBOARDING_LINKED": "T6",
                     "BRIDGE_ENTITY": "T7", "GEOGRAPHY_LINKED_CONCENTRATION": "T8", "ACTIVITY_TIMING_CONCENTRATION": "T9",
                     "COMMON_BENEFICIAL_OWNER": "T10", "LAYERED_COMPLEX_OWNERSHIP": "T11", "OWNERSHIP_OPERATIONAL_OVERLAP": "T12",
                     "OPAQUE_JURISDICTION_ROUTING": "T13", "OWNERSHIP_CHANGE_STRUCTURAL_SHIFT": "T14"}

DETECTOR_VARIANTS = {
    "spike_ratio": [2.0, 4.0], "sponsor_window_days": [30, 180], "rapid_window_days": [14, 60],
    "onboarding_window_days": [30, 180], "min_peer_size": [5, 12], "louvain_resolution": [0.5, 2.0],
    "louvain_seed": [1, 2], "open_set_seed": [1, 2], "open_set_quantile": [0.95, 0.99],
    "corroboration_step": [0.05, 0.2], "benign_penalty": [0.15, 0.5], "cluster_min_relation_types": [2, 4],
}
CASE_VARIANTS = {"merge_overlap": [0.3, 0.7], "tag_min_score": [0.5, 0.8], "multi_typology_bonus": [0.0, 0.1]}
GRAPH_VARIANTS = {"er_conflict_override_evidence": [2]}


class Runner:
    """Caches graphs per (t, graph config) and findings per (t, graph config, detector config)."""

    def __init__(self):
        self.graphs, self.findings = {}, {}

    def graph(self, t, gcfg):
        key = (t, json.dumps(gcfg, sort_keys=True))
        if key not in self.graphs:
            self.graphs[key] = gb.build_graph(gb.DEFAULT_DATA_DIR, t, gcfg or None)
        return self.graphs[key]

    def run(self, t, gcfg, dcfg, exclude=()):
        key = (t, json.dumps(gcfg, sort_keys=True), json.dumps(dcfg, sort_keys=True), tuple(sorted(exclude)))
        if key not in self.findings:
            only = set(dt.DETECTORS) - set(exclude)
            self.findings[key] = dt.run_detectors(self.graph(t, gcfg), gb.DEFAULT_DATA_DIR, dcfg or None, only)
        return self.findings[key]

    def period_cases(self, period, gcfg=None, dcfg=None, ccfg=None, exclude=()):
        cfg = {**cs.CASE_CONFIG, **(ccfg or {})}
        t0, t1 = cs.PERIODS[period]
        g1 = self.graph(t1, gcfg)
        f1 = self.run(t1, gcfg, dcfg, exclude)
        f0 = self.run(t0, gcfg, dcfg, exclude) if t0 is not None else None
        onboarding = g1.attributes["third_party"].onboarding_date.rename(lambda i: i.split(":", 1)[1])
        em = cs.emerged(f1, f0, onboarding, t0, cfg)
        if t0 is not None:
            em = em[~cs.context_only(em, g1, onboarding, t0)]
        return cs.assemble_cases(em, cfg), g1

    def evaluate(self, period, **kw):
        cases, g1 = self.period_cases(period, **kw)
        present = set(g1.attributes["third_party"].index.str.split(":", n=1).str[1])
        planted = cs.planted_cases(period, cs.PERIODS[period][1], present)
        return cs.evaluate(cases, planted, 0.5)


def sensitivity(runner):
    rows = []
    variants = [("baseline", None, None, None, None)]
    variants += [(k, v, {k: v}, None, None) for k, vs in DETECTOR_VARIANTS.items() for v in vs]
    variants += [(k, v, None, {k: v}, None) for k, vs in CASE_VARIANTS.items() for v in vs]
    variants += [(k, v, None, None, {k: v}) for k, vs in GRAPH_VARIANTS.items() for v in vs]
    for name, value, dcfg, ccfg, gcfg in variants:
        started = time.time()
        for period in EVAL_PERIODS:
            m, _ = runner.evaluate(period, gcfg=gcfg, dcfg=dcfg, ccfg=ccfg)
            rows.append(dict(parameter=name, value=value, period=period, matched=m["matched"], planted=m["planted_cases"],
                             AP=m["AP"], **{f"R@{k}": m[f"R@{k}"] for k in (10, 25, 50, 100)}, mean_IoU=m["mean_IoU"],
                             tag_accuracy=m["tag_accuracy_primary"], first_hit=m["first_hit_rank"]))
        print(f"{name}={value}: {time.time() - started:.0f}s", flush=True)
    return pd.DataFrame(rows)


def leave_one_typology_out(runner):
    rows = []
    settings = {"all_detectors": (), "without_X": None, "without_X_and_open_set": None}
    for typology, det in TYPOLOGY_DETECTOR.items():
        for setting in settings:
            exclude = () if setting == "all_detectors" else ((det,) if setting == "without_X" else (det, "O"))
            found = total = 0
            for period in EVAL_PERIODS:
                m, per_case = runner.evaluate(period, dcfg=None, exclude=exclude)
                sub = per_case[per_case.typology_tags.str.split("|").map(lambda ts: typology in ts)]
                total += len(sub)
                found += int((sub["rank"].notna() & (sub["rank"] <= 100)).sum())
            rows.append(dict(typology=typology, detector=det, setting=setting, planted=total, recovered_top100=found))
        print(typology, flush=True)
    out = pd.DataFrame(rows)
    return out.pivot_table(index=["typology", "detector", "planted"], columns="setting", values="recovered_top100").reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--what", choices=["sensitivity", "loto", "both"], default="both")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runner = Runner()
    if args.what in ("loto", "both"):
        loto = leave_one_typology_out(runner)
        loto.to_csv(OUT / "leave_one_typology_out.csv", index=False)
        print(loto.to_string(index=False))
    if args.what in ("sensitivity", "both"):
        sens = sensitivity(runner)
        sens.to_csv(OUT / "sensitivity.csv", index=False)
        print(sens.to_string(index=False))


if __name__ == "__main__":
    main()
