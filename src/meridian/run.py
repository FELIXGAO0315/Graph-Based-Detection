"""Command-line entry point: one command from raw data to submission and local score.

    meridian --config configs/default.yaml [--out artifacts/runs/NAME] [--set key=value ...]
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import pandas as pd
import yaml

from . import __version__
from .cases import assemble, in_window
from .checks import run_checks, to_markdown
from .context import Context
from .data import as_of, load_bundle
from .detectors import run_detectors
from .ranking import SUBMISSION_COLUMNS, rank

ROOT = Path(__file__).resolve().parents[2]


def load_config(path, overrides=()) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    for item in overrides:
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _calibration(cases, cfg: dict):
    """Return (hit_rates, per_case) for the configured calibration mode.

    prior     no labels: every detector gets the same prior
    file      hit rates fitted earlier and saved with calibrate.save
    insample  fit on the configured labels and rank with the same labels (optimistic)
    crossfit  fit on the labels, cross-fitted by planted case (for honest starter evaluation)
    """
    rcfg = cfg.get("ranking", {})
    mode = rcfg.get("calibration", "prior")
    if mode == "prior":
        return None, None
    from . import calibrate
    if mode == "file":
        return calibrate.load(ROOT / rcfg["calibration_file"]), None
    from .scorer import ScorerConfig, load_labels
    truth = load_labels(ROOT / cfg["labels"])
    scfg = ScorerConfig(**cfg.get("scorer", {}))
    if mode == "insample":
        return calibrate.fit(cases, truth, scfg), None
    if mode == "crossfit":
        return None, calibrate.crossfit(cases, truth, scfg, k=rcfg.get("folds", 3), seed=cfg.get("seed", 0))
    raise ValueError(f"unknown calibration mode {mode}")


def pipeline(cfg: dict, bundle=None):
    """Run detection end to end. Returns (submission, cases, candidates, context, checks)."""
    b = bundle if bundle is not None else load_bundle(ROOT / cfg["bundle"])
    checks = run_checks(b)
    view = as_of(b, cfg["snapshot"])
    ctx = Context(view, cfg)
    cands = run_detectors(ctx, cfg.get("detectors", "all"), cfg.get("exclude_detectors", ()))
    if not cfg.get("ranking", {}).get("include_policy", True):
        from .detectors.base import POLICY_TAG
        cands = [c for c in cands if POLICY_TAG not in c.tags]
    cases = assemble(cands, **cfg.get("cases", {}))
    cases = in_window(cases, cfg.get("window"))
    hit_rates, per_case = _calibration(cases, cfg)
    sub = rank(cases, cfg.get("ranking", {}), hit_rates, per_case)
    return sub, cases, cands, ctx, checks


def evaluate(sub: pd.DataFrame, cands, cfg: dict) -> dict:
    from .scorer import ScorerConfig, coverage, load_labels, prepare_submission, score

    truth = load_labels(ROOT / cfg["labels"])
    scfg = ScorerConfig(**cfg.get("scorer", {}))
    res = score(prepare_submission(sub[SUBMISSION_COLUMNS]), truth, scfg)
    cov = coverage([c.members for c in cands], truth, scfg)
    return {"score": res, "coverage": cov}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--out", default=None, help="output directory (default artifacts/runs/<timestamp>)")
    ap.add_argument("--set", action="append", default=[], help="override a config key, e.g. features.ownership=false")
    ap.add_argument("--no-score", action="store_true", help="skip the local scorer")
    ap.add_argument("--packages", type=int, default=10, help="write analyst case packages for the top N cases")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.set)
    t0 = time.time()
    sub, cases, cands, ctx, checks = pipeline(cfg)
    runtime = time.time() - t0

    out = Path(a.out) if a.out else ROOT / "artifacts" / "runs" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    sub[SUBMISSION_COLUMNS].to_csv(out / "submission.csv", index=False)
    sub.to_csv(out / "ranked_cases.csv", index=False)
    (out / "checks.md").write_text(to_markdown(checks))
    # Resolution lists for manual review.
    ctx.reg_matches.to_csv(out / "registry_matches.csv", index=False)
    ctx.identity_pairs.to_csv(out / "identity_pairs.csv", index=False)
    ctx.entity_map[ctx.entity_map["n_records"] > 1].to_csv(out / "resolved_entities.csv", index=False)
    with open(out / "cases.jsonl", "w") as fh:
        for c in cases:
            fh.write(json.dumps({
                "members": sorted(c.members), "tags": list(c.tags), "score": c.score,
                "emerged_at": c.emerged_at.date().isoformat() if c.emerged_at is not None else None,
                "detectors": list(c.detectors), "reason": c.reason, "benign": c.benign,
                "calibrated": c.calibrated, "benign_factor": c.benign_factor,
                "evidence": [cd.evidence for cd in c.candidates + c.support],
            }, default=str) + "\n")

    meta = {
        "version": __version__, "git_commit": _git_commit(), "config": cfg,
        "input_hashes": ctx.b.input_hashes, "snapshot": str(ctx.t.date()),
        "runtime_sec": round(runtime, 2), "python": platform.python_version(),
        "pandas": pd.__version__, "n_candidates": len(cands), "n_cases": len(cases),
        "n_submitted": len(sub),
        "checks": checks["status"].value_counts().to_dict(),
    }
    print(f"candidates={len(cands)} cases={len(cases)} submitted={len(sub)} runtime={runtime:.1f}s -> {out}")

    # The canonical heterogeneous graph behind this run, with its content hash.
    from .graph import build_graph, export, validate
    g = build_graph(ctx)
    meta["graph"] = export(g, out / "graph")
    problems = validate(g)
    meta["graph"]["problems"] = problems
    print(f"graph {meta['graph']['graph_hash']}: {len(g.nodes)} nodes, {len(g.edges)} edges"
          + (f"; problems: {problems}" if problems else "") + f" -> {out / 'graph'}")

    if a.packages:
        from .explain import config_hash, write_packages
        ordered = sorted((c for c in cases if c.rank is not None), key=lambda c: c.rank)
        pk = write_packages(ctx, ordered, out / "cases", {
            "snapshot": str(ctx.t.date()), "run": out.name, "config_hash": config_hash(cfg),
            "git_commit": meta["git_commit"], "input_hashes": ctx.b.input_hashes,
            "graph_hash": meta["graph"]["graph_hash"]}, top=a.packages)
        print(f"case packages for the top {min(a.packages, len(ordered))} cases -> {pk}")

    if not a.no_score and cfg.get("labels"):
        from .scorer import summary_line
        ev = evaluate(sub, cands, cfg)
        res = ev["score"]
        cov = ev["coverage"]
        cov.to_csv(out / "coverage.csv", index=False)
        res.pop("_matches").to_csv(out / "matches.csv", index=False)
        meta["score"] = res
        meta["coverage"] = float(cov["covered"].mean())
        print("[dev scorer, starter labels]", summary_line(res))
        print(f"candidate coverage={cov['covered'].mean():.3f} "
              f"({int(cov['covered'].sum())}/{len(cov)} planted cases reachable by some candidate)")
    (out / "run.json").write_text(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
