"""Detector hit-rate calibration from labelled cases.

Reads labels, so like scorer.py it sits on the evaluation side: nothing here
feeds candidate generation.

A case is a hit when some planted case overlaps it at the scorer threshold
(rank-free). Hit rates are Beta-smoothed per lead detector.

`crossfit` avoids in-sample optimism when calibrating and evaluating on the
same labels: planted cases are split into folds; a case that matches a planted
case in fold f is scored with rates estimated without fold f, and all other
cases with the mean over folds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cases import Case
from .scorer import ScorerConfig, overlap


def _best_match(case: Case, truth: pd.DataFrame, cfg: ScorerConfig):
    best, best_id = 0.0, None
    for t in truth.itertuples():
        o = overlap(case.members, t.members, cfg.measure)
        if o > best:
            best, best_id = o, t.case_id
    return best_id if best >= cfg.threshold else None


def _rates(cases, matched, exclude: set, a: float, b: float) -> dict:
    counts: dict[str, list] = {}
    for c, m in zip(cases, matched):
        if m in exclude:
            continue
        h = counts.setdefault(c.lead.detector, [0, 0])
        h[0] += m is not None
        h[1] += 1
    return {d: (h + a) / (n + a + b) for d, (h, n) in counts.items()}


def fit(cases: list[Case], truth: pd.DataFrame, cfg: ScorerConfig | None = None, a: float = 1.0,
        b: float = 1.0) -> dict:
    cfg = cfg or ScorerConfig()
    matched = [_best_match(c, truth, cfg) for c in cases]
    return _rates(cases, matched, set(), a, b)


def crossfit(cases: list[Case], truth: pd.DataFrame, cfg: ScorerConfig | None = None, k: int = 3,
             seed: int = 0, a: float = 1.0, b: float = 1.0) -> list[dict]:
    """Per-case hit-rate tables (one dict per case, aligned with `cases`)."""
    cfg = cfg or ScorerConfig()
    matched = [_best_match(c, truth, cfg) for c in cases]
    ids = np.array(sorted(truth["case_id"]))
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(ids), k)
    fold_of = {cid: f for f, fold in enumerate(folds) for cid in fold}
    tables = [_rates(cases, matched, set(folds[f]), a, b) for f in range(k)]
    detectors = {c.lead.detector for c in cases}
    mean_table = {d: float(np.mean([t.get(d, a / (a + b)) for t in tables])) for d in detectors}
    return [tables[fold_of[m]] if m is not None else mean_table for m in matched]


def save(rates: dict, path: str | Path, meta: dict | None = None) -> None:
    Path(path).write_text(json.dumps({"hit_rates": rates, "meta": meta or {}}, indent=2, default=str))


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())["hit_rates"]
