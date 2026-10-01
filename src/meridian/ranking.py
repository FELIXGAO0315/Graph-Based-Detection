"""Unified ranking: calibration, type interleaving, submission table.

Raw detector scores have different scales. Each candidate inside a case gets
    s = hit_rate(detector) * (0.5 + 0.5 * within-detector percentile) * benign weight
where hit_rate is either a fixed prior (no labels) or estimated from labelled
cases by `calibrate.py`, and the benign weight only reflects explanations for
that candidate's own signal (a coworking address does not explain an ownership
convergence). The case takes its best candidate plus part of the noisy-OR gain
from other detectors that corroborate it.

The ranked list then favours type diversity: a case's effective score decays
with the number of already selected cases of the same primary tag, and is
halved when it nearly duplicates a selected case.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cases import Case, _iou, benign_weight

SUBMISSION_COLUMNS = ["case_rank", "member_record_ids", "typology_tags", "score", "reason_codes"]

# Label-free starting point: every detector gets the same prior, so the order
# comes only from within-detector strength, benign factors and interleaving.
DEFAULT_PRIOR = 0.5


def percentiles(cases: list[Case]) -> dict[int, float]:
    """Within-detector percentile of each candidate's raw score, over all cases."""
    by_det: dict[str, list] = {}
    for case in cases:
        for c in case.candidates:
            by_det.setdefault(c.detector, []).append(c.score)
    arrays = {d: np.array(v) for d, v in by_det.items()}
    return {id(c): float((arrays[c.detector] <= c.score).mean()) if len(arrays[c.detector]) > 1 else 1.0
            for case in cases for c in case.candidates}


def calibrate_cases(cases: list[Case], hit_rates: dict | None = None, prior: float = DEFAULT_PRIOR,
                    per_case: list[dict] | None = None, corroboration: float = 0.5,
                    detector_priors: dict | None = None) -> None:
    """Set `calibrated` and `tag_scores` on every case. `per_case` (aligned with cases) overrides `hit_rates`."""
    hit_rates = hit_rates or {}
    pct = percentiles(cases)
    for i, case in enumerate(cases):
        table = per_case[i] if per_case is not None else {**(detector_priors or {}), **hit_rates}
        per_det: dict[str, tuple] = {}
        case.tag_scores = {}
        for c in case.candidates + case.support:
            w = benign_weight(c.benign)
            v = table.get(c.detector, prior) * (0.5 + 0.5 * pct.get(id(c), 0.5)) * w
            for t in c.tags:
                case.tag_scores[t] = max(case.tag_scores.get(t, 0.0), v)
            if c in case.support:
                continue
            if v > per_det.get(c.detector, (0.0, 1.0))[0]:
                per_det[c.detector] = (v, w)
        vals = sorted(per_det.values(), reverse=True)
        best, best_w = vals[0]
        noisy_or = 1 - float(np.prod([1 - v for v, _ in vals]))
        case.calibrated = best + corroboration * (noisy_or - best)
        case.benign_factor = best_w


def order(cases: list[Case], interleave: bool = True, decay: float = 0.85, dup_penalty: float = 0.5,
          dup_days: int = 365, contained: float = 0.6, max_cases: int | None = None) -> list[Case]:
    remaining = sorted(cases, key=lambda c: (-c.calibrated, sorted(c.members)))
    if not interleave:
        return remaining[:max_cases] if max_cases else remaining
    chosen: list[Case] = []
    tag_count: dict[str, int] = {}
    limit = max_cases or len(remaining)
    while remaining and len(chosen) < limit:
        best_i, best_v = 0, -1.0
        for i, c in enumerate(remaining):
            v = c.calibrated * decay ** tag_count.get(c.primary_tag, 0)
            # Near-duplicates and fragments of a chosen case add little for a reviewer
            # (and cannot match a second planted case under one-to-one matching).
            if any((_iou(c.members, s.members) >= 0.5 or len(c.members & s.members) >= contained * len(c.members)) and
                   (c.emerged_at is None or s.emerged_at is None or abs((c.emerged_at - s.emerged_at).days) <= dup_days)
                   for s in chosen):
                v *= dup_penalty
            if v > best_v:
                best_i, best_v = i, v
        c = remaining.pop(best_i)
        chosen.append(c)
        tag_count[c.primary_tag] = tag_count.get(c.primary_tag, 0) + 1
    return chosen


def select_tags(case: Case, ratio: float = 0.85) -> tuple:
    """Tags whose strongest evidence is within `ratio` of the case's strongest, strongest first.

    OPEN_SET is a fallback label: it is dropped when a typed tag remains.
    """
    from .detectors.base import OPEN_SET_TAG
    ts = case.tag_scores or {t: 1.0 for t in case.tags}
    best = max(ts.values())
    keep = sorted((t for t, v in ts.items() if v >= ratio * best), key=lambda t: -ts[t])
    typed = [t for t in keep if t != OPEN_SET_TAG]
    return tuple(typed or keep)


def to_submission(ordered: list[Case], tag_ratio: float = 0.85) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(ordered, start=1):
        rows.append({
            "case_rank": i,
            "member_record_ids": "|".join(sorted(c.members)),
            "typology_tags": "|".join(select_tags(c, tag_ratio)),
            "all_tags": "|".join(c.tags),
            "score": round(float(c.calibrated), 4),
            "reason_codes": c.reason,
            "detectors": "|".join(c.detectors),
            "emerged_at": c.emerged_at.date().isoformat() if c.emerged_at is not None else "",
            "raw_score": round(float(c.score), 3),
            "benign_factor": round(c.benign_factor, 3),
        })
    return pd.DataFrame(rows, columns=SUBMISSION_COLUMNS + ["all_tags", "detectors", "emerged_at", "raw_score",
                                                            "benign_factor"])


def rank(cases: list[Case], cfg: dict | None = None, hit_rates: dict | None = None,
         per_case: list[dict] | None = None) -> pd.DataFrame:
    cfg = cfg or {}
    calibrate_cases(cases, hit_rates, cfg.get("prior", DEFAULT_PRIOR), per_case, cfg.get("corroboration", 0.5),
                    cfg.get("detector_priors"))
    ordered = order(cases, interleave=cfg.get("interleave", True), decay=cfg.get("decay", 0.85),
                    dup_penalty=cfg.get("dup_penalty", 0.5), contained=cfg.get("contained", 0.6),
                    max_cases=cfg.get("max_cases"))
    for c in cases:
        c.rank = None
    for i, c in enumerate(ordered, start=1):
        c.rank = i
    return to_submission(ordered, cfg.get("tag_ratio", 0.85))
