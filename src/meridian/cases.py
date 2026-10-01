"""Case assembly: merge candidates, emergence dates, benign explanations, window filter."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .detectors.base import Candidate


@dataclass
class Case:
    members: frozenset
    tags: tuple
    score: float                      # max raw detector score (not comparable across detectors)
    emerged_at: pd.Timestamp | None
    candidates: list = field(default_factory=list)
    support: list = field(default_factory=list)   # candidates inside the case that add tags but not members
    calibrated: float = 0.0
    benign_factor: float = 1.0
    rank: int | None = None                        # position in the submitted list, None if cut
    tag_scores: dict = field(default_factory=dict)  # tag -> strongest calibrated evidence for it

    @property
    def lead(self) -> Candidate:
        return self.candidates[0]

    @property
    def detectors(self) -> tuple:
        return tuple(sorted({c.detector for c in self.candidates + self.support}))

    @property
    def primary_tag(self) -> str:
        return self.lead.tags[0]

    @property
    def reason(self) -> str:
        parts = list(dict.fromkeys(c.reason for c in self.candidates + self.support))
        if self.benign:
            parts.append("benign to check: " + "; ".join(self.benign))
        return " | ".join(parts)

    @property
    def benign(self) -> list:
        return list(dict.fromkeys(b for c in self.candidates for b in c.benign))


def _iou(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b)


def _close(a, b, days: int) -> bool:
    if a is None or b is None:
        return True
    return abs((a - b).days) <= days


def _concurrent(case: "Case", c: Candidate, days: int) -> bool:
    """Same time frame: emergence dates within `days`, or one pattern was still
    holding when the other emerged. Two patterns on the same vendors whose
    periods do not meet (a layered structure that ended when the vendor was
    sold) stay separate cases."""
    if _close(case.emerged_at, c.emerged_at, days):
        return True
    for a in case.candidates:
        if a.timeless:
            continue
        first, second = (a, c) if (a.emerged_at or pd.Timestamp.min) <= (c.emerged_at or pd.Timestamp.min) else (c, a)
        if second.emerged_at is None or first.active_until is None or second.emerged_at < first.active_until:
            return True
    return False


def _percentiles(cands: list[Candidate]) -> dict[int, float]:
    by_det: dict[str, list] = {}
    for c in cands:
        by_det.setdefault(c.detector, []).append(c.score)
    return {id(c): sum(s <= c.score for s in by_det[c.detector]) / len(by_det[c.detector]) for c in cands}


def assemble(cands: list[Candidate], merge_iou: float = 0.8, union_iou: float = 0.5, max_growth: float = 1.5,
             merge_days: int = 365, support_days: int = 365) -> list[Case]:
    """Merge candidates that describe the same group at about the same time.

    Candidates are visited by within-detector percentile (raw scores are not
    comparable across detectors). A candidate joins an existing case when the
    emergence dates are within merge_days and either
      - IoU >= merge_iou: same members; the case keeps its boundary, or
      - IoU >= union_iou, both are groups, and they come from different
        detectors: the case takes the union of members. Planted multi-typology
        cases are unions of what several relations show.
    The time condition (`_concurrent`) keeps apart two patterns on the same
    vendor whose periods do not meet (on the starter data one vendor carries a
    layered-ownership case that ended when it was sold, and the ownership-change
    case of that sale), while an owner group and the address overlap it later
    develops stay one case. Tags are the union, the emergence date is
    the earliest, and every candidate's evidence is kept.

    Candidates whose members are a strict subset of a larger case (e.g. a
    single-record ownership finding inside an owner group) stay as their own
    case and also add their tag to the larger case as support, if
    they carry no benign flag and score at or above their detector's median.
    """
    pct = _percentiles(cands)
    cases: list[Case] = []
    # Timed candidates first, so timeless ones only attach to cases whose time frame is already set.
    for c in sorted(cands, key=lambda c: (c.timeless, -pct[id(c)], -c.score, c.detector, sorted(c.members))):
        if c.timeless:
            host = next((k for k in cases if _iou(k.members, c.members) >= merge_iou), None)
            if host is not None:
                host.candidates.append(c)
                host.tags = tuple(dict.fromkeys(host.tags + c.tags))
                continue
        for case in cases:
            if not _concurrent(case, c, merge_days):
                continue
            iou = _iou(case.members, c.members)
            # Union is judged against the lead candidate and capped, so that groups
            # sharing one hub (a sponsor, an owner) do not chain into one big case.
            lead = case.lead.members
            union = (_iou(lead, c.members) >= union_iou and len(c.members) > 1 and len(lead) > 1
                     and len(case.members | c.members) <= max_growth * len(lead)
                     and c.detector not in {x.detector for x in case.candidates})
            if iou >= merge_iou or union:
                case.candidates.append(c)
                if union and iou < merge_iou:
                    case.members = case.members | c.members
                case.tags = tuple(dict.fromkeys(case.tags + c.tags))
                case.score = max(case.score, c.score)
                if c.emerged_at is not None and (case.emerged_at is None or c.emerged_at < case.emerged_at):
                    case.emerged_at = c.emerged_at
                break
        else:
            cases.append(Case(members=c.members, tags=c.tags, score=c.score,
                              emerged_at=c.emerged_at, candidates=[c]))
    for case in cases:
        if len(case.members) < 2:
            continue
        for c in cands:
            if (c.members < case.members and c not in case.candidates and not c.benign and pct[id(c)] >= 0.5
                    and _close(case.emerged_at, c.emerged_at, support_days)):
                case.support.append(c)
                case.tags = tuple(dict.fromkeys(case.tags + c.tags))
    return cases


def benign_weight(benign: list, per_item: float = 0.7, floor: float = 0.3) -> float:
    """Down-weight for documented benign explanations. Signals are never removed."""
    return max(floor, per_item ** len(benign))


def in_window(cases: list[Case], window) -> list[Case]:
    """Keep cases whose emergence date falls in [start, end]. Undated cases are dropped."""
    if not window:
        return cases
    lo, hi = (pd.Timestamp(x) for x in window)
    return [c for c in cases if c.emerged_at is not None and lo <= c.emerged_at <= hi]
