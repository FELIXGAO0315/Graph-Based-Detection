"""Local development scorer.

Dossier §9.1 describes the official matching only in words: work down the
submitted ranking and match each submitted case one-to-one to a planted case
whose member set overlaps "sufficiently". The overlap measure and threshold
here are our own, public and configurable; results are labelled as coming
from the development scorer.

This is the only module that reads labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

K_VALUES = (10, 25, 50, 100)


def split_ids(s) -> list[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    return [x for x in str(s).split("|") if x]


def load_labels(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df = df.copy()
    df["members"] = df["member_record_ids"].map(lambda s: frozenset(split_ids(s)))
    df["tags"] = df["typology_tags"].map(lambda s: frozenset(split_ids(s)))
    return df


def read_submission(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return prepare_submission(df)


def prepare_submission(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df["case_rank"].duplicated().any():
        raise ValueError("submission ranks must be unique")
    df = df.sort_values("case_rank").reset_index(drop=True)
    df["members"] = df["member_record_ids"].map(lambda s: frozenset(split_ids(s)))
    df["tags"] = df["typology_tags"].map(lambda s: frozenset(split_ids(s)))
    return df


def overlap(a: frozenset, b: frozenset, measure: str) -> float:
    inter = len(a & b)
    if inter == 0:
        return 0.0
    if measure == "iou":
        return inter / len(a | b)
    if measure == "overlap_coef":
        return inter / min(len(a), len(b))
    raise ValueError(measure)


@dataclass
class ScorerConfig:
    measure: str = "iou"
    threshold: float = 0.5
    k_values: tuple = K_VALUES


def match(sub: pd.DataFrame, truth: pd.DataFrame, cfg: ScorerConfig) -> pd.DataFrame:
    """Greedy one-to-one matching down the submitted ranking.

    Each submitted case takes the still-unmatched planted case with the highest
    overlap at or above the threshold (ties broken by case_id). A planted case
    matched once cannot be matched again, so resubmitting it scores nothing.
    """
    truth_members = list(truth["members"])
    truth_ids = list(truth["case_id"])
    taken: set[int] = set()
    rows = []
    for r in sub.itertuples():
        best, best_j = 0.0, None
        for j, tm in enumerate(truth_members):
            if j in taken:
                continue
            o = overlap(r.members, tm, cfg.measure)
            if o >= cfg.threshold and (o > best or (o == best and best_j is not None
                                                     and truth_ids[j] < truth_ids[best_j])):
                best, best_j = o, j
        if best_j is not None:
            taken.add(best_j)
        rows.append({"case_rank": r.case_rank, "matched_case": truth_ids[best_j] if best_j is not None else None,
                     "overlap": best if best_j is not None else 0.0})
    return pd.DataFrame(rows)


def score(sub: pd.DataFrame, truth: pd.DataFrame, cfg: ScorerConfig | None = None) -> dict:
    cfg = cfg or ScorerConfig()
    sub = sub.sort_values("case_rank").reset_index(drop=True)
    m = match(sub, truth, cfg)
    n_truth = len(truth)
    hit = m["matched_case"].notna().to_numpy()
    out: dict = {"n_submitted": len(sub), "n_planted": n_truth,
                 "measure": cfg.measure, "threshold": cfg.threshold}

    for k in cfg.k_values:
        h = hit[:k].sum()
        out[f"P@{k}"] = h / k
        out[f"R@{k}"] = h / n_truth if n_truth else np.nan
        if len(sub) < k:
            out[f"short@{k}"] = True  # fewer cases than k: reported, never padded

    # Average precision over the ranked list, normalised by all planted cases.
    prec_at_hits = [hit[: i + 1].sum() / (i + 1) for i in np.flatnonzero(hit)]
    out["AP"] = float(np.sum(prec_at_hits) / n_truth) if n_truth else np.nan
    out["first_hit_rank"] = int(sub.loc[np.flatnonzero(hit)[0], "case_rank"]) if hit.any() else None
    out["recall"] = hit.sum() / n_truth if n_truth else np.nan
    out["fp_top50"] = int((~hit[:50]).sum())

    truth_by_id = truth.set_index("case_id")
    matched = m[hit].merge(sub[["case_rank", "members", "tags"]], on="case_rank")
    ious, mrec, exact, tag_exact, tag_jacc, tag_any = [], [], [], [], [], []
    for r in matched.itertuples():
        t = truth_by_id.loc[r.matched_case]
        ious.append(overlap(r.members, t.members, "iou"))
        mrec.append(len(r.members & t.members) / len(t.members))
        exact.append(r.members == t.members)
        tag_exact.append(r.tags == t.tags)
        tag_jacc.append(len(r.tags & t.tags) / len(r.tags | t.tags) if (r.tags | t.tags) else 0.0)
        tag_any.append(bool(r.tags & t.tags))
    out["mean_iou"] = float(np.mean(ious)) if ious else np.nan
    out["member_recall"] = float(np.mean(mrec)) if mrec else np.nan
    out["exact_recovery"] = float(np.mean(exact)) if exact else np.nan
    out["tag_exact"] = float(np.mean(tag_exact)) if tag_exact else np.nan
    out["tag_jaccard"] = float(np.mean(tag_jacc)) if tag_jacc else np.nan
    out["tag_any"] = float(np.mean(tag_any)) if tag_any else np.nan

    matched_ids = set(m["matched_case"].dropna())
    by_tag = {}
    for tag in sorted({t for ts in truth["tags"] for t in ts}):
        cases = truth[truth["tags"].map(lambda ts: tag in ts)]
        by_tag[tag] = cases["case_id"].isin(matched_ids).mean()
    out["recall_by_tag"] = by_tag
    out["macro_tag_recall"] = float(np.mean(list(by_tag.values()))) if by_tag else np.nan
    out["worst_tag_recall"] = float(min(by_tag.values())) if by_tag else np.nan
    if "difficulty_tier" in truth.columns:
        out["recall_by_difficulty"] = truth.groupby("difficulty_tier")["case_id"].apply(
            lambda s: s.isin(matched_ids).mean()).to_dict()
    out["_matches"] = m
    return out


def coverage(candidates: list[frozenset], truth: pd.DataFrame, cfg: ScorerConfig | None = None) -> pd.DataFrame:
    """Rank-free coverage: can any candidate match each planted case?

    Separates "not found" from "ranked too low".
    """
    cfg = cfg or ScorerConfig()
    rows = []
    for t in truth.itertuples():
        best = max((overlap(c, t.members, cfg.measure) for c in candidates), default=0.0)
        best_iou = max((overlap(c, t.members, "iou") for c in candidates), default=0.0)
        rows.append({"case_id": t.case_id, "tags": "|".join(sorted(t.tags)), "n_members": len(t.members),
                     "best_overlap": best, "best_iou": best_iou, "covered": best >= cfg.threshold})
    return pd.DataFrame(rows)


def summary_line(res: dict) -> str:
    keys = ["P@10", "P@25", "R@25", "R@50", "R@100", "AP", "first_hit_rank", "mean_iou",
            "macro_tag_recall", "worst_tag_recall", "tag_any", "fp_top50"]
    parts = []
    for k in keys:
        v = res.get(k)
        parts.append(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
    return " ".join(parts)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Score a submission CSV against labels (development scorer).")
    ap.add_argument("submission")
    ap.add_argument("labels")
    ap.add_argument("--measure", default="iou")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args(argv)
    res = score(read_submission(a.submission), load_labels(a.labels), ScorerConfig(a.measure, a.threshold))
    print(summary_line(res))
    print("recall_by_tag:", {k: round(v, 2) for k, v in res["recall_by_tag"].items()})


if __name__ == "__main__":
    main()
