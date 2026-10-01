"""Record-level outliers (ladder level A): no relation to any other vendor is used."""
from __future__ import annotations

from ..context import Context
from .base import OPEN_SET_TAG, Candidate


def detect_record_outlier(ctx: Context) -> list[Candidate]:
    from ..features import record_features, record_outlier_scores

    share = ctx.param("record_outlier", "top_share", 0.03)
    feat = record_features(ctx)
    score = record_outlier_scores(feat, ctx.cfg.get("seed", 0))
    top = score.sort_values(ascending=False).head(max(10, int(share * len(score))))
    z = (feat - feat.median()) / (feat.quantile(0.9) - feat.quantile(0.1)).replace(0, 1)
    out = []
    for rec, s in top.items():
        drivers = z.loc[rec].abs().sort_values(ascending=False).head(3)
        out.append(Candidate(
            members=ctx.expand([rec]), tags=(OPEN_SET_TAG,), score=float(s),
            emerged_at=ctx.vendors.loc[rec, "onboarding_date"], detector="record_outlier",
            reason=("record-level outlier (isolation forest on the record's own attributes and payments); "
                    "most unusual: " + ", ".join(f"{k}={feat.at[rec, k]:.2f}" for k in drivers.index)),
            evidence={"outlier_score": float(s), "drivers": {k: float(feat.at[rec, k]) for k in drivers.index}},
            features={"outlier": float(s)}, timeless=True,
        ))
    return out
