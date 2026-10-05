"""SPONSOR_CONCENTRATION: a sponsor takes on many new vendors in a short window, relative to role peers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..context import Context
from .base import Candidate, tail_threshold

CENTRAL_ROLES = {"category_manager", "procurement_ss", "system_default"}


def first_engagements(ctx: Context) -> pd.DataFrame:
    """First engagement of each (sponsor, vendor entity) pair.

    Sponsorships are the instances of the meta-path
    VendorRecord -CONTRACTED_UNDER-> Engagement -SPONSORED_BY-> Employee;
    e1 and e2 are the two edges of each instance.
    """
    from ..graph_ops import hetero
    mp = hetero(ctx).metapath("CONTRACTED_UNDER", "SPONSORED_BY")
    inst = pd.DataFrame({"engagement_id": mp["n1"].str.slice(len("engagement:")), "e1": mp["e1"], "e2": mp["e2"]})
    e = ctx.engagements.merge(inst, on="engagement_id", how="inner")
    e["entity"] = e["record_id"].map(ctx.entity_of)
    return (e.sort_values("start_date")
            .groupby(["sponsor_employee_id", "entity"], as_index=False).first())


def best_window(dates: np.ndarray, days: int) -> tuple[int, int]:
    """(start index, count) of the densest window of `days` in sorted dates."""
    best, bi = 0, 0
    for i in range(len(dates)):
        j = np.searchsorted(dates, dates[i] + np.timedelta64(days, "D"), side="right")
        if j - i > best:
            best, bi = j - i, i
    return bi, best


def first_reach(dates: np.ndarray, days: int, k: int):
    """Earliest date at which some window of `days` holds k dates (the k-th date of that window)."""
    for i in range(len(dates) - k + 1):
        if dates[i + k - 1] - dates[i] <= np.timedelta64(days, "D"):
            return pd.Timestamp(dates[i + k - 1])
    return None


def detect(ctx: Context) -> list[Candidate]:
    days = ctx.param("sponsor", "window_days", 60)
    min_new = ctx.param("sponsor", "min_new", 4)
    fe = first_engagements(ctx)

    stats = []
    for sp, g in fe.groupby("sponsor_employee_id"):
        d = g.sort_values("start_date")
        dates = d["start_date"].to_numpy()
        i, n = best_window(dates, days)
        stats.append({"sponsor": sp, "max_new": n, "win": d.iloc[i:i + n], "dates": dates,
                      "role": ctx.role_of.get(sp), "n_total": len(d)})
    st = pd.DataFrame(stats)
    if st.empty:
        return []
    # Each sponsor's densest window against the median of its role peers (all sponsors
    # when the role has fewer than 5 others); the threshold is a quantile of these ratios.
    overall = float(st["max_new"].median())
    by_role = {role: np.sort(g.to_numpy()) for role, g in st.groupby("role")["max_new"]}
    memo: dict = {}
    peer = []
    for role, n in zip(st["role"], st["max_new"]):
        vals = by_role.get(role) if pd.notna(role) else None
        if vals is None or len(vals) - 1 < 5:
            peer.append((overall, "all sponsors (role group < 5)"))
            continue
        if (role, n) not in memo:  # median of the role's other sponsors
            memo[(role, n)] = float(np.median(np.delete(vals, np.searchsorted(vals, n))))
        peer.append((memo[(role, n)], f"role {role} (n={len(vals) - 1})"))
    st["peer_med"] = [m for m, _ in peer]
    st["peer_def"] = [d for _, d in peer]
    st["ratio"] = st["max_new"] / st["peer_med"].clip(lower=1.0)
    min_ratio = tail_threshold(st["ratio"], ctx.param("sponsor", "ratio_quantile", None),
                               ctx.param("sponsor", "min_ratio", 3.0))
    out = []
    for r in st.itertuples():
        if r.max_new < min_new:
            continue
        peer_med, peer_def, ratio = r.peer_med, r.peer_def, r.ratio
        if ratio < min_ratio:
            continue
        win = r.win
        lo, hi = win["start_date"].min(), win["start_date"].max()
        members = ctx.expand(win["record_id"])
        role_row = ctx.roles.set_index("employee_id").loc[r.sponsor]
        cats = win["cat_primary"].dropna()
        top_cat = cats.mode().iloc[0] if len(cats) else None
        cat_share = float((cats == top_cat).mean()) if len(cats) else 0.0
        benign = []
        if r.role in CENTRAL_ROLES:
            benign.append(f"central role {r.role}")
        scope = str(role_row["scope_markets"])
        if scope == "*" or scope.startswith(("R:", "S:")):
            benign.append(f"regional/global scope {scope}")
        mkt = win["market"].mode().iloc[0]
        same_day = win["start_date"].value_counts().iloc[0] / len(win)
        if same_day >= 0.8:
            benign.append(f"{same_day:.0%} of the new vendors start on one day: batch entry (migration or integration)")
        benign += ctx.benign_events("sponsor", lo, hi, mkt, cats=zip(win["parent"], win["cat_primary"]),
                                    roles=[r.role])
        emerged = first_reach(r.dates, days, min_new)
        out.append(Candidate(
            members=members, tags=("SPONSOR_CONCENTRATION",), score=float(np.log2(ratio) + np.log2(r.max_new)),
            emerged_at=emerged, active_until=hi, detector="sponsor_concentration",
            reason=(f"{r.role} {r.sponsor} ({role_row['home_market']}) took on {r.max_new} new vendors within "
                    f"{days} days ({lo.date()}–{hi.date()}), {ratio:.1f}x the peer median of {peer_med:g} "
                    f"for {peer_def} (threshold {min_ratio:.1f}x); top category {top_cat} ({cat_share:.0%})"),
            evidence={"sponsor": r.sponsor, "role": r.role, "window": (lo, hi), "peer_median": peer_med,
                      "ratio_threshold": min_ratio,
                      "peer_definition": peer_def, "engagements": list(win["engagement_id"])},
            benign=benign,
            features={"max_new": r.max_new, "ratio": ratio, "cat_share": cat_share},
            edges=tuple(win["e1"]) + tuple(win["e2"]),
        ))
    return out
