"""SPONSOR_APPROVER_COUPLING and APPROVER_CONCENTRATION.

A sponsor's engagements keep going to one approver although several were
eligible. Measured per (sponsor, approver) pair, because on the starter data
one approver sits in two separate planted pairs: an approver-level view would
merge them into one oversized case.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom

from ..context import Context
from .base import Candidate


def detect(ctx: Context) -> list[Candidate]:
    min_pair = ctx.param("approval", "min_pair", 5)
    min_share = ctx.param("approval", "min_sponsor_share", 0.5)
    min_eligible = ctx.param("approval", "min_eligible", 3)
    ap = ctx.approvals
    fin = ap[(ap["approval_type"] == "financial") & ap["sponsor_employee_id"].notna()].copy()
    fin["under"] = fin["ap_approver_level"] < fin["required_approver_level"]
    fin["out_of_role"] = (fin["approval_date"] < fin["ap_valid_from"]) | (
        fin["ap_valid_to"].notna() & (fin["approval_date"] > fin["ap_valid_to"]))
    known = fin[fin["approver_employee_id"].notna()]
    eng_s = known.groupby("sponsor_employee_id")["engagement_id"].nunique()

    out = []
    pairs = known.groupby(["sponsor_employee_id", "approver_employee_id"])
    for (s, a), g in pairs:
        n = g["engagement_id"].nunique()
        if n < min_pair:
            continue
        n_s = eng_s[s]
        share = n / n_s
        if share < min_share:
            continue
        mkts = set(g["market"])
        pool = known[known["market"].isin(mkts) & (known["sponsor_employee_id"] != s)]
        lo, hi = g["approval_date"].min(), g["approval_date"].max()
        period = pool[(pool["approval_date"] >= lo) & (pool["approval_date"] <= hi)]
        eligible = period[period["ap_approver_level"] >= g["required_approver_level"].min()][
            "approver_employee_id"].nunique()
        if eligible < min_eligible:
            continue
        p_a = (period["approver_employee_id"] == a).mean() if len(period) else 1.0 / max(eligible, 1)
        p_a = min(max(p_a, 1.0 / max(eligible, 1) / 4), 0.99)
        surprise = float(-np.log10(max(binom.sf(n - 1, n_s, p_a), 1e-300)))
        n_under = int(g["under"].sum())
        n_oor = int(g["out_of_role"].sum())

        # The sponsor's engagements with no recorded financial approver may belong to the same pair.
        s_all = ctx.engagements[ctx.engagements["sponsor_employee_id"] == s]
        unapproved = s_all[~s_all["engagement_id"].isin(known["engagement_id"])]
        members = ctx.expand(set(g["record_id"]) | set(unapproved["record_id"]))
        dates = np.sort(g["approval_date"].to_numpy())
        emerged = pd.Timestamp(dates[min_pair - 1])
        tags = ["SPONSOR_APPROVER_COUPLING"]
        if n_under > 0:
            tags.append("APPROVER_CONCENTRATION")
        benign = []
        if ctx.role_of.get(s) == "system_default":
            benign.append("system_default sponsor")
        if eligible <= 3:
            benign.append(f"small approver pool ({eligible})")
        out.append(Candidate(
            members=members, tags=tuple(tags), score=surprise + np.log2(1 + n_under) + np.log2(1 + n_oor),
            emerged_at=emerged, detector="sponsor_approver_pair",
            reason=(f"sponsor {s} ({ctx.role_of.get(s)}) routed {n} of {n_s} approved engagements to approver "
                    f"{a} ({ctx.role_of.get(a)}) although {eligible} approvers were eligible in "
                    f"{','.join(sorted(mkts))} (expected share {p_a:.0%}, surprise 10^-{surprise:.1f})"
                    + (f"; {n_under} approvals below the required level" if n_under else "")
                    + (f"; {n_oor} approvals outside the approver's role dates" if n_oor else "")),
            evidence={"sponsor": s, "approver": a, "pair_engagements": sorted(set(g["engagement_id"])),
                      "unapproved_engagements": sorted(set(unapproved["engagement_id"])),
                      "eligible_approvers": eligible, "expected_share": p_a,
                      "under_authority": n_under, "out_of_role": n_oor},
            benign=benign,
            features={"pair_n": n, "sponsor_share": share, "eligible": eligible, "surprise": surprise,
                      "under": n_under, "out_of_role": n_oor},
        ))
    return out
