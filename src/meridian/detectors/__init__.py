"""Detector registry. Each detector takes a Context and returns a list of Candidates."""
from __future__ import annotations

from . import activity, approval, cluster, market, ownership, policy, record, shared_address, sponsor
from .open_set import OPEN_SET_DETECTORS

# Typed detectors and policy detectors. The exact-address baseline
# (the first, simplest detector) is kept for the representation ladder but not run by default.
DETECTORS = {
    "shared_address": shared_address.detect,
    "sponsor_concentration": sponsor.detect,
    "sponsor_approver_pair": approval.detect,
    "dense_cluster": cluster.detect_dense,
    "rapid_onboarding": cluster.detect_rapid_onboarding,
    "bridge": market.detect_bridge,
    "geography": market.detect_geography,
    "sync_spikes": activity.detect_sync_spikes,
    "post_termination": activity.detect_post_termination,
    "common_owner": ownership.detect_common_owner,
    "layered_ownership": ownership.detect_layered,
    "opaque_routing": ownership.detect_opaque,
    "ownership_change": ownership.detect_change,
    "fmv_cap": policy.detect_fmv,
}
OPTIONAL = {
    "address_exact": shared_address.detect_exact,
    "record_outlier": record.detect_record_outlier,
}
OWNERSHIP_DETECTORS = {"common_owner", "layered_ownership", "opaque_routing", "ownership_change"}


def registry() -> dict:
    return {**DETECTORS, **OPTIONAL, **OPEN_SET_DETECTORS}


def run_detectors(ctx, names="all", exclude=()) -> list:
    """Run the selected detectors. `exclude` drops candidates by detector name
    (a function can emit several, e.g. common_owner and ownership_overlap)."""
    reg = registry()
    if names in (None, "all"):
        selected = {**DETECTORS, **(OPEN_SET_DETECTORS if ctx.cfg.get("features", {}).get("open_set", True) else {})}
    else:
        selected = {n: reg[n] for n in names}
    out = []
    for name, fn in selected.items():
        if not ctx.use_ownership and name in OWNERSHIP_DETECTORS:
            continue
        cands = fn(ctx)
        if not ctx.use_ownership:
            cands = [c for c in cands if not c.uses_ownership]
        out.extend(c for c in cands if c.detector not in exclude)
    return out
