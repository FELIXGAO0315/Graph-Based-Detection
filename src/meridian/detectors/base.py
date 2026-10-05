"""Shared types for detectors."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# The 14 tags used by the starter labels. Fixed vocabulary.
TAGS = (
    "SHARED_ADDRESS",
    "SPONSOR_CONCENTRATION",
    "APPROVER_CONCENTRATION",
    "SPONSOR_APPROVER_COUPLING",
    "DENSE_MULTI_RELATION_CLUSTER",
    "RAPID_ONBOARDING_LINKED",
    "BRIDGE_ENTITY",
    "GEOGRAPHY_LINKED_CONCENTRATION",
    "ACTIVITY_TIMING_CONCENTRATION",
    "COMMON_BENEFICIAL_OWNER",
    "LAYERED_COMPLEX_OWNERSHIP",
    "OWNERSHIP_OPERATIONAL_OVERLAP",
    "OPAQUE_JURISDICTION_ROUTING",
    "OWNERSHIP_CHANGE_STRUCTURAL_SHIFT",
)
OWNERSHIP_TAGS = frozenset({
    "COMMON_BENEFICIAL_OWNER", "LAYERED_COMPLEX_OWNERSHIP", "OWNERSHIP_OPERATIONAL_OVERLAP",
    "OPAQUE_JURISDICTION_ROUTING", "OWNERSHIP_CHANGE_STRUCTURAL_SHIFT",
})
# Tags for single-record policy findings, which the dossier scores separately.
POLICY_TAG = "POLICY_VIOLATION"
OPEN_SET_TAG = "OPEN_SET"


@dataclass
class Candidate:
    """One detector hit, carrying everything needed later for assembly and explanation.

    `score` is the detector's own strength measure (higher = more unusual); it is
    not comparable across detectors until ranking calibrates it.
    """

    members: frozenset
    tags: tuple
    score: float
    emerged_at: pd.Timestamp | None
    detector: str
    reason: str
    evidence: dict = field(default_factory=dict)
    benign: list = field(default_factory=list)
    features: dict = field(default_factory=dict)
    uses_ownership: bool = False
    # Last date the pattern held; None = still holding at t. Event-like patterns
    # (bursts, spikes) end with their window; structures end when they change.
    active_until: pd.Timestamp | None = None
    # A timeless signal (an outlier score) has no emergence of its own: it adds
    # evidence to a case on the same members but never sets the case's time frame.
    timeless: bool = False
    # Ids of the graph edges that support this candidate (Dossier Appendix C: the
    # evidence edges used by each reason); case packages draw them as a subgraph.
    edges: tuple = ()

    def __post_init__(self):
        self.members = frozenset(self.members)
        self.tags = tuple(self.tags)
        self.edges = tuple(dict.fromkeys(self.edges))
        if self.emerged_at is not None and not pd.isna(self.emerged_at):
            self.emerged_at = pd.Timestamp(self.emerged_at)
        else:
            self.emerged_at = None
        if self.active_until is not None and not pd.isna(self.active_until):
            self.active_until = pd.Timestamp(self.active_until)
        else:
            self.active_until = None
