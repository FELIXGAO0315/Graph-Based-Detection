import pandas as pd

from meridian.cases import Case, assemble
from meridian.detectors.base import Candidate
from meridian.ranking import calibrate_cases, order, percentiles, select_tags

T = pd.Timestamp


def cand(members, tag, det, emerged, until=None, score=1.0, timeless=False, benign=(), edges=()):
    return Candidate(members=members, tags=(tag,), score=score, emerged_at=T(emerged), active_until=until,
                     detector=det, reason=det, benign=list(benign), timeless=timeless, edges=tuple(edges))


def test_same_members_in_periods_that_do_not_meet_stay_apart():
    layered = cand({"R1"}, "LAYERED_COMPLEX_OWNERSHIP", "layered", "2014-03-31", until=T("2019-07-08"))
    change = cand({"R1"}, "OWNERSHIP_CHANGE_STRUCTURAL_SHIFT", "change", "2019-07-08", until=T("2019-10-06"))
    assert len(assemble([layered, change])) == 2


def test_ongoing_pattern_absorbs_a_later_one_on_the_same_members():
    owner = cand({"A", "B", "C"}, "COMMON_BENEFICIAL_OWNER", "owner", "2022-02-18")
    address = cand({"A", "B", "C"}, "SHARED_ADDRESS", "address", "2025-02-27")
    cases = assemble([owner, address])
    assert len(cases) == 1 and set(cases[0].tags) == {"COMMON_BENEFICIAL_OWNER", "SHARED_ADDRESS"}


def test_timeless_signal_does_not_bridge_two_periods():
    layered = cand({"R1"}, "LAYERED_COMPLEX_OWNERSHIP", "layered", "2014-03-31", until=T("2019-07-08"))
    change = cand({"R1"}, "OWNERSHIP_CHANGE_STRUCTURAL_SHIFT", "change", "2019-07-08", until=T("2019-10-06"))
    outlier = cand({"R1"}, "OPEN_SET", "outlier", "2014-03-31", timeless=True)
    assert len(assemble([layered, change, outlier])) == 2


def test_overlapping_groups_from_different_detectors_merge_into_the_union():
    a = cand({"A", "B", "C", "D"}, "SPONSOR_APPROVER_COUPLING", "pair", "2025-01-01")
    b = cand({"A", "B", "C", "E"}, "ACTIVITY_TIMING_CONCENTRATION", "spikes", "2025-02-01")
    cases = assemble([a, b])
    assert len(cases) == 1 and cases[0].members == frozenset("ABCDE")


def test_open_set_community_does_not_widen_a_typed_group():
    address = cand(set("ABCDEF"), "SHARED_ADDRESS", "address", "2025-01-01")
    community = cand(set("ABCDEFGH"), "OPEN_SET", "community", "2025-01-01")
    cases = assemble([address, community])
    assert len(cases) == 1 and cases[0].members == frozenset("ABCDEF") and len(cases[0].candidates) == 2


def test_candidate_joins_the_best_matching_case():
    first = cand(set("ABCDE"), "SPONSOR_CONCENTRATION", "x", "2025-01-01", score=3)
    second = cand(set("ABCDF"), "SPONSOR_CONCENTRATION", "x", "2025-01-01", score=2)
    other = cand(set("XY"), "SHARED_ADDRESS", "y", "2025-01-01", score=5)
    late = cand(set("ABCDF"), "SHARED_ADDRESS", "y", "2025-01-01", score=1)
    best = assemble([first, second, other, late])
    host = next(k for k in best if late in k.candidates)
    assert host.members == frozenset("ABCDF")
    first_found = assemble([first, second, other, late], best_match=False)
    assert next(k for k in first_found if late in k.candidates).members == frozenset("ABCDEF")


def test_findings_on_the_same_structure_join_when_enabled():
    layered = cand({"X"}, "LAYERED_COMPLEX_OWNERSHIP", "layered", "2026-01-01", score=2, edges=["e1", "e2", "e3", "e4"])
    owner = cand({"X", "Y"}, "COMMON_BENEFICIAL_OWNER", "owner", "2026-02-27",
                 edges=["e1", "e2", "e3", "e4", "e5", "e6"])
    assert len(assemble([layered, owner])) == 2
    cases = assemble([layered, owner], evidence_union=0.8)
    assert len(cases) == 1 and cases[0].members == frozenset("XY")


def test_shared_evidence_is_counted_once():
    def case_with(edges_b):
        a = cand({"A", "B"}, "COMMON_BENEFICIAL_OWNER", "owner", "2022-01-01", edges=["o1", "o2"])
        b = cand({"A", "B"}, "OWNERSHIP_OPERATIONAL_OVERLAP", "overlap", "2022-01-01", edges=edges_b)
        cases = assemble([a, b])
        calibrate_cases(cases)
        return cases[0].calibrated
    assert case_with(["o1", "o2"]) < case_with(["s1", "s2"])


def test_small_detector_pools_get_a_mid_percentile():
    lone = Case(members=frozenset("AB"), tags=("X",), score=1, emerged_at=None,
                candidates=[cand({"A", "B"}, "POLICY_VIOLATION", "fmv", "2025-01-01", score=9)])
    assert percentiles([lone])[id(lone.lead)] == 0.5
    assert percentiles([lone], min_pool=1)[id(lone.lead)] == 0.5


def test_benign_explanation_only_discounts_its_own_signal():
    owner = cand({"A", "B"}, "COMMON_BENEFICIAL_OWNER", "owner", "2022-01-01")
    address = cand({"A", "B"}, "SHARED_ADDRESS", "address", "2022-01-01", benign=["address type coworking"])
    cases = assemble([owner, address])
    calibrate_cases(cases)
    only_address = assemble([cand({"C", "D"}, "SHARED_ADDRESS", "address", "2022-01-01",
                                  benign=["address type coworking"])])
    calibrate_cases(only_address)
    assert cases[0].calibrated > only_address[0].calibrated


def test_fragments_of_a_chosen_case_are_pushed_down():
    big = Case(members=frozenset("ABCDEF"), tags=("X",), score=1, emerged_at=T("2025-01-01"), calibrated=0.9,
               candidates=[cand("ABCDEF", "X", "x", "2025-01-01")])
    frag = Case(members=frozenset("ABC"), tags=("Y",), score=1, emerged_at=T("2025-01-01"), calibrated=0.6,
                candidates=[cand("ABC", "Y", "y", "2025-01-01")])
    other = Case(members=frozenset("GH"), tags=("Y",), score=1, emerged_at=T("2025-01-01"), calibrated=0.5,
                 candidates=[cand("GH", "Y", "y", "2025-01-01")])
    assert [c.members for c in order([big, frag, other])] == [big.members, other.members, frag.members]


def test_select_tags_drops_open_set_and_weak_tags():
    c = Case(members=frozenset("AB"), tags=("SHARED_ADDRESS", "OPEN_SET", "BRIDGE_ENTITY"), score=1,
             emerged_at=None, tag_scores={"SHARED_ADDRESS": 0.6, "OPEN_SET": 0.58, "BRIDGE_ENTITY": 0.2})
    assert select_tags(c) == ("SHARED_ADDRESS",)
