import pandas as pd
import pytest

from meridian.data import as_of
from meridian.context import Context
from meridian.detectors.base import OPEN_SET_TAG, POLICY_TAG, TAGS
from meridian.run import ROOT, load_config, pipeline


@pytest.fixture(scope="module")
def run(bundle):
    cfg = load_config(ROOT / "configs" / "default.yaml")
    return pipeline(cfg, bundle)


def test_submission_matches_template(run, bundle):
    sub = run[0]
    template = pd.read_csv(ROOT / "student_bundle" / "submission_template.csv")
    assert list(sub.columns[: len(template.columns)]) == list(template.columns)
    assert sub["case_rank"].is_unique and (sub["case_rank"] == range(1, len(sub) + 1)).all()
    ids = set(bundle.third_parties["record_id"])
    for m in sub["member_record_ids"]:
        assert set(m.split("|")) <= ids
    vocab = set(TAGS) | {POLICY_TAG, OPEN_SET_TAG}
    for t in sub["typology_tags"]:
        assert set(t.split("|")) <= vocab


def test_every_case_has_an_emergence_date_and_reason(run):
    sub = run[0]
    assert (sub["emerged_at"] != "").all()
    assert sub["reason_codes"].str.len().gt(20).all()


def test_every_tag_has_a_detector(run):
    cands = run[2]
    found = {t for c in cands for t in c.tags}
    assert set(TAGS) <= found


def test_ownership_switch_removes_ownership_candidates(bundle):
    cfg = load_config(ROOT / "configs" / "default.yaml", ["features.ownership=false"])
    _, _, cands, _, _ = pipeline(cfg, bundle)
    assert not any(c.uses_ownership for c in cands)
    assert not any(c.detector in {"common_owner", "layered_ownership", "opaque_routing", "ownership_change"}
                   for c in cands)


def test_layered_structure_found_in_history(bundle):
    # TP-98B17648 was layered through a trust until 2019, then acquired via a BVI holdco.
    from meridian.detectors.ownership import detect_change, detect_layered
    ctx = Context(as_of(bundle, "2026-06-30"))
    layered = [c for c in detect_layered(ctx) if "TP-98B17648" in c.members]
    change = [c for c in detect_change(ctx) if "TP-98B17648" in c.members]
    assert layered and change
    assert layered[0].emerged_at < change[0].emerged_at


def test_detectors_see_no_future(bundle):
    t = pd.Timestamp("2023-06-30")
    cfg = load_config(ROOT / "configs" / "default.yaml", ["snapshot=2023-06-30"])
    _, cases, cands, ctx, _ = pipeline(cfg, bundle)
    late = set(bundle.third_parties[bundle.third_parties["onboarding_date"] > t]["record_id"])
    assert not any(c.members & late for c in cands)
    assert all(c.emerged_at is None or c.emerged_at <= t for c in cands)


def test_open_set_detectors_run_and_are_tagged(run):
    cands = run[2]
    open_set = [c for c in cands if c.detector in {"rare_sharing_community", "entity_change_point",
                                                   "duplicate_paid", "entity_outlier"}]
    assert open_set and all(c.tags == (OPEN_SET_TAG,) for c in open_set)
    assert all(c.timeless for c in open_set if c.detector == "entity_outlier")


def test_case_packages_have_required_sections(run, tmp_path):
    from meridian.explain import write_packages
    sub, cases, cands, ctx, _ = run
    ordered = sorted((c for c in cases if c.rank is not None), key=lambda c: c.rank)
    write_packages(ctx, ordered, tmp_path, {"snapshot": "2026-06-30", "run": "test", "config_hash": "x",
                                             "git_commit": "x", "input_hashes": {}}, top=3)
    text = (tmp_path / "case_001.md").read_text()
    for section in ["## Members", "## Why flagged", "## Benign explanations to check first", "## Evidence",
                    "## Timeline", "## Ownership", "## Relationship sketch", "## Provenance"]:
        assert section in text
    assert (tmp_path / "README.md").exists()
