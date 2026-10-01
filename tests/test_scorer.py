import numpy as np
import pandas as pd

from meridian.scorer import ScorerConfig, prepare_submission, score


def as_submission(df):
    return prepare_submission(pd.DataFrame({
        "case_rank": range(1, len(df) + 1),
        "member_record_ids": df["member_record_ids"].to_numpy(),
        "typology_tags": df["typology_tags"].to_numpy(),
        "score": np.linspace(1, 0, len(df)),
        "reason_codes": "",
    }))


def test_labels_as_submission_is_perfect(labels):
    res = score(as_submission(labels), labels)
    assert res["recall"] == 1.0
    assert res["AP"] == 1.0
    assert res["mean_iou"] == 1.0
    assert res["exact_recovery"] == 1.0
    assert res["tag_exact"] == 1.0
    assert res["worst_tag_recall"] == 1.0
    assert res["fp_top50"] == 0
    assert res["first_hit_rank"] == 1


def test_random_submission_is_near_zero(bundle, labels):
    rng = np.random.default_rng(0)
    ids = bundle.third_parties["record_id"].to_numpy()
    rows = []
    for i in range(100):
        m = rng.choice(ids, size=rng.integers(1, 8), replace=False)
        rows.append({"case_rank": i + 1, "member_record_ids": "|".join(m),
                     "typology_tags": "SHARED_ADDRESS", "score": 1 - i / 100, "reason_codes": ""})
    res = score(prepare_submission(pd.DataFrame(rows)), labels)
    assert res["R@100"] <= 0.1
    assert res["AP"] <= 0.05


def test_duplicate_submissions_count_once(labels):
    one = labels.iloc[[0, 0, 0]]
    res = score(as_submission(one), labels)
    assert res["P@10"] == 0.1
    assert res["fp_top50"] == 2


def test_partial_overlap_threshold(labels):
    row = labels.iloc[[0]].copy()   # 6 members
    members = row.iloc[0]["member_record_ids"].split("|")
    row["member_record_ids"] = "|".join(members[:3])  # IoU 0.5
    assert score(as_submission(row), labels, ScorerConfig(threshold=0.5))["recall"] > 0
    assert score(as_submission(row), labels, ScorerConfig(threshold=0.6))["recall"] == 0
