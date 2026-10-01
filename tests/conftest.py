from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "student_bundle"
LABELS = BUNDLE / "labelled_cases_TRAINING_SET.parquet"


@pytest.fixture(scope="session")
def bundle():
    from meridian.data import load_bundle
    return load_bundle(BUNDLE)


@pytest.fixture(scope="session")
def labels():
    from meridian.scorer import load_labels
    return load_labels(LABELS)
