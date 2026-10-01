import pandas as pd

from meridian.checks import run_checks
from meridian.data import as_of, valid_at


def test_starter_has_no_failed_checks(bundle):
    report = run_checks(bundle)
    assert (report["status"] != "fail").all(), report[report["status"] == "fail"]


def test_as_of_drops_future_observations(bundle):
    t = pd.Timestamp("2022-06-30")
    v = as_of(bundle, t)
    assert (v.tp_engagements["observed_at"] <= t).all()
    assert (v.tp_address_history["observed_at"] <= t).all()
    assert (v.tp_engagement_approvals["observed_at"] <= t).all()
    assert (v.reg_interests["statement_date"] <= t).all()
    assert (v.third_parties["onboarding_date"] <= t).all()


def test_as_of_blanks_end_dates_after_t(bundle):
    t = pd.Timestamp("2022-06-30")
    v = as_of(bundle, t)
    for df, col in [(v.tp_engagements, "end_date"), (v.tp_address_history, "valid_to"),
                    (v.third_parties, "terminated_date"), (v.reg_interests, "end_date")]:
        assert not (df[col] > t).any()
    # A vendor terminated after t is active at t.
    later = bundle.third_parties[bundle.third_parties["terminated_date"] > t]["record_id"]
    st = v.third_parties.set_index("record_id").loc[lambda d: d.index.isin(later), "status"]
    assert (st == "active").all()


def test_monthly_spend_available_after_month_end(bundle):
    v = as_of(bundle, "2022-06-15")
    assert v.tp_payments_monthly["month"].max() == pd.Timestamp("2022-05-01")


def test_valid_at_uses_exclusive_end():
    df = pd.DataFrame({"valid_from": pd.to_datetime(["2020-01-01"]), "valid_to": pd.to_datetime(["2021-01-01"])})
    assert len(valid_at(df, "2020-12-31")) == 1
    assert len(valid_at(df, "2021-01-01")) == 0
