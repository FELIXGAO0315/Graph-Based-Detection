import pandas as pd

from meridian.detectors.shared_address import _episodes

T = pd.Timestamp


class OneEntityPerRecord:
    def n_entities(self, records):
        return len(records)


def stays(*rows):
    return pd.DataFrame([{"record_id": r, "valid_from": T(a), "valid_to": T(b) if b else pd.NaT} for r, a, b in rows])


def test_two_groups_at_one_address_are_two_episodes():
    g = stays(("A", "2015-01-01", "2016-01-01"), ("B", "2015-02-01", "2016-01-01"), ("C", "2015-03-01", "2016-01-01"),
              ("D", "2020-01-01", None), ("E", "2020-02-01", None), ("F", "2020-03-01", None))
    eps = _episodes(OneEntityPerRecord(), g, 3)
    assert [(e["members"], e["reached"], e["ended"]) for e in eps] == [
        ({"A", "B", "C"}, T("2015-03-01"), T("2016-01-01")),
        ({"D", "E", "F"}, T("2020-03-01"), None)]


def test_a_same_day_handover_does_not_split_an_episode():
    g = stays(("A", "2015-01-01", None), ("B", "2015-01-01", None), ("C", "2015-01-01", "2017-01-01"),
              ("D", "2017-01-01", None))
    eps = _episodes(OneEntityPerRecord(), g, 3)
    assert len(eps) == 1 and eps[0]["ended"] is None and eps[0]["peak"] == 3
