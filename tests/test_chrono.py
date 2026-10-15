import pandas as pd

from f1pred.chrono import known_rounds, race_index, race_order
from tests.helpers import _date, make_results


def postponed(results, season=2025, rnd=3, after=5):
    """La manche `rnd` est courue une semaine après la manche `after`, sous son numéro."""
    late = (pd.Timestamp(_date(season, after)) + pd.Timedelta(days=7)).date().isoformat()
    out = results.copy()
    out.loc[(out["season"] == season) & (out["round"] == rnd), "date"] = late
    return out


def test_race_order_follows_race_dates():
    results = make_results(seasons=(2025,), n_rounds=6)
    assert race_order(results, 2025) == [1, 2, 3, 4, 5, 6]
    assert race_order(postponed(results), 2025) == [1, 2, 4, 5, 3, 6]


def test_known_rounds_are_those_raced_up_to_the_checkpoint():
    results = postponed(make_results(seasons=(2025,), n_rounds=6))
    assert known_rounds(results, 2025, 4) == [1, 2, 4]
    assert known_rounds(results, 2025, 3) == [1, 2, 4, 5, 3]
    assert known_rounds(results, 2025, 6) == [1, 2, 4, 5, 3, 6]


def test_race_index_is_chronological_across_seasons():
    results = postponed(make_results(seasons=(2024, 2025), n_rounds=5), after=4)
    idx = race_index(results).set_index(["season", "round"])["race_idx"]
    order = [(2024, r) for r in range(1, 6)] + [(2025, r) for r in (1, 2, 4, 3, 5)]
    assert [idx[k] for k in order] == list(range(10))
