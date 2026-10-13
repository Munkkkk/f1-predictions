import pandas as pd
import pytest

from f1pred.features import FEATURE_COLUMNS, add_future_rows, build_features
from tests.helpers import make_results, make_schedule


def test_one_row_per_gp_driver_and_no_missing_features():
    results = make_results()
    feats = build_features(results)
    gp_rows = (~results["is_sprint"]).sum()
    assert len(feats) == gp_rows
    assert not feats[FEATURE_COLUMNS].isna().any().any()


def test_no_leakage_from_future_races():
    results = make_results()
    base = build_features(results)
    last = (results["season"] == 2025) & (results["round"] == 8)
    tampered = results.copy()
    tampered.loc[last, "position"] = 99
    tampered.loc[last, "points"] = 999.0
    tampered.loc[last, "finished"] = False
    after = build_features(tampered)
    cols = ["season", "round", "driver_id", *FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(base[cols], after[cols])


def test_rolling_form_by_hand():
    rows = []
    for rnd, pos in enumerate([1, 3, 5, 7], start=1):
        rows.append(dict(season=2026, round=rnd, race_name="GP", date=f"2026-0{rnd}-01",
                         circuit_id="c", driver_id="a", driver_name="A", constructor_id="t",
                         position=pos, points=0.0, finished=True, is_sprint=False))
    feats = build_features(pd.DataFrame(rows)).set_index("round")
    assert feats.loc[1, "drv_form3"] == pytest.approx(1.0)  # aucun passé : (1 + 1) / 2
    assert feats.loc[2, "drv_form3"] == pytest.approx(1.0)
    assert feats.loc[4, "drv_form3"] == pytest.approx(3.0)  # moyenne de 1, 3, 5


def test_teammate_delta_handles_single_driver_team():
    rows = [dict(season=2026, round=r, race_name="GP", date=f"2026-0{r}-01", circuit_id="c",
                 driver_id="solo", driver_name="Solo", constructor_id="alone", position=4,
                 points=12.0, finished=True, is_sprint=False) for r in (1, 2, 3)]
    feats = build_features(pd.DataFrame(rows))
    assert (feats["teammate_delta"] == 0).all()


def test_points_share_includes_sprints_from_previous_rounds():
    results = make_results(seasons=(2025,), n_rounds=4, sprint_rounds=(1,))
    feats = build_features(results)
    r1 = results[results["round"] == 1].groupby("driver_id")["points"].sum()
    expected = r1 / r1.sum()
    got = feats[feats["round"] == 2].set_index("driver_id")["points_share"]
    pd.testing.assert_series_equal(got.sort_index(), expected.sort_index(), check_names=False)
    assert (feats[feats["round"] == 1]["points_share"] == 0).all()


def test_add_future_rows_creates_unknown_outcomes():
    results = make_results(seasons=(2025,), n_rounds=5)
    future = make_schedule(2025, 8).query("round > 5")
    lineup = (results[results["round"] == 5]
              .drop_duplicates("driver_id")[["driver_id", "driver_name", "constructor_id"]])
    full = add_future_rows(results, future, lineup)
    new = full[full["round"] > 5]
    assert len(new) == 3 * len(lineup)
    assert new["position"].isna().all()
    feats = build_features(full)
    assert not feats[FEATURE_COLUMNS].isna().any().any()


def test_neutral_form_scales_with_field_size():
    results = make_results(seasons=(2025,), n_rounds=1, n_drivers=22, sprint_rounds=())
    feats = build_features(results)
    assert (feats["drv_form3"] == 11.5).all()
