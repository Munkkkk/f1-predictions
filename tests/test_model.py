import numpy as np
import pandas as pd

from f1pred.features import build_features
from f1pred.model import HOLDOUT_RACES, predict, sample_weights, train
from tests.helpers import make_results


def test_sample_weights_boost_current_and_decay_past():
    w = sample_weights(pd.Series([2026, 2025, 2024, 2016]), current_season=2026)
    assert w[0] == 5.0
    assert w[1] == 1.0
    assert w[2] == 0.8
    assert 0 < w[3] < w[2]


def test_train_and_predict_ranges():
    feats = build_features(make_results(seasons=(2023, 2024, 2025)))
    models = train(feats, current_season=2025)
    preds = predict(models, feats.tail(10))
    assert list(preds.columns) == ["season", "round", "driver_id", "pred_pos", "p_dnf"]
    assert (preds["pred_pos"] >= 1).all()
    assert preds["p_dnf"].between(0, 1).all()
    assert models.sigma > 0


def test_model_learns_strong_driver_is_ahead():
    feats = build_features(make_results(seasons=(2023, 2024, 2025)))
    models = train(feats, current_season=2025)
    last = feats[(feats["season"] == 2025) & (feats["round"] == 8)]
    preds = predict(models, last).set_index("driver_id")
    assert preds.loc["d0", "pred_pos"] < preds.loc["d9", "pred_pos"]


def test_no_dnf_observed_uses_constant_rate():
    feats = build_features(make_results(seasons=(2024, 2025)))
    feats["finished"] = True
    models = train(feats, current_season=2025)
    assert models.clf is None
    preds = predict(models, feats.tail(5))
    assert np.allclose(preds["p_dnf"], models.dnf_base)


def test_shocks_are_finite_and_non_negative():
    models = train(build_features(make_results(seasons=(2023, 2024, 2025))), current_season=2025)
    shocks = [models.tau_driver, models.tau_team, models.dnf_shock]
    assert np.isfinite(shocks).all()
    assert min(shocks) >= 0.0
    assert models.dnf_shock <= 1.0


def test_shocks_are_zero_without_holdout():
    models = train(build_features(make_results(seasons=(2024, 2025))), current_season=2025)
    assert (models.tau_driver, models.tau_team, models.dnf_shock) == (0.0, 0.0, 0.0)


def test_driver_persistently_worse_in_holdout_gives_driver_shock():
    results = make_results(seasons=(2023, 2024, 2025))
    gp = results[~results["is_sprint"]]
    last_races = gp[["season", "round"]].drop_duplicates().tail(HOLDOUT_RACES)
    in_holdout = results.set_index(["season", "round"]).index.isin(
        list(last_races.itertuples(index=False, name=None)))
    slumping = in_holdout & (results["driver_id"] == "d0").to_numpy()
    base = train(build_features(results), current_season=2025)
    results.loc[slumping, "position"] += 6
    slump = train(build_features(results), current_season=2025)
    assert slump.tau_driver > 0.5
    assert slump.tau_driver > base.tau_driver
