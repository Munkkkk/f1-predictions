import numpy as np
import pandas as pd
import pytest

from f1pred.shocks import Shocks, estimate_shocks, residual_records, usable_records


def planted(tau_team=1.5, tau_driver=1.0, sigma=3.0, n_ckpt=60, n_teams=10, horizon=6,
            p_dnf=0.1, dnf_shock=0.0, seed=0):
    """Résidus multi-étapes synthétiques : effet écurie + effet pilote par checkpoint + bruit."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_ckpt):
        team_fx = rng.normal(0, tau_team, n_teams)
        for t in range(n_teams):
            for k in range(2):
                drv_fx = rng.normal(0, tau_driver)
                mult = np.exp(rng.normal(0, dnf_shock) - dnf_shock**2 / 2)
                for race in range(c + 1, c + 1 + horizon):
                    rows.append({
                        "season": 2025, "checkpoint": c, "round": race,
                        "driver_id": f"d{t}_{k}", "team": f"t{t}",
                        "resid": team_fx[t] + drv_fx + rng.normal(0, sigma),
                        "finished": bool(rng.random() >= min(p_dnf * mult, 0.9)),
                        "p_dnf": p_dnf,
                    })
    return pd.DataFrame(rows)


def test_recovers_planted_team_and_driver_form():
    s = estimate_shocks(planted(), fallback_sigma=9.9)
    assert s.tau_team == pytest.approx(1.5, abs=0.3)
    assert s.tau_driver == pytest.approx(1.0, abs=0.3)
    assert s.sigma == pytest.approx(3.0, abs=0.3)


def test_no_planted_effects_gives_small_shocks():
    s = estimate_shocks(planted(tau_team=0.0, tau_driver=0.0), fallback_sigma=9.9)
    assert s.tau_team < 0.4 and s.tau_driver < 0.4


def test_iid_sigma_completes_total_variance():
    rec = planted(seed=3)
    s = estimate_shocks(rec, fallback_sigma=9.9)
    total = rec.loc[rec["finished"], "resid"].var(ddof=1)
    assert s.sigma**2 + s.tau_driver**2 + s.tau_team**2 == pytest.approx(total, rel=0.02)


def test_recovers_planted_reliability_shock():
    s = estimate_shocks(planted(dnf_shock=0.6, n_ckpt=200, seed=1), fallback_sigma=9.9)
    assert s.dnf_shock == pytest.approx(0.6, abs=0.2)
    calm = estimate_shocks(planted(dnf_shock=0.0, n_ckpt=200, seed=1), fallback_sigma=9.9)
    assert calm.dnf_shock < 0.3
    assert 0.0 <= s.dnf_shock <= 1.0


def test_empty_records_fall_back_to_no_shock():
    empty = planted().iloc[0:0]
    assert estimate_shocks(empty, fallback_sigma=3.7) == Shocks(sigma=3.7)


def test_usable_records_never_include_races_after_checkpoint():
    prev = planted(n_ckpt=3).assign(season=2025)
    cur = planted(n_ckpt=8).assign(season=2026)
    rec = pd.concat([prev, cur], ignore_index=True)
    used = usable_records(rec, season=2026, upto_round=5)
    assert (used.loc[used["season"] == 2026, "round"] <= 5).all()
    assert (used.loc[used["season"] == 2026, "checkpoint"] < 5).all()
    assert len(used[used["season"] == 2025]) == len(prev)


def test_residual_records_pair_forecasts_with_actual_results():
    preds = pd.DataFrame({"round": [6, 6, 7, 13], "driver_id": ["a", "b", "a", "a"],
                          "pred_pos": [2.0, 5.0, 3.0, 1.0], "p_dnf": [0.1, 0.2, 0.1, 0.1]})
    results = pd.DataFrame({
        "season": 2026, "round": [6, 6, 6, 7, 13], "driver_id": ["a", "b", "a", "a", "a"],
        "constructor_id": "t", "position": [4, 9, 1, 3, 1],
        "finished": [True, False, True, True, True], "is_sprint": [False, False, True, False,
                                                                   False]})
    rec = residual_records(preds, results, season=2026, checkpoint=5)
    assert list(rec["round"]) == [6, 6, 7]  # manche 13 hors horizon, sprint ignoré
    assert list(rec["resid"]) == [2.0, 4.0, 0.0]
    assert list(rec["finished"]) == [True, False, True]
    assert set(rec["checkpoint"]) == {5} and set(rec["team"]) == {"t"}
