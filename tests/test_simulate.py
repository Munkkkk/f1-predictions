import numpy as np
import pytest

from f1pred.simulate import (
    RaceInput,
    _simulate_totals,
    contenders,
    race_finish_probs,
    remaining_max_points,
    simulate_season,
)


def zeros_cb(n):
    return np.zeros((n, 6), dtype=int)


def gp(pred, dnf=None, sprint=False):
    pred = np.asarray(pred, dtype=float)
    dnf = np.zeros_like(pred) if dnf is None else np.asarray(dnf, dtype=float)
    return RaceInput(pred_pos=pred, p_dnf=dnf, is_sprint=sprint)


def test_insurmountable_lead_is_certain_and_decided():
    res = simulate_season(["a", "b"], np.array([200, 0]), zeros_cb(2),
                          [gp([2.0, 1.0])], sigma=2.0, n_sims=2000)
    assert res.p_title["a"] == 1.0
    assert res.decided is True


def test_open_fight_is_not_decided():
    res = simulate_season(["a", "b"], np.array([30, 10]), zeros_cb(2),
                          [gp([1.5, 1.5])], sigma=2.0, n_sims=2000)
    assert res.decided is False


def test_identical_drivers_split_title_evenly():
    races = [gp([1.5, 1.5]) for _ in range(5)]
    res = simulate_season(["a", "b"], np.array([50, 50]), zeros_cb(2), races,
                          sigma=2.0, n_sims=10_000, seed=1)
    assert res.p_title["a"] == pytest.approx(0.5, abs=0.03)


def test_points_table_applied_without_noise():
    res = simulate_season(["a", "b", "c"], np.zeros(3, dtype=int), zeros_cb(3),
                          [gp([1.0, 2.0, 3.0])], sigma=0.0, n_sims=100)
    assert res.expected_points == {"a": 25.0, "b": 18.0, "c": 15.0}


def test_sprint_uses_sprint_points():
    res = simulate_season(["a", "b", "c"], np.zeros(3, dtype=int), zeros_cb(3),
                          [gp([1.0, 2.0, 3.0], sprint=True)], sigma=0.0, n_sims=100)
    assert res.expected_points == {"a": 8.0, "b": 7.0, "c": 6.0}


def test_dnf_scores_zero():
    res = simulate_season(["a", "b"], np.zeros(2, dtype=int), zeros_cb(2),
                          [gp([1.0, 2.0], dnf=[1.0, 0.0])], sigma=0.0, n_sims=100)
    assert res.expected_points == {"a": 0.0, "b": 25.0}


def test_countback_breaks_tie_with_wins():
    cb = zeros_cb(2)
    cb[1, 0] = 1  # b a une victoire, a n'en a aucune
    res = simulate_season(["a", "b"], np.array([25, 25]), cb, [], sigma=1.0, n_sims=100)
    assert res.p_title == {"a": 0.0, "b": 1.0}


def test_probabilities_sum_to_one_and_reproducible():
    races = [gp([1.0, 2.0, 3.0, 4.0]) for _ in range(3)]
    args = (["a", "b", "c", "d"], np.array([40, 35, 30, 0]), zeros_cb(4), races)
    r1 = simulate_season(*args, sigma=3.0, n_sims=3000, seed=7)
    r2 = simulate_season(*args, sigma=3.0, n_sims=3000, seed=7)
    assert sum(r1.p_title.values()) == pytest.approx(1.0)
    assert r1 == r2


def test_race_finish_probs():
    win, podium = race_finish_probs(gp([1.0, 2.0, 3.0, 4.0]), sigma=0.0, n_sims=100)
    assert list(win) == [1.0, 0.0, 0.0, 0.0]
    assert list(podium) == [1.0, 1.0, 1.0, 0.0]


# Sorties de simulate_season figées avant l'ajout des chocs (commit 1f2d73d).
PINNED_P_TITLE = {"a": 0.6563333333333333, "b": 0.26866666666666666,
                  "c": 0.07466666666666667, "d": 0.0003333333333333333}
PINNED_EXPECTED = {"a": 101.638, "b": 90.546, "c": 76.245, "d": 52.76433333333333}


def test_default_shock_kwargs_reproduce_previous_output():
    races = [gp([1.0, 2.5, 3.0, 4.0], dnf=[0.1, 0.1, 0.2, 0.05], sprint=i == 1)
             for i in range(4)]
    args = (["a", "b", "c", "d"], np.array([40, 35, 30, 0]), zeros_cb(4), races)
    base = simulate_season(*args, sigma=3.0, n_sims=3000, seed=11)
    same = simulate_season(*args, sigma=3.0, n_sims=3000, seed=11, teams=None,
                           tau_driver=0.0, tau_team=0.0, dnf_shock=0.0)
    with_teams = simulate_season(*args, sigma=3.0, n_sims=3000, seed=11,
                                 teams=["x", "x", "y", "y"])
    assert base.p_title == PINNED_P_TITLE and base.expected_points == PINNED_EXPECTED
    assert same.p_title == base.p_title and same.expected_points == base.expected_points
    assert with_teams == base


def test_driver_form_shock_helps_trailing_driver():
    # b doit gagner les 3 courses (écart max 3 x 7 = 21 > 20) : la forme persistante aide.
    races = [gp([1.5, 1.5]) for _ in range(3)]
    args = (["a", "b"], np.array([60, 40]), zeros_cb(2), races)
    calm = simulate_season(*args, sigma=2.0, n_sims=20_000, seed=3)
    shaky = simulate_season(*args, sigma=2.0, n_sims=20_000, seed=3, tau_driver=3.0)
    assert 0 < calm.p_title["b"] < shaky.p_title["b"]


def test_team_shock_correlates_teammates():
    races = [gp([2.0, 2.5, 2.0, 2.5]) for _ in range(5)]
    total, _, _ = _simulate_totals(np.zeros(4, dtype=int), zeros_cb(4), races, sigma=1.0,
                                   n_sims=5000, seed=0, teams=["A", "A", "B", "B"],
                                   tau_team=4.0)
    corr = np.corrcoef(total[:, 0], total[:, 1])[0, 1]
    assert corr > 0.3


def test_dnf_shock_keeps_mean_and_adds_dispersion():
    n_races, n = 10, 6
    races = [gp(np.arange(1.0, n + 1), dnf=np.full(n, 0.1)) for _ in range(n_races)]

    def dnf_counts(shock):
        *_, dnfs = _simulate_totals(np.zeros(n, dtype=int), zeros_cb(n), races, sigma=0.0,
                                    n_sims=20_000, seed=5, dnf_shock=shock)
        return dnfs

    base, shocked = dnf_counts(0.0), dnf_counts(0.8)
    assert abs(shocked.mean() / n_races - 0.1) < 0.02
    assert shocked.var() > base.var()


def test_contenders_and_tie():
    assert list(contenders(np.array([340, 238, 221, 138]), 150)) == [True, True, True, False]
    assert list(contenders(np.array([340, 190]), 150)) == [True, True]


def test_remaining_max_points():
    races = [gp([1.0, 2.0]), gp([1.0, 2.0], sprint=True), gp([1.0, 2.0])]
    assert remaining_max_points(races, 2) == 58
    assert remaining_max_points([], 22) == 0
