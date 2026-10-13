import numpy as np
import pytest

from f1pred.simulate import RaceInput, race_finish_probs, simulate_season


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
