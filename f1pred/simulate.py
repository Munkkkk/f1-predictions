"""Simulation Monte Carlo vectorisée de la fin de saison."""

from dataclasses import dataclass

import numpy as np

from f1pred.rules import COUNTBACK_DEPTH, points_table

COUNTBACK_BASE = 32  # > nombre maximal de victoires possible sur une saison


@dataclass
class RaceInput:
    pred_pos: np.ndarray
    p_dnf: np.ndarray
    is_sprint: bool


@dataclass
class SimResult:
    p_title: dict[str, float]
    expected_points: dict[str, float]
    decided: bool


def _draw_race(race: RaceInput, sigma: float, n_sims: int, rng: np.random.Generator):
    """Renvoie (rangs 0-indexés, masque d'abandon), chacun de forme (n_sims, n)."""
    n = len(race.pred_pos)
    perf = race.pred_pos[None, :] + rng.normal(0.0, sigma, (n_sims, n))
    dnf = rng.random((n_sims, n)) < race.p_dnf[None, :]
    perf = np.where(dnf, np.inf, perf)
    order = np.argsort(perf, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(n), (n_sims, n)), axis=1)
    return ranks, dnf


def _is_decided(points: np.ndarray, races: list[RaceInput]) -> bool:
    if not races:
        return True
    n = len(points)
    remaining = sum(int(points_table(n, r.is_sprint)[0]) for r in races)
    leader = int(np.argmax(points))
    others = np.delete(points, leader)
    return bool(np.all(points[leader] > others + remaining))


def simulate_season(drivers: list[str], points: np.ndarray, countback: np.ndarray,
                    races: list[RaceInput], sigma: float, n_sims: int = 10_000,
                    seed: int = 0) -> SimResult:
    rng = np.random.default_rng(seed)
    n = len(drivers)
    total = np.tile(np.asarray(points, dtype=np.int64), (n_sims, 1))
    counts = np.tile(np.asarray(countback, dtype=np.int64), (n_sims, 1, 1))

    for race in races:
        ranks, dnf = _draw_race(race, sigma, n_sims, rng)
        table = points_table(n, race.is_sprint)
        total += np.where(dnf, 0, table[ranks])
        if not race.is_sprint:
            for k in range(min(COUNTBACK_DEPTH, n)):
                counts[:, :, k] += (ranks == k) & ~dnf

    weights = COUNTBACK_BASE ** np.arange(COUNTBACK_DEPTH - 1, -1, -1, dtype=np.int64)
    key = total * COUNTBACK_BASE**COUNTBACK_DEPTH + (counts * weights).sum(axis=2)
    champions = key.argmax(axis=1)
    p = np.bincount(champions, minlength=n) / n_sims
    expected = total.mean(axis=0)
    return SimResult(
        p_title={d: float(p[i]) for i, d in enumerate(drivers)},
        expected_points={d: float(expected[i]) for i, d in enumerate(drivers)},
        decided=_is_decided(np.asarray(points), races),
    )


def race_finish_probs(race: RaceInput, sigma: float, n_sims: int = 10_000,
                      seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ranks, dnf = _draw_race(race, sigma, n_sims, rng)
    win = ((ranks == 0) & ~dnf).mean(axis=0)
    podium = ((ranks < 3) & ~dnf).mean(axis=0)
    return win, podium
