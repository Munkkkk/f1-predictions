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


SIM_CHUNK = 50_000  # simulations par lot, pour borner la mémoire
MAX_SHOCKED_DNF = 0.9  # plafond du taux d'abandon après choc de fiabilité


def _draw_race(race: RaceInput, sigma: float, n_sims: int, rng: np.random.Generator,
               shift: np.ndarray | None = None, p_dnf: np.ndarray | None = None):
    """Renvoie (rangs 0-indexés, masque d'abandon), chacun de forme (n_sims, n).

    `shift` (n_sims, n) décale la performance de chaque simulation ; `p_dnf` (n_sims, n)
    remplace le taux d'abandon de la course. Sans eux, le flux aléatoire est inchangé.
    """
    n = len(race.pred_pos)
    perf = race.pred_pos[None, :] + rng.normal(0.0, sigma, (n_sims, n))
    if shift is not None:
        perf = perf + shift
    dnf = rng.random((n_sims, n)) < (race.p_dnf[None, :] if p_dnf is None else p_dnf)
    perf = np.where(dnf, np.inf, perf)
    order = np.argsort(perf, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(n), (n_sims, n)), axis=1)
    return ranks, dnf


def remaining_max_points(races: list[RaceInput], n_drivers: int) -> int:
    """Points encore en jeu : la somme des points du vainqueur de chaque manche restante."""
    return sum(int(points_table(n_drivers, r.is_sprint)[0]) for r in races)


def contenders(points: np.ndarray, remaining_max: int) -> np.ndarray:
    """Pilotes encore mathématiquement en course (égalité possible : départage)."""
    points = np.asarray(points)
    return points + remaining_max >= points.max()


def _is_decided(points: np.ndarray, races: list[RaceInput]) -> bool:
    if not races:
        return True
    remaining = remaining_max_points(races, len(points))
    leader = int(np.argmax(points))
    others = np.delete(points, leader)
    return bool(np.all(points[leader] > others + remaining))


def _draw_shocks(n_sims: int, n: int, rng: np.random.Generator, teams: list[str] | None,
                 tau_driver: float, tau_team: float, dnf_shock: float):
    """Chocs persistants d'une simulation : (décalage de forme, multiplicateur d'abandon).

    Rien n'est tiré (et None est renvoyé) pour un choc nul, afin de garder le flux aléatoire.
    """
    shift = None
    if tau_driver > 0:
        shift = rng.normal(0.0, tau_driver, (n_sims, n))
    if tau_team > 0 and teams is not None:
        _, team_idx = np.unique(np.asarray(teams), return_inverse=True)
        team_shift = rng.normal(0.0, tau_team, (n_sims, team_idx.max() + 1))[:, team_idx]
        shift = team_shift if shift is None else shift + team_shift
    mult = None
    if dnf_shock > 0:
        # Multiplicateur log-normal de moyenne 1.
        mult = np.exp(rng.normal(0.0, dnf_shock, (n_sims, n)) - dnf_shock**2 / 2)
    return shift, mult


def _shocked_dnf(p_dnf: np.ndarray, mult: np.ndarray | None) -> np.ndarray | None:
    if mult is None:
        return None
    shocked = np.clip(p_dnf[None, :] * mult, 0.0, MAX_SHOCKED_DNF)
    return np.where(p_dnf[None, :] >= 1.0, 1.0, shocked)  # pilote absent : toujours dehors


def _simulate_totals(points: np.ndarray, countback: np.ndarray, races: list[RaceInput],
                     sigma: float, n_sims: int, seed, *, teams: list[str] | None = None,
                     tau_driver: float = 0.0, tau_team: float = 0.0, dnf_shock: float = 0.0):
    """Renvoie (points finaux, compteurs de départage, nombre d'abandons) par simulation.

    `seed` est une graine ou un `np.random.Generator` (repris tel quel par `default_rng`).
    """
    rng = np.random.default_rng(seed)
    n = len(points)
    total = np.tile(np.asarray(points, dtype=np.int64), (n_sims, 1))
    counts = np.tile(np.asarray(countback, dtype=np.int64), (n_sims, 1, 1))
    dnf_count = np.zeros((n_sims, n), dtype=np.int64)
    shift, mult = _draw_shocks(n_sims, n, rng, teams, tau_driver, tau_team, dnf_shock)

    for race in races:
        ranks, dnf = _draw_race(race, sigma, n_sims, rng, shift, _shocked_dnf(race.p_dnf, mult))
        table = points_table(n, race.is_sprint)
        total += np.where(dnf, 0, table[ranks])
        dnf_count += dnf
        if not race.is_sprint:
            for k in range(min(COUNTBACK_DEPTH, n)):
                counts[:, :, k] += (ranks == k) & ~dnf
    return total, counts, dnf_count


def simulate_season(drivers: list[str], points: np.ndarray, countback: np.ndarray,
                    races: list[RaceInput], sigma: float, n_sims: int = 10_000,
                    seed: int = 0, *, teams: list[str] | None = None, tau_driver: float = 0.0,
                    tau_team: float = 0.0, dnf_shock: float = 0.0,
                    chunk_size: int = SIM_CHUNK) -> SimResult:
    """Simule la fin de saison. Les chocs `tau_*` (en places) et `dnf_shock` (log-écart-type)
    sont tirés une fois par simulation et s'appliquent à toutes ses courses restantes.
    Les simulations sont faites par lots de `chunk_size`, avec un seul flux aléatoire."""
    n = len(drivers)
    rng = np.random.default_rng(seed)
    weights = COUNTBACK_BASE ** np.arange(COUNTBACK_DEPTH - 1, -1, -1, dtype=np.int64)
    titles = np.zeros(n, dtype=np.int64)
    points_sum = np.zeros(n, dtype=np.int64)
    for start in range(0, n_sims, chunk_size):
        size = min(chunk_size, n_sims - start)
        total, counts, _ = _simulate_totals(points, countback, races, sigma, size, rng,
                                            teams=teams, tau_driver=tau_driver,
                                            tau_team=tau_team, dnf_shock=dnf_shock)
        key = total * COUNTBACK_BASE**COUNTBACK_DEPTH + (counts * weights).sum(axis=2)
        titles += np.bincount(key.argmax(axis=1), minlength=n)
        points_sum += total.sum(axis=0)
    p = titles / n_sims
    expected = points_sum / n_sims
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
