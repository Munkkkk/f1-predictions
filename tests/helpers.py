"""Jeux de données synthétiques pour les tests (aucun appel réseau)."""

import numpy as np
import pandas as pd

from f1pred.data import RESULT_COLUMNS, SCHEDULE_COLUMNS
from f1pred.rules import points_for


def _date(season: int, rnd: int) -> str:
    return (pd.Timestamp(season, 3, 1) + pd.Timedelta(weeks=2 * rnd)).date().isoformat()


def make_results(seasons=(2024, 2025), n_rounds=8, n_drivers=10, sprint_rounds=(3,), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for rnd in range(1, n_rounds + 1):
            sessions = [True, False] if rnd in sprint_rounds else [False]
            for sprint in sessions:
                order = np.argsort(np.arange(n_drivers) + rng.normal(0, 1.5, n_drivers))
                for pos, di in enumerate(order, start=1):
                    finished = bool(rng.random() > 0.08)
                    rows.append({
                        "season": season, "round": rnd, "race_name": f"GP {rnd}",
                        "date": _date(season, rnd), "circuit_id": f"c{rnd}",
                        "driver_id": f"d{di}", "driver_name": f"Pilote {di}",
                        "constructor_id": f"t{di // 2}", "position": pos,
                        "points": float(points_for(pos, sprint)) if finished else 0.0,
                        "finished": finished, "is_sprint": sprint,
                    })
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def make_schedule(season, n_rounds, sprint_rounds=(3,)):
    rows = [{"season": season, "round": r, "race_name": f"GP {r}", "date": _date(season, r),
             "circuit_id": f"c{r}", "has_sprint": r in sprint_rounds}
            for r in range(1, n_rounds + 1)]
    return pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)
