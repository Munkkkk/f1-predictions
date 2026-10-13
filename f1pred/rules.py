"""Règles sportives du championnat pilotes (valeurs 2026, FIA Section A, art. A2)."""

import numpy as np

GP_POINTS = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)
SPRINT_POINTS = (8, 7, 6, 5, 4, 3, 2, 1)

# Départage : victoires en GP, puis 2es places, etc. On s'arrête à la 6e place.
COUNTBACK_DEPTH = 6

# Une manche avec moins de résultats est considérée comme pas encore publiée.
MIN_RESULTS = 15


def points_for(position: int, sprint: bool = False) -> int:
    table = SPRINT_POINTS if sprint else GP_POINTS
    return table[position - 1] if 1 <= position <= len(table) else 0


def points_table(n_drivers: int, sprint: bool) -> np.ndarray:
    return np.array([points_for(p, sprint) for p in range(1, n_drivers + 1)], dtype=np.int64)
