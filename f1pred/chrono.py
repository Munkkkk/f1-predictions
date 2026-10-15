"""Ordre chronologique des manches : la date de course, pas le numéro.

Une manche reportée garde son numéro mais se court plus tard ; tout ce qui raisonne en
« avant / après » passe par ici. Égalité de date : le numéro départage. Quand les dates
suivent les numéros (cas normal), cet ordre est celui des numéros.
"""

import pandas as pd


def race_index(df: pd.DataFrame) -> pd.DataFrame:
    """(season, round, race_idx) : rang chronologique de chaque manche présente dans `df`."""
    races = df.groupby(["season", "round"], as_index=False)["date"].max()
    races["_t"] = pd.to_datetime(races["date"])
    races = races.sort_values(["season", "_t", "round"], kind="stable").reset_index(drop=True)
    races["race_idx"] = range(len(races))
    return races[["season", "round", "race_idx"]]


def race_order(results: pd.DataFrame, season: int) -> list[int]:
    """Manches courues de `season`, de la plus ancienne à la plus récente."""
    races = race_index(results[results["season"] == season])
    return [int(r) for r in races["round"]]


def known_rounds(results: pd.DataFrame, season: int, upto_round: int) -> list[int]:
    """Manches de `season` connues après la manche `upto_round`, dans l'ordre chronologique."""
    order = race_order(results, season)
    return order[:order.index(upto_round) + 1]


def rounds_after(results: pd.DataFrame, season: int, checkpoint: int, n: int) -> list[int]:
    """Les `n` manches courues juste après la manche `checkpoint`."""
    order = race_order(results, season)
    start = order.index(checkpoint) + 1
    return order[start:start + n]
