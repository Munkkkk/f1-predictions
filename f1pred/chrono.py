"""Ordre chronologique des manches : la date de course, pas le numéro.

Une manche reportée garde son numéro mais se court plus tard ; tout ce qui raisonne en
« avant / après » passe par ici. Égalité de date : le numéro départage. Quand les dates
suivent les numéros (cas normal), cet ordre est celui des numéros.
"""

from collections.abc import Collection
from datetime import date, timedelta

import pandas as pd

CANCEL_GRACE_DAYS = 7  # manche non courue plus de 7 jours après sa date : annulée


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


def round_date(results: pd.DataFrame, season: int, rnd: int) -> date:
    """Date de course de la manche `rnd` de `season`, d'après ses résultats."""
    cur = results[(results["season"] == season) & (results["round"] == rnd)]
    return pd.Timestamp(cur["date"].max()).date()


def cancelled_rounds(schedule: pd.DataFrame, raced: Collection[int], before: date) -> list[int]:
    """Manches du calendrier jamais courues dont la date est antérieure à `before`.

    Une manche reportée avec une nouvelle date (à venir) n'en fait pas partie.
    """
    dates = pd.to_datetime(schedule["date"]).dt.date
    gone = schedule[~schedule["round"].isin(list(raced)) & (dates < before)]
    return sorted(int(r) for r in gone["round"])


def cancel_cutoff(today: date) -> date:
    """Au checkpoint publié : une manche datée d'avant ce jour et non courue est annulée."""
    return today - timedelta(days=CANCEL_GRACE_DAYS)
