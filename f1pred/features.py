"""Features par (manche, pilote). Chaque valeur n'utilise que les manches antérieures."""

import numpy as np
import pandas as pd

from f1pred.data import RESULT_COLUMNS

FEATURE_COLUMNS = [
    "drv_form3", "drv_form5", "team_form5", "circuit_hist",
    "teammate_delta", "dnf_rate10", "points_share",
]
DEFAULT_DNF_RATE = 0.1


def add_future_rows(results: pd.DataFrame, schedule: pd.DataFrame, lineup: pd.DataFrame):
    rows = [
        {
            "season": race.season, "round": race.round, "race_name": race.race_name,
            "date": race.date, "circuit_id": race.circuit_id, "driver_id": d.driver_id,
            "driver_name": d.driver_name, "constructor_id": d.constructor_id,
            "position": np.nan, "points": np.nan, "finished": np.nan, "is_sprint": False,
        }
        for race in schedule.itertuples()
        for d in lineup.itertuples()
    ]
    future = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    return pd.concat([results, future], ignore_index=True)


def _past_rolling(df: pd.DataFrame, by, col: str, window: int) -> pd.Series:
    return df.groupby(by)[col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    )


def build_features(results: pd.DataFrame) -> pd.DataFrame:
    gp = results[~results["is_sprint"].astype(bool)].copy()
    gp = gp.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)
    gp["position"] = gp["position"].astype(float)
    gp["race_idx"] = gp.groupby(["season", "round"], sort=True).ngroup()

    gp["drv_form3"] = _past_rolling(gp, "driver_id", "position", 3)
    gp["drv_form5"] = _past_rolling(gp, "driver_id", "position", 5)

    team = gp.groupby(["constructor_id", "race_idx"], as_index=False)["position"].mean()
    team = team.sort_values(["constructor_id", "race_idx"])
    team["team_form5"] = _past_rolling(team, "constructor_id", "position", 5)
    gp = gp.merge(team[["constructor_id", "race_idx", "team_form5"]],
                  on=["constructor_id", "race_idx"], how="left")

    gp["circuit_hist"] = gp.groupby(["driver_id", "circuit_id"])["position"].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    grp = gp.groupby(["race_idx", "constructor_id"])["position"]
    team_sum, team_n = grp.transform("sum"), grp.transform("count")
    mate_mean = (team_sum - gp["position"]) / (team_n - 1).where(team_n > 1)
    gp["_mate_delta"] = gp["position"] - mate_mean
    gp["teammate_delta"] = _past_rolling(gp, "driver_id", "_mate_delta", 5)

    gp["_dnf"] = 1.0 - gp["finished"].astype(float)
    gp["dnf_rate10"] = _past_rolling(gp, "driver_id", "_dnf", 10)

    pts = (results.assign(points=results["points"].fillna(0.0))
           .groupby(["season", "round", "driver_id"], as_index=False)["points"].sum()
           .sort_values(["season", "round"]))
    pts["drv_before"] = pts.groupby(["season", "driver_id"])["points"].cumsum() - pts["points"]
    totals = pts.groupby(["season", "round"], as_index=False)["points"].sum()
    totals["total_before"] = totals.groupby("season")["points"].cumsum() - totals["points"]
    pts = pts.merge(totals[["season", "round", "total_before"]], on=["season", "round"])
    pts["points_share"] = np.where(
        pts["total_before"] > 0, pts["drv_before"] / pts["total_before"].where(
            pts["total_before"] > 0), 0.0)
    gp = gp.merge(pts[["season", "round", "driver_id", "points_share"]],
                  on=["season", "round", "driver_id"], how="left")

    neutral = (gp.groupby("race_idx")["driver_id"].transform("size") + 1) / 2
    for col in ("drv_form3", "drv_form5", "team_form5"):
        gp[col] = gp[col].fillna(neutral)
    gp["circuit_hist"] = gp["circuit_hist"].fillna(gp["drv_form5"])
    gp["teammate_delta"] = gp["teammate_delta"].fillna(0.0)
    gp["dnf_rate10"] = gp["dnf_rate10"].fillna(DEFAULT_DNF_RATE)
    gp["points_share"] = gp["points_share"].fillna(0.0)

    gp = gp.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)
    return gp[[*RESULT_COLUMNS, "race_idx", *FEATURE_COLUMNS]]
