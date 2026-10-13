"""Orchestration : données → features → modèle → simulation → historique → site."""

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from f1pred.data import JolpicaClient, load_results, load_schedule, update_cache
from f1pred.features import add_future_rows, build_features
from f1pred.history import load_history, save_history, upsert
from f1pred.model import DEFAULT_SIGMA, predict, train
from f1pred.rules import COUNTBACK_DEPTH
from f1pred.simulate import (
    RaceInput,
    SimResult,
    contenders,
    race_finish_probs,
    remaining_max_points,
    simulate_season,
)
from f1pred.site import render_site

FIRST_SEASON = 2014
ABSENT_POSITION = 99.0  # pilote sans baquet : toujours dernier, et abandon garanti


class ValidationError(RuntimeError):
    """Les sorties de la simulation sont incohérentes : on ne publie pas."""


@dataclass
class SeasonState:
    drivers: list[str]
    names: dict[str, str]
    teams: dict[str, str]
    points: np.ndarray
    countback: np.ndarray
    last_round: int


@dataclass
class Checkpoint:
    round: int
    state: SeasonState
    sim: SimResult
    races: list[RaceInput]
    sigma: float
    tau_driver: float = 0.0
    tau_team: float = 0.0
    dnf_shock: float = 0.0


@dataclass
class Outcome:
    updated: bool
    message: str


def season_state(results: pd.DataFrame, season: int) -> SeasonState:
    cur = results[results["season"] == season]
    drivers = sorted(cur["driver_id"].unique())
    latest = cur.sort_values(["round", "is_sprint"]).drop_duplicates("driver_id", keep="last")
    names = dict(zip(latest["driver_id"], latest["driver_name"], strict=True))
    teams = dict(zip(latest["driver_id"], latest["constructor_id"], strict=True))
    points = (cur.groupby("driver_id")["points"].sum().reindex(drivers).fillna(0)
              .round().astype(np.int64).to_numpy())
    gp = cur[~cur["is_sprint"].astype(bool) & cur["finished"].astype(bool)]
    countback = np.zeros((len(drivers), COUNTBACK_DEPTH), dtype=np.int64)
    for i, driver in enumerate(drivers):
        positions = gp.loc[gp["driver_id"] == driver, "position"].astype(int)
        for k in range(COUNTBACK_DEPTH):
            countback[i, k] = int((positions == k + 1).sum())
    return SeasonState(drivers, names, teams, points, countback, int(cur["round"].max()))


def predict_from(results: pd.DataFrame, schedule: pd.DataFrame, season: int, upto_round: int,
                 n_sims: int, seed: int) -> Checkpoint:
    """Probabilités de titre avec uniquement les données disponibles après `upto_round`."""
    past = results[(results["season"] < season)
                   | ((results["season"] == season) & (results["round"] <= upto_round))]
    state = season_state(past, season)
    last_gp = past[(past["season"] == season) & (past["round"] == upto_round)
                   & ~past["is_sprint"].astype(bool)]
    lineup = last_gp.drop_duplicates("driver_id")[["driver_id", "driver_name", "constructor_id"]]
    future = schedule[schedule["round"] > upto_round].sort_values("round")

    races: list[RaceInput] = []
    sigma = DEFAULT_SIGMA
    shocks = {"tau_driver": 0.0, "tau_team": 0.0, "dnf_shock": 0.0}
    if not future.empty:
        feats = build_features(add_future_rows(past, future, lineup))
        models = train(feats, current_season=season)
        sigma = models.sigma
        shocks = {k: getattr(models, k) for k in shocks}
        preds = predict(models, feats[feats["position"].isna()])
        for race in future.itertuples():
            p = (preds[preds["round"] == race.round].set_index("driver_id")
                 .reindex(state.drivers))
            gp_input = RaceInput(
                pred_pos=p["pred_pos"].fillna(ABSENT_POSITION).to_numpy(dtype=float),
                p_dnf=p["p_dnf"].fillna(1.0).to_numpy(dtype=float),
                is_sprint=False,
            )
            if race.has_sprint:
                races.append(replace(gp_input, is_sprint=True))
            races.append(gp_input)

    sim = simulate_season(state.drivers, state.points, state.countback, races,
                          sigma=sigma, n_sims=n_sims, seed=seed,
                          teams=[state.teams[d] for d in state.drivers], **shocks)
    return Checkpoint(upto_round, state, sim, races, sigma, **shocks)


def validate(cp: Checkpoint) -> None:
    p = np.array(list(cp.sim.p_title.values()), dtype=float)
    e = np.array(list(cp.sim.expected_points.values()), dtype=float)
    if not (np.isfinite(p).all() and np.isfinite(e).all()):
        raise ValidationError(f"Valeurs non finies après la manche {cp.round}")
    if abs(p.sum() - 1.0) > 1e-3:
        raise ValidationError(f"Probabilités de somme {p.sum():.4f} après la manche {cp.round}")
    if set(cp.sim.p_title) != set(cp.state.drivers):
        raise ValidationError(f"Pilotes manquants après la manche {cp.round}")


def backtest_metrics(checkpoints: list[Checkpoint], results: pd.DataFrame, season: int,
                     n_sims: int, seed: int) -> dict:
    """Score de Brier de la prédiction de la course suivante, pour chaque checkpoint."""
    gp = results[(results["season"] == season) & ~results["is_sprint"].astype(bool)]
    bw, bw0, bp, bp0 = [], [], [], []
    for cp in checkpoints:
        nxt = gp[gp["round"] == cp.round + 1]
        next_gp = next((r for r in cp.races if not r.is_sprint), None)
        if nxt.empty or next_gp is None:
            continue
        drivers = cp.state.drivers
        win, podium = race_finish_probs(next_gp, cp.sigma, n_sims=n_sims, seed=seed)
        finish = nxt.set_index("driver_id").reindex(drivers)
        classified = finish["finished"].fillna(False).astype(bool).to_numpy()
        pos = finish["position"].fillna(99).to_numpy()
        y_win = ((pos == 1) & classified).astype(float)
        y_pod = ((pos <= 3) & classified).astype(float)
        n = len(drivers)
        bw.append(float(((win - y_win) ** 2).sum()))
        bw0.append(float(((1 / n - y_win) ** 2).sum()))
        bp.append(float(((podium - y_pod) ** 2).mean()))
        bp0.append(float(((3 / n - y_pod) ** 2).mean()))
    if not bw:
        return {"n_races": 0, "brier_winner": float("nan"), "brier_winner_baseline": float("nan"),
                "brier_podium": float("nan"), "brier_podium_baseline": float("nan")}
    return {"n_races": len(bw), "brier_winner": float(np.mean(bw)),
            "brier_winner_baseline": float(np.mean(bw0)), "brier_podium": float(np.mean(bp)),
            "brier_podium_baseline": float(np.mean(bp0))}


def _history_entry(cp: Checkpoint, season: int, schedule: pd.DataFrame) -> dict:
    row = schedule.loc[schedule["round"] == cp.round, ["race_name", "date"]]
    name = row["race_name"]
    return {
        "season": season, "round": cp.round,
        "date": str(row["date"].iloc[0]) if not row.empty else "",
        "race_name": str(name.iloc[0]) if not name.empty else f"Manche {cp.round}",
        "probabilities": {d: round(p, 4) for d, p in cp.sim.p_title.items()},
    }


def _context(final: Checkpoint, season: int, history: list[dict], metrics: dict,
             schedule: pd.DataFrame, n_sims: int, today: date, first_season: int) -> dict:
    st = final.state
    remaining = remaining_max_points(final.races, len(st.drivers))
    alive = contenders(st.points, remaining) | final.sim.decided
    drivers = sorted(
        ({"driver_id": d, "name": st.names[d], "team": st.teams[d],
          "points": int(st.points[i]), "max_points": int(st.points[i]) + remaining,
          "eliminated": not bool(alive[i]), "p_title": final.sim.p_title[d],
          "expected_points": final.sim.expected_points[d]} for i, d in enumerate(st.drivers)),
        key=lambda x: (-x["p_title"], -x["points"]),
    )
    n_sprints = sum(r.is_sprint for r in final.races)
    last = schedule.loc[schedule["round"] == final.round, "race_name"]
    return {
        "season": season, "updated_at": today.isoformat(),
        "last_race": str(last.iloc[0]) if not last.empty else f"manche {final.round}",
        "decided": final.sim.decided,
        "champion": drivers[0]["name"] if final.sim.decided else None,
        "drivers": drivers,
        "history": [e for e in history if e["season"] == season],
        "names": st.names, "backtest": metrics, "sigma": final.sigma, "n_sims": n_sims,
        "train_seasons": f"{first_season}–{season}",
        "remaining_points": remaining, "remaining_gp": len(final.races) - n_sprints,
        "remaining_sprints": n_sprints, "tau_driver": final.tau_driver,
        "tau_team": final.tau_team, "dnf_shock": final.dnf_shock,
    }


def run(data_dir: Path, out_dir: Path, offline: bool, seed: int = 0, n_sims: int = 10_000,
        force: bool = False, today: date | None = None, client=None) -> Outcome:
    today = today or date.today()
    raw = data_dir / "raw"
    if not offline:
        update_cache(raw, range(FIRST_SEASON, today.year + 1), today.year,
                     client or JolpicaClient())
    results = load_results(raw)
    season = today.year if (results["season"] == today.year).any() else today.year - 1
    schedule = load_schedule(raw, season)
    rounds = sorted(int(r) for r in results.loc[results["season"] == season, "round"].unique())
    last = rounds[-1]

    hist_path = data_dir / "history.json"
    history = load_history(hist_path)
    if not force and any(e["season"] == season and e["round"] == last for e in history):
        return Outcome(False, f"Aucune nouvelle manche : saison {season}, manche {last}.")

    checkpoints = []
    for rnd in rounds:
        cp = predict_from(results, schedule, season, rnd, n_sims, seed)
        validate(cp)
        checkpoints.append(cp)
        history = upsert(history, _history_entry(cp, season, schedule))
    metrics = backtest_metrics(checkpoints, results, season, n_sims, seed)

    first_season = int(results["season"].min())
    render_site(out_dir, _context(checkpoints[-1], season, history, metrics, schedule,
                                  n_sims, today, first_season))
    save_history(hist_path, history)
    return Outcome(True, f"Site mis à jour : saison {season}, manche {last}.")
