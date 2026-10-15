"""Orchestration : données → features → modèle → simulation → historique → site."""

import hashlib
import json
import logging
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from f1pred.data import RESULT_COLUMNS, JolpicaClient, load_results, load_schedule, update_cache
from f1pred.features import add_future_rows, build_features
from f1pred.history import load_history, save_history, upsert
from f1pred.model import DEFAULT_SIGMA, predict, train
from f1pred.rules import COUNTBACK_DEPTH
from f1pred.shocks import HORIZON as SHOCK_HORIZON
from f1pred.shocks import Shocks, estimate_shocks, residual_records, usable_records
from f1pred.simulate import (
    RaceInput,
    SimResult,
    contenders,
    race_finish_probs,
    remaining_max_points,
    simulate_season,
)
from f1pred.site import render_site

log = logging.getLogger(__name__)

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
    sigma: float  # erreur d'une prévision à une course (backtest)
    shocks: Shocks  # aléa par course et chocs persistants utilisés par la simulation


@dataclass
class Forecast:
    """Prévisions des courses restantes, faites avec les données connues après `round`."""
    round: int
    state: SeasonState
    races: list[RaceInput]
    sigma: float
    preds: pd.DataFrame  # une ligne par (manche future, pilote) : pred_pos, p_dnf


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


LINEUP_WINDOW = 5  # GP récents qui fixent les titulaires de chaque écurie
SEATS_PER_TEAM = 2


def current_lineup(results: pd.DataFrame, season: int, upto_round: int) -> pd.DataFrame:
    """Titulaires pour la suite : par écurie, les 2 pilotes aux plus de départs sur les
    `LINEUP_WINDOW` derniers GP de la saison (égalité : le départ le plus récent).

    Un titulaire absent une course garde son baquet ; un réserviste d'un week-end ne le garde
    pas. Un pilote ne compte que pour sa dernière écurie.
    """
    gp = results[(results["season"] == season) & (results["round"] <= upto_round)
                 & ~results["is_sprint"].astype(bool)]
    window = sorted(gp["round"].unique())[-LINEUP_WINDOW:]
    gp = gp[gp["round"].isin(window)].sort_values("round")
    latest = gp.drop_duplicates("driver_id", keep="last").set_index("driver_id")
    gp = gp[gp["constructor_id"].to_numpy()
            == latest.loc[gp["driver_id"], "constructor_id"].to_numpy()]
    starts = (gp.groupby("driver_id")["round"].agg(starts="count", last="max")
              .join(latest[["driver_name", "constructor_id"]]).reset_index())
    starts = starts.sort_values(["starts", "last", "driver_id"], ascending=[False, False, True])
    lineup = starts.groupby("constructor_id", sort=False).head(SEATS_PER_TEAM)
    return lineup.sort_values("driver_id")[["driver_id", "driver_name", "constructor_id"]]


def forecast(results: pd.DataFrame, schedule: pd.DataFrame, season: int,
             upto_round: int) -> Forecast:
    """Entraîne le modèle sur les données connues après `upto_round` et prévoit la suite."""
    past = results[(results["season"] < season)
                   | ((results["season"] == season) & (results["round"] <= upto_round))]
    state = season_state(past, season)
    lineup = current_lineup(past, season, upto_round)
    raced = past.loc[past["season"] == season, "round"].unique()
    future = schedule[~schedule["round"].isin(raced)].sort_values("round")  # reports inclus

    races: list[RaceInput] = []
    sigma = DEFAULT_SIGMA
    preds = pd.DataFrame(columns=["season", "round", "driver_id", "pred_pos", "p_dnf"])
    if not future.empty:
        feats = build_features(add_future_rows(past, future, lineup))
        models = train(feats, current_season=season)
        sigma = models.sigma
        preds = predict(models, feats[feats["position"].isna()])
        for race in future.itertuples():
            p = (preds[preds["round"] == race.round].set_index("driver_id")
                 .reindex(state.drivers))
            gp_input = RaceInput(
                pred_pos=p["pred_pos"].fillna(ABSENT_POSITION).to_numpy(dtype=float),
                p_dnf=p["p_dnf"].fillna(1.0).to_numpy(dtype=float),
                is_sprint=False,
                round=int(race.round),
            )
            if race.has_sprint:
                races.append(replace(gp_input, is_sprint=True))
            races.append(gp_input)
    return Forecast(upto_round, state, races, sigma, preds)


def simulate_forecast(fc: Forecast, n_sims: int, seed: int,
                      shocks: Shocks | None = None) -> Checkpoint:
    """Simule la fin de saison ; sans `shocks` (ou saison finie), aucun choc."""
    if shocks is None or not fc.races:
        shocks = Shocks(sigma=fc.sigma)
    st = fc.state
    sim = simulate_season(st.drivers, st.points, st.countback, fc.races, sigma=shocks.sigma,
                          n_sims=n_sims, seed=seed, teams=[st.teams[d] for d in st.drivers],
                          tau_driver=shocks.tau_driver, tau_team=shocks.tau_team,
                          dnf_shock=shocks.dnf_shock)
    return Checkpoint(fc.round, st, sim, fc.races, fc.sigma, shocks)


def predict_from(results: pd.DataFrame, schedule: pd.DataFrame, season: int, upto_round: int,
                 n_sims: int, seed: int, shocks: Shocks | None = None) -> Checkpoint:
    """Probabilités de titre avec uniquement les données disponibles après `upto_round`."""
    return simulate_forecast(forecast(results, schedule, season, upto_round), n_sims, seed,
                             shocks)


def _schedule_from_results(results: pd.DataFrame, season: int) -> pd.DataFrame:
    """Calendrier d'une saison passée, reconstitué à partir de ses résultats."""
    cur = results[results["season"] == season]
    sprint_rounds = set(cur.loc[cur["is_sprint"].astype(bool), "round"])
    gp = cur[~cur["is_sprint"].astype(bool)].drop_duplicates("round").sort_values("round")
    return gp[["season", "round", "race_name", "date", "circuit_id"]].assign(
        has_sprint=gp["round"].isin(sprint_rounds).to_numpy())


def _season_records(results: pd.DataFrame, season: int) -> list[pd.DataFrame]:
    """Erreurs de prévision multi-courses d'une saison passée (un checkpoint par manche)."""
    if not (results["season"] == season).any():
        return []
    schedule = _schedule_from_results(results, season)
    rounds = schedule["round"].tolist()
    return [residual_records(forecast(results, schedule, season, rnd).preds, results, season,
                             rnd) for rnd in rounds[:-1]]


def season_checkpoints(results: pd.DataFrame, schedule: pd.DataFrame, season: int,
                       rounds: list[int], n_sims: int, seed: int,
                       n_sims_final: int | None = None) -> list[Checkpoint]:
    """Un checkpoint par manche ; les chocs de la manche r n'utilisent que les courses <= r.

    Les erreurs de prévision viennent de la saison précédente et des checkpoints déjà calculés
    de la saison en cours. La dernière manche utilise `n_sims_final` simulations.
    """
    records = _season_records(results, season - 1)
    checkpoints = []
    for rnd in rounds:
        fc = forecast(results, schedule, season, rnd)
        shocks = None
        if records:
            pool = usable_records(pd.concat(records, ignore_index=True), season, rnd)
            shocks = estimate_shocks(pool, fallback_sigma=fc.sigma)
        sims = n_sims_final if rnd == rounds[-1] and n_sims_final else n_sims
        checkpoints.append(simulate_forecast(fc, sims, seed, shocks))
        records.append(residual_records(fc.preds, results, season, rnd))
    return checkpoints


def validate(cp: Checkpoint) -> None:
    p = np.array(list(cp.sim.p_title.values()), dtype=float)
    e = np.array(list(cp.sim.expected_points.values()), dtype=float)
    if not (np.isfinite(p).all() and np.isfinite(e).all()):
        raise ValidationError(f"Valeurs non finies après la manche {cp.round}")
    if abs(p.sum() - 1.0) > 1e-3:
        raise ValidationError(f"Probabilités de somme {p.sum():.4f} après la manche {cp.round}")
    if set(cp.sim.p_title) != set(cp.state.drivers):
        raise ValidationError(f"Pilotes manquants après la manche {cp.round}")
    for field in ("sigma", "tau_driver", "tau_team", "dnf_shock"):
        value = getattr(cp.shocks, field)
        if not (np.isfinite(value) and value >= 0):
            raise ValidationError(f"{field} = {value} invalide après la manche {cp.round}")


def check_contiguous(rounds: list[int], schedule: pd.DataFrame, season: int) -> list[int]:
    """Manches 1..dernière du calendrier non courues (reportées) ; erreur si inexplicable.

    Une manche manquante encore au calendrier (reportée, ou annulée mais pas encore retirée)
    n'est qu'un avertissement : la prévision la garde à venir. Une manche courue absente du
    calendrier, ou un trou que le calendrier n'explique pas, reste une erreur.
    """
    last = rounds[-1]
    scheduled = {int(r) for r in schedule["round"]}
    missing = sorted(set(range(1, last + 1)) - set(rounds))
    unexplained = sorted((set(rounds) - scheduled) | (set(missing) - scheduled))
    if unexplained:
        raise ValidationError(f"Saison {season} : manches courues {rounds} non contiguës "
                              f"(manquantes : {missing}, hors calendrier : {unexplained})")
    return missing


def backtest_metrics(checkpoints: list[Checkpoint], results: pd.DataFrame, season: int,
                     n_sims: int, seed: int) -> dict:
    """Score de Brier de la prédiction de la course suivante, pour chaque checkpoint."""
    gp = results[(results["season"] == season) & ~results["is_sprint"].astype(bool)]
    bw, bw0, bp, bp0 = [], [], [], []
    for cp in checkpoints:
        next_gp = next((r for r in cp.races if not r.is_sprint), None)
        if next_gp is None or next_gp.round is None:
            continue
        nxt = gp[gp["round"] == next_gp.round]
        if nxt.empty:
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
        key=lambda x: (-x["points"], -x["p_title"]),  # ordre du classement officiel
    )
    n_sprints = sum(r.is_sprint for r in final.races)
    last = schedule.loc[schedule["round"] == final.round, "race_name"]
    return {
        "season": season, "updated_at": today.isoformat(),
        "last_race": str(last.iloc[0]) if not last.empty else f"manche {final.round}",
        "decided": final.sim.decided,
        "champion": (max(drivers, key=lambda x: x["p_title"])["name"]
                     if final.sim.decided else None),
        "drivers": drivers,
        "history": [e for e in history if e["season"] == season],
        "names": st.names, "backtest": metrics, "sigma": final.shocks.sigma,
        "n_sims": n_sims,
        "train_seasons": f"{first_season}–{season}",
        "remaining_points": remaining, "remaining_gp": len(final.races) - n_sprints,
        "remaining_sprints": n_sprints, "tau_driver": final.shocks.tau_driver,
        "tau_team": final.shocks.tau_team, "dnf_shock": final.shocks.dnf_shock,
        "shock_horizon": SHOCK_HORIZON, "shock_seasons": f"{season - 1}–{season}",
    }


def results_fingerprint(results: pd.DataFrame, season: int) -> str:
    """sha256 des résultats de la saison, triés et sérialisés de façon déterministe."""
    cur = results.loc[results["season"] == season, RESULT_COLUMNS]
    cur = cur.sort_values(["season", "round", "is_sprint", "driver_id"])
    return hashlib.sha256(cur.to_csv(index=False).encode("utf-8")).hexdigest()


def _load_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


FINAL_SIMS = 200_000  # checkpoint publié : assez de tirages pour les probabilités rares


def run(data_dir: Path, out_dir: Path, offline: bool, seed: int = 0, n_sims: int = 10_000,
        force: bool = False, today: date | None = None, client=None,
        n_sims_final: int = FINAL_SIMS) -> Outcome:
    """`n_sims` pour l'historique et le backtest ; `n_sims_final` pour la dernière manche."""
    today = today or date.today()
    raw = data_dir / "raw"
    if not offline:
        update_cache(raw, range(FIRST_SEASON, today.year + 1), today.year,
                     client or JolpicaClient(), today)
    results = load_results(raw)
    season = today.year if (results["season"] == today.year).any() else today.year - 1
    schedule = load_schedule(raw, season)
    rounds = sorted(int(r) for r in results.loc[results["season"] == season, "round"].unique())
    last = rounds[-1]
    postponed = check_contiguous(rounds, schedule, season)
    warning = ""
    if postponed:
        warning = f" Attention : manche(s) {postponed} non courue(s), gardée(s) à venir."
        log.warning("Saison %s : manche(s) %s non courue(s), gardée(s) à venir.",
                    season, postponed)

    hist_path = data_dir / "history.json"
    history = load_history(hist_path)
    # On ne saute le calcul que si la manche ET les résultats (corrections, sprint tardif)
    # sont ceux du dernier calcul publié.
    state_path = data_dir / "state.json"
    fingerprint = results_fingerprint(results, season)
    seen = any(e["season"] == season and e["round"] == last for e in history)
    if not force and seen and _load_state(state_path).get("results_sha256") == fingerprint:
        return Outcome(False, f"Aucune nouvelle manche : saison {season}, manche {last}."
                       + warning)

    checkpoints = season_checkpoints(results, schedule, season, rounds, n_sims, seed,
                                     n_sims_final)
    for cp in checkpoints:
        validate(cp)
        history = upsert(history, _history_entry(cp, season, schedule))
    metrics = backtest_metrics(checkpoints, results, season, n_sims, seed)

    first_season = int(results["season"].min())
    render_site(out_dir, _context(checkpoints[-1], season, history, metrics, schedule,
                                  n_sims_final, today, first_season))
    save_history(hist_path, history)
    _save_state(state_path, {"season": season, "round": last, "results_sha256": fingerprint})
    return Outcome(True, f"Site mis à jour : saison {season}, manche {last}." + warning)
