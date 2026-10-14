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


class JolpicaResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status
        self.url = ""

    def raise_for_status(self):
        import requests
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self.payload


def jolpica_page(results: pd.DataFrame, sprint: bool, offset: int, limit: int,
                 total: int | None = None) -> dict:
    """Page Jolpica (results ou sprint) construite à partir d'un DataFrame de résultats."""
    key = "SprintResults" if sprint else "Results"
    sub = results[results["is_sprint"].astype(bool) == sprint].sort_values(["round", "position"])
    races: dict = {}
    for t in sub.iloc[offset:offset + limit].itertuples():
        given, family = t.driver_name.split(" ", 1)
        race = races.setdefault(t.round, {
            "season": str(t.season), "round": str(t.round), "raceName": t.race_name,
            "date": t.date, "Circuit": {"circuitId": t.circuit_id}, key: []})
        race[key].append({
            "Driver": {"driverId": t.driver_id, "givenName": given, "familyName": family},
            "Constructor": {"constructorId": t.constructor_id}, "position": str(t.position),
            "points": str(t.points), "positionText": str(t.position) if t.finished else "R"})
    n = len(sub) if total is None else total
    return {"MRData": {"total": str(n), "RaceTable": {"Races": list(races.values())}}}


class JolpicaFake:
    """Session factice qui sert results/sprint/races d'une saison ; `hook` peut altérer une page."""

    def __init__(self, results: pd.DataFrame, schedule: pd.DataFrame, hook=None):
        self.headers = {}
        self.results, self.schedule, self.hook = results, schedule, hook
        self.calls = []

    def get(self, url, params=None, timeout=None, **kwargs):
        offset, limit = params["offset"], params["limit"]
        self.calls.append((url, offset))
        if self.hook:
            resp = self.hook(url, offset)
            if resp is not None:
                return resp
        if url.endswith("/races/"):
            races = [{"season": str(s.season), "round": str(s.round), "raceName": s.race_name,
                      "date": s.date, "Circuit": {"circuitId": s.circuit_id},
                      **({"Sprint": {}} if s.has_sprint else {})}
                     for s in self.schedule.itertuples()]
            return JolpicaResponse({"MRData": {"total": str(len(races)), "RaceTable": {
                "Races": races[offset:offset + limit]}}})
        return JolpicaResponse(jolpica_page(self.results, url.endswith("/sprint/"), offset, limit))
