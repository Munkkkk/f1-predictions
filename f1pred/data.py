"""Récupération des données Jolpica (successeur d'Ergast) et cache parquet."""

import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import requests

from f1pred.rules import MIN_RESULTS

BASE_URL = "https://api.jolpi.ca/ergast/f1"
PAGE_LIMIT = 100
USER_AGENT = "f1pred/0.1 (prédictions championnat F1)"

RESULT_COLUMNS = [
    "season", "round", "race_name", "date", "circuit_id", "driver_id", "driver_name",
    "constructor_id", "position", "points", "finished", "is_sprint",
]
SCHEDULE_COLUMNS = ["season", "round", "race_name", "date", "circuit_id", "has_sprint"]


class FetchError(RuntimeError):
    """L'API n'a pas répondu correctement après plusieurs tentatives."""


class JolpicaClient:
    def __init__(self, session=None, retries=3, backoff=1.0, min_interval=0.3, sleep=time.sleep):
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.retries = retries
        self.backoff = backoff
        self.min_interval = min_interval
        self.sleep = sleep

    def _get(self, path: str, offset: int) -> dict:
        url = f"{BASE_URL}/{path}/"
        last_error = None
        for attempt in range(self.retries):
            self.sleep(self.min_interval)
            try:
                response = self.session.get(
                    url, params={"limit": PAGE_LIMIT, "offset": offset}, timeout=30
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.retries - 1:
                    self.sleep(self.backoff * 2**attempt)
        raise FetchError(f"Échec de {path} après {self.retries} tentatives") from last_error

    def get_pages(self, path: str) -> list[dict]:
        pages = [self._get(path, 0)]
        total = int(pages[0]["MRData"]["total"])
        offset = PAGE_LIMIT
        while offset < total:
            pages.append(self._get(path, offset))
            offset += PAGE_LIMIT
        return pages


def parse_results(payload: dict, sprint: bool) -> pd.DataFrame:
    key = "SprintResults" if sprint else "Results"
    rows = []
    for race in payload["MRData"]["RaceTable"]["Races"]:
        for res in race.get(key, []):
            driver = res["Driver"]
            rows.append({
                "season": int(race["season"]),
                "round": int(race["round"]),
                "race_name": race["raceName"],
                "date": race["date"],
                "circuit_id": race["Circuit"]["circuitId"],
                "driver_id": driver["driverId"],
                "driver_name": f"{driver['givenName']} {driver['familyName']}",
                "constructor_id": res["Constructor"]["constructorId"],
                "position": int(res["position"]),
                "points": float(res["points"]),
                "finished": res["positionText"].isdigit(),
                "is_sprint": sprint,
            })
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def parse_schedule(payload: dict) -> pd.DataFrame:
    rows = [
        {
            "season": int(race["season"]),
            "round": int(race["round"]),
            "race_name": race["raceName"],
            "date": race["date"],
            "circuit_id": race["Circuit"]["circuitId"],
            "has_sprint": "Sprint" in race,
        }
        for race in payload["MRData"]["RaceTable"]["Races"]
    ]
    return pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)


def drop_incomplete_rounds(results: pd.DataFrame) -> pd.DataFrame:
    """Ignore toute manche dont le GP n'a pas encore au moins MIN_RESULTS résultats."""
    # astype(bool) : après concat avec un sprint vide, la colonne est en object et ~ casse.
    gp_counts = results[~results["is_sprint"].astype(bool)].groupby(["season", "round"]).size()
    complete = set(gp_counts[gp_counts >= MIN_RESULTS].index)
    keep = [(s, r) in complete for s, r in zip(results["season"], results["round"], strict=True)]
    return results[keep].reset_index(drop=True)


def fetch_season(client: JolpicaClient, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    gp = [parse_results(p, sprint=False) for p in client.get_pages(f"{season}/results")]
    sprint = [parse_results(p, sprint=True) for p in client.get_pages(f"{season}/sprint")]
    schedule = pd.concat(
        [parse_schedule(p) for p in client.get_pages(f"{season}/races")], ignore_index=True
    )
    results = pd.concat(gp + sprint, ignore_index=True)
    return drop_incomplete_rounds(results), schedule


def update_cache(
    raw_dir: Path, seasons: Iterable[int], current_season: int, client: JolpicaClient
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        results_path = raw_dir / f"results_{season}.parquet"
        if results_path.exists() and season != current_season:
            continue
        results, schedule = fetch_season(client, season)
        if not schedule.empty:
            schedule.to_parquet(raw_dir / f"schedule_{season}.parquet", index=False)
        if not results.empty:
            results.to_parquet(results_path, index=False)


def load_results(raw_dir: Path) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(raw_dir.glob("results_*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["season", "round", "is_sprint", "position"]).reset_index(drop=True)


def load_schedule(raw_dir: Path, season: int) -> pd.DataFrame:
    return pd.read_parquet(raw_dir / f"schedule_{season}.parquet")
