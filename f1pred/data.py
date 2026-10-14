"""Récupération des données Jolpica (successeur d'Ergast) et cache parquet."""

import time
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path

import numpy as np
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
                    url, params={"limit": PAGE_LIMIT, "offset": offset}, timeout=30,
                    allow_redirects=False,
                )
                if 300 <= response.status_code < 400:  # jamais suivie : hôte non vérifié
                    raise FetchError(f"{path} : redirection refusée "
                                     f"(HTTP {response.status_code})")
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.retries - 1:
                    self.sleep(self.backoff * 2**attempt)
        raise FetchError(f"Échec de {path} après {self.retries} tentatives") from last_error

    def get_pages(self, path: str) -> list[dict]:
        """Toutes les pages d'une ressource ; le total de référence est celui de la dernière."""
        pages = [self._get(path, 0)]
        first_total = total = _total(pages[0])
        _check_not_empty(pages[0], path, 0, total)
        offset = PAGE_LIMIT
        while offset < total:
            page = self._get(path, offset)
            total = _total(page)
            _check_not_empty(page, path, offset, total)
            pages.append(page)
            offset += PAGE_LIMIT
        if total != first_total:  # une manche publiée pendant la pagination décale les pages
            raise FetchError(f"{path} : total passé de {first_total} à {total} en cours de "
                             "pagination")
        return pages


def _total(page: dict) -> int:
    return int(page["MRData"]["total"])


def _check_not_empty(page: dict, path: str, offset: int, total: int) -> None:
    """Une page 200 OK sans course alors que des lignes sont attendues est un incident."""
    races = page["MRData"].get("RaceTable", {}).get("Races")
    if races is not None and not races and offset < total:
        raise FetchError(f"{path} : page vide à l'offset {offset} (total {total})")


def _check_count(path: str, pages: list[dict], received: int) -> None:
    expected = _total(pages[-1])
    if received != expected:
        raise FetchError(f"{path} : {received} lignes reçues pour un total annoncé de {expected}")


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


GRACE_DAYS = 2  # course le dimanche : le lundi reste strict, le mardi publie quand même


def drop_incomplete_rounds(results: pd.DataFrame, schedule: pd.DataFrame | None = None,
                           today: date | None = None) -> pd.DataFrame:
    """Ignore toute manche dont le GP ne semble pas encore entièrement publié.

    Chaque GP doit avoir au moins MIN_RESULTS lignes. La dernière manche de chaque saison,
    la seule qui peut être en cours de publication, doit en plus avoir au moins autant de
    lignes que le GP complet précédent et, si le calendrier en fait un week-end sprint,
    ses lignes de sprint. Les manches plus anciennes plus courtes (forfait réel) restent.
    Si `today` est donné, la dernière manche est aussi acceptée (avec MIN_RESULTS lignes)
    dès que sa course date d'au moins GRACE_DAYS jours : plateau réduit, sprint annulé ou
    dernière manche de la saison ne bloquent pas indéfiniment.
    """
    # astype(bool) : après concat avec un sprint vide, la colonne est en object et ~ casse.
    is_sprint = results["is_sprint"].astype(bool)
    gp_counts = results[~is_sprint].groupby(["season", "round"]).size()
    sprint_counts = results[is_sprint].groupby(["season", "round"]).size()
    sprint_weekends = set()
    if schedule is not None and not schedule.empty:
        flagged = schedule[schedule["has_sprint"].astype(bool)]
        sprint_weekends = set(zip(flagged["season"], flagged["round"], strict=True))
    complete = set()
    for _, counts in gp_counts.groupby(level="season"):
        counts = counts[counts >= MIN_RESULTS]
        if counts.empty:
            continue
        keys = list(counts.index)
        complete.update(keys[:-1])
        latest = keys[-1]
        reference = int(counts.iloc[-2]) if len(counts) > 1 else MIN_RESULTS
        has_sprint = sprint_counts.get(latest, 0) >= MIN_RESULTS
        settled = False
        if today is not None:
            raced = results.loc[(results["season"] == latest[0])
                                & (results["round"] == latest[1]), "date"]
            settled = pd.Timestamp(raced.max()).date() <= today - timedelta(days=GRACE_DAYS)
        full = counts.iloc[-1] >= reference and (latest not in sprint_weekends or has_sprint)
        if full or settled:
            complete.add(latest)
    keep = [(s, r) in complete for s, r in zip(results["season"], results["round"], strict=True)]
    return results[np.array(keep, dtype=bool)].reset_index(drop=True)


def fetch_season(client: JolpicaClient, season: int,
                 today: date | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Résultats (GP + sprints) et calendrier d'une saison ; FetchError si incomplets."""
    frames = []
    for path, sprint in ((f"{season}/results", False), (f"{season}/sprint", True)):
        pages = client.get_pages(path)
        parsed = [parse_results(p, sprint=sprint) for p in pages]
        _check_count(path, pages, sum(len(df) for df in parsed))
        frames += parsed
    pages = client.get_pages(f"{season}/races")
    schedule = pd.concat([parse_schedule(p) for p in pages], ignore_index=True)
    _check_count(f"{season}/races", pages, len(schedule))
    results = pd.concat(frames, ignore_index=True)
    return drop_incomplete_rounds(results, schedule, today or date.today()), schedule


def _season_complete(raw_dir: Path, season: int) -> bool:
    """Saison en cache complète : résultats jusqu'à la dernière manche du calendrier."""
    results_path = raw_dir / f"results_{season}.parquet"
    schedule_path = raw_dir / f"schedule_{season}.parquet"
    if not (results_path.exists() and schedule_path.exists()):
        return False
    cached_round = pd.read_parquet(results_path, columns=["round"])["round"].max()
    scheduled_round = pd.read_parquet(schedule_path, columns=["round"])["round"].max()
    return cached_round == scheduled_round


def _sessions(df: pd.DataFrame) -> set[tuple[int, bool]]:
    return set(zip(df["round"].astype(int), df["is_sprint"].astype(bool), strict=True))


def update_cache(
    raw_dir: Path, seasons: Iterable[int], current_season: int, client: JolpicaClient,
    today: date | None = None,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        results_path = raw_dir / f"results_{season}.parquet"
        if season != current_season and _season_complete(raw_dir, season):
            continue
        results, schedule = fetch_season(client, season, today)
        if results_path.exists():
            cached = pd.read_parquet(results_path, columns=["round", "is_sprint"])
            missing = sorted(_sessions(cached) - _sessions(results))
            if missing:  # le cache ne recule jamais : une réponse incomplète ne l'écrase pas
                raise FetchError(f"Saison {season} : la réponse ne contient plus "
                                 f"(manche, sprint) = {missing} ; cache conservé")
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
