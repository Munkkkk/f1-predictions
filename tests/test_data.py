import json
from pathlib import Path

import pandas as pd
import pytest
import requests

from f1pred.data import (
    RESULT_COLUMNS,
    FetchError,
    JolpicaClient,
    drop_incomplete_rounds,
    load_results,
    load_schedule,
    parse_results,
    parse_schedule,
    update_cache,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeResponse:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Renvoie les réponses dans l'ordre et mémorise les appels."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return self.responses.pop(0)


def make_client(responses):
    return JolpicaClient(session=FakeSession(responses), sleep=lambda s: None)


def test_parse_results_columns_and_types():
    df = parse_results(load("results_page1.json"), sprint=False)
    assert list(df.columns) == RESULT_COLUMNS
    first = df.iloc[0]
    assert first.season == 2026 and first["round"] == 1
    assert first.driver_id == "russell" and first.driver_name == "George Russell"
    assert first.position == 1 and first.points == 25.0
    assert bool(first.finished) is True and bool(first.is_sprint) is False
    assert bool(df.iloc[1].finished) is False  # positionText "R" = abandon


def test_parse_sprint_results():
    df = parse_results(load("sprint.json"), sprint=True)
    assert len(df) == 1
    assert bool(df.iloc[0].is_sprint) is True
    assert df.iloc[0].points == 8.0


def test_parse_schedule_flags_sprint_weekends():
    df = parse_schedule(load("schedule.json"))
    assert list(df["has_sprint"]) == [False, True]
    assert list(df["circuit_id"]) == ["albert_park", "shanghai"]


def test_get_pages_follows_pagination():
    client = make_client([FakeResponse(200, {"MRData": {"total": "150"}}),
                          FakeResponse(200, {"MRData": {"total": "150"}})])
    pages = client.get_pages("2026/results")
    assert len(pages) == 2
    offsets = [params["offset"] for _, params in client.session.calls]
    assert offsets == [0, 100]
    assert all(params["limit"] == 100 for _, params in client.session.calls)


def test_client_sets_custom_user_agent():
    client = make_client([])
    assert "f1pred" in client.session.headers["User-Agent"]


def test_retries_on_429_then_succeeds():
    client = make_client([FakeResponse(429), FakeResponse(500),
                          FakeResponse(200, {"MRData": {"total": "1"}})])
    pages = client.get_pages("2026/races")
    assert len(pages) == 1
    assert len(client.session.calls) == 3


def test_raises_fetch_error_after_retries():
    client = make_client([FakeResponse(503)] * 3)
    with pytest.raises(FetchError, match="2026/races"):
        client.get_pages("2026/races")


def _rows(season, rnd, n, sprint):
    return [dict(season=season, round=rnd, race_name="GP", date="2026-01-01", circuit_id="c",
                 driver_id=f"d{i}", driver_name=f"D {i}", constructor_id="t", position=i + 1,
                 points=0.0, finished=True, is_sprint=sprint) for i in range(n)]


def test_drop_incomplete_rounds_drops_whole_round():
    # Manche 2 : sprint complet mais GP partiel (publication en cours) → manche entière ignorée.
    df = pd.DataFrame(
        _rows(2026, 1, 22, False) + _rows(2026, 2, 22, True) + _rows(2026, 2, 5, False)
    )
    out = drop_incomplete_rounds(df)
    assert set(out["round"]) == {1}


def test_drop_incomplete_rounds_drops_sprint_only_round():
    df = pd.DataFrame(_rows(2026, 1, 22, False) + _rows(2026, 2, 22, True))
    assert set(drop_incomplete_rounds(df)["round"]) == {1}


def test_update_cache_skips_cached_past_seasons(tmp_path):
    pd.DataFrame(_rows(2025, 1, 22, False)).to_parquet(tmp_path / "results_2025.parquet")
    pd.DataFrame([dict(season=2025, round=1, race_name="GP", date="2025-03-01",
                       circuit_id="c", has_sprint=False)]).to_parquet(
        tmp_path / "schedule_2025.parquet")
    schedule = load("schedule.json")
    results = load("results_page1.json")
    results["MRData"]["total"] = "2"
    empty = {"MRData": {"total": "0", "RaceTable": {"Races": []}}}
    # Seule 2026 est téléchargée : results, sprint, races.
    client = make_client([FakeResponse(200, results), FakeResponse(200, empty),
                          FakeResponse(200, schedule)])
    update_cache(tmp_path, [2025, 2026], current_season=2026, client=client)
    urls = [url for url, _ in client.session.calls]
    assert all("/2026/" in url for url in urls)
    assert load_schedule(tmp_path, 2026)["round"].tolist() == [1, 2]
    # Les résultats 2026 (2 lignes < MIN_RESULTS) sont jugés incomplets : seule 2025 reste.
    assert set(load_results(tmp_path)["season"]) == {2025}
