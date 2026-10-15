import json
from datetime import date
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

    def get(self, url, params=None, timeout=None, **kwargs):
        self.calls.append((url, dict(params or {})))
        self.kwargs = kwargs
        return self.responses.pop(0)


def make_client(responses, sleep=None):
    return JolpicaClient(session=FakeSession(responses), sleep=sleep or (lambda s: None))


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
    sleeps = []
    client = make_client([FakeResponse(429), FakeResponse(500),
                          FakeResponse(200, {"MRData": {"total": "1"}})], sleep=sleeps.append)
    pages = client.get_pages("2026/races")
    assert len(pages) == 1
    assert len(client.session.calls) == 3
    # Chaque tentative attend min_interval ; entre deux échecs, backoff 1.0 puis 2.0.
    assert sleeps == [0.3, 1.0, 0.3, 2.0, 0.3]


def test_raises_fetch_error_after_retries():
    sleeps = []
    client = make_client([FakeResponse(503)] * 3, sleep=sleeps.append)
    with pytest.raises(FetchError, match="2026/races"):
        client.get_pages("2026/races")
    assert len(client.session.calls) == 3
    # Pas de backoff après la dernière tentative.
    assert sleeps == [0.3, 1.0, 0.3, 2.0, 0.3]


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


def test_update_cache_refetches_past_season_missing_last_round(tmp_path):
    # Saison 2025 mise en cache alors que la manche 2 n'était pas encore publiée.
    pd.DataFrame(_rows(2025, 1, 22, False)).to_parquet(tmp_path / "results_2025.parquet")
    pd.DataFrame([
        dict(season=2025, round=1, race_name="GP", date="2025-03-01", circuit_id="c",
             has_sprint=False),
        dict(season=2025, round=2, race_name="GP", date="2025-03-08", circuit_id="c",
             has_sprint=False),
    ]).to_parquet(tmp_path / "schedule_2025.parquet")
    empty = {"MRData": {"total": "0", "RaceTable": {"Races": []}}}
    client = make_client([FakeResponse(200, empty)] * 3)
    # La réponse (vide) perdrait la manche 1 en cache : refusée, mais elle a bien été demandée.
    with pytest.raises(FetchError):
        update_cache(tmp_path, [2025], current_season=2026, client=client)
    urls = [url for url, _ in client.session.calls]
    assert urls and all("/2025/" in url for url in urls)


# --- Complétude des données (audit de pannes #1, #3, #6) ---------------------------------

def _season(n_rounds=6, n_drivers=22, sprint_rounds=(2,)):
    from tests.helpers import make_results, make_schedule
    res = make_results(seasons=(2026,), n_rounds=n_rounds, n_drivers=n_drivers,
                       sprint_rounds=sprint_rounds)
    return res, make_schedule(2026, 10, sprint_rounds=sprint_rounds)


def _fake_client(res, sched, hook=None):
    from tests.helpers import JolpicaFake
    return JolpicaClient(session=JolpicaFake(res, sched, hook), sleep=lambda s: None)


def test_fetch_season_round_trips_through_fake_api():
    from f1pred.data import fetch_season
    res, sched = _season()
    got, schedule = fetch_season(_fake_client(res, sched), 2026)
    assert len(got) == len(res)
    assert schedule["round"].tolist() == list(range(1, 11))


def test_glitched_empty_page_raises_and_leaves_cache_untouched(tmp_path):
    # Repro r3 : une page 200 OK vide au milieu de la pagination supprimait des manches.
    from tests.helpers import JolpicaResponse
    res, sched = _season()
    res.to_parquet(tmp_path / "results_2026.parquet", index=False)
    before = (tmp_path / "results_2026.parquet").read_bytes()
    total = int((~res["is_sprint"]).sum())

    def hook(url, offset):
        if url.endswith("2026/results/") and offset == 100:
            return JolpicaResponse({"MRData": {"total": str(total), "RaceTable": {"Races": []}}})

    with pytest.raises(FetchError):
        update_cache(tmp_path, [2026], current_season=2026, client=_fake_client(res, sched, hook))
    assert (tmp_path / "results_2026.parquet").read_bytes() == before


def test_fetch_raises_when_row_count_differs_from_total():
    from f1pred.data import fetch_season
    from tests.helpers import JolpicaResponse, jolpica_page
    res, sched = _season()
    gp = res[~res["is_sprint"]]

    def hook(url, offset):  # la page 2 annonce le bon total mais perd 3 lignes
        if url.endswith("2026/results/") and offset == 100:
            page = jolpica_page(gp, False, 100, 100)
            page["MRData"]["RaceTable"]["Races"][-1]["Results"] = (
                page["MRData"]["RaceTable"]["Races"][-1]["Results"][:-3])
            return JolpicaResponse(page)

    with pytest.raises(FetchError, match="2026/results"):
        fetch_season(_fake_client(res, sched, hook), 2026)


def test_fetch_raises_when_total_grows_during_pagination():
    from f1pred.data import fetch_season
    from tests.helpers import JolpicaResponse, jolpica_page
    res, sched = _season()
    gp = res[~res["is_sprint"]]

    def hook(url, offset):  # une manche est publiée entre la page 1 et la page 2
        if url.endswith("2026/results/") and offset == 100:
            return JolpicaResponse(jolpica_page(gp, False, 100, 100, total=len(gp) + 22))

    with pytest.raises(FetchError):
        fetch_season(_fake_client(res, sched, hook), 2026)


def test_update_cache_never_drops_a_cached_round(tmp_path):
    res, sched = _season()
    res.to_parquet(tmp_path / "results_2026.parquet", index=False)
    before = (tmp_path / "results_2026.parquet").read_bytes()
    shorter = res[res["round"] < 6]  # l'API ne renvoie plus la manche 6 (cohérente avec total)
    with pytest.raises(FetchError, match="6"):
        update_cache(tmp_path, [2026], current_season=2026, client=_fake_client(shorter, sched))
    assert (tmp_path / "results_2026.parquet").read_bytes() == before


def test_partial_latest_gp_is_dropped():
    # Repro r2 : 15 lignes sur 22 (= MIN_RESULTS) passaient pour une manche complète.
    df = pd.DataFrame(_rows(2026, 1, 22, False) + _rows(2026, 2, 22, False)
                      + _rows(2026, 3, 15, False))
    assert set(drop_incomplete_rounds(df)["round"]) == {1, 2}


def test_first_round_uses_min_results():
    df = pd.DataFrame(_rows(2026, 1, 15, False))
    assert set(drop_incomplete_rounds(df)["round"]) == {1}
    df = pd.DataFrame(_rows(2026, 1, 14, False))
    assert drop_incomplete_rounds(df).empty


def test_earlier_short_round_is_kept():
    # Un forfait réel (19 voitures après 20) n'est pas une publication partielle : seule la
    # dernière manche, la seule qui peut être en cours de publication, est comparée.
    df = pd.DataFrame(_rows(2024, 1, 20, False) + _rows(2024, 2, 19, False)
                      + _rows(2024, 3, 20, False))
    assert set(drop_incomplete_rounds(df)["round"]) == {1, 2, 3}


def test_latest_sprint_weekend_needs_sprint_rows():
    sched = pd.DataFrame([dict(season=2026, round=r, race_name="GP", date="2026-01-01",
                               circuit_id="c", has_sprint=r == 2) for r in (1, 2)])
    df = pd.DataFrame(_rows(2026, 1, 22, False) + _rows(2026, 2, 22, False))
    assert set(drop_incomplete_rounds(df, sched)["round"]) == {1}
    df = pd.DataFrame(_rows(2026, 1, 22, False) + _rows(2026, 2, 22, True)
                      + _rows(2026, 2, 22, False))
    assert set(drop_incomplete_rounds(df, sched)["round"]) == {1, 2}


def _dated(rows, day):
    return [dict(r, date=day) for r in rows]


def test_short_final_round_is_kept_once_two_days_old():
    # Repro I1 : une dernière manche à 21 lignes (après 22) n'était jamais publiée.
    df = pd.DataFrame(_rows(2025, 1, 22, False) + _dated(_rows(2025, 2, 21, False), "2025-12-07"))
    assert set(drop_incomplete_rounds(df, today=date(2025, 12, 9))["round"]) == {1, 2}


def test_missing_sprint_is_kept_once_two_days_old():
    sched = pd.DataFrame([dict(season=2026, round=r, race_name="GP", date="2026-05-03",
                               circuit_id="c", has_sprint=r == 2) for r in (1, 2)])
    df = pd.DataFrame(_rows(2026, 1, 22, False) + _dated(_rows(2026, 2, 22, False), "2026-05-03"))
    assert set(drop_incomplete_rounds(df, sched, today=date(2026, 5, 5))["round"]) == {1, 2}


def test_recent_short_round_is_still_held():
    # Le lundi (J+1) reste strict ; seul le mardi (J+2) publie une manche courte.
    df = pd.DataFrame(_rows(2025, 1, 22, False) + _dated(_rows(2025, 2, 21, False), "2025-12-07"))
    assert set(drop_incomplete_rounds(df, today=date(2025, 12, 8))["round"]) == {1}
    df = pd.DataFrame(_rows(2025, 1, 22, False) + _dated(_rows(2025, 2, 14, False), "2025-12-01"))
    assert set(drop_incomplete_rounds(df, today=date(2025, 12, 9))["round"]) == {1}


def test_saturday_race_keeps_monday_strict():
    # Course le samedi (Las Vegas) : le lundi reste strict, seul le mardi publie.
    df = pd.DataFrame(_rows(2026, 1, 22, False) + _dated(_rows(2026, 2, 21, False), "2026-11-21"))
    assert set(drop_incomplete_rounds(df, today=date(2026, 11, 23))["round"]) == {1}
    assert set(drop_incomplete_rounds(df, today=date(2026, 11, 24))["round"]) == {1, 2}


def test_client_refuses_redirects():
    # Une redirection (vers un autre hôte ou non) n'est jamais suivie : erreur immédiate.
    client = make_client([FakeResponse(302, {"MRData": {"total": "0"}})] * 3)
    with pytest.raises(FetchError, match="redirection"):
        client.get_pages("2026/results")
    assert len(client.session.calls) == 1
    assert client.session.kwargs.get("allow_redirects") is False
