# F1 Predictions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un pipeline Python qui récupère les résultats F1, entraîne un modèle LightGBM, simule la fin de saison en Monte Carlo et publie un site statique des probabilités de titre pilotes, mis à jour automatiquement par GitHub Actions.

**Architecture:** Package `f1pred/` en modules à responsabilité unique (`rules`, `data`, `features`, `model`, `simulate`, `history`, `site`), orchestrés par `pipeline.py` et exposés par `cli.py`. Les données brutes (parquet) et `history.json` sont committés ; le site est généré dans `site/` et déployé sur GitHub Pages.

**Tech Stack:** Python ≥ 3.12, uv, pandas, numpy, LightGBM, requests, Jinja2, Plotly, pyarrow, pytest, ruff, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-13-f1-predictions-design.md`
**Faits vérifiés (barème, calendrier, API) :** `docs/research/2026-10-13-f1-data-and-rules.md`

**Écart assumé avec la spec :** le backtest et l'orchestration vivent dans `f1pred/pipeline.py` (et non dans `model.py`), car ils combinent modèle et simulation ; `model.py` reste limité à l'entraînement et à la prédiction. La saison en cours est re-téléchargée en entier à chaque exécution (≈ 8 requêtes) ; seules les saisons passées sont mises en cache définitivement.

## Global Constraints

- Python : `requires-python = ">=3.12"` (le poste local a Python 3.13 ; la CI utilise 3.12).
- Barème GP 2026 : 25-18-15-12-10-8-6-4-2-1 ; pas de point du meilleur tour.
- Barème sprint 2026 : 8-7-6-5-4-3-2-1.
- Départage : nombre de victoires en GP, puis de 2es places, etc. (sprints exclus). Implémenté jusqu'à la 6e place (`COUNTBACK_DEPTH = 6`).
- Grille 2026 : 22 voitures, 11 écuries. Aucun code ne doit supposer 20 pilotes.
- API : `https://api.jolpi.ca/ergast/f1`, `limit` ≤ 100, ≤ 4 req/s et ≤ 500 req/h, User-Agent personnalisé obligatoire.
- Historique d'entraînement : saisons 2014 à la saison en cours.
- Simulation : 10 000 tirages par défaut, seed configurable.
- Aucun appel réseau dans les tests.
- Textes du site en français.
- Toutes les commandes se lancent depuis la racine du repo avec `uv run …`.

## Review Focus

1. **Pilote remplacé en cours de saison** (présent au classement, absent de la dernière manche) : il garde ses points, marque 0 point dans les courses simulées, sans crash. → test dans la Task 7.
2. **Lundi matin, sprint publié mais pas le GP** (ou GP partiel) : la manche entière est ignorée jusqu'au passage suivant. → test dans la Task 2.
3. **Janvier, saison pas commencée** : le pipeline retombe sur la saison précédente au lieu de planter. → test dans la Task 7.
4. **Écurie avec un seul pilote classé sur une manche** : `teammate_delta` ne divise pas par zéro. → test dans la Task 3.
5. **API qui répond 429 ou 500** : nouvelles tentatives avec backoff, puis `FetchError` claire. → test dans la Task 2.

---

## File Structure

```
pyproject.toml                    dépendances, config ruff/pytest
.gitignore
README.md
f1pred/__init__.py
f1pred/__main__.py                `python -m f1pred`
f1pred/rules.py                   barèmes, départage, constantes
f1pred/data.py                    client Jolpica, parsing, cache parquet
f1pred/features.py                features sans fuite + lignes futures
f1pred/model.py                   LightGBM position + abandon
f1pred/simulate.py                Monte Carlo saison + probas d'une course
f1pred/history.py                 lecture/écriture history.json
f1pred/site.py                    rendu Jinja2 + Plotly
f1pred/templates/base.html
f1pred/templates/index.html
f1pred/templates/methode.html
f1pred/pipeline.py                checkpoints, backtest, validation, run()
f1pred/cli.py                     argparse, sortie GITHUB_OUTPUT
tests/__init__.py
tests/helpers.py                  générateur de résultats synthétiques
tests/fixtures/results_page1.json
tests/fixtures/results_page2.json
tests/fixtures/sprint.json
tests/fixtures/schedule.json
tests/test_rules.py
tests/test_data.py
tests/test_features.py
tests/test_model.py
tests/test_simulate.py
tests/test_history.py
tests/test_site.py
tests/test_pipeline.py
.github/workflows/test.yml
.github/workflows/update.yml
```

---

### Task 1: Scaffold + règles sportives

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `f1pred/__init__.py`, `f1pred/rules.py`, `tests/__init__.py`
- Test: `tests/test_rules.py`

**Interfaces:**
- Consumes: rien.
- Produces (`f1pred/rules.py`) :
  - `GP_POINTS: tuple[int, ...]`, `SPRINT_POINTS: tuple[int, ...]`
  - `COUNTBACK_DEPTH: int = 6`, `MIN_RESULTS: int = 15`
  - `points_for(position: int, sprint: bool = False) -> int`
  - `points_table(n_drivers: int, sprint: bool) -> numpy.ndarray` (int64, longueur `n_drivers`, index 0 = 1re place)

- [ ] **Step 1: Installer uv si absent**

Run: `uv --version || python -m pip install --user uv`
Expected: une version de uv s'affiche (relancer `uv --version` après installation ; si la commande reste introuvable, utiliser `python -m uv` à la place de `uv` dans tout le plan).

- [ ] **Step 2: Créer `pyproject.toml`**

```toml
[project]
name = "f1pred"
version = "0.1.0"
description = "Probabilités de titre mondial F1 (LightGBM + Monte Carlo)"
requires-python = ">=3.12"
dependencies = [
    "pandas>=2.2",
    "numpy>=1.26",
    "lightgbm>=4.3",
    "requests>=2.31",
    "jinja2>=3.1",
    "plotly>=5.20",
    "pyarrow>=15",
]

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.4"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["f1pred"]

[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 3: Créer `.gitignore`, `f1pred/__init__.py`, `tests/__init__.py`**

`.gitignore` :
```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
site/
```

`f1pred/__init__.py` :
```python
"""Prédictions du championnat du monde de F1."""
```

`tests/__init__.py` : fichier vide.

- [ ] **Step 4: Écrire le test qui échoue**

`tests/test_rules.py` :
```python
from f1pred.rules import GP_POINTS, SPRINT_POINTS, points_for, points_table


def test_gp_points_2026():
    assert GP_POINTS == (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)
    assert points_for(1) == 25
    assert points_for(10) == 1
    assert points_for(11) == 0


def test_sprint_points_2026():
    assert SPRINT_POINTS == (8, 7, 6, 5, 4, 3, 2, 1)
    assert points_for(1, sprint=True) == 8
    assert points_for(9, sprint=True) == 0


def test_points_table_matches_grid_size():
    table = points_table(22, sprint=False)
    assert len(table) == 22
    assert table[0] == 25 and table[9] == 1 and table[21] == 0
    assert list(points_table(3, sprint=True)) == [8, 7, 6]
```

- [ ] **Step 5: Lancer le test pour vérifier qu'il échoue**

Run: `uv sync && uv run pytest tests/test_rules.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'f1pred.rules'`

- [ ] **Step 6: Implémenter `f1pred/rules.py`**

```python
"""Règles sportives du championnat pilotes (valeurs 2026, FIA Section A, art. A2)."""

import numpy as np

GP_POINTS = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)
SPRINT_POINTS = (8, 7, 6, 5, 4, 3, 2, 1)

# Départage : victoires en GP, puis 2es places, etc. On s'arrête à la 6e place.
COUNTBACK_DEPTH = 6

# Une manche avec moins de résultats est considérée comme pas encore publiée.
MIN_RESULTS = 15


def points_for(position: int, sprint: bool = False) -> int:
    table = SPRINT_POINTS if sprint else GP_POINTS
    return table[position - 1] if 1 <= position <= len(table) else 0


def points_table(n_drivers: int, sprint: bool) -> np.ndarray:
    return np.array([points_for(p, sprint) for p in range(1, n_drivers + 1)], dtype=np.int64)
```

- [ ] **Step 7: Lancer les tests et ruff**

Run: `uv run pytest tests/test_rules.py -v && uv run ruff check .`
Expected: 3 passed ; ruff `All checks passed!`

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock .gitignore f1pred tests
git commit -m "feat: scaffold du projet et barèmes 2026"
```

---

### Task 2: Données Jolpica (client, parsing, cache)

**Files:**
- Create: `f1pred/data.py`, `tests/fixtures/results_page1.json`, `tests/fixtures/results_page2.json`, `tests/fixtures/sprint.json`, `tests/fixtures/schedule.json`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `f1pred.rules.MIN_RESULTS`.
- Produces (`f1pred/data.py`) :
  - `RESULT_COLUMNS: list[str]` = `["season", "round", "race_name", "date", "circuit_id", "driver_id", "driver_name", "constructor_id", "position", "points", "finished", "is_sprint"]`
  - `SCHEDULE_COLUMNS: list[str]` = `["season", "round", "race_name", "date", "circuit_id", "has_sprint"]`
  - `class FetchError(RuntimeError)`
  - `class JolpicaClient(session=None, retries=3, backoff=1.0, min_interval=0.3, sleep=time.sleep)` avec `get_pages(path: str) -> list[dict]`
  - `parse_results(payload: dict, sprint: bool) -> pd.DataFrame` (colonnes `RESULT_COLUMNS`)
  - `parse_schedule(payload: dict) -> pd.DataFrame` (colonnes `SCHEDULE_COLUMNS`)
  - `drop_incomplete_rounds(results: pd.DataFrame) -> pd.DataFrame`
  - `fetch_season(client, season: int) -> tuple[pd.DataFrame, pd.DataFrame]` (résultats GP+sprint, calendrier)
  - `update_cache(raw_dir: Path, seasons: Iterable[int], current_season: int, client: JolpicaClient) -> None`
  - `load_results(raw_dir: Path) -> pd.DataFrame`
  - `load_schedule(raw_dir: Path, season: int) -> pd.DataFrame`
  - Fichiers : `raw_dir/results_{season}.parquet`, `raw_dir/schedule_{season}.parquet`

Types des colonnes : `season`, `round` int ; `position` int (rang officiel, y compris abandons) ; `points` float ; `finished` bool (vrai si `positionText` est numérique = classé) ; `is_sprint` bool ; `date` str ISO.

- [ ] **Step 1: Créer les fixtures (format Ergast/Jolpica réel, réduit)**

`tests/fixtures/results_page1.json` :
```json
{"MRData": {"limit": "2", "offset": "0", "total": "3", "RaceTable": {"season": "2026", "Races": [
  {"season": "2026", "round": "1", "raceName": "Australian Grand Prix", "date": "2026-03-08",
   "Circuit": {"circuitId": "albert_park"},
   "Results": [
     {"position": "1", "positionText": "1", "points": "25",
      "Driver": {"driverId": "russell", "givenName": "George", "familyName": "Russell"},
      "Constructor": {"constructorId": "mercedes"}, "status": "Finished"},
     {"position": "2", "positionText": "R", "points": "0",
      "Driver": {"driverId": "antonelli", "givenName": "Andrea Kimi", "familyName": "Antonelli"},
      "Constructor": {"constructorId": "mercedes"}, "status": "Engine"}
   ]}
]}}}
```

`tests/fixtures/results_page2.json` :
```json
{"MRData": {"limit": "2", "offset": "2", "total": "3", "RaceTable": {"season": "2026", "Races": [
  {"season": "2026", "round": "1", "raceName": "Australian Grand Prix", "date": "2026-03-08",
   "Circuit": {"circuitId": "albert_park"},
   "Results": [
     {"position": "3", "positionText": "3", "points": "15",
      "Driver": {"driverId": "hamilton", "givenName": "Lewis", "familyName": "Hamilton"},
      "Constructor": {"constructorId": "ferrari"}, "status": "+1 Lap"}
   ]}
]}}}
```

`tests/fixtures/sprint.json` :
```json
{"MRData": {"limit": "100", "offset": "0", "total": "1", "RaceTable": {"season": "2026", "Races": [
  {"season": "2026", "round": "2", "raceName": "Chinese Grand Prix", "date": "2026-03-15",
   "Circuit": {"circuitId": "shanghai"},
   "SprintResults": [
     {"position": "1", "positionText": "1", "points": "8",
      "Driver": {"driverId": "antonelli", "givenName": "Andrea Kimi", "familyName": "Antonelli"},
      "Constructor": {"constructorId": "mercedes"}, "status": "Finished"}
   ]}
]}}}
```

`tests/fixtures/schedule.json` :
```json
{"MRData": {"limit": "100", "offset": "0", "total": "2", "RaceTable": {"season": "2026", "Races": [
  {"season": "2026", "round": "1", "raceName": "Australian Grand Prix", "date": "2026-03-08",
   "Circuit": {"circuitId": "albert_park"}},
  {"season": "2026", "round": "2", "raceName": "Chinese Grand Prix", "date": "2026-03-15",
   "Circuit": {"circuitId": "shanghai"}, "Sprint": {"date": "2026-03-14", "time": "03:00:00Z"}}
]}}}
```

- [ ] **Step 2: Écrire les tests qui échouent**

`tests/test_data.py` :
```python
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
    df = pd.DataFrame(_rows(2026, 1, 22, False) + _rows(2026, 2, 22, True) + _rows(2026, 2, 5, False))
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
```

- [ ] **Step 3: Lancer les tests pour vérifier qu'ils échouent**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'f1pred.data'`

- [ ] **Step 4: Implémenter `f1pred/data.py`**

```python
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
    gp_counts = results[~results["is_sprint"]].groupby(["season", "round"]).size()
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
```

- [ ] **Step 5: Lancer les tests**

Run: `uv run pytest tests/test_data.py -v && uv run ruff check .`
Expected: 10 passed ; ruff OK.

- [ ] **Step 6: Commit**

```bash
git add f1pred/data.py tests/test_data.py tests/fixtures
git commit -m "feat: client Jolpica, parsing et cache parquet"
```

---

### Task 3: Features sans fuite d'information

**Files:**
- Create: `f1pred/features.py`, `tests/helpers.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `f1pred.data.RESULT_COLUMNS`, `f1pred.rules.points_for`.
- Produces :
  - `tests/helpers.make_results(seasons=(2024, 2025), n_rounds=8, n_drivers=10, sprint_rounds=(3,), seed=0) -> pd.DataFrame` (colonnes `RESULT_COLUMNS` ; pilote `d0` le plus fort ; écuries `t0`…, 2 pilotes chacune)
  - `tests/helpers.make_schedule(season: int, n_rounds: int, sprint_rounds=(3,)) -> pd.DataFrame` (colonnes `SCHEDULE_COLUMNS`)
  - `f1pred.features.FEATURE_COLUMNS: list[str]` = `["drv_form3", "drv_form5", "team_form5", "circuit_hist", "teammate_delta", "dnf_rate10", "points_share"]`
  - `add_future_rows(results, schedule, lineup) -> pd.DataFrame` — ajoute une ligne GP par (manche de `schedule`, pilote de `lineup`) avec `position`, `points`, `finished` à NaN. `lineup` a les colonnes `driver_id, driver_name, constructor_id`.
  - `build_features(results) -> pd.DataFrame` — une ligne par (manche GP, pilote), colonnes `RESULT_COLUMNS` (sans les lignes sprint) + `race_idx` (int, ordre chronologique) + `FEATURE_COLUMNS`. Aucune valeur manquante dans `FEATURE_COLUMNS`.

- [ ] **Step 1: Créer `tests/helpers.py`**

```python
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
```

- [ ] **Step 2: Écrire les tests qui échouent**

`tests/test_features.py` :
```python
import pandas as pd
import pytest

from f1pred.features import FEATURE_COLUMNS, add_future_rows, build_features
from tests.helpers import make_results, make_schedule


def test_one_row_per_gp_driver_and_no_missing_features():
    results = make_results()
    feats = build_features(results)
    gp_rows = (~results["is_sprint"]).sum()
    assert len(feats) == gp_rows
    assert not feats[FEATURE_COLUMNS].isna().any().any()


def test_no_leakage_from_future_races():
    results = make_results()
    base = build_features(results)
    last = (results["season"] == 2025) & (results["round"] == 8)
    tampered = results.copy()
    tampered.loc[last, "position"] = 99
    tampered.loc[last, "points"] = 999.0
    tampered.loc[last, "finished"] = False
    after = build_features(tampered)
    cols = ["season", "round", "driver_id", *FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(base[cols], after[cols])


def test_rolling_form_by_hand():
    rows = []
    for rnd, pos in enumerate([1, 3, 5, 7], start=1):
        rows.append(dict(season=2026, round=rnd, race_name="GP", date=f"2026-0{rnd}-01",
                         circuit_id="c", driver_id="a", driver_name="A", constructor_id="t",
                         position=pos, points=0.0, finished=True, is_sprint=False))
    feats = build_features(pd.DataFrame(rows)).set_index("round")
    assert feats.loc[1, "drv_form3"] == pytest.approx(10.5)  # aucun passé : valeur neutre
    assert feats.loc[2, "drv_form3"] == pytest.approx(1.0)
    assert feats.loc[4, "drv_form3"] == pytest.approx(3.0)  # moyenne de 1, 3, 5


def test_teammate_delta_handles_single_driver_team():
    rows = [dict(season=2026, round=r, race_name="GP", date=f"2026-0{r}-01", circuit_id="c",
                 driver_id="solo", driver_name="Solo", constructor_id="alone", position=4,
                 points=12.0, finished=True, is_sprint=False) for r in (1, 2, 3)]
    feats = build_features(pd.DataFrame(rows))
    assert (feats["teammate_delta"] == 0).all()


def test_points_share_includes_sprints_from_previous_rounds():
    results = make_results(seasons=(2025,), n_rounds=4, sprint_rounds=(1,))
    feats = build_features(results)
    r1 = results[results["round"] == 1].groupby("driver_id")["points"].sum()
    expected = r1 / r1.sum()
    got = feats[feats["round"] == 2].set_index("driver_id")["points_share"]
    pd.testing.assert_series_equal(got.sort_index(), expected.sort_index(), check_names=False)
    assert (feats[feats["round"] == 1]["points_share"] == 0).all()


def test_add_future_rows_creates_unknown_outcomes():
    results = make_results(seasons=(2025,), n_rounds=5)
    future = make_schedule(2025, 8).query("round > 5")
    lineup = (results[results["round"] == 5]
              .drop_duplicates("driver_id")[["driver_id", "driver_name", "constructor_id"]])
    full = add_future_rows(results, future, lineup)
    new = full[full["round"] > 5]
    assert len(new) == 3 * len(lineup)
    assert new["position"].isna().all()
    feats = build_features(full)
    assert not feats[FEATURE_COLUMNS].isna().any().any()
```

- [ ] **Step 3: Lancer les tests pour vérifier qu'ils échouent**

Run: `uv run pytest tests/test_features.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'f1pred.features'`

- [ ] **Step 4: Implémenter `f1pred/features.py`**

```python
"""Features par (manche, pilote). Chaque valeur n'utilise que les manches antérieures."""

import numpy as np
import pandas as pd

from f1pred.data import RESULT_COLUMNS

FEATURE_COLUMNS = [
    "drv_form3", "drv_form5", "team_form5", "circuit_hist",
    "teammate_delta", "dnf_rate10", "points_share",
]
NEUTRAL_POSITION = 10.5
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

    gp["drv_form3"] = gp["drv_form3"].fillna(NEUTRAL_POSITION)
    gp["drv_form5"] = gp["drv_form5"].fillna(NEUTRAL_POSITION)
    gp["team_form5"] = gp["team_form5"].fillna(NEUTRAL_POSITION)
    gp["circuit_hist"] = gp["circuit_hist"].fillna(gp["drv_form5"])
    gp["teammate_delta"] = gp["teammate_delta"].fillna(0.0)
    gp["dnf_rate10"] = gp["dnf_rate10"].fillna(DEFAULT_DNF_RATE)
    gp["points_share"] = gp["points_share"].fillna(0.0)

    gp = gp.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)
    return gp[[*RESULT_COLUMNS, "race_idx", *FEATURE_COLUMNS]]
```

- [ ] **Step 5: Lancer les tests**

Run: `uv run pytest tests/test_features.py -v && uv run ruff check .`
Expected: 6 passed ; ruff OK.

- [ ] **Step 6: Commit**

```bash
git add f1pred/features.py tests/helpers.py tests/test_features.py
git commit -m "feat: features de forme sans fuite d'information"
```

---

### Task 4: Modèle LightGBM

**Files:**
- Create: `f1pred/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `f1pred.features.FEATURE_COLUMNS`, `build_features` (dans les tests), `tests.helpers.make_results`.
- Produces (`f1pred/model.py`) :
  - `@dataclass Models(reg, clf, dnf_base: float, sigma: float)` — `clf` vaut `None` si aucune classe d'abandon n'est observée.
  - `sample_weights(seasons: pd.Series, current_season: int, boost: float = 5.0, decay: float = 0.8) -> np.ndarray`
  - `train(features: pd.DataFrame, current_season: int) -> Models` — n'utilise que les lignes où `position` est connue.
  - `predict(models: Models, rows: pd.DataFrame) -> pd.DataFrame` — colonnes `season, round, driver_id, pred_pos, p_dnf` ; `pred_pos ≥ 1`, `p_dnf ∈ [0, 1]`.

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/test_model.py` :
```python
import numpy as np
import pandas as pd

from f1pred.features import build_features
from f1pred.model import predict, sample_weights, train
from tests.helpers import make_results


def test_sample_weights_boost_current_and_decay_past():
    w = sample_weights(pd.Series([2026, 2025, 2024, 2016]), current_season=2026)
    assert w[0] == 5.0
    assert w[1] == 1.0
    assert w[2] == 0.8
    assert 0 < w[3] < w[2]


def test_train_and_predict_ranges():
    feats = build_features(make_results(seasons=(2023, 2024, 2025)))
    models = train(feats, current_season=2025)
    preds = predict(models, feats.tail(10))
    assert list(preds.columns) == ["season", "round", "driver_id", "pred_pos", "p_dnf"]
    assert (preds["pred_pos"] >= 1).all()
    assert preds["p_dnf"].between(0, 1).all()
    assert models.sigma > 0


def test_model_learns_strong_driver_is_ahead():
    feats = build_features(make_results(seasons=(2023, 2024, 2025)))
    models = train(feats, current_season=2025)
    last = feats[(feats["season"] == 2025) & (feats["round"] == 8)]
    preds = predict(models, last).set_index("driver_id")
    assert preds.loc["d0", "pred_pos"] < preds.loc["d9", "pred_pos"]


def test_no_dnf_observed_uses_constant_rate():
    feats = build_features(make_results(seasons=(2024, 2025)))
    feats["finished"] = True
    models = train(feats, current_season=2025)
    assert models.clf is None
    preds = predict(models, feats.tail(5))
    assert np.allclose(preds["p_dnf"], models.dnf_base)
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'f1pred.model'`

- [ ] **Step 3: Implémenter `f1pred/model.py`**

```python
"""Deux modèles LightGBM : position d'arrivée (régression) et abandon (classification)."""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from f1pred.features import FEATURE_COLUMNS

PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=10,
              random_state=0, verbose=-1)
HOLDOUT_RACES = 10
DEFAULT_SIGMA = 3.0


@dataclass
class Models:
    reg: lgb.LGBMRegressor
    clf: lgb.LGBMClassifier | None
    dnf_base: float
    sigma: float


def sample_weights(seasons: pd.Series, current_season: int, boost: float = 5.0,
                   decay: float = 0.8) -> np.ndarray:
    """La saison en cours pèse `boost` ; les précédentes décroissent depuis 1.0."""
    age = current_season - seasons.to_numpy()
    return np.where(age <= 0, boost, decay ** np.maximum(age - 1, 0))


def _fit_reg(X, y, w) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**PARAMS).fit(X, y, sample_weight=w)


def train(features: pd.DataFrame, current_season: int) -> Models:
    known = features.dropna(subset=["position"]).reset_index(drop=True)
    X = known[FEATURE_COLUMNS]
    y = known["position"].astype(float)
    w = sample_weights(known["season"], current_season)
    finished = known["finished"].astype(bool).to_numpy()
    dnf = (~finished).astype(int)

    races = np.sort(known["race_idx"].unique())
    sigma = DEFAULT_SIGMA
    if len(races) > 2 * HOLDOUT_RACES:
        cutoff = races[-HOLDOUT_RACES]
        is_train = (known["race_idx"] < cutoff).to_numpy() & finished
        is_hold = (known["race_idx"] >= cutoff).to_numpy() & finished
        holdout_model = _fit_reg(X[is_train], y[is_train], w[is_train])
        sigma = float(np.std(y[is_hold] - holdout_model.predict(X[is_hold])))

    reg = _fit_reg(X[finished], y[finished], w[finished])
    dnf_base = float(np.average(dnf, weights=w))
    clf = None
    if 0 < dnf.sum() < len(dnf):
        clf = lgb.LGBMClassifier(**PARAMS).fit(X, dnf, sample_weight=w)
    return Models(reg=reg, clf=clf, dnf_base=dnf_base, sigma=sigma)


def predict(models: Models, rows: pd.DataFrame) -> pd.DataFrame:
    X = rows[FEATURE_COLUMNS]
    out = rows[["season", "round", "driver_id"]].copy().reset_index(drop=True)
    out["pred_pos"] = np.clip(models.reg.predict(X), 1.0, None)
    if models.clf is None:
        out["p_dnf"] = models.dnf_base
    else:
        out["p_dnf"] = models.clf.predict_proba(X)[:, 1]
    return out
```

- [ ] **Step 4: Lancer les tests**

Run: `uv run pytest tests/test_model.py -v && uv run ruff check .`
Expected: 4 passed ; ruff OK.

- [ ] **Step 5: Commit**

```bash
git add f1pred/model.py tests/test_model.py
git commit -m "feat: modèles LightGBM position et abandon"
```

---

### Task 5: Simulation Monte Carlo

**Files:**
- Create: `f1pred/simulate.py`
- Test: `tests/test_simulate.py`

**Interfaces:**
- Consumes: `f1pred.rules.COUNTBACK_DEPTH`, `points_table`.
- Produces (`f1pred/simulate.py`) :
  - `@dataclass RaceInput(pred_pos: np.ndarray, p_dnf: np.ndarray, is_sprint: bool)` — tableaux alignés sur la liste `drivers`.
  - `@dataclass SimResult(p_title: dict[str, float], expected_points: dict[str, float], decided: bool)`
  - `simulate_season(drivers: list[str], points: np.ndarray, countback: np.ndarray, races: list[RaceInput], sigma: float, n_sims: int = 10_000, seed: int = 0) -> SimResult` — `points` : int (n,) ; `countback` : int (n, COUNTBACK_DEPTH), colonne k = nombre de (k+1)es places en GP.
  - `race_finish_probs(race: RaceInput, sigma: float, n_sims: int = 10_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]` — (P(victoire), P(podium)) par pilote.

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/test_simulate.py` :
```python
import numpy as np
import pytest

from f1pred.simulate import RaceInput, race_finish_probs, simulate_season


def zeros_cb(n):
    return np.zeros((n, 6), dtype=int)


def gp(pred, dnf=None, sprint=False):
    pred = np.asarray(pred, dtype=float)
    dnf = np.zeros_like(pred) if dnf is None else np.asarray(dnf, dtype=float)
    return RaceInput(pred_pos=pred, p_dnf=dnf, is_sprint=sprint)


def test_insurmountable_lead_is_certain_and_decided():
    res = simulate_season(["a", "b"], np.array([200, 0]), zeros_cb(2),
                          [gp([2.0, 1.0])], sigma=2.0, n_sims=2000)
    assert res.p_title["a"] == 1.0
    assert res.decided is True


def test_open_fight_is_not_decided():
    res = simulate_season(["a", "b"], np.array([30, 10]), zeros_cb(2),
                          [gp([1.5, 1.5])], sigma=2.0, n_sims=2000)
    assert res.decided is False


def test_identical_drivers_split_title_evenly():
    races = [gp([1.5, 1.5]) for _ in range(5)]
    res = simulate_season(["a", "b"], np.array([50, 50]), zeros_cb(2), races,
                          sigma=2.0, n_sims=10_000, seed=1)
    assert res.p_title["a"] == pytest.approx(0.5, abs=0.03)


def test_points_table_applied_without_noise():
    res = simulate_season(["a", "b", "c"], np.zeros(3, dtype=int), zeros_cb(3),
                          [gp([1.0, 2.0, 3.0])], sigma=0.0, n_sims=100)
    assert res.expected_points == {"a": 25.0, "b": 18.0, "c": 15.0}


def test_sprint_uses_sprint_points():
    res = simulate_season(["a", "b", "c"], np.zeros(3, dtype=int), zeros_cb(3),
                          [gp([1.0, 2.0, 3.0], sprint=True)], sigma=0.0, n_sims=100)
    assert res.expected_points == {"a": 8.0, "b": 7.0, "c": 6.0}


def test_dnf_scores_zero():
    res = simulate_season(["a", "b"], np.zeros(2, dtype=int), zeros_cb(2),
                          [gp([1.0, 2.0], dnf=[1.0, 0.0])], sigma=0.0, n_sims=100)
    assert res.expected_points == {"a": 0.0, "b": 25.0}


def test_countback_breaks_tie_with_wins():
    cb = zeros_cb(2)
    cb[1, 0] = 1  # b a une victoire, a n'en a aucune
    res = simulate_season(["a", "b"], np.array([25, 25]), cb, [], sigma=1.0, n_sims=100)
    assert res.p_title == {"a": 0.0, "b": 1.0}


def test_probabilities_sum_to_one_and_reproducible():
    races = [gp([1.0, 2.0, 3.0, 4.0]) for _ in range(3)]
    args = (["a", "b", "c", "d"], np.array([40, 35, 30, 0]), zeros_cb(4), races)
    r1 = simulate_season(*args, sigma=3.0, n_sims=3000, seed=7)
    r2 = simulate_season(*args, sigma=3.0, n_sims=3000, seed=7)
    assert sum(r1.p_title.values()) == pytest.approx(1.0)
    assert r1 == r2


def test_race_finish_probs():
    win, podium = race_finish_probs(gp([1.0, 2.0, 3.0, 4.0]), sigma=0.0, n_sims=100)
    assert list(win) == [1.0, 0.0, 0.0, 0.0]
    assert list(podium) == [1.0, 1.0, 1.0, 0.0]
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `uv run pytest tests/test_simulate.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'f1pred.simulate'`

- [ ] **Step 3: Implémenter `f1pred/simulate.py`**

```python
"""Simulation Monte Carlo vectorisée de la fin de saison."""

from dataclasses import dataclass

import numpy as np

from f1pred.rules import COUNTBACK_DEPTH, points_table

COUNTBACK_BASE = 32  # > nombre maximal de victoires possible sur une saison


@dataclass
class RaceInput:
    pred_pos: np.ndarray
    p_dnf: np.ndarray
    is_sprint: bool


@dataclass
class SimResult:
    p_title: dict[str, float]
    expected_points: dict[str, float]
    decided: bool


def _draw_race(race: RaceInput, sigma: float, n_sims: int, rng: np.random.Generator):
    """Renvoie (rangs 0-indexés, masque d'abandon), chacun de forme (n_sims, n)."""
    n = len(race.pred_pos)
    perf = race.pred_pos[None, :] + rng.normal(0.0, sigma, (n_sims, n))
    dnf = rng.random((n_sims, n)) < race.p_dnf[None, :]
    perf = np.where(dnf, np.inf, perf)
    order = np.argsort(perf, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(n), (n_sims, n)), axis=1)
    return ranks, dnf


def _is_decided(points: np.ndarray, races: list[RaceInput]) -> bool:
    if not races:
        return True
    n = len(points)
    remaining = sum(int(points_table(n, r.is_sprint)[0]) for r in races)
    leader = int(np.argmax(points))
    others = np.delete(points, leader)
    return bool(np.all(points[leader] > others + remaining))


def simulate_season(drivers: list[str], points: np.ndarray, countback: np.ndarray,
                    races: list[RaceInput], sigma: float, n_sims: int = 10_000,
                    seed: int = 0) -> SimResult:
    rng = np.random.default_rng(seed)
    n = len(drivers)
    total = np.tile(np.asarray(points, dtype=np.int64), (n_sims, 1))
    counts = np.tile(np.asarray(countback, dtype=np.int64), (n_sims, 1, 1))

    for race in races:
        ranks, dnf = _draw_race(race, sigma, n_sims, rng)
        table = points_table(n, race.is_sprint)
        total += np.where(dnf, 0, table[ranks])
        if not race.is_sprint:
            for k in range(min(COUNTBACK_DEPTH, n)):
                counts[:, :, k] += (ranks == k) & ~dnf

    weights = COUNTBACK_BASE ** np.arange(COUNTBACK_DEPTH - 1, -1, -1, dtype=np.int64)
    key = total * COUNTBACK_BASE**COUNTBACK_DEPTH + (counts * weights).sum(axis=2)
    champions = key.argmax(axis=1)
    p = np.bincount(champions, minlength=n) / n_sims
    expected = total.mean(axis=0)
    return SimResult(
        p_title={d: float(p[i]) for i, d in enumerate(drivers)},
        expected_points={d: float(expected[i]) for i, d in enumerate(drivers)},
        decided=_is_decided(np.asarray(points), races),
    )


def race_finish_probs(race: RaceInput, sigma: float, n_sims: int = 10_000,
                      seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ranks, dnf = _draw_race(race, sigma, n_sims, rng)
    win = ((ranks == 0) & ~dnf).mean(axis=0)
    podium = ((ranks < 3) & ~dnf).mean(axis=0)
    return win, podium
```

- [ ] **Step 4: Lancer les tests**

Run: `uv run pytest tests/test_simulate.py -v && uv run ruff check .`
Expected: 9 passed ; ruff OK.

- [ ] **Step 5: Commit**

```bash
git add f1pred/simulate.py tests/test_simulate.py
git commit -m "feat: simulation Monte Carlo de la fin de saison"
```

---

### Task 6: Historique + rendu du site

**Files:**
- Create: `f1pred/history.py`, `f1pred/site.py`, `f1pred/templates/base.html`, `f1pred/templates/index.html`, `f1pred/templates/methode.html`
- Test: `tests/test_history.py`, `tests/test_site.py`

**Interfaces:**
- Consumes: rien des autres modules (données simples uniquement).
- Produces :
  - `f1pred.history.load_history(path: Path) -> list[dict]` (liste vide si le fichier n'existe pas)
  - `f1pred.history.upsert(history: list[dict], entry: dict) -> list[dict]` — clé `(season, round)`, triée par `(season, round)`
  - `f1pred.history.save_history(path: Path, history: list[dict]) -> None` (JSON indenté, UTF-8)
  - Format d'une entrée : `{"season": int, "round": int, "race_name": str, "probabilities": {driver_id: float}}`
  - `f1pred.site.render_site(out_dir: Path, context: dict) -> None` — écrit `index.html` et `methode.html`.
  - Contexte attendu :
    ```python
    {
      "season": int, "updated_at": str, "last_race": str,
      "decided": bool, "champion": str | None,
      "drivers": [  # trié par p_title décroissant
        {"driver_id": str, "name": str, "team": str, "points": int,
         "p_title": float, "expected_points": float},
      ],
      "history": list[dict],      # entrées de la saison affichée
      "names": dict[str, str],    # driver_id -> nom affiché
      "backtest": {"n_races": int, "brier_winner": float, "brier_winner_baseline": float,
                   "brier_podium": float, "brier_podium_baseline": float},
      "sigma": float, "n_sims": int, "train_seasons": str,
    }
    ```

**Avant d'écrire les graphiques :** charger le skill `dataviz` et appliquer ses règles (couleurs, axes, lisibilité clair/sombre).

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/test_history.py` :
```python
from f1pred.history import load_history, save_history, upsert


def entry(season, rnd, p=0.5):
    return {"season": season, "round": rnd, "race_name": f"GP {rnd}",
            "probabilities": {"a": p, "b": 1 - p}}


def test_missing_file_gives_empty_history(tmp_path):
    assert load_history(tmp_path / "history.json") == []


def test_upsert_is_idempotent_and_sorted():
    h = upsert([], entry(2026, 3))
    h = upsert(h, entry(2026, 1))
    h = upsert(h, entry(2026, 3, p=0.9))
    assert [(e["season"], e["round"]) for e in h] == [(2026, 1), (2026, 3)]
    assert h[1]["probabilities"]["a"] == 0.9


def test_roundtrip(tmp_path):
    path = tmp_path / "history.json"
    save_history(path, [entry(2026, 1)])
    assert load_history(path) == [entry(2026, 1)]
```

`tests/test_site.py` :
```python
from f1pred.site import render_site


def context(decided=False):
    return {
        "season": 2026, "updated_at": "2026-10-13", "last_race": "Singapore Grand Prix",
        "decided": decided, "champion": "Andrea Kimi Antonelli" if decided else None,
        "drivers": [
            {"driver_id": "antonelli", "name": "Andrea Kimi Antonelli", "team": "mercedes",
             "points": 340, "p_title": 0.91, "expected_points": 410.0},
            {"driver_id": "russell", "name": "George Russell", "team": "mercedes",
             "points": 238, "p_title": 0.09, "expected_points": 320.0},
        ],
        "history": [
            {"season": 2026, "round": 1, "race_name": "Australian Grand Prix",
             "probabilities": {"antonelli": 0.3, "russell": 0.7}},
            {"season": 2026, "round": 2, "race_name": "Chinese Grand Prix",
             "probabilities": {"antonelli": 0.5, "russell": 0.5}},
        ],
        "names": {"antonelli": "Andrea Kimi Antonelli", "russell": "George Russell"},
        "backtest": {"n_races": 16, "brier_winner": 0.612, "brier_winner_baseline": 0.955,
                     "brier_podium": 0.081, "brier_podium_baseline": 0.124},
        "sigma": 3.4, "n_sims": 10000, "train_seasons": "2014–2026",
    }


def test_renders_both_pages_with_every_driver(tmp_path):
    render_site(tmp_path, context())
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    methode = (tmp_path / "methode.html").read_text(encoding="utf-8")
    assert "Andrea Kimi Antonelli" in index and "George Russell" in index
    assert "91" in index
    assert "0.612" in methode or "0,612" in methode
    assert 'lang="fr"' in index


def test_champion_banner_only_when_decided(tmp_path):
    render_site(tmp_path, context(decided=False))
    assert "Champion du monde" not in (tmp_path / "index.html").read_text(encoding="utf-8")
    render_site(tmp_path, context(decided=True))
    assert "Champion du monde" in (tmp_path / "index.html").read_text(encoding="utf-8")
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `uv run pytest tests/test_history.py tests/test_site.py -v`
Expected: FAIL avec `ModuleNotFoundError`

- [ ] **Step 3: Implémenter `f1pred/history.py`**

```python
"""Évolution des probabilités de titre, une entrée par manche."""

import json
from pathlib import Path


def load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def upsert(history: list[dict], entry: dict) -> list[dict]:
    key = (entry["season"], entry["round"])
    kept = [e for e in history if (e["season"], e["round"]) != key]
    return sorted([*kept, entry], key=lambda e: (e["season"], e["round"]))


def save_history(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
```

- [ ] **Step 4: Implémenter `f1pred/site.py`**

```python
"""Génération du site statique (Jinja2 + Plotly)."""

from pathlib import Path

import plotly.graph_objects as go
from jinja2 import Environment, PackageLoader, select_autoescape

ACCENT = "#e10600"
MUTED = "#8a8f98"
TOP_N_BARS = 10
TOP_N_LINES = 6

_env = Environment(loader=PackageLoader("f1pred", "templates"),
                   autoescape=select_autoescape(["html"]))


def _layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=MUTED, size=13),
        showlegend=False,
    )
    return fig


def _bars(drivers: list[dict]) -> str:
    top = drivers[:TOP_N_BARS][::-1]
    fig = go.Figure(go.Bar(
        x=[d["p_title"] * 100 for d in top], y=[d["name"] for d in top], orientation="h",
        marker_color=[ACCENT if i == len(top) - 1 else MUTED for i in range(len(top))],
        text=[f"{d['p_title'] * 100:.1f} %" for d in top], textposition="outside",
        hovertemplate="%{y} : %{x:.1f} %<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 115], showgrid=False, visible=False)
    fig.update_yaxes(showgrid=False)
    return _layout(fig, 40 * len(top) + 40).to_html(full_html=False, include_plotlyjs=False)


def _lines(history: list[dict], names: dict[str, str], drivers: list[dict]) -> str:
    fig = go.Figure()
    x = [e["race_name"].replace(" Grand Prix", "") for e in history]
    for i, d in enumerate(drivers[:TOP_N_LINES]):
        y = [e["probabilities"].get(d["driver_id"], 0.0) * 100 for e in history]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers", name=names.get(d["driver_id"], d["driver_id"]),
            line=dict(width=3 if i == 0 else 1.5, color=ACCENT if i == 0 else None),
            hovertemplate="%{fullData.name} : %{y:.1f} %<extra></extra>",
        ))
    fig.update_yaxes(range=[0, 100], ticksuffix=" %", gridcolor="rgba(138,143,152,0.2)")
    fig.update_xaxes(showgrid=False)
    fig = _layout(fig, 380)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.2))
    return fig.to_html(full_html=False, include_plotlyjs=False)


def render_site(out_dir: Path, context: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "bars_html": _bars(context["drivers"]),
        "lines_html": _lines(context["history"], context["names"], context["drivers"]),
    }
    for page in ("index.html", "methode.html"):
        html = _env.get_template(page).render(**context, **charts)
        (out_dir / page).write_text(html, encoding="utf-8")
```

- [ ] **Step 5: Créer les gabarits**

`f1pred/templates/base.html` :
```html
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{% block title %}Prédictions F1 {{ season }}{% endblock %}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
  <style>
    :root { --bg: #ffffff; --fg: #15151e; --muted: #6b7079; --card: #f4f4f6; --accent: #e10600; }
    @media (prefers-color-scheme: dark) {
      :root { --bg: #0f0f14; --fg: #f2f2f5; --muted: #9a9fa8; --card: #1a1a22; }
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--fg);
           font: 16px/1.55 Inter, system-ui, -apple-system, sans-serif; }
    main { max-width: 860px; margin: 0 auto; padding: 24px 16px 64px; }
    nav { display: flex; gap: 20px; margin-bottom: 32px; font-size: 14px; }
    nav a { color: var(--muted); text-decoration: none; }
    nav a.active, nav a:hover { color: var(--fg); }
    h1 { font-size: clamp(28px, 6vw, 44px); line-height: 1.1; margin: 0 0 8px; }
    h2 { font-size: 20px; margin: 40px 0 12px; }
    .sub { color: var(--muted); margin: 0 0 24px; }
    .card { background: var(--card); border-radius: 16px; padding: 16px; }
    .banner { background: var(--accent); color: #fff; border-radius: 16px; padding: 16px 20px;
              font-weight: 600; margin-bottom: 24px; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    td, th { padding: 8px 4px; text-align: left; border-bottom: 1px solid var(--card); }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
    footer { color: var(--muted); font-size: 13px; margin-top: 48px; }
  </style>
</head>
<body>
<main>
  <nav>
    <a href="index.html" class="{% block nav_index %}{% endblock %}">Probabilités</a>
    <a href="methode.html" class="{% block nav_methode %}{% endblock %}">Méthode</a>
  </nav>
  {% block content %}{% endblock %}
  <footer>Mis à jour le {{ updated_at }} après le {{ last_race }}. Données : Jolpica F1 API.
    Projet non officiel, sans lien avec la Formule 1 ou la FIA.</footer>
</main>
</body>
</html>
```

`f1pred/templates/index.html` :
```html
{% extends "base.html" %}
{% block nav_index %}active{% endblock %}
{% block content %}
  {% if decided %}
  <div class="banner">🏆 {{ champion }} est Champion du monde {{ season }}.</div>
  {% endif %}
  <h1>Qui sera champion du monde {{ season }} ?</h1>
  <p class="sub">Probabilité de titre pilotes, d'après {{ "{:,}".format(n_sims).replace(",", " ") }}
    simulations de la fin de saison.</p>

  <div class="card">{{ bars_html | safe }}</div>

  <h2>Évolution au fil de la saison</h2>
  <div class="card">{{ lines_html | safe }}</div>

  <h2>Détail</h2>
  <table>
    <thead><tr><th>Pilote</th><th class="num">Points</th>
      <th class="num">Points attendus</th><th class="num">Titre</th></tr></thead>
    <tbody>
    {% for d in drivers %}
      <tr><td>{{ d.name }}</td><td class="num">{{ d.points }}</td>
        <td class="num">{{ "%.0f"|format(d.expected_points) }}</td>
        <td class="num">{{ "%.1f"|format(d.p_title * 100) }} %</td></tr>
    {% endfor %}
    </tbody>
  </table>
{% endblock %}
```

`f1pred/templates/methode.html` :
```html
{% extends "base.html" %}
{% block title %}Méthode · Prédictions F1 {{ season }}{% endblock %}
{% block nav_methode %}active{% endblock %}
{% block content %}
  <h1>Méthode</h1>
  <p class="sub">Comment les probabilités sont calculées, et à quel point s'y fier.</p>

  <h2>1. Un modèle par course</h2>
  <p>Deux modèles LightGBM sont entraînés sur toutes les courses depuis {{ train_seasons }} :
    l'un prédit la position d'arrivée, l'autre la probabilité d'abandon. Ils n'utilisent que des
    informations connues avant la course : forme récente du pilote (3 et 5 dernières courses),
    forme de l'écurie, historique sur le circuit, écart avec le coéquipier, taux d'abandon récent
    et part des points de la saison. La position sur la grille n'est pas utilisée, car elle est
    inconnue pour les courses futures.</p>

  <h2>2. Le règlement 2026 change la donne</h2>
  <p>Nouveaux moteurs, nouvelles voitures : l'historique d'avant 2026 prédit mal la hiérarchie
    actuelle. Les courses de la saison en cours pèsent donc 5 fois plus à l'entraînement, et les
    saisons anciennes de moins en moins.</p>

  <h2>3. Simuler la fin de saison</h2>
  <p>Pour chaque course restante, on part de la position prédite de chaque pilote, on ajoute un
    aléa (écart-type de {{ "%.1f"|format(sigma) }} places, mesuré sur les erreurs passées du
    modèle), on tire au sort les abandons, puis on attribue les points du barème officiel. On
    répète {{ n_sims }} fois. La probabilité de titre est la part des simulations où le pilote
    finit en tête, avec le départage officiel (nombre de victoires, puis de 2es places…).</p>

  <h2>4. Précision mesurée</h2>
  <p>Après chacune des {{ backtest.n_races }} dernières courses de la saison, on a refait la
    prédiction de la course suivante avec uniquement les données disponibles à ce moment-là.
    Score de Brier : plus il est bas, mieux c'est ; la référence donne la même chance à tout
    le monde.</p>
  <table>
    <thead><tr><th>Prédiction</th><th class="num">Modèle</th><th class="num">Référence</th></tr></thead>
    <tbody>
      <tr><td>Vainqueur de la course</td><td class="num">{{ "%.3f"|format(backtest.brier_winner) }}</td>
        <td class="num">{{ "%.3f"|format(backtest.brier_winner_baseline) }}</td></tr>
      <tr><td>Podium</td><td class="num">{{ "%.3f"|format(backtest.brier_podium) }}</td>
        <td class="num">{{ "%.3f"|format(backtest.brier_podium_baseline) }}</td></tr>
    </tbody>
  </table>

  <h2>5. Limites</h2>
  <p>Le modèle ignore les qualifications, la météo, les pénalités moteur et les évolutions
    techniques en cours de saison. Il suppose des courses disputées sur toute la distance.
    Les probabilités décrivent une incertitude, pas une certitude.</p>
{% endblock %}
```

- [ ] **Step 6: Lancer les tests**

Run: `uv run pytest tests/test_history.py tests/test_site.py -v && uv run ruff check .`
Expected: 5 passed ; ruff OK.

- [ ] **Step 7: Commit**

```bash
git add f1pred/history.py f1pred/site.py f1pred/templates tests/test_history.py tests/test_site.py
git commit -m "feat: historique des probabilités et rendu du site"
```

---

### Task 7: Pipeline, backtest, validation et CLI

**Files:**
- Create: `f1pred/pipeline.py`, `f1pred/cli.py`, `f1pred/__main__.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: tout ce qui précède :
  - `data.update_cache`, `data.load_results`, `data.load_schedule`, `data.JolpicaClient`
  - `features.add_future_rows`, `features.build_features`
  - `model.train`, `model.predict`
  - `simulate.RaceInput`, `simulate.simulate_season`, `simulate.race_finish_probs`, `simulate.SimResult`
  - `history.load_history`, `history.upsert`, `history.save_history`
  - `site.render_site`
  - `rules.COUNTBACK_DEPTH`
- Produces (`f1pred/pipeline.py`) :
  - `FIRST_SEASON = 2014`
  - `class ValidationError(RuntimeError)`
  - `@dataclass SeasonState(drivers: list[str], names: dict[str, str], teams: dict[str, str], points: np.ndarray, countback: np.ndarray, last_round: int)`
  - `@dataclass Checkpoint(round: int, state: SeasonState, sim: SimResult, races: list[RaceInput], sigma: float)`
  - `@dataclass Outcome(updated: bool, message: str)`
  - `season_state(results: pd.DataFrame, season: int) -> SeasonState`
  - `predict_from(results, schedule, season: int, upto_round: int, n_sims: int, seed: int) -> Checkpoint`
  - `validate(cp: Checkpoint) -> None`
  - `backtest_metrics(checkpoints: list[Checkpoint], results, season: int, n_sims: int, seed: int) -> dict`
  - `run(data_dir: Path, out_dir: Path, offline: bool, seed: int = 0, n_sims: int = 10_000, force: bool = False, today: date | None = None, client=None) -> Outcome`
- Produces (`f1pred/cli.py`) : `main(argv: list[str] | None = None) -> int`.

- [ ] **Step 1: Écrire les tests qui échouent**

`tests/test_pipeline.py` :
```python
from datetime import date

import numpy as np
import pandas as pd
import pytest

from f1pred.cli import main
from f1pred.history import load_history
from f1pred.pipeline import ValidationError, predict_from, run, season_state, validate
from tests.helpers import make_results, make_schedule


def write_raw(raw, results, schedules):
    raw.mkdir(parents=True, exist_ok=True)
    for season, df in results.groupby("season"):
        df.to_parquet(raw / f"results_{season}.parquet", index=False)
    for sched in schedules:
        sched.to_parquet(raw / f"schedule_{int(sched['season'].iloc[0])}.parquet", index=False)


@pytest.fixture
def data_dir(tmp_path):
    past = make_results(seasons=(2023, 2024), n_rounds=8)
    current = make_results(seasons=(2025,), n_rounds=5, seed=3)
    results = pd.concat([past, current], ignore_index=True)
    write_raw(tmp_path / "data" / "raw", results,
              [make_schedule(2024, 8), make_schedule(2025, 8)])
    return tmp_path / "data"


def test_season_state_counts_points_and_wins():
    results = make_results(seasons=(2025,), n_rounds=3, sprint_rounds=())
    state = season_state(results, 2025)
    cur = results.set_index("driver_id")
    assert state.last_round == 3
    i = state.drivers.index("d0")
    assert state.points[i] == int(cur.loc["d0", "points"].sum())
    is_d0_win = (results["driver_id"] == "d0") & (results["position"] == 1)
    wins = (is_d0_win & results["finished"]).sum()
    assert state.countback[i, 0] == wins


def test_predict_from_gives_valid_probabilities(data_dir):
    from f1pred.data import load_results, load_schedule
    results = load_results(data_dir / "raw")
    cp = predict_from(results, load_schedule(data_dir / "raw", 2025), 2025, 5,
                      n_sims=500, seed=0)
    validate(cp)
    assert len(cp.races) == 3  # manches 6, 7, 8 (aucun sprint après la 5)
    assert sum(cp.sim.p_title.values()) == pytest.approx(1.0)


def test_replaced_driver_keeps_points_but_scores_nothing(data_dir):
    from f1pred.data import load_results, load_schedule
    results = load_results(data_dir / "raw")
    # d9 est remplacé par "rookie" à la manche 5.
    mask = (results["season"] == 2025) & (results["round"] == 5) & (results["driver_id"] == "d9")
    results.loc[mask, ["driver_id", "driver_name"]] = ["rookie", "Rookie"]
    cp = predict_from(results, load_schedule(data_dir / "raw", 2025), 2025, 5,
                      n_sims=500, seed=0)
    validate(cp)
    i = cp.state.drivers.index("d9")
    before = cp.state.points[i]
    assert cp.sim.expected_points["d9"] == pytest.approx(before)
    assert "rookie" in cp.sim.p_title


def test_validate_rejects_bad_probabilities(data_dir):
    from f1pred.data import load_results, load_schedule
    results = load_results(data_dir / "raw")
    cp = predict_from(results, load_schedule(data_dir / "raw", 2025), 2025, 5,
                      n_sims=200, seed=0)
    cp.sim.p_title[cp.state.drivers[0]] += 0.5
    with pytest.raises(ValidationError):
        validate(cp)
    cp.sim.p_title[cp.state.drivers[0]] = float("nan")
    with pytest.raises(ValidationError):
        validate(cp)


def test_run_offline_builds_site_then_skips(data_dir, tmp_path):
    out = tmp_path / "site"
    first = run(data_dir, out, offline=True, n_sims=300, today=date(2025, 6, 1))
    assert first.updated is True
    index = (out / "index.html").read_text(encoding="utf-8")
    for i in range(10):
        assert f"Pilote {i}" in index
    assert (out / "methode.html").exists()
    history = load_history(data_dir / "history.json")
    assert [e["round"] for e in history] == [1, 2, 3, 4, 5]

    second = run(data_dir, out, offline=True, n_sims=300, today=date(2025, 6, 1))
    assert second.updated is False

    forced = run(data_dir, out, offline=True, n_sims=300, today=date(2025, 6, 1), force=True)
    assert forced.updated is True


def test_run_falls_back_to_previous_season_in_january(data_dir, tmp_path):
    outcome = run(data_dir, tmp_path / "site", offline=True, n_sims=200, today=date(2026, 1, 15))
    assert outcome.updated is True
    assert "2025" in outcome.message


def test_cli_writes_github_output(data_dir, tmp_path, monkeypatch):
    gh_out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(gh_out))
    monkeypatch.setattr("f1pred.cli.date", type("D", (), {"today": staticmethod(
        lambda: date(2025, 6, 1))}))
    code = main(["run", "--offline", "--sims", "200", "--data-dir", str(data_dir),
                 "--out", str(tmp_path / "site")])
    assert code == 0
    assert "updated=true" in gh_out.read_text(encoding="utf-8")


def test_backtest_metrics_are_finite(data_dir):
    from f1pred.data import load_results, load_schedule
    from f1pred.pipeline import backtest_metrics
    results = load_results(data_dir / "raw")
    sched = load_schedule(data_dir / "raw", 2025)
    cps = [predict_from(results, sched, 2025, r, n_sims=200, seed=0) for r in (1, 2, 3)]
    m = backtest_metrics(cps, results, 2025, n_sims=200, seed=0)
    assert m["n_races"] == 3
    assert np.isfinite([m["brier_winner"], m["brier_podium"]]).all()
    assert 0 < m["brier_winner_baseline"] < 1
```

- [ ] **Step 2: Lancer les tests pour vérifier qu'ils échouent**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL avec `ModuleNotFoundError: No module named 'f1pred.pipeline'`

- [ ] **Step 3: Implémenter `f1pred/pipeline.py`**

```python
"""Orchestration : données → features → modèle → simulation → historique → site."""

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from f1pred.data import JolpicaClient, load_results, load_schedule, update_cache
from f1pred.features import add_future_rows, build_features
from f1pred.history import load_history, save_history, upsert
from f1pred.model import predict, train
from f1pred.rules import COUNTBACK_DEPTH
from f1pred.simulate import RaceInput, SimResult, race_finish_probs, simulate_season
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

    feats = build_features(add_future_rows(past, future, lineup))
    models = train(feats, current_season=season)
    preds = predict(models, feats[feats["position"].isna()])

    races: list[RaceInput] = []
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
                          sigma=models.sigma, n_sims=n_sims, seed=seed)
    return Checkpoint(upto_round, state, sim, races, models.sigma)


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
    name = schedule.loc[schedule["round"] == cp.round, "race_name"]
    return {
        "season": season, "round": cp.round,
        "race_name": str(name.iloc[0]) if not name.empty else f"Manche {cp.round}",
        "probabilities": {d: round(p, 4) for d, p in cp.sim.p_title.items()},
    }


def _context(final: Checkpoint, season: int, history: list[dict], metrics: dict,
             schedule: pd.DataFrame, n_sims: int, today: date, first_season: int) -> dict:
    st = final.state
    drivers = sorted(
        ({"driver_id": d, "name": st.names[d], "team": st.teams[d],
          "points": int(st.points[i]), "p_title": final.sim.p_title[d],
          "expected_points": final.sim.expected_points[d]} for i, d in enumerate(st.drivers)),
        key=lambda x: (-x["p_title"], -x["points"]),
    )
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
```

- [ ] **Step 4: Implémenter `f1pred/cli.py` et `f1pred/__main__.py`**

`f1pred/cli.py` :
```python
"""Ligne de commande : `python -m f1pred run`."""

import argparse
import os
from datetime import date
from pathlib import Path

from f1pred import pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="f1pred")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Met à jour les données, les prédictions et le site")
    run.add_argument("--offline", action="store_true", help="N'appelle pas l'API")
    run.add_argument("--force", action="store_true", help="Recalcule même sans nouvelle manche")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--sims", type=int, default=10_000)
    run.add_argument("--data-dir", default="data")
    run.add_argument("--out", default="site")
    args = parser.parse_args(argv)

    outcome = pipeline.run(Path(args.data_dir), Path(args.out), offline=args.offline,
                           seed=args.seed, n_sims=args.sims, force=args.force,
                           today=date.today())
    print(outcome.message)
    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a", encoding="utf-8") as fh:
            fh.write(f"updated={'true' if outcome.updated else 'false'}\n")
    return 0
```

`f1pred/__main__.py` :
```python
from f1pred.cli import main

raise SystemExit(main())
```

- [ ] **Step 5: Lancer les tests**

Run: `uv run pytest tests/test_pipeline.py -v && uv run ruff check .`
Expected: 8 passed ; ruff OK.

- [ ] **Step 6: Lancer toute la suite**

Run: `uv run pytest -q`
Expected: tous les tests passent (≈ 45).

- [ ] **Step 7: Commit**

```bash
git add f1pred/pipeline.py f1pred/cli.py f1pred/__main__.py tests/test_pipeline.py
git commit -m "feat: pipeline complet, backtest, validation et CLI"
```

---

### Task 8: Premier run réel, workflows GitHub Actions, README

**Files:**
- Create: `.github/workflows/test.yml`, `.github/workflows/update.yml`, `README.md`
- Create (généré) : `data/raw/*.parquet`, `data/history.json`

**Interfaces:**
- Consumes: `python -m f1pred run` (Task 7), sortie `updated=true|false` dans `$GITHUB_OUTPUT`.
- Produces: données réelles committées ; workflows prêts pour GitHub.

- [ ] **Step 1: Premier run réel (≈ 80 requêtes à l'API, ≈ 3 à 5 min)**

Run: `uv run python -m f1pred run`
Expected: `Site mis à jour : saison 2026, manche 17.` ; `data/raw/` contient `results_2014.parquet` à `results_2026.parquet` ; `site/index.html` existe.

- [ ] **Step 2: Vérifier la cohérence avec le classement officiel**

Run: `uv run python -c "from pathlib import Path; from f1pred.data import load_results; from f1pred.pipeline import season_state; s = season_state(load_results(Path('data/raw')), 2026); top = sorted(zip(s.points, s.drivers), reverse=True)[:5]; print(top)"`
Expected : `antonelli` 340, `russell` 238, `hamilton` 229, `max_verstappen` 221, `leclerc` 215 (valeurs du document de recherche, §4). Si les chiffres diffèrent, s'arrêter et signaler l'écart.

- [ ] **Step 3: Ouvrir le site et le vérifier visuellement**

Ouvrir `site/index.html` dans le navigateur intégré (`preview_start` avec l'URL `file:///…/site/index.html`) : barres triées, courbe sur 17 manches, tableau complet, page « Méthode » avec des scores de Brier inférieurs à la référence. Vérifier aussi en largeur mobile (375 px).

- [ ] **Step 4: Créer `.github/workflows/test.yml`**

```yaml
name: tests
on:
  push:
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv python install 3.12
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run pytest -q
```

- [ ] **Step 5: Créer `.github/workflows/update.yml`**

```yaml
name: update
on:
  schedule:
    - cron: "0 8 * * 1,2"   # lundi et mardi, 8 h UTC
  workflow_dispatch:
    inputs:
      force:
        description: "Recalculer même sans nouvelle manche"
        type: boolean
        default: false
permissions:
  contents: write
  pages: write
  id-token: write
concurrency:
  group: update
  cancel-in-progress: false
jobs:
  update:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deploy.outputs.page_url }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv python install 3.12
      - run: uv sync
      - run: uv run pytest -q
      - id: run
        run: uv run python -m f1pred run ${{ inputs.force && '--force' || '' }}
      - if: steps.run.outputs.updated == 'true'
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add data
          git diff --cached --quiet || git commit -m "data: mise à jour après la dernière manche"
          git push
      - if: steps.run.outputs.updated == 'true'
        uses: actions/configure-pages@v5
      - if: steps.run.outputs.updated == 'true'
        uses: actions/upload-pages-artifact@v3
        with:
          path: site
      - if: steps.run.outputs.updated == 'true'
        id: deploy
        uses: actions/deploy-pages@v4
```

- [ ] **Step 6: Créer `README.md`**

````markdown
# Prédictions F1

Probabilité de chaque pilote de remporter le championnat du monde de F1, mise à jour
automatiquement après chaque Grand Prix.

**Méthode :** deux modèles LightGBM (position d'arrivée, abandon) entraînés depuis 2014,
puis 10 000 simulations Monte Carlo de la fin de saison. Détails sur la page « Méthode » du site.

## Lancer en local

```bash
uv sync
uv run python -m f1pred run            # télécharge les nouvelles données et génère site/
uv run python -m f1pred run --offline  # sans réseau, à partir de data/raw/
uv run pytest
```

## Automatisation

- `.github/workflows/update.yml` : lundi et mardi 8 h UTC (et à la demande), met à jour
  `data/`, puis déploie `site/` sur GitHub Pages.
- `.github/workflows/test.yml` : ruff + pytest à chaque push.

Données : [Jolpica F1 API](https://github.com/jolpica/jolpica-f1). Projet non officiel.
````

- [ ] **Step 7: Commit**

```bash
git add .github README.md data
git commit -m "feat: données réelles, workflows GitHub Actions et README"
```

---

### Task 9: Publication sur GitHub (⚠️ demander l'accord explicite de l'utilisateur avant)

**Files:** aucun.

**Interfaces:**
- Consumes: repo local complet (Tasks 1 à 8), CLI `gh` connectée au compte de l'utilisateur.
- Produces: repo public `f1-predictions`, GitHub Pages activé, premier déploiement.

Créer un repo public est une publication : **montrer à l'utilisateur le nom du repo et sa visibilité, et attendre un « oui » explicite avant le Step 1.**

- [ ] **Step 1: Créer le repo et pousser**

Run: `gh repo create f1-predictions --public --source . --push`
Expected: URL du repo affichée.

- [ ] **Step 2: Activer GitHub Pages en mode « GitHub Actions »**

Run: `gh api -X POST "repos/{owner}/f1-predictions/pages" -f build_type=workflow`
Expected: JSON contenant `"build_type": "workflow"`.

- [ ] **Step 3: Lancer le premier déploiement**

Run: `gh workflow run update -f force=true && gh run watch --exit-status`
Expected: le run se termine en succès ; `gh api "repos/{owner}/f1-predictions/pages" --jq .html_url` donne l'URL du site.

- [ ] **Step 4: Vérifier le site en ligne**

Ouvrir l'URL dans le navigateur intégré ; vérifier que la page affiche les mêmes probabilités que `site/index.html` en local.
