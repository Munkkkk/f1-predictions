import json
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
    # d9 est remplacé pour de bon par "rookie" à partir de la manche 3 : sur les 5 derniers GP,
    # rookie (3 départs) passe devant d9 (2 départs) chez t4.
    mask = ((results["season"] == 2025) & (results["round"] >= 3)
            & (results["driver_id"] == "d9"))
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
    small = dict(offline=True, n_sims=300, n_sims_final=300, today=date(2025, 6, 1))
    first = run(data_dir, out, **small)
    assert first.updated is True
    index = (out / "index.html").read_text(encoding="utf-8")
    for i in range(10):
        assert f"Pilote {i}" in index
    assert (out / "methode.html").exists()
    history = load_history(data_dir / "history.json")
    assert [e["round"] for e in history] == [1, 2, 3, 4, 5]
    assert history[0]["date"] == make_schedule(2025, 8)["date"].iloc[0]

    second = run(data_dir, out, **small)
    assert second.updated is False

    forced = run(data_dir, out, **small, force=True)
    assert forced.updated is True


def test_run_falls_back_to_previous_season_in_january(data_dir, tmp_path):
    outcome = run(data_dir, tmp_path / "site", offline=True, n_sims=200, n_sims_final=200,
                  today=date(2026, 1, 15))
    assert outcome.updated is True
    assert "2025" in outcome.message


def test_cli_writes_github_output(data_dir, tmp_path, monkeypatch):
    gh_out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(gh_out))
    monkeypatch.setattr("f1pred.cli.date", type("D", (), {"today": staticmethod(
        lambda: date(2025, 6, 1))}))
    code = main(["run", "--offline", "--sims", "200", "--sims-final", "200",
                 "--data-dir", str(data_dir), "--out", str(tmp_path / "site")])
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


def test_backtest_scores_the_next_gp_against_its_own_round():
    # Le prochain GP prévu est la manche 7 (une manche reportée passe avant la 6, par exemple) :
    # il doit être comparé aux résultats de la manche 7, pas de la manche cp.round + 1.
    from types import SimpleNamespace

    from f1pred.pipeline import backtest_metrics
    from f1pred.simulate import RaceInput
    results = make_results(seasons=(2025,), n_rounds=8, sprint_rounds=())
    gp7 = results[results["round"] == 7].set_index("driver_id")
    drivers = sorted(results["driver_id"].unique())
    pred = gp7.loc[drivers, "position"].to_numpy(dtype=float)  # prévision parfaite de la 7
    race = RaceInput(pred_pos=pred, p_dnf=np.zeros(len(drivers)), is_sprint=False, round=7)
    cp = SimpleNamespace(round=5, races=[race], sigma=0.01,
                         state=SimpleNamespace(drivers=drivers))
    m = backtest_metrics([cp], results, 2025, n_sims=200, seed=0)
    winner_7 = gp7.loc[drivers]
    assert m["n_races"] == 1
    if bool(winner_7.loc[winner_7["position"] == 1, "finished"].iloc[0]):
        assert m["brier_winner"] == pytest.approx(0.0, abs=1e-9)
    assert m["brier_podium"] < 0.05


def test_forecast_tags_each_race_with_its_round(data_dir):
    from f1pred.data import load_results, load_schedule
    from f1pred.pipeline import forecast
    fc = forecast(load_results(data_dir / "raw"), load_schedule(data_dir / "raw", 2025), 2025, 5)
    assert [r.round for r in fc.races] == [6, 7, 8]


def test_predict_from_complete_season_is_decided(tmp_path):
    results = make_results(seasons=(2024, 2025), n_rounds=8)
    cp = predict_from(results, make_schedule(2025, 8), 2025, 8, n_sims=200, seed=0)
    validate(cp)
    assert cp.sim.decided is True
    assert cp.races == []
    assert sum(cp.sim.p_title.values()) == pytest.approx(1.0)


def test_run_on_completed_previous_season_shows_champion(tmp_path):
    results = make_results(seasons=(2024, 2025), n_rounds=8)
    write_raw(tmp_path / "data" / "raw", results,
              [make_schedule(2024, 8), make_schedule(2025, 8)])
    out = tmp_path / "site"
    outcome = run(tmp_path / "data", out, offline=True, n_sims=200, n_sims_final=200,
                  today=date(2026, 1, 15))
    assert outcome.updated is True
    assert "Champion du monde" in (out / "index.html").read_text(encoding="utf-8")


def test_context_reports_remaining_points_and_eliminations(data_dir):
    from f1pred.data import load_results
    from f1pred.pipeline import _context
    results = load_results(data_dir / "raw")
    sched = make_schedule(2025, 8, sprint_rounds=(3, 7))
    cp = predict_from(results, sched, 2025, 5, n_sims=200, seed=0)
    ctx = _context(cp, 2025, [], {}, sched, 200, date(2025, 6, 1), 2023)
    assert (ctx["remaining_gp"], ctx["remaining_sprints"]) == (3, 1)
    assert ctx["remaining_points"] == 25 * 3 + 8 * 1
    leader = max(cp.state.points)
    for d in ctx["drivers"]:
        assert d["max_points"] == d["points"] + ctx["remaining_points"]
        assert d["eliminated"] == (d["max_points"] < leader)
    assert any(d["eliminated"] for d in ctx["drivers"])
    assert {"tau_driver", "tau_team", "dnf_shock"} <= ctx.keys()
    assert ctx["tau_driver"] == cp.shocks.tau_driver
    assert ctx["sigma"] == cp.shocks.sigma


def test_context_never_marks_eliminated_once_decided():
    from f1pred.pipeline import _context
    results = make_results(seasons=(2024, 2025), n_rounds=8)
    sched = make_schedule(2025, 8)
    cp = predict_from(results, sched, 2025, 8, n_sims=200, seed=0)
    ctx = _context(cp, 2025, [], {}, sched, 200, date(2026, 1, 15), 2024)
    assert ctx["remaining_points"] == 0
    assert not any(d["eliminated"] for d in ctx["drivers"])


def test_final_checkpoint_uses_more_simulations(data_dir, tmp_path, monkeypatch):
    import f1pred.pipeline as pl
    calls, contexts = [], []
    real = pl.simulate_forecast

    def spy(fc, n_sims, seed, shocks=None):
        calls.append((fc.round, n_sims))
        return real(fc, n_sims, seed, shocks)

    monkeypatch.setattr(pl, "simulate_forecast", spy)
    monkeypatch.setattr(pl, "render_site", lambda out, ctx: contexts.append(ctx))
    run(data_dir, tmp_path / "site", offline=True, n_sims=200, n_sims_final=400,
        today=date(2025, 6, 1))
    assert calls == [(1, 200), (2, 200), (3, 200), (4, 200), (5, 400)]
    assert contexts[0]["n_sims"] == 400


def test_cli_passes_final_sims(data_dir, tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr("f1pred.cli.pipeline.run",
                        lambda *a, **kw: seen.update(kw) or type("O", (), {
                            "updated": False, "message": "ok"})())
    assert main(["run", "--offline", "--sims-final", "1234"]) == 0
    assert seen["n_sims_final"] == 1234
    main(["run", "--offline"])
    assert seen["n_sims_final"] == 200_000


def test_checkpoint_shocks_ignore_later_results(data_dir):
    from f1pred.data import load_results, load_schedule
    from f1pred.pipeline import season_checkpoints
    results = load_results(data_dir / "raw")
    sched = load_schedule(data_dir / "raw", 2025)
    base = season_checkpoints(results, sched, 2025, [1, 2, 3], n_sims=100, seed=0)
    tampered = results.copy()
    later = (tampered["season"] == 2025) & (tampered["round"] > 3)
    tampered.loc[later, "position"] = 21 - tampered.loc[later, "position"]
    tampered.loc[later, "finished"] = False
    after = season_checkpoints(tampered, sched, 2025, [1, 2, 3], n_sims=100, seed=0)
    assert [cp.shocks for cp in base] == [cp.shocks for cp in after]
    assert all(cp.shocks.sigma != cp.sigma for cp in base)  # estimées, pas le repli


def test_context_sorts_drivers_like_the_standings(data_dir):
    from f1pred.data import load_results, load_schedule
    from f1pred.pipeline import _context
    results = load_results(data_dir / "raw")
    sched = load_schedule(data_dir / "raw", 2025)
    cp = predict_from(results, sched, 2025, 5, n_sims=300, seed=0)
    i, j = 0, 1
    points = np.full(len(cp.state.drivers), 50)
    points[j] = 80  # meneur aux points, mais moins de chances de titre
    cp.state.points = points
    a, b = cp.state.drivers[i], cp.state.drivers[j]
    cp.sim.p_title.update({d: 0.0 for d in cp.sim.p_title})
    cp.sim.p_title.update({a: 0.7, b: 0.3})
    ctx = _context(cp, 2025, [], {}, sched, 300, date(2025, 6, 1), 2023)
    order = [d["driver_id"] for d in ctx["drivers"]]
    assert order[:2] == [b, a]  # points d'abord, puis probabilité de titre
    points = [d["points"] for d in ctx["drivers"]]
    assert points == sorted(points, reverse=True)


def test_predict_from_forwards_teams_and_shocks(data_dir, monkeypatch):
    import f1pred.pipeline as pl
    from f1pred.data import load_results, load_schedule
    from f1pred.shocks import Shocks
    seen = {}
    real = pl.simulate_season

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(pl, "simulate_season", spy)
    results = load_results(data_dir / "raw")
    shocks = Shocks(sigma=2.5, tau_driver=1.1, tau_team=1.2, dnf_shock=0.3)
    cp = predict_from(results, load_schedule(data_dir / "raw", 2025), 2025, 5, n_sims=100,
                      seed=0, shocks=shocks)
    assert seen["teams"] == [cp.state.teams[d] for d in cp.state.drivers]
    assert (seen["sigma"], seen["tau_driver"], seen["tau_team"], seen["dnf_shock"]) == (
        2.5, 1.1, 1.2, 0.3)
    assert cp.shocks == shocks


def test_run_refuses_a_round_gap_absent_from_the_schedule(data_dir, tmp_path):
    # Un trou que le calendrier n'explique pas (manche 3 courue nulle part) reste une erreur.
    raw = data_dir / "raw"
    cur = pd.read_parquet(raw / "results_2025.parquet")
    cur[cur["round"] != 3].to_parquet(raw / "results_2025.parquet", index=False)
    sched = pd.read_parquet(raw / "schedule_2025.parquet")
    sched[sched["round"] != 3].to_parquet(raw / "schedule_2025.parquet", index=False)
    with pytest.raises(ValidationError, match="3"):
        run(data_dir, tmp_path / "site", offline=True, n_sims=100, n_sims_final=100,
            today=date(2025, 6, 1))
    assert not (data_dir / "history.json").exists()
    assert not (tmp_path / "site").exists()


def test_run_warns_about_a_postponed_round(data_dir, tmp_path, caplog):
    # Repro I2 : une manche reportée (encore au calendrier) bloquait toute exécution.
    raw = data_dir / "raw"
    cur = pd.read_parquet(raw / "results_2025.parquet")
    cur[cur["round"] != 3].to_parquet(raw / "results_2025.parquet", index=False)
    with caplog.at_level("WARNING", logger="f1pred.pipeline"):
        out = run(data_dir, tmp_path / "site", offline=True, n_sims=100, n_sims_final=100,
                  today=date(2025, 6, 1))
    assert out.updated is True
    assert "3" in out.message and "non courue" in out.message
    assert any("non courue" in r.getMessage() for r in caplog.records)
    assert (tmp_path / "site" / "index.html").exists()


def _small_run(data_dir, tmp_path, **kw):
    return run(data_dir, tmp_path / "site", offline=True, n_sims=100, n_sims_final=100,
               today=date(2025, 6, 1), **kw)


def test_run_recomputes_after_a_correction_of_the_last_round(data_dir, tmp_path):
    # Repro r1 : une disqualification après coup n'était jamais prise en compte.
    assert _small_run(data_dir, tmp_path).updated is True
    raw = data_dir / "raw" / "results_2025.parquet"
    cur = pd.read_parquet(raw)
    winner = (cur["round"] == 5) & ~cur["is_sprint"] & (cur["position"] == 1)
    cur.loc[winner, ["points", "finished"]] = [0.0, False]
    cur.to_parquet(raw, index=False)
    out = _small_run(data_dir, tmp_path)
    assert out.updated is True
    assert _small_run(data_dir, tmp_path).updated is False  # stable une fois recalculé


def test_run_recomputes_when_a_late_sprint_arrives(data_dir, tmp_path):
    raw = data_dir / "raw" / "results_2025.parquet"
    full = pd.read_parquet(raw)
    full[~(full["is_sprint"] & (full["round"] == 3))].to_parquet(raw, index=False)
    assert _small_run(data_dir, tmp_path).updated is True
    full.to_parquet(raw, index=False)
    assert _small_run(data_dir, tmp_path).updated is True


def test_results_fingerprint_ignores_row_order_and_other_seasons(data_dir):
    from f1pred.data import load_results
    from f1pred.pipeline import results_fingerprint
    results = load_results(data_dir / "raw")
    fp = results_fingerprint(results, 2025)
    assert fp == results_fingerprint(results.sample(frac=1, random_state=1), 2025)
    assert fp == results_fingerprint(results[results["season"] != 2024], 2025)
    changed = results.copy()
    changed.loc[changed.index[-1], "points"] += 1
    assert fp != results_fingerprint(changed, 2025)


def _miss_round(results, driver, rnd, reserve="reserve_x"):
    """`driver` manque la manche `rnd` de 2025 : un réserviste prend son volant ce week-end."""
    out = results.copy()
    m = (out["season"] == 2025) & (out["round"] == rnd) & (out["driver_id"] == driver)
    out.loc[m, ["driver_id", "driver_name", "points"]] = [reserve, "Reserve X", 0.0]
    return out


def test_regular_who_misses_the_last_race_keeps_a_real_forecast(data_dir):
    # Repro r4 : un titulaire absent une seule course était traité comme parti pour de bon.
    from f1pred.data import load_results, load_schedule
    results = _miss_round(load_results(data_dir / "raw"), "d0", 5)
    cp = predict_from(results, load_schedule(data_dir / "raw", 2025), 2025, 5,
                      n_sims=500, seed=0)
    validate(cp)
    i = cp.state.drivers.index("d0")
    assert cp.races[0].pred_pos[i] < 30
    assert cp.races[0].p_dnf[i] < 1.0
    assert cp.sim.expected_points["d0"] > cp.state.points[i]
    # Le réserviste ne garde pas le baquet pour le reste de la saison.
    j = cp.state.drivers.index("reserve_x")
    assert cp.races[0].p_dnf[j] == 1.0
    assert cp.sim.expected_points["reserve_x"] == pytest.approx(cp.state.points[j])


def test_regular_who_missed_a_race_mid_window_keeps_his_seat(data_dir):
    from f1pred.data import load_results, load_schedule
    from f1pred.pipeline import forecast
    results = _miss_round(load_results(data_dir / "raw"), "d3", 2)
    fc = forecast(results, load_schedule(data_dir / "raw", 2025), 2025, 5)
    assert set(fc.preds["driver_id"]) == {f"d{i}" for i in range(10)}


def test_lineup_keeps_the_two_drivers_with_most_starts_in_the_last_five_gp():
    from f1pred.pipeline import current_lineup
    results = make_results(seasons=(2025,), n_rounds=6, sprint_rounds=())
    # Chez t0 : d0 court les manches 1-3, "new" les manches 4-6. Fenêtre = manches 2-6 :
    # d0 2 départs, new 3, d1 5.
    m = (results["driver_id"] == "d0") & (results["round"] >= 4)
    results.loc[m, ["driver_id", "driver_name"]] = ["new", "New"]
    lineup = current_lineup(results, 2025, 6)
    assert set(lineup.loc[lineup["constructor_id"] == "t0", "driver_id"]) == {"d1", "new"}
    # d0 récupère la manche 4 : d0 3 départs (2-4), new 2 (5-6). Les départs priment.
    m = (results["driver_id"] == "new") & (results["round"] == 4)
    results.loc[m, ["driver_id", "driver_name"]] = ["d0", "Pilote 0"]
    lineup = current_lineup(results, 2025, 6)
    assert set(lineup.loc[lineup["constructor_id"] == "t0", "driver_id"]) == {"d1", "d0"}
    assert len(lineup) == 10


def test_lineup_tie_goes_to_most_recent_start():
    from f1pred.pipeline import current_lineup
    results = make_results(seasons=(2025,), n_rounds=4, sprint_rounds=())
    # Chez t0, le volant de d1 passe à x (manches 1-2) puis à y (manches 3-4) : égalité
    # 2-2 pour le 2e baquet, y a couru le plus récemment.
    for rnd, who in [(1, "x"), (2, "x"), (3, "y"), (4, "y")]:
        m = (results["driver_id"] == "d1") & (results["round"] == rnd)
        results.loc[m, ["driver_id", "driver_name"]] = [who, who.upper()]
    lineup = current_lineup(results, 2025, 4)
    assert set(lineup.loc[lineup["constructor_id"] == "t0", "driver_id"]) == {"d0", "y"}


def test_driver_who_changed_team_counts_only_for_his_current_team():
    from f1pred.pipeline import current_lineup
    results = make_results(seasons=(2025,), n_rounds=5, sprint_rounds=())
    # d2 (t1) passe chez t0 à la manche 4 à la place de d1, qui prend son volant chez t1.
    m2 = (results["driver_id"] == "d2") & (results["round"] >= 4)
    m1 = (results["driver_id"] == "d1") & (results["round"] >= 4)
    results.loc[m2, "constructor_id"] = "t0"
    results.loc[m1, "constructor_id"] = "t1"
    lineup = current_lineup(results, 2025, 5).set_index("driver_id")["constructor_id"]
    assert lineup.index.is_unique
    assert lineup["d2"] == "t0" and lineup["d1"] == "t1"


def test_postponed_round_stays_in_the_future(data_dir):
    # La manche 3 (week-end sprint) est reportée : courue plus tard, elle reste à prévoir.
    from f1pred.data import load_results, load_schedule
    results = load_results(data_dir / "raw")
    results = results[~((results["season"] == 2025) & (results["round"] == 3))]
    cp = predict_from(results, load_schedule(data_dir / "raw", 2025), 2025, 5,
                      n_sims=200, seed=0)
    validate(cp)
    assert sum(not r.is_sprint for r in cp.races) == 4  # manches 3, 6, 7, 8
    assert sum(r.is_sprint for r in cp.races) == 1  # sprint de la manche 3


@pytest.mark.parametrize("field", ["sigma", "tau_driver", "tau_team", "dnf_shock"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.5])
def test_validate_rejects_bad_shocks(data_dir, field, value):
    from dataclasses import replace

    from f1pred.data import load_results, load_schedule
    results = load_results(data_dir / "raw")
    cp = predict_from(results, load_schedule(data_dir / "raw", 2025), 2025, 5,
                      n_sims=200, seed=0)
    validate(cp)
    cp.shocks = replace(cp.shocks, **{field: value})
    with pytest.raises(ValidationError, match=field):
        validate(cp)


def _postpone_round_3(data_dir):
    """Manche 3 courue une semaine après la 5 (calendrier mis à jour), sous son numéro."""
    from tests.test_chrono import postponed
    raw = data_dir / "raw"
    postponed(pd.read_parquet(raw / "results_2025.parquet")).to_parquet(
        raw / "results_2025.parquet", index=False)
    sched = pd.read_parquet(raw / "schedule_2025.parquet")
    postponed(sched).to_parquet(raw / "schedule_2025.parquet", index=False)


def test_round_raced_late_is_unknown_at_earlier_checkpoints(data_dir):
    from f1pred.data import load_results, load_schedule
    from f1pred.pipeline import forecast
    _postpone_round_3(data_dir)
    results, sched = load_results(data_dir / "raw"), load_schedule(data_dir / "raw", 2025)
    fc = forecast(results, sched, 2025, 4)
    cur = results[(results["season"] == 2025) & results["round"].isin([1, 2, 4])]
    expected = cur.groupby("driver_id")["points"].sum().reindex(fc.state.drivers).fillna(0)
    assert fc.state.points.tolist() == expected.round().astype(int).tolist()
    assert fc.state.last_round == 4
    assert [r.round for r in fc.races if not r.is_sprint] == [5, 3, 6, 7, 8]
    assert [r.round for r in fc.races if r.is_sprint] == [3]
    assert forecast(results, sched, 2025, 3).state.last_round == 3


def test_run_publishes_the_most_recently_raced_round(data_dir, tmp_path):
    _postpone_round_3(data_dir)
    out = _small_run(data_dir, tmp_path)
    assert out.updated is True and "manche 3" in out.message
    state = json.loads((data_dir / "state.json").read_text(encoding="utf-8"))
    assert state["round"] == 3
    rounds = [e["round"] for e in load_history(data_dir / "history.json")]
    assert rounds == [1, 2, 4, 5, 3]
