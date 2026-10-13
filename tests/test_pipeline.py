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
