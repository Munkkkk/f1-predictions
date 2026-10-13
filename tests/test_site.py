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


def test_no_backtest_shows_fallback_not_nan(tmp_path):
    ctx = context()
    nan = float("nan")
    ctx["backtest"] = {"n_races": 0, "brier_winner": nan, "brier_winner_baseline": nan,
                       "brier_podium": nan, "brier_podium_baseline": nan}
    render_site(tmp_path, ctx)
    methode = (tmp_path / "methode.html").read_text(encoding="utf-8")
    assert "nan" not in methode.lower()
    assert "Pas encore assez de courses terminées" in methode


def test_line_colors_follow_driver_not_rank():
    from f1pred.site import _line_colors

    def drv(i):
        return {"driver_id": i}

    a = _line_colors([drv("x"), drv("b"), drv("c"), drv("d")])
    b = _line_colors([drv("x"), drv("d"), drv("b"), drv("c")])
    assert a == b
    assert len(set(a.values())) == len(a)
