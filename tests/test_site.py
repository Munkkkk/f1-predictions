from f1pred.site import format_pct, render_site


def context(decided=False):
    return {
        "season": 2026, "updated_at": "2026-10-13", "last_race": "Singapore Grand Prix",
        "decided": decided, "champion": "Andrea Kimi Antonelli" if decided else None,
        "drivers": [
            {"driver_id": "antonelli", "name": "Andrea Kimi Antonelli", "team": "mercedes",
             "points": 340, "p_title": 0.91, "expected_points": 410.0,
             "max_points": 490, "eliminated": False},
            {"driver_id": "russell", "name": "George Russell", "team": "mercedes",
             "points": 238, "p_title": 0.09, "expected_points": 320.0,
             "max_points": 388, "eliminated": False},
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
        "remaining_points": 150, "remaining_gp": 6, "remaining_sprints": 0,
        "tau_driver": 1.83, "tau_team": 1.21, "dnf_shock": 0.62,
        "shock_horizon": 6, "shock_seasons": "2025–2026",
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


def test_format_pct_cases():
    assert format_pct(1.0, decided=False) == "> 99,9 %"
    assert format_pct(0.0, decided=False) == "< 0,1 %"
    assert format_pct(0.123, decided=False) == "12,3 %"
    assert format_pct(1.0, decided=True) == "100 %"
    assert format_pct(0.0, decided=True) == "0 %"


def test_undecided_certain_title_never_shows_100(tmp_path):
    ctx = context(decided=False)
    ctx["drivers"][0]["p_title"] = 1.0
    ctx["drivers"][1]["p_title"] = 0.0
    render_site(tmp_path, ctx)
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "&gt; 99,9 %" in index or "> 99,9 %" in index
    table = index.split("<tbody>")[1].split("</tbody>")[0]
    assert "100" not in table and "99,9" in table and "0,1" in table


def test_remaining_points_line_only_while_open(tmp_path):
    render_site(tmp_path, context(decided=False))
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "Il reste <strong>150 points</strong> en jeu (6 GP)" in index
    assert "sprint" not in index.split("Il reste")[1].split("</p>")[0]
    ctx = context(decided=False)
    ctx.update(remaining_points=166, remaining_gp=6, remaining_sprints=2)
    render_site(tmp_path, ctx)
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "Il reste <strong>166 points</strong> en jeu (6 GP, 2 sprints)" in index
    render_site(tmp_path, context(decided=True))
    assert "Il reste" not in (tmp_path / "index.html").read_text(encoding="utf-8")


def test_eliminated_driver_and_max_points_column(tmp_path):
    ctx = context()
    ctx["drivers"][1].update(p_title=0.0, eliminated=True)
    render_site(tmp_path, ctx)
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "Points max" in index
    table = index.split("<tbody>")[1].split("</tbody>")[0]
    russell = table.split("George Russell")[1].split("</tr>")[0]
    assert "Éliminé" in russell and "388" in russell
    antonelli = table.split("Andrea Kimi Antonelli")[1].split("</tr>")[0]
    assert "Éliminé" not in antonelli and "490" in antonelli


def test_methode_explains_persistent_shocks(tmp_path):
    render_site(tmp_path, context())
    methode = (tmp_path / "methode.html").read_text(encoding="utf-8")
    assert "±1,8 place" in methode and "±1,2 place" in methode
    assert "×e<sup>±0,6</sup>" in methode
    assert "abandons" in methode


def test_methode_describes_shock_window_and_hides_values_once_decided(tmp_path):
    render_site(tmp_path, context())
    methode = (tmp_path / "methode.html").read_text(encoding="utf-8")
    assert "10 dernières courses" not in methode
    assert "6 courses suivantes" in methode and "2025–2026" in methode
    render_site(tmp_path, context(decided=True))
    methode = (tmp_path / "methode.html").read_text(encoding="utf-8")
    assert "±1,8" not in methode and "±0,0" not in methode


def test_bars_and_lines_highlight_first_driver_in_points_order():
    from f1pred.site import ACCENT, MUTED, _bars, _line_colors
    drivers = [
        {"driver_id": "russell", "name": "George Russell", "points": 250, "p_title": 0.3},
        {"driver_id": "antonelli", "name": "Andrea Kimi Antonelli", "points": 240,
         "p_title": 0.7},
    ]
    html = _bars(drivers, decided=False)  # barres de bas en haut : le meneur en dernier
    assert '"y":["Andrea Kimi Antonelli","George Russell"]' in html
    assert f'"color":["{MUTED}","{ACCENT}"]' in html
    assert _line_colors(drivers)["russell"] == ACCENT


def test_table_follows_context_order(tmp_path):
    ctx = context()
    ctx["drivers"].reverse()
    render_site(tmp_path, ctx)
    table = (tmp_path / "index.html").read_text(encoding="utf-8").split("<tbody>")[1]
    assert table.index("George Russell") < table.index("Andrea Kimi Antonelli")


EVIL = '<a href="https://evil.example">x</a>'
CSP = ("default-src 'none'; script-src 'self' 'unsafe-inline'; "
       "style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'none'; "
       "base-uri 'none'; form-action 'none'")


def test_plotly_is_self_hosted_and_loaded_only_where_charts_are(tmp_path):
    render_site(tmp_path, context())
    js = (tmp_path / "plotly.min.js").read_text(encoding="utf-8")
    assert "plotly" in js[:2000].lower() and len(js) > 1_000_000
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    methode = (tmp_path / "methode.html").read_text(encoding="utf-8")
    assert '<script src="plotly.min.js"' in index
    assert "plotly.min.js" not in methode
    assert "cdn.plot.ly" not in index + methode
    assert "https://" not in index.split("<body>")[0]  # aucune ressource externe dans <head>


def test_pages_carry_csp_and_referrer_policy(tmp_path):
    render_site(tmp_path, context())
    for page in ("index.html", "methode.html"):
        html = (tmp_path / page).read_text(encoding="utf-8")
        assert '<meta name="referrer" content="no-referrer">' in html
        assert f'<meta http-equiv="Content-Security-Policy" content="{CSP}">' in html


def test_api_strings_are_escaped_in_chart_labels(tmp_path):
    from f1pred.site import _bars_figure, _lines_figure
    ctx = context()
    ctx["drivers"][0]["name"] = EVIL
    ctx["names"]["antonelli"] = EVIL
    ctx["history"][0]["race_name"] = EVIL
    bars = _bars_figure(ctx["drivers"], decided=False).to_json()
    lines = _lines_figure(ctx["history"], ctx["names"], ctx["drivers"]).to_json()
    for fig_json in (bars, lines):
        assert "<a" not in fig_json and r"\u003ca" not in fig_json
        assert "&lt;a href" in fig_json
    render_site(tmp_path, ctx)
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert EVIL not in index
    assert r"\u003ca href" not in index
