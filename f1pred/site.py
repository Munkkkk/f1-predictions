"""Génération du site statique (Jinja2 + Plotly)."""

import html
from pathlib import Path

import plotly.graph_objects as go
import plotly.offline
from jinja2 import Environment, PackageLoader, select_autoescape

ACCENT = "#e10600"
MUTED = "#8a8f98"
# Teintes catégorielles (bleu, vert, violet, ambre, magenta), toutes à >= 3:1 de contraste
# contre les cartes claire (#f4f4f6) et sombre (#1a1a22). Attribuées par identité du pilote
# (ordre alphabétique des driver_id tracés) ; le leader garde l'accent.
SERIES = ["#2a78d6", "#0e8f6a", "#7c6ad8", "#b07a00", "#c0508a"]
TOP_N_BARS = 10
TOP_N_LINES = 6

CHART_CONFIG = {"displayModeBar": False}
PLOTLY_JS = "plotly.min.js"  # servi par le site lui-même, version de plotly.py


def _label(text: str) -> str:
    """Plotly.js interprète un sous-ensemble de HTML dans les libellés : on échappe les
    chaînes venues de l'API (noms de pilotes, de GP)."""
    return html.escape(str(text), quote=False)


def format_pct(p: float, decided: bool) -> str:
    """Pourcentage à la française ; évite d'afficher 100 % / 0 % tant que le titre est ouvert."""
    if decided:
        v = p * 100
        return f"{round(v)} %" if abs(v - round(v)) < 1e-9 else f"{v:.1f}".replace(".", ",") + " %"
    if p >= 0.9995:
        return "> 99,9 %"
    if p < 0.0005:
        return "< 0,1 %"
    return f"{p * 100:.1f}".replace(".", ",") + " %"


def _fr(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _int_fr(n: int) -> str:
    return f"{int(n):,}".replace(",", " ")


_env = Environment(loader=PackageLoader("f1pred", "templates"),
                   autoescape=select_autoescape(["html"]))
_env.filters["pct"] = format_pct
_env.filters["fr"] = _fr
_env.filters["int_fr"] = _int_fr


def _layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=MUTED, size=13),
        showlegend=False,
    )
    return fig


def _bars_figure(drivers: list[dict], decided: bool) -> go.Figure:
    top = drivers[:TOP_N_BARS][::-1]
    fig = go.Figure(go.Bar(
        x=[d["p_title"] * 100 for d in top], y=[_label(d["name"]) for d in top],
        orientation="h",
        marker_color=[ACCENT if i == len(top) - 1 else MUTED for i in range(len(top))],
        text=[_label(format_pct(d["p_title"], decided)) for d in top], textposition="auto",
        cliponaxis=False,
        hovertemplate="%{y} : %{text}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 108], showgrid=False, visible=False)
    fig.update_yaxes(showgrid=False, automargin=True)
    return _layout(fig, 40 * len(top) + 40)


def _bars(drivers: list[dict], decided: bool) -> str:
    return _bars_figure(drivers, decided).to_html(
        full_html=False, include_plotlyjs=False, config=CHART_CONFIG)


def _line_colors(drivers: list[dict]) -> dict[str, str]:
    plotted = [d["driver_id"] for d in drivers[:TOP_N_LINES]]
    colors = {plotted[0]: ACCENT} if plotted else {}
    for driver_id, color in zip(sorted(plotted[1:]), SERIES, strict=False):
        colors[driver_id] = color
    return colors


def _lines_figure(history: list[dict], names: dict[str, str],
                  drivers: list[dict]) -> go.Figure:
    fig = go.Figure()
    colors = _line_colors(drivers)
    x = [_label(e["race_name"].replace(" Grand Prix", "")) for e in history]
    for i, d in enumerate(drivers[:TOP_N_LINES]):
        y = [e["probabilities"].get(d["driver_id"], 0.0) * 100 for e in history]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers",
            name=_label(names.get(d["driver_id"], d["driver_id"])),
            line=dict(width=3 if i == 0 else 2, color=colors[d["driver_id"]]),
            marker=dict(size=8),
            hovertemplate="%{fullData.name} : %{y:.1f} %<extra></extra>",
        ))
    fig.update_yaxes(range=[0, 100], ticksuffix=" %", gridcolor="rgba(138,143,152,0.2)")
    fig.update_xaxes(showgrid=False)
    fig = _layout(fig, 380)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.2), hovermode="x unified")
    return fig


def _lines(history: list[dict], names: dict[str, str], drivers: list[dict]) -> str:
    return _lines_figure(history, names, drivers).to_html(
        full_html=False, include_plotlyjs=False, config=CHART_CONFIG)


def render_site(out_dir: Path, context: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / PLOTLY_JS).write_text(plotly.offline.get_plotlyjs(), encoding="utf-8",
                                     newline="")
    charts = {
        "bars_html": _bars(context["drivers"], context["decided"]),
        "lines_html": _lines(context["history"], context["names"], context["drivers"]),
    }
    for page in ("index.html", "methode.html"):
        text = _env.get_template(page).render(**context, **charts, plotly_js=PLOTLY_JS)
        (out_dir / page).write_text(text, encoding="utf-8")
