"""Plotly-Figuren und Tabellen. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import ahp_constants as C
import ahp_matrix as mx
from ahp_evaluation import Analysis, de

ARROW = {1: "↑", -1: "↓"}
SHORT = {"wahr": "wahre Gewichte", "gleich": "gleiche Gewichte", "ev": "Eigenvektor", "gm": "geometrisches Mittel"}
QUESTION_SHORT = {"wichtigkeit": "Wichtigkeit", "schwenk": "Schwenk"}
METHOD_COLORS = {"wahr": C.COLOR_TRUE, "gleich": C.COLOR_FIT, "ev": C.COLOR_EV, "gm": C.COLOR_GM}


def alt_names(n: int) -> list[str]:
    return [f"A{i + 1}" for i in range(n)]


def _fixed(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_matrix(a: Analysis) -> go.Figure:
    """Die Urteilsmatrix (nach Verstärkung) als Heatmap; Farbe = Logarithmus des Eintrags, abgeschnittene Urteile tragen ein Sternchen."""
    m = a.settings.m
    star = np.zeros((m, m), dtype=bool)
    for (i, j), c in zip(mx.upper_pairs(m), a.clipped):
        star[i, j] = star[j, i] = bool(c)
    text = [[mx.format_entry(a.A[i, j], a.settings.jscale) + ("*" if star[i, j] else "") for j in range(m)] for i in range(m)]
    labels = [f"K{k + 1}" for k in range(m)]
    lim = float(np.log10(max(C.SAATY_MAX, np.max(a.A))))
    fig = go.Figure(go.Heatmap(z=np.log10(a.A), x=labels, y=labels, text=text, texttemplate="%{text}", zmin=-lim, zmax=lim, colorscale="RdBu_r", showscale=False,
                               hovertemplate="Zeile %{y} gegenüber Spalte %{x}: %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    power = "" if a.settings.gamma == 1.0 else f" hoch γ = {de(a.settings.gamma, 1)}"
    fig.update_layout(title=dict(text=f"Urteilsmatrix{power} (* = abgeschnitten)", font=dict(size=13)))
    return _fixed(fig, 340)


def build_weights(a: Analysis) -> go.Figure:
    """Wahre Gewichte des Entscheiders, die erfragten (falls anders) und die zwei abgeleiteten."""
    x = [f"K{k + 1}" for k in range(a.settings.m)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=a.case.w, name="wahre Gewichte (Entscheider)", marker_color=C.COLOR_TRUE))
    if not np.allclose(a.reference, a.case.w):
        fig.add_trace(go.Bar(x=x, y=a.reference, name="erfragt (Schwenk in der Stichprobe)", marker_color=C.COLOR_FIT))
    fig.add_trace(go.Bar(x=x, y=a.ev, name="Eigenvektor", marker_color=C.COLOR_EV))
    fig.add_trace(go.Bar(x=x, y=a.gm, name="geometrisches Mittel", marker_color=C.COLOR_GM))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Gewicht", range=[0, 1])
    return _fixed(fig, 340)


def build_weight_table(a: Analysis) -> pd.DataFrame:
    m = a.settings.m
    rank = lambda v: (np.argsort(np.argsort(-np.asarray(v), kind="stable")) + 1)  # noqa: E731
    cols = {"Kriterium": [f"K{k + 1}" for k in range(m)], "wahres Gewicht": a.case.w, "erfragt": a.reference, "Eigenvektor": a.ev, "geometrisches Mittel": a.gm,
            "Rang Eigenvektor": rank(a.ev), "Rang geom. Mittel": rank(a.gm)}
    return pd.DataFrame(cols)


def style_weights(df: pd.DataFrame):
    fmt = {c: (lambda v: de(v, 3)) for c in ("wahres Gewicht", "erfragt", "Eigenvektor", "geometrisches Mittel")}
    return df.style.format(fmt)


def build_method_table(a: Analysis) -> pd.DataFrame:
    names = alt_names(a.case.matrix.n)
    rows = []
    for k, r in a.results.items():
        rows.append({"Gewichte": SHORT[k], "Platz 1": names[r.top1], "Trifft den Besten": "ja" if r.hit else "nein", "τ gegen Nutzen": de(r.tau, 2),
                     "Nutzenverlust": de(r.loss, 3), "Abstand zu den erfragten Gewichten": de(r.tv, 3)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Experimente

def build_ev_gm(rows: list[dict]) -> go.Figure:
    """Anteil der Matrizen, in denen Eigenvektor und geometrisches Mittel mindestens ein Paar entgegengesetzt ordnen."""
    fig = go.Figure()
    for m in sorted({r["m"] for r in rows}):
        rs = [r for r in rows if r["m"] == m]
        fig.add_trace(go.Scatter(x=[r["sigma"] for r in rs], y=[100 * r["rev"] for r in rs], mode="lines+markers", name=f"m = {m}"))
    fig.update_xaxes(title="Rauschen σ der Urteile", tickvals=list(C.EXP_SIGMAS))
    fig.update_yaxes(title="entgegengesetzt geordnetes Paar (% der Matrizen)", rangemode="tozero")
    return _fixed(fig, 320)


def ev_gm_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Kriterien m": r["m"], "σ": de(r["sigma"], 1), "Abstand Eigenvektor": de(r["tv_ev"], 3), "Abstand geom. Mittel": de(r["tv_gm"], 3),
                          "Platz 1 Eigenvektor": f"{de(100 * r['hit_ev'], 1)} %", "Platz 1 geom. Mittel": f"{de(100 * r['hit_gm'], 1)} %",
                          "Platz 1 wahre Gewichte": f"{de(100 * r['hit_wahr'], 1)} %", "andere Ordnung": f"{de(100 * r['rev'], 1)} %"} for r in rows])


def build_comparisons(rows: list[dict]) -> go.Figure:
    fig = go.Figure(go.Bar(x=[f"{r['label']} ({de(r['comparisons'], 0)})" for r in rows], y=[r["tv"] for r in rows], marker_color=C.COLOR_EV))
    fig.update_yaxes(title="Abstand zu den wahren Gewichten (TV, kleiner ist besser)", rangemode="tozero")
    return _fixed(fig, 300)


def build_intensify(rows: list[dict]) -> go.Figure:
    labels = [f"γ = {de(r['gamma'], 1)}" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[100 * r["ev"] for r in rows], name="Eigenvektor", marker_color=C.COLOR_EV))
    fig.add_trace(go.Bar(x=labels, y=[100 * r["gm"] for r in rows], name="geometrisches Mittel", marker_color=C.COLOR_GM))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Ordnung kehrt sich um (% der Matrizen)", rangemode="tozero")
    return _fixed(fig, 320)


def counterexample_table(ce: dict) -> pd.DataFrame:
    fmt = lambda v: " ; ".join(de(x, 3) for x in v)  # noqa: E731
    order = lambda o: " > ".join(f"K{k}" for k in o)  # noqa: E731
    return pd.DataFrame([
        {"Matrix": "A", "Verfahren": "Eigenvektor", "Gewichte": fmt(ce["ev1"]), "Ordnung": order(ce["order_ev1"])},
        {"Matrix": "A², jedes Urteil quadriert", "Verfahren": "Eigenvektor", "Gewichte": fmt(ce["ev2"]), "Ordnung": order(ce["order_ev2"])},
        {"Matrix": "A", "Verfahren": "geometrisches Mittel", "Gewichte": fmt(ce["gm1"]), "Ordnung": order(ce["order_gm1"])},
        {"Matrix": "A², jedes Urteil quadriert", "Verfahren": "geometrisches Mittel", "Gewichte": fmt(ce["gm2"]), "Ordnung": order(ce["order_gm2"])},
    ])


def build_consistency(rows: list[dict]) -> go.Figure:
    labels = [f"σ = {de(r['sigma'], 1)}" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["tv_ok"] for r in rows], name="CR ≤ 0,1", marker_color=C.COLOR_EV))
    fig.add_trace(go.Bar(x=labels, y=[r["tv_bad"] for r in rows], name="CR > 0,1", marker_color=C.COLOR_WARN))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Abstand des Eigenvektors zu w (TV)", rangemode="tozero")
    return _fixed(fig, 320)


def consistency_table(cons: list[dict], comp: list[dict]) -> pd.DataFrame:
    rows = [{"Fall": f"verrauscht, σ = {de(r['sigma'], 1)}", "mittleres CR": de(r["cr_mean"], 3), "CR > 0,1": f"{de(100 * r['over'], 1)} %",
             "Abstand bei CR ≤ 0,1": de(r["tv_ok"], 3), "Abstand bei CR > 0,1": de(r["tv_bad"], 3)} for r in cons]
    rows += [{"Fall": f"konsistent, verdichtet β = {de(r['beta'], 2)}", "mittleres CR": de(r["cr"], 3), "CR > 0,1": "0,0 %", "Abstand bei CR ≤ 0,1": de(r["tv"], 3),
              "Abstand bei CR > 0,1": "-"} for r in comp]
    return pd.DataFrame(rows)


def build_question(rows: list[dict], world: str = "roh") -> go.Figure:
    names = {"wahr": "wahre Gewichte", "wichtigkeit": "AHP mit Wichtigkeitsurteilen", "schwenk": "AHP mit Schwenkurteilen", "ziel": "Schwenk-Gewichte (ohne Urteil)"}
    colors = {"wahr": C.COLOR_TRUE, "wichtigkeit": C.COLOR_WARN, "schwenk": C.COLOR_EV, "ziel": C.COLOR_FIT}
    fig = go.Figure()
    for k in ("wahr", "wichtigkeit", "schwenk", "ziel"):
        rs = sorted([r for r in rows if r["world"] == world and r["method"] == k], key=lambda r: -r["delta"])
        fig.add_trace(go.Scatter(x=[de(r["delta"], 2) for r in rs], y=[100 * r["hit"] for r in rs], mode="lines+markers", name=names[k], line=dict(color=colors[k])))
    fig.update_xaxes(title="Anteil der Streuung im ersten Kriterium (1 = unverändert)")
    fig.update_yaxes(title="Platz 1 getroffen (%)", range=[0, 102])
    return _fixed(fig, 340)


def build_scale(rows: list[dict]) -> go.Figure:
    labels = [f"m = {r['m']}" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["tv_saaty"] for r in rows], name="Saaty-Skala 1 bis 9", marker_color=C.COLOR_WARN))
    fig.add_trace(go.Bar(x=labels, y=[r["tv_stetig"] for r in rows], name="stetige Urteile", marker_color=C.COLOR_EV))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Abstand des Eigenvektors zu w (TV)", rangemode="tozero")
    return _fixed(fig, 300)
