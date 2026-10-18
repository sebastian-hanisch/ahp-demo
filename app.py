"""AHP: Paarvergleiche, Eigenvektor, Konsistenz - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Viertes Stück der Linie Mehrkriterien-Ranking (Kind von „Gewichte erheben“). Dieselbe neutrale Entscheidungsmatrix; ein simulierter Entscheider kennt seine Gewichte und
urteilt über ihre Verhältnisse, verrauscht und auf die Saaty-Skala gerundet. Das Urteilsmodell ist angenommen, nicht am Menschen gemessen. Kern: ahp_matrix.py,
ahp_weights.py, ahp_score.py; Generator: entscheidungsmatrix_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import ahp_constants as C
from ahp_evaluation import (
    Settings,
    analyse,
    counterexample,
    de,
    experiment_comparisons,
    experiment_compression,
    experiment_consistency,
    experiment_ev_gm,
    experiment_intensify,
    experiment_question,
    experiment_scale,
    random_index_table,
    verdict_consistency,
    verdict_ev_gm,
    verdict_intensify,
    verdict_question,
)
from ahp_pdf_export import build_pdf
from ahp_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from ahp_visualization import (
    SHORT,
    alt_names,
    build_comparisons,
    build_consistency,
    build_ev_gm,
    build_intensify,
    build_matrix,
    build_method_table,
    build_question,
    build_scale,
    build_weight_table,
    build_weights,
    consistency_table,
    counterexample_table,
    ev_gm_table,
    style_weights,
)

st.set_page_config(page_title="AHP: Paarvergleiche – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Die vier Experimente rechnen …")
def _experiments():
    return {
        "ev_gm": experiment_ev_gm(),
        "comparisons": experiment_comparisons(),
        "intensify": experiment_intensify(),
        "counter": counterexample(),
        "consistency": experiment_consistency(),
        "compression": experiment_compression(),
        "ri": random_index_table(),
        "question": experiment_question(),
        "scale": experiment_scale(),
    }


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("⚖️ AHP: Paarvergleiche, Eigenvektor, Konsistenz")
st.markdown(
    """
**Im Urteilsmodell dieser Demo ist nicht die Ableitung (Eigenvektor oder geometrisches Mittel) das Problem, sondern Skala, Frage und Konsistenzmaß.** Dies ist das vierte Stück der Linie **Mehrkriterien-Ranking**: Statt Punkte zu
vergeben, vergleicht der Entscheider die Kriterien **paarweise** (wie viel wichtiger ist i als j?), und aus der Matrix dieser Urteile werden die Gewichte abgeleitet, entweder
über den **Eigenvektor** oder das **geometrische Mittel** der Zeilen. Die Demo misst beides an der **Entscheidung**, die die gewichtete Summe damit trifft. Der Entscheider ist
simuliert, ebenso sein **Urteil**: Rauschen und Rundung sind angenommen, nicht am Menschen gemessen.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert AHP", expanded=True):
    st.markdown(
        """
1. **Paarvergleiche:** Für jedes Paar von Kriterien nennt der Entscheider ein Verhältnis auf der Saaty-Skala (1 = gleich wichtig, 9 = extrem wichtiger, dazu die Kehrwerte).
   Bei m Kriterien sind das m(m − 1)/2 Urteile; die Matrix ist reziprok: a_ji = 1 / a_ij.
2. **Gewichte:** Der **Eigenvektor** zum größten Eigenwert der Matrix oder das **geometrische Mittel** jeder Zeile, jeweils auf Summe 1. Ist die Matrix vollständig konsistent
   (a_ik = a_ij · a_jk), liefern beide dasselbe; bei 3 Kriterien liefern sie es immer.
3. **Konsistenz:** CI = (λmax − m) / (m − 1) misst, wie weit die Urteile von der Konsistenz entfernt sind; CR = CI / RI teilt durch den **Random Index** RI, den mittleren CI
   zufälliger Matrizen. CR ≤ 0,1 ist eine Konvention, keine Naturkonstante; die Demo simuliert den Random Index selbst.
4. **Gemessen wird an der Entscheidung:** Platz 1 der gewichteten Summe (Min-Max) gegen den Besten des Entscheiders, Kendalls τ der Rangfolge und der Nutzenverlust.
5. **Der Entscheider ist simuliert:** Er kennt seine Gewichte; jedes Paarurteil ist das wahre Verhältnis mit lognormalem Rauschen (σ), gerundet und bei 9 abgeschnitten.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Matrix**")
    n = st.slider("Alternativen n", *bounds("n_slider"), key="n_slider", help="Wie viele Alternativen zur Wahl stehen.")
    m = st.slider("Kriterien m", *bounds("m_slider"), key="m_slider",
                  help="Zahl der Kriterien, also Größe der Paarvergleichsmatrix. Bei 3 Kriterien sind Eigenvektor und geometrisches Mittel identisch.")
    delta = st.selectbox("Streuung des ersten Kriteriums", C.DELTA_CHOICES, key="delta_select", format_func=lambda d: "unverändert" if d == 1.0 else f"{de(100 * d, 0)} % der Streuung",
                         help="Wie stark sich die Alternativen im ersten Kriterium unterscheiden. Bei kleiner Streuung ist die Spannweite in der Stichprobe schmal: dort entscheidet "
                              "die Frage, ob die Urteile die Spannweite enthalten.")
    st.markdown("**Urteile**")
    sigma = st.selectbox("Rauschen σ der Urteile", C.SIGMA_CHOICES, key="sigma_select", format_func=lambda s: de(s, 1),
                         help="Streuung des Logarithmus jedes Verhältnisurteils (lognormal). 0: exakte Verhältnisse, nur gerundet; 0,8: ein Verhältnis von 2 wird oft als 1 (27 %) oder als 4 und mehr (24 %) genannt.")
    jscale = st.selectbox("Urteilsskala", C.JUDGMENT_SCALES, key="jscale_select", format_func=lambda k: C.JUDGMENT_SCALE_LABELS[k],
                          help="Saaty: ganze Zahlen 1 bis 9 und Kehrwerte, größere Verhältnisse werden bei 9 abgeschnitten. Stetig: jedes Verhältnis, als Gegenprobe zur Skala.")
    question = st.selectbox("Frage an den Entscheider", C.QUESTIONS, key="question_select", format_func=lambda k: C.QUESTION_LABELS[k],
                            help="Wichtigkeit: Verhältnis der Gewichte, ohne Bezug zu den Alternativen. Schwenk: Verhältnis der Schwenks worst → best in der Stichprobe, "
                                 "also Gewicht mal Spannweite des Wertes.")
    st.markdown("**Fehlerquellen**")
    gamma = st.selectbox("Verstärkung γ der Urteile", C.GAMMA_CHOICES, key="gamma_select", format_func=lambda g: "keine (γ = 1)" if g == 1.0 else f"γ = {de(g, 1)}",
                         help="Jedes Urteil wird mit γ potenziert (die Richtung bleibt). Das geometrische Mittel ändert dadurch seine Ordnung nie, der Eigenvektor kann sie ändern.")
    beta = st.selectbox("Verdichtung β der erfragten Gewichte", C.BETA_CHOICES, key="beta_select", format_func=lambda b: "keine (β = 1)" if b == 1.0 else f"β = {de(b, 2)}",
                        help="Die Urteile beziehen sich auf w hoch β statt auf w: ein systematischer Fehler, der die Unterschiede unterschätzt. Ohne Rauschen bleibt die Matrix konsistent.")
    st.markdown("**Entscheider (simuliert)**")
    world = st.selectbox("Wertskala", C.WORLDS, key="world_select", format_func=lambda k: C.WORLD_LABELS[k],
                         help="Auf welcher Skala der verborgene Entscheider linear bewertet.")
    st.markdown("**Anzeige**")
    threshold = st.selectbox("Schwelle für das Konsistenzverhältnis", C.THRESHOLD_CHOICES, key="threshold_select", format_func=lambda t: de(t, 2),
                             help="Die Schwelle 0,1 ist eine Konvention. Der Regler ändert nur die Anzeige, nicht die Gewichte und nicht die Experimente.")
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Matrix würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "m_slider": int(m), "sigma_select": float(sigma), "jscale_select": jscale, "question_select": question, "world_select": world,
                   "delta_select": float(delta), "gamma_select": float(gamma), "beta_select": float(beta), "threshold_select": float(threshold), "seed_input": int(seed)})

settings = Settings(int(n), int(m), float(sigma), jscale, question, world, float(delta), float(gamma), float(beta), float(threshold), int(seed))
a = _analysis(settings)
names = alt_names(a.case.matrix.n)
ev_r, gm_r, ref = a.results["ev"], a.results["gm"], a.results["wahr"]

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Platz 1: Eigenvektor", names[ev_r.top1], delta=f"bester nach Nutzen: {names[a.best]}", delta_color="off",
          help="Platz 1 der gewichteten Summe (Min-Max) mit den Eigenvektor-Gewichten; „bester nach Nutzen“ ist die Alternative, die der simulierte Entscheider am höchsten bewertet.")
c2.metric("Platz 1: geom. Mittel", names[gm_r.top1], delta=f"τ gegen Nutzen: {de(gm_r.tau, 2)} (Eigenvektor {de(ev_r.tau, 2)})", delta_color="off",
          help="τ ist Kendalls τ-b der ganzen Rangfolge gegen den verborgenen Nutzen.")
c3.metric("Konsistenz CR", de(a.cr, 3), delta=f"Schwelle {de(settings.threshold, 2)}: {'darunter' if a.cr <= settings.threshold else 'darüber'}", delta_color="off",
          help="CR = CI / RI mit dem simulierten Random Index. Eine Aussage über Widersprüche zwischen den Urteilen, keine Garantie für richtige Gewichte.")
c4.metric("λmax", de(a.lmax, 3), delta=f"CI = {de(a.ci, 3)}, RI = {de(a.ri, 2)} (simuliert)", delta_color="off",
          help=f"Größter Eigenwert der Matrix; bei vollständiger Konsistenz gleich m = {settings.m}. Der Random Index ist der Mittelwert über {C.RI_DRAWS:,} zufällige Matrizen "
               f"(Standardfehler {de(a.ri_se, 3)}).".replace(",", "."))

left, right = st.columns(2)
with left:
    st.plotly_chart(build_matrix(a), width="stretch", key="matrix_chart")
with right:
    st.plotly_chart(build_weights(a), width="stretch", key="weights_chart")

if settings.gamma != 1.0:
    st.info(f"Verstärkung γ = {de(settings.gamma, 1)}: Gegenüber der unverstärkten Matrix ordnet der Eigenvektor "
            f"{'mindestens ein Paar entgegengesetzt' if a.reversal_ev else 'alle Paare gleich'}, das geometrische Mittel "
            f"{'mindestens ein Paar entgegengesetzt' if a.reversal_gm else 'alle Paare gleich'} (Gleichstände zählen nicht als Umkehr). "
            "Konsistenzverhältnis und λmax gelten für die verstärkte Matrix.")

tab_w, tab_m = st.tabs(["Gewichte je Kriterium", "Alle Gewichte im Vergleich"])
with tab_w:
    st.dataframe(style_weights(build_weight_table(a)), hide_index=True, width="stretch")
    if a.clipped.any():
        st.caption(f"{int(a.clipped.sum())} Urteil(e) lagen über 9 und wurden auf 9 abgeschnitten (Sternchen in der Matrix).")
    if settings.question == "schwenk":
        st.caption("Gefragt wurde nach dem Schwenk: „erfragt“ sind die Gewichte mal Spannweite des Wertes in der Stichprobe, auf Summe 1. Sie sind das Ziel der Ableitung.")
with tab_m:
    st.dataframe(build_method_table(a), hide_index=True, width="stretch")
    st.caption("Dieselbe Matrix, derselbe Entscheider. „Abstand zu den erfragten Gewichten“ ist der halbe L1-Abstand; bei der Frage nach dem Schwenk sind die wahren Gewichte "
               "nicht das Ziel der Ableitung.")

if st.button("📄 PDF des aktuellen Falls erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="ahp.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Ändert die Art der Ableitung die Entscheidung, und sagt CR etwas über die Güte?")
if ev_r.hit and gm_r.hit:
    st.success(f"Beide Ableitungen treffen den Besten ({names[a.best]})" + (", ebenso die wahren Gewichte." if ref.hit else "; die wahren Gewichte verfehlen ihn."))
elif ev_r.hit != gm_r.hit:
    st.warning(f"Die Ableitungen entscheiden verschieden: {'Eigenvektor' if ev_r.hit else 'geometrisches Mittel'} trifft den Besten ({names[a.best]}), "
               f"{'geometrisches Mittel' if ev_r.hit else 'Eigenvektor'} wählt {names[gm_r.top1 if ev_r.hit else ev_r.top1]}.")
else:
    st.warning(f"Weder Eigenvektor noch geometrisches Mittel treffen den Besten ({names[a.best]}); die wahren Gewichte {'treffen' if ref.hit else 'verfehlen'} ihn.")
st.caption(f"Konsistenzverhältnis {de(a.cr, 3)} ({'unter' if a.cr <= settings.threshold else 'über'} der Schwelle {de(settings.threshold, 2)}); Abstand des Eigenvektors zu den "
           f"erfragten Gewichten {de(ev_r.tv, 3)}. Das Konsistenzverhältnis misst Widersprüche zwischen den Urteilen; ein kleiner Wert schließt systematisch verzerrte Urteile "
           "nicht aus (Experiment 3).")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Die vier Experimente")
st.caption("Je 200 feste Matrizen (SWEEP_SEEDS, n = 20, Min-Max) oder 2 000 bis 5 000 feste Ziehungen, unabhängig von den Reglern. Die Experimente starten erst auf Klick.")
if st.button("▶ Experimente rechnen", key="exp_start"):
    st.session_state["exp_rows"] = _experiments()
exp = st.session_state.get("exp_rows")
if exp:
    st.markdown("### Experiment 1: Eigenvektor oder geometrisches Mittel?")
    st.caption("Saaty-Skala, Frage nach der Wichtigkeit, Entscheider in den Rohwerten, 200 Matrizen. Standardfehler der Trefferquoten 1,3 bis 3,5 Punkte: kleinere Unterschiede "
               "sind nicht belastbar.")
    st.plotly_chart(build_ev_gm(exp["ev_gm"]), width="stretch", key="evgm_chart")
    verdict(verdict_ev_gm(exp["ev_gm"]))
    st.dataframe(ev_gm_table(exp["ev_gm"]), hide_index=True, width="stretch")
    st.markdown("**Weniger Vergleiche** (Log-Kleinste-Quadrate auf den vorhandenen Paaren; 5 Kriterien, σ = 0,2; in Klammern die Zahl der Vergleiche)")
    st.plotly_chart(build_comparisons(exp["comparisons"]), width="stretch", key="comparisons_chart")

    st.markdown("### Experiment 2: Wenn die Urteile verstärkt werden")
    st.caption("5 Kriterien, σ = 0,4, Saaty-Skala, 5 000 Matrizen: Anteil, bei dem die elementweise Potenz γ ein Paar der Gewichte strikt umkehrt.")
    st.plotly_chart(build_intensify(exp["intensify"]), width="stretch", key="intensify_chart")
    verdict(verdict_intensify(exp["intensify"]))
    st.caption("Ein festes Gegenbeispiel (4 Kriterien; A12 = A13 = A14 = 1/3, A23 = 1/2, A24 = 7, A34 = 3):")
    st.dataframe(counterexample_table(exp["counter"]), hide_index=True, width="stretch")

    st.markdown("### Experiment 3: Konsistenz ist keine Richtigkeit")
    st.caption("5 Kriterien, Saaty-Skala. Verrauscht: 5 000 Matrizen je σ. Verdichtet: 2 000 Ziehungen, die Matrix aus w hoch β ohne Rauschen und Rundung (vollständig konsistent).")
    st.plotly_chart(build_consistency(exp["consistency"]), width="stretch", key="consistency_chart")
    verdict(verdict_consistency(exp["consistency"], exp["compression"]))
    st.dataframe(consistency_table(exp["consistency"], exp["compression"]), hide_index=True, width="stretch")
    st.markdown("**Der Random Index hängt an der Erzeugung der Zufallsmatrizen.** Simuliert (Einträge unabhängig und gleichverteilt aus den 17 Skalenwerten) gegen zwei "
                "Tabellen aus Suchzusammenfassungen von Quellen (nicht im Original geprüft):")
    st.dataframe([{"m": r["m"], "simuliert": de(r["sim"], 3), "Standardfehler": de(r["se"], 3), "Tabelle A": de(r["table_a"], 2), "Tabelle B": de(r["table_b"], 2)}
                  for r in exp["ri"]], hide_index=True, width="stretch")

    st.markdown("### Experiment 4: Die Frage entscheidet, und die Skala 1 bis 9")
    st.caption("5 Kriterien, σ = 0,2, Saaty-Skala, Eigenvektor; das erste Kriterium wird auf einen Anteil seiner Streuung gestaucht. Entscheider in den Rohwerten.")
    st.plotly_chart(build_question(exp["question"], "roh"), width="stretch", key="question_chart")
    verdict(verdict_question(exp["question"]))
    gd = {x["method"]: x for x in exp["question"] if x["world"] == "guete" and x["delta"] == min(d["delta"] for d in exp["question"])}
    st.caption(f"Entscheider in der Güte, {de(100 * min(d['delta'] for d in exp['question']), 0)} % der Streuung: wahre Gewichte {de(100 * gd['wahr']['hit'], 0)} %, "
               f"Wichtigkeitsurteile {de(100 * gd['wichtigkeit']['hit'], 0)} %, Schwenkurteile {de(100 * gd['schwenk']['hit'], 0)} % (die Güte-Skala ist bei den logarithmischen "
               "Kriterien nicht linear in den Rohwerten, daher erreichen auch die Schwenk-Gewichte dort nicht 100 %).")
    st.markdown("**Die Skala 1 bis 9** (σ = 0,2; 3 000 Ziehungen je m; dieselben Rauschwerte, einmal gerundet und abgeschnitten, einmal stetig):")
    st.plotly_chart(build_scale(exp["scale"]), width="stretch", key="scale_chart")
    st.caption("Anteil der Paare mit wahrem Verhältnis über 9: " + ", ".join(f"m = {r['m']}: {de(100 * r['over_nine'], 0)} %" for r in exp["scale"]) +
               f". Die Skala allein macht den Fehler {de(min(r['tv_saaty'] / r['tv_stetig'] for r in exp['scale']), 1)}- bis "
               f"{de(max(r['tv_saaty'] / r['tv_stetig'] for r in exp['scale']), 1)}-mal so groß wie bei stetigen Urteilen.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Urteilsmodell stimmt** (lognormales Rauschen, gleiches σ für alle Paare, keine Zyklen als Absicht) | Es ist angenommen, nicht gemessen. Echte Urteile haben Anker, runde Zahlen und systematische Verzerrungen; die Demo sagt nichts über Menschen. | – |
| **Die Konsistenz echter Urteile** | Wie konsistent Menschen urteilen, ist nicht simuliert; ob CR ≤ 0,1 echte Fehler trennt, ist eine Frage an Menschen. | – |
| **Der Random Index ist eine Konstante** | Er hängt an der Erzeugung der Zufallsmatrizen; die Demo simuliert ihn (gleichverteilte Skalenwerte) und zeigt zwei Tabellen daneben. | – |
| **Die Saaty-Skala 1 bis 9** | Größere Verhältnisse werden abgeschnitten; andere Skalen (geometrisch, balanciert) sind nicht gemessen. Stetige Urteile dienen nur als Gegenprobe. | – |
| **Gewichte aus einer einzigen Matrix** | Die Hierarchie und die Synthese über die Alternativen (paarweise je Kriterium) sind nicht enthalten; die Rangumkehr durch eine hinzugefügte Alternative steht dort. | Rangumkehr |
| **Ein Punkt statt eines Bereichs** | Eine Matrix liefert genau einen Gewichtsvektor; die Unsicherheit der Urteile wird nicht weitergegeben. | Unsichere Gewichte |
| **Eine Normierung (Min-Max)** | Die Wahl der Normierung ist im zweiten Stück gemessen; hier ist sie fest. | Gewichtete Summe |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Paarvergleichsmatrix.** Eine positive reziproke Matrix $A = (a_{ij})$ mit $a_{ii} = 1$ und $a_{ji} = 1/a_{ij}$; sie ist **konsistent**, wenn $a_{ik} = a_{ij}\, a_{jk}$ für
alle $i, j, k$, also $a_{ij} = z_i / z_j$ für einen Gewichtsvektor $z$.

**Eigenvektor.** $A v = \lambda_{\max} v$ mit $v > 0$, $\sum_i v_i = 1$ (Perron-Frobenius: $\lambda_{\max}$ ist einfach und reell). Es gilt $\lambda_{\max} \ge m$, mit Gleichheit
genau bei Konsistenz.

**Geometrisches Mittel.** $g_i = \left(\prod_j a_{ij}\right)^{1/m}$, normiert auf Summe 1. Das ist die Lösung von $\min_v \sum_{i<j} (\ln a_{ij} - (v_i - v_j))^2$ mit
$\sum v_i = 0$ und Gewichten $e^{v}$ (Kleinste Quadrate im Logarithmus). Bei $m = 3$ ist es der Eigenvektor.

**Verstärkung.** Die elementweise Potenz $A^{(\gamma)} = (a_{ij}^\gamma)$ ist wieder reziprok. Für das geometrische Mittel gilt $g(A^{(\gamma)}) \propto g(A)^\gamma$ und damit
dieselbe Ordnung; für den Eigenvektor gibt es kein solches Gesetz.

**Konsistenzverhältnis.** $\mathrm{CI} = (\lambda_{\max} - m)/(m - 1)$, $\mathrm{CR} = \mathrm{CI} / \mathrm{RI}_m$ mit dem Random Index $\mathrm{RI}_m$ als Mittelwert des CI
über Zufallsmatrizen (hier: Einträge oberhalb der Diagonalen unabhängig und gleichverteilt aus $\{1/9, \dots, 1, \dots, 9\}$). Die Schwelle $\mathrm{CR} \le 0{,}1$ ist
eine Konvention.

**Urteilsmodell.** Das Urteil über das Paar $(i, j)$ ist $a_{ij} = \mathrm{R}\!\left(\dfrac{z_i}{z_j}\, e^{\sigma \varepsilon_{ij}}\right)$ mit $\varepsilon_{ij} \sim N(0,1)$ und der Rundung
$\mathrm{R}$ auf die Saaty-Skala (kaufmännisch, abgeschnitten bei 9 und $1/9$). Die erfragten Gewichte sind $z \propto w^\beta$ (Verdichtung) oder $w_k \Delta_k$ (Schwenk).

**Abstand.** $\mathrm{TV}(\hat w, w) = \tfrac12 \sum_k |\hat w_k - w_k|$.

**Literatur** (nur aus Zusammenfassungen, nicht im Original geprüft): T. L. Saaty, *The Analytic Hierarchy Process*, McGraw-Hill, 1980; G. Crawford, C. Williams,
*The Analysis of Subjective Judgment Matrices*, RAND R-2572-1, 1985 (geometrisches Mittel als Alternative zum Eigenvektor); C. A. Bana e Costa, J.-C. Vansnick, *A critical
analysis of the eigenvalue method used to derive priorities in AHP*, European Journal of Operational Research 187(3), 2008, S. 1422–1428 (Ordnungserhaltung);
G. W. Fischer, *Range sensitivity of attribute weights in multiattribute value models*, Organizational Behavior and Human Decision Processes 62(3), 1995, S. 252–266
(direkte Wichtigkeit bereichsunempfindlich).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Mehrkriterien-Ranking: von Pareto zum Vergleich](https://sebastianhanisch.net/konzepte-mehrkriterien-ranking.html)."
)
