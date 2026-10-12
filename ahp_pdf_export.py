"""PDF-Export des aktuellen Falls (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import ahp_constants as C
from ahp_evaluation import Analysis, de

_REPLACE = {"↑": "+", "↓": "-", "→": "->", "–": "-", "—": "-", "τ": "tau", "σ": "sigma", "δ": "delta", "γ": "gamma", "β": "beta", "λ": "lambda", "≥": ">=",
            "≤": "<=", "−": "-", "€": "EUR", "·": "*", "²": "^2"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    inst = a.case.matrix
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("AHP: Paarvergleiche, Eigenvektor, Konsistenz"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Alternativen n = {s.n}, Kriterien m = {s.m}, Rauschen sigma = {de(s.sigma, 1)}, Urteilsskala: {C.JUDGMENT_SCALE_LABELS[s.jscale]}, "
                            f"Frage: {C.QUESTION_LABELS[s.question]}, Wertskala: {C.WORLD_LABELS[s.world]}, Streuung des ersten Kriteriums delta = {de(s.delta, 2)}, "
                            f"Verstaerkung gamma = {de(s.gamma, 1)}, Verdichtung beta = {de(s.beta, 2)}, Seed {s.seed}. Alles simuliert."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.multi_cell(0, 5, _s(f"lambda_max = {de(a.lmax, 3)}, CI = {de(a.ci, 3)}, Random Index (simuliert, m = {s.m}) = {de(a.ri, 3)}, CR = {de(a.cr, 3)} "
                            f"(Schwelle {de(s.threshold, 2)}: {'darunter' if a.cr <= s.threshold else 'darueber'})."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    head = ["Kriterium", "wahres Gewicht", "erfragt", "Eigenvektor", "geom. Mittel"]
    widths = [40, 36, 36, 36, 36]
    pdf.set_font("Helvetica", "B", 8)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for k in range(s.m):
        row = [f"K{k + 1} ({'+' if inst.directions[k] > 0 else '-'}) {inst.units[k]}", de(a.case.w[k], 3), de(a.reference[k], 3), de(a.ev[k], 3), de(a.gm[k], 3)]
        for v, w in zip(row, widths):
            pdf.cell(w, 5.5, _s(v), border=1)
        pdf.ln()
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, _s("Entscheidung (gewichtete Summe, Min-Max)"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    for k, res in a.results.items():
        label = {"wahr": "wahre Gewichte", "gleich": "gleiche Gewichte", "ev": "Eigenvektor", "gm": "geometrisches Mittel"}[k]
        pdf.cell(0, 5, _s(f"{label}: Platz 1 {'getroffen' if res.hit else 'nicht getroffen'}, tau {de(res.tau, 2)}, Verlust {de(res.loss, 3)}, "
                          f"Abstand zu den erfragten Gewichten {de(res.tv, 3)}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Das Urteilsmodell ist angenommen (lognormales Rauschen, Rundung auf die Skala), nicht am Menschen gemessen. Quelle: ahp-demo, Sebastian Hanisch."),
                   new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
