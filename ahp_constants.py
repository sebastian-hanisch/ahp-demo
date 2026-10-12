"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

N_MIN, N_MAX, DEFAULT_N = 5, 40, 20           # Score und Min-Max: O(n m); die Paarvergleichsmatrix hängt nur an m
M_MIN, M_MAX, DEFAULT_M = 3, 8, 5             # bei m = 3 sind Eigenvektor und geometrisches Mittel identisch: Voreinstellung 5
SEED_MAX, DEFAULT_SEED = 999999, 3

SIGMA_CHOICES = (0.0, 0.1, 0.2, 0.4, 0.8)     # Streuung des Logarithmus jedes Verhältnisurteils
DEFAULT_SIGMA = 0.2
DELTA_CHOICES = (1.0, 0.5, 0.2, 0.05)         # Anteil der Streuung des ersten Kriteriums (1 = unverändert)
DEFAULT_DELTA = 1.0
GAMMA_CHOICES = (1.0, 0.5, 1.5, 2.0, 3.0)     # elementweise Potenz der Urteile (1 = keine Verstärkung)
DEFAULT_GAMMA = 1.0
BETA_CHOICES = (1.0, 0.5, 0.25)               # Verdichtung der erfragten Gewichte: z proportional w^beta (1 = keine)
DEFAULT_BETA = 1.0
THRESHOLD_CHOICES = (0.05, 0.1, 0.15, 0.2)    # Schwelle für das Konsistenzverhältnis (0,1 ist eine Konvention, keine Naturkonstante)
DEFAULT_THRESHOLD = 0.1

JUDGMENT_SCALES = ("saaty", "stetig")
JUDGMENT_SCALE_LABELS = {
    "saaty": "Saaty-Skala (ganze Zahlen 1 bis 9 und Kehrwerte)",
    "stetig": "stetig (jedes Verhältnis, keine Rundung, kein Abschneiden)",
}
DEFAULT_JUDGMENT_SCALE = "saaty"

QUESTIONS = ("wichtigkeit", "schwenk")
QUESTION_LABELS = {
    "wichtigkeit": "Wie wichtig ist Kriterium i gegenüber j? (Wichtigkeit)",
    "schwenk": "Wie viel ist der Schwenk worst → best in i gegenüber j wert? (Schwenk in der Stichprobe)",
}
DEFAULT_QUESTION = "wichtigkeit"

WORLDS = ("guete", "roh")                      # Wertskala des simulierten Entscheiders (wie in Stück 3)
WORLD_LABELS = {
    "guete": "Linear in der Güte (Quantil innerhalb des Kriteriums)",
    "roh": "Linear in den Rohwerten (feste Nenn-Spannweite)",
}
DEFAULT_WORLD = "roh"

SAATY_MAX = 9                                  # Obergrenze der Urteilsskala; Untergrenze ist 1/9
SAATY_VALUES = tuple([1.0 / k for k in range(9, 1, -1)] + [1.0] + [float(k) for k in range(2, 10)])   # 17 Skalenwerte
CR_THRESHOLD = 0.1                             # Konvention der Experimente (der Regler THRESHOLD_CHOICES ändert nur die Anzeige)

# Random Index: Die Demo SIMULIERT ihn (gleichverteilte Skalenwerte, Eigenvektor-lambda_max). Die Tabellenwerte unten stammen aus
# Suchzusammenfassungen zweier Quellen (nicht im Original geprüft) und dienen nur dem Vergleich; sie stimmen untereinander nicht überein.
RI_DRAWS = 20000
RI_TABLE_A = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
RI_TABLE_B = {3: 0.52, 4: 0.89, 5: 1.11, 6: 1.25, 7: 1.35, 8: 1.45, 9: 1.49}

# Feste Instanzen für alle Experimente: Mittel über SWEEP_SEEDS, nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(200000, 200200))
EXP_N = 20                                     # feste Instanz mit 20 Alternativen, unabhängig vom Regler
EXP_M = (3, 5, 7)
EXP_SIGMAS = (0.0, 0.2, 0.4, 0.8)
EXP_M_MAIN = 5
EXP_GAMMAS = (0.5, 1.5, 2.0, 3.0)
EXP_GAMMA_SIGMA = 0.4
EXP_GAMMA_DRAWS = 5000                         # Matrizen je Zelle der Ordnungsumkehr-Messung (Standardfehler unter 0,5 Punkt)
EXP_CONSISTENCY_DRAWS = 5000
EXP_BETAS = (1.0, 0.5, 0.25)
EXP_BETA_DRAWS = 2000
EXP_QUESTION_SIGMA = 0.2
EXP_SCALE_M = (3, 5, 8)
EXP_SCALE_DRAWS = 3000
EXP_DESIGN_DRAWS = 2000

COLOR_TRUE = "#D68A2E"
COLOR_EV = "#2F6B65"
COLOR_GM = "#4c78a8"
COLOR_FIT = "#8A96A6"
COLOR_WARN = "#c0392b"
COLOR_LINE = "#4c78a8"

# Gegenbeispiel zur Ordnungsstabilität (4 x 4, obere Dreiecksmatrix A12, A13, A14, A23, A24, A34); Eigenvektor von A hat 3 > 2, von A^2 hat 2 > 3.
COUNTEREXAMPLE = (1 / 3, 1 / 3, 1 / 3, 1 / 2, 7.0, 3.0)

PRESETS = {
    "Fast konsistent (Voreinstellung)": {"n": DEFAULT_N, "m": DEFAULT_M, "sigma": DEFAULT_SIGMA, "jscale": DEFAULT_JUDGMENT_SCALE, "question": DEFAULT_QUESTION,
                                         "world": DEFAULT_WORLD, "delta": 1.0, "gamma": 1.0, "beta": 1.0, "threshold": DEFAULT_THRESHOLD, "seed": DEFAULT_SEED},
    "Stark verrauscht": {"n": 20, "m": 5, "sigma": 0.8, "jscale": "saaty", "question": "wichtigkeit", "world": "roh", "delta": 1.0, "gamma": 1.0, "beta": 1.0,
                         "threshold": 0.1, "seed": 5},
    "Verstärkung kippt den Eigenvektor": {"n": 20, "m": 5, "sigma": 0.4, "jscale": "saaty", "question": "wichtigkeit", "world": "roh", "delta": 1.0, "gamma": 2.0,
                                          "beta": 1.0, "threshold": 0.1, "seed": 371},
    "Konsistent, aber falsch": {"n": 20, "m": 5, "sigma": 0.0, "jscale": "stetig", "question": "wichtigkeit", "world": "roh", "delta": 1.0, "gamma": 1.0,
                                "beta": 0.5, "threshold": 0.1, "seed": 1},
}
PRESET_HELP = {
    "Fast konsistent (Voreinstellung)": "20 Alternativen, 5 Kriterien, mäßiges Rauschen, Urteile auf der Saaty-Skala, Frage nach der Wichtigkeit: Eigenvektor und geometrisches Mittel "
                                        "wählen beide den Besten, das Konsistenzverhältnis liegt unter der Schwelle.",
    "Stark verrauscht": "Das Urteilsrauschen ist groß (σ = 0,8): das Konsistenzverhältnis liegt über der Schwelle, und beide Verfahren verfehlen den Besten, obwohl die wahren "
                        "Gewichte ihn treffen.",
    "Verstärkung kippt den Eigenvektor": "Alle Urteile werden mit γ = 2 potenziert (die Richtung bleibt): Die Ordnung des Eigenvektors ändert sich, die des geometrischen Mittels nicht.",
    "Konsistent, aber falsch": "Die erfragten Gewichte sind verdichtet (β = 0,5): Die Matrix ist vollständig konsistent (CR = 0), die Gewichte liegen trotzdem weit von den "
                               "wahren, und der Beste wird verfehlt.",
}
