"""Gewichte aus einer Paarvergleichsmatrix: Eigenvektor (Perron), geometrisches Mittel, Konsistenz, Random Index, Log-Kleinste-Quadrate.

Nur numpy, jede Regel eine eigene Funktion; Zufall nur über einen übergebenen `SplitMix64`.
"""

from __future__ import annotations

import numpy as np

import ahp_constants as C
from ahp_matrix import upper_pairs


def perron_vector(A) -> tuple[np.ndarray, float]:
    """Rechter Eigenvektor zum größten Eigenwert (Perron-Wurzel) einer positiven Matrix, auf Summe 1, und lambda_max."""
    A = np.asarray(A, dtype=float)
    vals, vecs = np.linalg.eig(A)
    k = int(np.argmax(vals.real))
    v = vecs[:, k].real
    return v / v.sum(), float(vals[k].real)


def geometric_mean_weights(A) -> np.ndarray:
    """Zeilen-Geometrisches-Mittel, auf Summe 1: die Lösung der Kleinste-Quadrate-Aufgabe im Logarithmus (Crawford und Williams)."""
    g = np.exp(np.log(np.asarray(A, dtype=float)).mean(axis=1))
    return g / g.sum()


def consistency_index(lmax: float, m: int) -> float:
    """CI = (lambda_max - m) / (m - 1); 0 genau bei einer konsistenten Matrix, sonst positiv."""
    return (lmax - m) / (m - 1)


def consistency_ratio(ci: float, ri: float) -> float:
    return ci / ri


def random_index(m: int, rng, draws: int = C.RI_DRAWS) -> tuple[float, float]:
    """Random Index durch Simulation: mittleres CI reziproker Matrizen, deren Einträge oberhalb der Diagonalen unabhängig und gleichverteilt aus den 17 Werten
    der Saaty-Skala gezogen werden. Gibt (Mittelwert, Standardfehler) zurück. Der Wert hängt an dieser Erzeugungsvorschrift: andere Vorschriften geben andere Werte."""
    pairs = upper_pairs(m)
    scale = np.array(C.SAATY_VALUES)
    idx = np.minimum((rng.uniform(draws * len(pairs)) * len(scale)).astype(int), len(scale) - 1).reshape(draws, len(pairs))
    vals = scale[idx]
    A = np.ones((draws, m, m))
    ii = np.array([p[0] for p in pairs])
    jj = np.array([p[1] for p in pairs])
    A[:, ii, jj] = vals
    A[:, jj, ii] = 1.0 / vals
    lmax = np.linalg.eigvals(A).real.max(axis=1)
    ci = (lmax - m) / (m - 1)
    return float(ci.mean()), float(ci.std(ddof=1) / np.sqrt(draws))


def strict_reversal(v1, v2, eps: float = 1e-9) -> bool:
    """Gibt es ein Paar (i, j), das in v1 strikt geordnet ist (Abstand > eps) und in v2 strikt umgekehrt? Gleichstände zählen nicht als Umkehr."""
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    d1 = v1[:, None] - v1[None, :]
    d2 = v2[:, None] - v2[None, :]
    return bool(((d1 > eps) & (d2 < -eps)).any())


def is_connected(m: int, pairs) -> bool:
    """Ist der Graph der verglichenen Paare zusammenhängend?"""
    parent = list(range(m))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        parent[find(i)] = find(j)
    return len({find(x) for x in range(m)}) == 1


def log_least_squares(m: int, pairs, values) -> np.ndarray:
    """Gewichte aus einer unvollständigen Matrix: minimiere Summe (log a_ij - (v_i - v_j))^2 über die vorhandenen Paare mit Summe v = 0, Gewichte exp(v) normiert.
    Mit allen Paaren ist das das Zeilen-Geometrische-Mittel. Die Paare müssen den Graphen zusammenhängend machen."""
    pairs = list(pairs)
    if not is_connected(m, pairs):
        raise ValueError("die verglichenen Paare verbinden nicht alle Kriterien")
    rows = np.zeros((len(pairs) + 1, m))
    rhs = np.zeros(len(pairs) + 1)
    for r, ((i, j), a) in enumerate(zip(pairs, values)):
        rows[r, i], rows[r, j] = 1.0, -1.0
        rhs[r] = np.log(a)
    rows[-1, :] = 1.0
    v = np.linalg.lstsq(rows, rhs, rcond=None)[0]
    e = np.exp(v)
    return e / e.sum()


def tv_distance(a, b) -> float:
    """Halber L1-Abstand zweier Gewichtsvektoren (Totalvariation), zwischen 0 und 1."""
    return float(0.5 * np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).sum())
