"""Paarvergleichsmatrix: Saaty-Rundung, reziproke Matrix aus der oberen Dreiecksmatrix, Urteile mit Rauschen, Verstärkung und Verdichtung.

Das Urteilsmodell ist ANGENOMMEN, nicht am Menschen gemessen: Der Entscheider urteilt über Verhältnisse z_i / z_j der erfragten Gewichte, lognormal verrauscht
(Streuung sigma des Logarithmus), auf die Saaty-Skala gerundet und abgeschnitten. Nur numpy; Zufall nur über übergebene Normalverteilte.
"""

from __future__ import annotations

import numpy as np

import ahp_constants as C


def saaty_round(r) -> np.ndarray:
    """Verhältnis r > 0 auf die Saaty-Skala {1/9, ..., 1/2, 1, 2, ..., 9}: Verhältnisse unter 1 über den Kehrwert, kaufmännisch gerundet (0,5 nach oben),
    abgeschnitten bei 9 bzw. 1/9."""
    r = np.asarray(r, dtype=float)
    big = np.clip(np.floor(np.maximum(r, 1.0 / r) + 0.5), 1.0, float(C.SAATY_MAX))
    return np.where(r >= 1.0, big, 1.0 / big)


def clipped_mask(r) -> np.ndarray:
    """True, wo das Verhältnis ohne Abschneiden auf einen Wert über 9 (oder unter 1/9) gerundet worden wäre."""
    r = np.asarray(r, dtype=float)
    return np.floor(np.maximum(r, 1.0 / r) + 0.5) > C.SAATY_MAX


def upper_pairs(m: int) -> list[tuple[int, int]]:
    """Paare i < j in Zeilenreihenfolge: (0,1), (0,2), ..., (0,m-1), (1,2), ..."""
    return [(i, j) for i in range(m) for j in range(i + 1, m)]


def reciprocal_matrix(upper, m: int) -> np.ndarray:
    """Reziproke Matrix mit Einsen auf der Diagonalen aus den m(m-1)/2 Einträgen oberhalb der Diagonalen (Zeilenreihenfolge)."""
    upper = np.asarray(upper, dtype=float)
    pairs = upper_pairs(m)
    if len(upper) != len(pairs):
        raise ValueError(f"für m = {m} werden {len(pairs)} Einträge erwartet, nicht {len(upper)}")
    A = np.ones((m, m))
    for (i, j), a in zip(pairs, upper):
        A[i, j], A[j, i] = a, 1.0 / a
    return A


def compress_weights(w, beta: float) -> np.ndarray:
    """Verdichtung der erfragten Gewichte: z proportional w^beta, auf Summe 1. beta = 1: keine; beta < 1: die Urteile unterschätzen die Unterschiede."""
    z = np.asarray(w, dtype=float) ** beta
    return z / z.sum()


def consistent_matrix(z) -> np.ndarray:
    """Vollständig konsistente Matrix A_ij = z_i / z_j."""
    z = np.asarray(z, dtype=float)
    return z[:, None] / z[None, :]


def judgment_matrix(z, sigma: float, eps, scale: str = "saaty"):
    """Urteilsmatrix des Entscheiders über die Gewichte z. `eps` sind die m(m-1)/2 Normalverteilten der Paare (Zeilenreihenfolge); die Matrix hängt damit nur
    an (z, sigma, eps, scale). Gibt (Matrix, Rohverhältnisse oberhalb der Diagonalen, Maske der abgeschnittenen Paare) zurück."""
    z = np.asarray(z, dtype=float)
    m = len(z)
    pairs = upper_pairs(m)
    ii = np.array([p[0] for p in pairs])
    jj = np.array([p[1] for p in pairs])
    raw = z[ii] / z[jj] * np.exp(sigma * np.asarray(eps, dtype=float))
    if scale == "saaty":
        values = saaty_round(raw)
        clipped = clipped_mask(raw)
    elif scale == "stetig":
        values = raw
        clipped = np.zeros(len(raw), dtype=bool)
    else:
        raise ValueError(f"unbekannte Urteilsskala: {scale}")
    return reciprocal_matrix(values, m), raw, clipped


def power_transform(A, gamma: float) -> np.ndarray:
    """Elementweise Potenz: verstärkt (gamma > 1) oder schwächt (gamma < 1) jedes Urteil, die Richtung bleibt, die Matrix bleibt reziprok."""
    return np.asarray(A, dtype=float) ** gamma


def format_entry(a: float, scale: str) -> str:
    """Anzeige eines Eintrags: auf der Saaty-Skala 3 oder 1/3, stetig mit zwei Nachkommastellen und Dezimalkomma."""
    if scale == "saaty" and abs(a * round(1.0 / a) - 1.0) < 1e-9 and a < 1.0:
        return f"1/{round(1.0 / a)}"
    if scale == "saaty" and abs(a - round(a)) < 1e-9:
        return f"{round(a)}"
    return f"{a:.2f}".replace(".", ",")
