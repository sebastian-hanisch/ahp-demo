"""Gewichte: Eigenvektor, geometrisches Mittel, Konsistenz, Random Index, Umkehr, Log-Kleinste-Quadrate (Handwerte der Mini-Instanzen)."""

import numpy as np
import pytest

import ahp_matrix as mx
import ahp_weights as wt
from entscheidungsmatrix_generator import SplitMix64


def test_geometric_mean_hand_values(A3, A4):
    assert wt.geometric_mean_weights(A3) == pytest.approx([0.6483, 0.2297, 0.1220], abs=5e-5)
    assert wt.geometric_mean_weights(A4) == pytest.approx([0.5251, 0.2906, 0.1186, 0.0656], abs=5e-5)


def test_perron_vector_hand_values(A3, A4):
    v3, l3 = wt.perron_vector(A3)
    assert v3 == pytest.approx([0.6483, 0.2297, 0.1220], abs=5e-5) and l3 == pytest.approx(3.0037, abs=5e-5)
    v4, l4 = wt.perron_vector(A4)
    assert v4 == pytest.approx([0.5240, 0.2916, 0.1189, 0.0655], abs=5e-5) and l4 == pytest.approx(4.0206, abs=5e-5)
    assert v4.sum() == pytest.approx(1.0) and (v4 > 0).all()


def test_consistency_index_and_ratio_hand_values(A3, A4):
    _, l3 = wt.perron_vector(A3)
    _, l4 = wt.perron_vector(A4)
    assert wt.consistency_index(l3, 3) == pytest.approx(0.0018, abs=5e-5)
    assert wt.consistency_ratio(wt.consistency_index(l3, 3), 0.52) == pytest.approx(0.0036, abs=5e-5)
    assert wt.consistency_ratio(wt.consistency_index(l3, 3), 0.58) == pytest.approx(0.0032, abs=5e-5)
    assert wt.consistency_index(l4, 4) == pytest.approx(0.0069, abs=5e-5)
    assert wt.consistency_ratio(wt.consistency_index(l4, 4), 0.89) == pytest.approx(0.0077, abs=5e-5)


def test_consistency_index_is_zero_for_a_consistent_matrix(w3):
    _, lmax = wt.perron_vector(mx.consistent_matrix(w3))
    assert lmax == pytest.approx(3.0) and wt.consistency_index(lmax, 3) == pytest.approx(0.0, abs=1e-12)


def test_compressed_matrix_is_consistent_but_wrong(w3):
    z = mx.compress_weights(w3, 0.5)
    v, lmax = wt.perron_vector(mx.consistent_matrix(z))
    assert wt.consistency_ratio(wt.consistency_index(lmax, 3), 0.52) == pytest.approx(0.0, abs=1e-9)
    assert v == pytest.approx(z) and wt.tv_distance(v, w3) == pytest.approx(0.1273, abs=5e-5)


def test_tv_distance_between_ev_and_gm_on_the_four_by_four(A4):
    assert wt.tv_distance(wt.perron_vector(A4)[0], wt.geometric_mean_weights(A4)) == pytest.approx(0.0013, abs=5e-5)


def test_tv_distance_properties():
    a = np.array([0.5, 0.3, 0.2])
    b = np.array([0.2, 0.3, 0.5])
    assert wt.tv_distance(a, a) == 0.0 and wt.tv_distance(a, b) == pytest.approx(0.3)
    assert wt.tv_distance(a, b) == wt.tv_distance(b, a)


def test_counterexample_reverses_the_eigenvector_but_not_the_geometric_mean(counter):
    e1, e2 = wt.perron_vector(counter)[0], wt.perron_vector(mx.power_transform(counter, 2.0))[0]
    assert np.argsort(-e1).tolist() == [2, 1, 3, 0] and np.argsort(-e2).tolist() == [1, 2, 3, 0]
    assert wt.strict_reversal(e1, e2)
    g1, g2 = wt.geometric_mean_weights(counter), wt.geometric_mean_weights(mx.power_transform(counter, 2.0))
    assert np.argsort(-g1).tolist() == [2, 1, 3, 0] and np.argsort(-g2).tolist() == [2, 1, 3, 0]
    assert not wt.strict_reversal(g1, g2)
    assert e1 == pytest.approx([0.0866, 0.3846, 0.4013, 0.1275], abs=5e-5) and e2 == pytest.approx([0.0203, 0.4717, 0.4625, 0.0455], abs=5e-5)


def test_strict_reversal_ignores_ties_and_is_symmetric():
    assert not wt.strict_reversal([0.5, 0.5, 0.0], [0.5, 0.5, 0.0])
    assert not wt.strict_reversal([0.4, 0.4, 0.2], [0.5, 0.3, 0.2])       # Gleichstand aufgelöst: keine Umkehr
    assert not wt.strict_reversal([0.5, 0.3, 0.2], [0.4, 0.4, 0.2])
    assert wt.strict_reversal([0.5, 0.3, 0.2], [0.3, 0.5, 0.2])
    assert wt.strict_reversal([0.3, 0.5, 0.2], [0.5, 0.3, 0.2])
    assert not wt.strict_reversal([0.5, 0.3, 0.2], [0.6, 0.3, 0.1])
    assert not wt.strict_reversal([0.5, 0.5], [0.5 + 1e-12, 0.5 - 1e-12])  # unterhalb von eps


def test_is_connected():
    assert wt.is_connected(4, [(0, 1), (1, 2), (2, 3)])
    assert wt.is_connected(4, [(0, 1), (0, 2), (0, 3)])
    assert not wt.is_connected(4, [(0, 1), (2, 3)])
    assert not wt.is_connected(3, [(0, 1)])


def test_log_least_squares_rejects_a_disconnected_design():
    with pytest.raises(ValueError):
        wt.log_least_squares(4, [(0, 1), (2, 3)], [2.0, 2.0])


def test_log_least_squares_star_recovers_the_stars_ratios():
    # Stern um K1 mit den Urteilen 2, 4, 8: z1 : z2 : z3 : z4 = 1 : 1/2 : 1/4 : 1/8
    z = wt.log_least_squares(4, [(0, 1), (0, 2), (0, 3)], [2.0, 4.0, 8.0])
    assert z == pytest.approx(np.array([8, 4, 2, 1]) / 15.0)


def test_log_least_squares_with_all_pairs_equals_the_geometric_mean(A4):
    z = wt.log_least_squares(4, mx.upper_pairs(4), [A4[i, j] for i, j in mx.upper_pairs(4)])
    assert z == pytest.approx(wt.geometric_mean_weights(A4), abs=1e-12)


def test_random_index_simulation_has_the_documented_size_and_error():
    ri, se = wt.random_index(3, SplitMix64(1), 4000)
    assert 0.45 < ri < 0.6 and 0 < se < 0.02
    assert wt.random_index(3, SplitMix64(1), 4000) == (ri, se)           # deterministisch
    assert wt.random_index(5, SplitMix64(1), 2000)[0] > ri               # wächst mit m


class _UniformOne:
    """Zufallsquelle, die an der oberen Kante des Wertebereichs liegt (uniform() = 1,0 statt < 1)."""

    def uniform(self, n):
        return np.ones(n)


def test_random_index_clamps_the_scale_index_at_the_upper_edge():
    """Liefert die Zufallsquelle genau 1,0, muss der Index auf den letzten Skalenwert (9) begrenzt werden, nicht aus der Skala laufen.
    Alle Einträge oberhalb der Diagonalen sind dann 9: CI der 3x3-Matrix [[1,9,9],[1/9,1,9],[1/9,1/9,1]] ist gleich für alle Ziehungen (Standardfehler 0)."""
    ri, se = wt.random_index(3, _UniformOne(), 5)
    A = np.array([[1.0, 9.0, 9.0], [1 / 9, 1.0, 9.0], [1 / 9, 1 / 9, 1.0]])
    expected = (np.linalg.eigvals(A).real.max() - 3) / 2
    assert ri == pytest.approx(expected)
    assert se == pytest.approx(0.0, abs=1e-12)
