"""Orakel (DEMO-PLAYBOOK §12): unabhängige Rechenwege für Eigenvektor, geometrisches Mittel, Potenzinvarianz, Random Index und unvollständige Matrizen.

Jede Eigenschaft hat eine Gegenprobe, die zeigt, dass der Test etwas trennen kann (EV = GM bei m = 3, nicht bei m = 4; Invarianz des geometrischen Mittels, nicht des
Eigenvektors; Simulation des Random Index, nicht die Tabelle 0,58)."""

import numpy as np
import pytest
import scipy.linalg

import ahp_matrix as mx
import ahp_weights as wt
from entscheidungsmatrix_generator import SplitMix64


def _random_reciprocal(m, rng, spread=2.0):
    return mx.reciprocal_matrix(np.exp(rng.uniform(-spread, spread, len(mx.upper_pairs(m)))), m)


def _power_iteration(A, steps=5000):
    v = np.ones(len(A)) / len(A)
    for _ in range(steps):
        w = A @ v
        w /= w.sum()
        if np.abs(w - v).max() < 1e-15:
            v = w
            break
        v = w
    return v, float((A @ v).sum() / v.sum())


# ----------------------------------------------------------------------------- 1. Eigenvektor

def test_perron_vector_equals_scipy_and_power_iteration_on_random_matrices():
    rng = np.random.default_rng(1)
    for m in (3, 4, 5, 6, 8):
        for _ in range(40):
            A = _random_reciprocal(m, rng)
            v, lmax = wt.perron_vector(A)
            vals, vecs = scipy.linalg.eig(A)
            k = int(np.argmax(vals.real))
            ref = vecs[:, k].real / vecs[:, k].real.sum()
            p, lp = _power_iteration(A)
            assert v == pytest.approx(ref, abs=1e-9) and lmax == pytest.approx(vals[k].real, abs=1e-9)
            assert v == pytest.approx(p, abs=1e-8) and lmax == pytest.approx(lp, abs=1e-8)
            assert (v > 0).all() and v.sum() == pytest.approx(1.0)


def test_lambda_max_is_at_least_m_and_equals_m_exactly_for_consistent_matrices():
    rng = np.random.default_rng(2)
    for _ in range(500):
        m = int(rng.integers(3, 9))
        assert wt.perron_vector(_random_reciprocal(m, rng))[1] >= m - 1e-9
        z = rng.dirichlet(np.ones(m))
        v, lmax = wt.perron_vector(mx.consistent_matrix(z))
        assert lmax == pytest.approx(m, abs=1e-9) and v == pytest.approx(z, abs=1e-9)


# ----------------------------------------------------------------------------- 2. Geometrisches Mittel

def test_geometric_mean_equals_definition_loop():
    rng = np.random.default_rng(3)
    for m in (3, 5, 7):
        A = _random_reciprocal(m, rng)
        g = np.array([np.prod([A[i, j] for j in range(m)]) ** (1.0 / m) for i in range(m)])
        assert wt.geometric_mean_weights(A) == pytest.approx(g / g.sum(), abs=1e-12)


def test_geometric_mean_minimises_the_log_least_squares_objective():
    rng = np.random.default_rng(4)
    m = 5
    A = _random_reciprocal(m, rng)
    pairs = mx.upper_pairs(m)

    def objective(z):
        v = np.log(z)
        return sum((np.log(A[i, j]) - (v[i] - v[j])) ** 2 for i, j in pairs)

    g = wt.geometric_mean_weights(A)
    base = objective(g)
    for _ in range(200):
        z = g * np.exp(rng.normal(0, 0.05, m))
        assert objective(z / z.sum()) >= base - 1e-12


# ----------------------------------------------------------------------------- 3. Potenzinvarianz

def test_geometric_mean_is_exactly_power_invariant():
    rng = np.random.default_rng(5)
    for _ in range(100):
        m = int(rng.integers(3, 9))
        A = _random_reciprocal(m, rng)
        g = wt.geometric_mean_weights(A)
        for gamma in (0.5, 1.5, 2.0, 3.0):
            p = g ** gamma
            assert wt.geometric_mean_weights(mx.power_transform(A, gamma)) == pytest.approx(p / p.sum(), abs=1e-12)


def test_eigenvector_is_not_power_invariant_and_random_search_finds_reversals():
    """Gegenprobe zur Invarianz: Auf Zufallsmatrizen (m = 5, Saaty-Skala) findet die Suche Fälle, in denen der Eigenvektor ein Paar umkehrt; beim geometrischen Mittel nie."""
    rng = np.random.default_rng(6)
    ev_rev = gm_rev = 0
    for _ in range(600):
        w = rng.dirichlet(np.ones(5))
        raw = np.array([w[i] / w[j] for i, j in mx.upper_pairs(5)]) * np.exp(0.4 * rng.standard_normal(10))
        A = mx.reciprocal_matrix(mx.saaty_round(raw), 5)
        B = mx.power_transform(A, 2.0)
        ev_rev += wt.strict_reversal(wt.perron_vector(A)[0], wt.perron_vector(B)[0])
        gm_rev += wt.strict_reversal(wt.geometric_mean_weights(A), wt.geometric_mean_weights(B))
    assert ev_rev >= 5 and gm_rev == 0


# ----------------------------------------------------------------------------- 4. EV = GM bei m = 3

def test_eigenvector_equals_geometric_mean_for_every_3x3_matrix():
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(5000):
        A = _random_reciprocal(3, rng, spread=np.log(9))
        worst = max(worst, float(np.abs(wt.perron_vector(A)[0] - wt.geometric_mean_weights(A)).max()))
    assert worst < 1e-12


def test_eigenvector_differs_from_geometric_mean_for_4x4_matrices():
    rng = np.random.default_rng(8)
    gaps = [wt.tv_distance(wt.perron_vector(A)[0], wt.geometric_mean_weights(A)) for A in (_random_reciprocal(4, rng, np.log(9)) for _ in range(200))]
    assert max(gaps) > 1e-3 and np.mean(gaps) > 1e-4


# ----------------------------------------------------------------------------- 5. Random Index

def _ri_reference(m, draws, seed):
    """Unabhängige Schleife mit numpy.random: Einträge gleichverteilt aus den 17 Skalenwerten, CI aus numpy.linalg.eig."""
    rng = np.random.default_rng(seed)
    scale = np.array([1 / k for k in range(9, 1, -1)] + [1.0] + [float(k) for k in range(2, 10)])
    cis = []
    for _ in range(draws):
        A = np.ones((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                a = scale[rng.integers(17)]
                A[i, j], A[j, i] = a, 1 / a
        cis.append((np.linalg.eigvals(A).real.max() - m) / (m - 1))
    return float(np.mean(cis)), float(np.std(cis, ddof=1) / np.sqrt(draws))


@pytest.mark.parametrize("m", [3, 5])
def test_random_index_simulation_agrees_with_an_independent_loop(m):
    ri, se = wt.random_index(m, SplitMix64(99), 20000)
    ref, ref_se = _ri_reference(m, 8000, 123)
    assert abs(ri - ref) < 3 * np.hypot(se, ref_se)


def test_simulated_random_index_at_m3_is_not_the_table_value_058():
    ri, se = wt.random_index(3, SplitMix64(99), 20000)
    assert abs(ri - 0.58) > 8 * se and abs(ri - 0.52) < 4 * se


# ----------------------------------------------------------------------------- 6. Unvollständige Matrizen

def test_incomplete_star_and_chain_equal_the_equation_system_solution():
    """Ein aufspannender Baum hat eine exakte Lösung: Stern z_j = z_0 / a_0j, Kette z_{i+1} = z_i / a_{i,i+1}; die Kleinste-Quadrate-Lösung muss sie treffen."""
    rng = np.random.default_rng(9)
    m = 6
    a = np.exp(rng.uniform(-1.5, 1.5, m - 1))
    star = wt.log_least_squares(m, [(0, j) for j in range(1, m)], list(a))
    ref = np.r_[1.0, 1.0 / a]
    assert star == pytest.approx(ref / ref.sum(), abs=1e-12)
    chain = wt.log_least_squares(m, [(i, i + 1) for i in range(m - 1)], list(a))
    ref = np.r_[1.0, np.cumprod(1.0 / a)]
    assert chain == pytest.approx(ref / ref.sum(), abs=1e-12)


def test_incomplete_with_extra_pairs_moves_towards_the_geometric_mean_of_the_full_matrix():
    rng = np.random.default_rng(10)
    m = 5
    A = _random_reciprocal(m, rng)
    full = wt.geometric_mean_weights(A)
    tree = [(0, j) for j in range(1, m)]
    assert wt.tv_distance(wt.log_least_squares(m, mx.upper_pairs(m), [A[i, j] for i, j in mx.upper_pairs(m)]), full) < 1e-12
    assert wt.tv_distance(wt.log_least_squares(m, tree, [A[i, j] for i, j in tree]), full) > 1e-3


# ----------------------------------------------------------------------------- 7. Verdichtung

def test_compression_is_consistent_and_the_distance_matches_the_formula():
    rng = np.random.default_rng(11)
    for _ in range(50):
        m = int(rng.integers(3, 9))
        w = rng.dirichlet(np.ones(m))
        for beta in (0.5, 0.25):
            z = w ** beta / (w ** beta).sum()
            v, lmax = wt.perron_vector(mx.consistent_matrix(mx.compress_weights(w, beta)))
            assert lmax == pytest.approx(m, abs=1e-9) and v == pytest.approx(z, abs=1e-9)
            assert wt.tv_distance(v, w) == pytest.approx(0.5 * np.abs(z - w).sum(), abs=1e-9)
