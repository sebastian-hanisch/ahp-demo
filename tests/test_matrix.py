"""Matrix-Bausteine: Saaty-Rundung, reziproke Matrix, Urteile, Verstärkung, Verdichtung (je Einheit gegen Handwerte)."""

import numpy as np
import pytest

import ahp_constants as C
import ahp_matrix as mx


@pytest.mark.parametrize("r,expected", [
    (1.0, 1.0), (2.49, 2.0), (2.5, 3.0), (8.6, 9.0), (9.4, 9.0), (12.0, 9.0), (100.0, 9.0),
    (0.5, 0.5), (0.34, 1 / 3), (0.45, 0.5), (1 / 8.6, 1 / 9), (1 / 12, 1 / 9), (0.0001, 1 / 9),
    (1.4, 1.0), (1.5, 2.0), (0.7, 1.0), (0.66, 0.5),
])
def test_saaty_round_hand_values(r, expected):
    assert mx.saaty_round(r) == pytest.approx(expected, abs=1e-12)


def test_saaty_round_is_idempotent_and_keeps_scale_values():
    for v in C.SAATY_VALUES:
        assert mx.saaty_round(v) == pytest.approx(v)
    rng = np.random.default_rng(0)
    r = np.exp(rng.uniform(-4, 4, 500))
    once = mx.saaty_round(r)
    assert np.allclose(mx.saaty_round(once), once)
    assert set(np.round(once, 12)) <= set(np.round(C.SAATY_VALUES, 12))


def test_saaty_round_is_reciprocal_away_from_half_ties():
    r = np.array([0.31, 0.77, 1.3, 2.2, 4.7, 11.0])
    assert np.allclose(mx.saaty_round(1.0 / r), 1.0 / mx.saaty_round(r))


def test_clipped_mask_marks_ratios_that_would_round_above_nine():
    r = np.array([9.49, 9.5, 20.0, 1 / 9.49, 1 / 9.5, 1 / 20.0, 3.0, 1.0])
    assert mx.clipped_mask(r).tolist() == [False, True, True, False, True, True, False, False]


def test_upper_pairs_order_and_count():
    assert mx.upper_pairs(4) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert len(mx.upper_pairs(8)) == 28


def test_reciprocal_matrix_structure(A4):
    assert A4.shape == (4, 4) and np.allclose(np.diag(A4), 1.0)
    assert np.allclose(A4 * A4.T, 1.0)
    assert A4[0, 3] == 8.0 and A4[3, 0] == 0.125 and A4[2, 3] == 2.0 and A4[1, 2] == 3.0


def test_reciprocal_matrix_rejects_wrong_length():
    with pytest.raises(ValueError):
        mx.reciprocal_matrix([2, 3], 3)


def test_compress_weights_hand_values(w3):
    z = mx.compress_weights(w3, 0.5)
    assert z == pytest.approx([0.4727, 0.3343, 0.1930], abs=5e-5)
    assert z.sum() == pytest.approx(1.0)
    assert mx.compress_weights(w3, 1.0) == pytest.approx(w3)
    assert np.argsort(-z).tolist() == np.argsort(-w3).tolist()


def test_consistent_matrix_satisfies_transitivity(w3):
    A = mx.consistent_matrix(w3)
    assert A[0, 1] == pytest.approx(2.0) and A[0, 2] == pytest.approx(6.0) and A[1, 2] == pytest.approx(3.0)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                assert A[i, k] == pytest.approx(A[i, j] * A[j, k])


def test_judgment_matrix_without_noise_and_continuous_is_consistent(w3):
    A, raw, clipped = mx.judgment_matrix(w3, 0.0, np.zeros(3), "stetig")
    assert np.allclose(A, mx.consistent_matrix(w3))
    assert raw == pytest.approx([2.0, 6.0, 3.0]) and not clipped.any()


def test_judgment_matrix_saaty_rounds_and_clips():
    A, raw, clipped = mx.judgment_matrix([0.8, 0.15, 0.05], 0.0, np.zeros(3), "saaty")   # Verhältnisse 5,33; 16; 3
    assert raw == pytest.approx([0.8 / 0.15, 16.0, 3.0])
    assert A[0, 1] == 5.0 and A[0, 2] == 9.0 and A[1, 2] == 3.0 and A[2, 0] == pytest.approx(1 / 9)
    assert clipped.tolist() == [False, True, False]


def test_judgment_matrix_noise_enters_through_the_logarithm():
    z = np.array([0.5, 0.3, 0.2])
    eps = np.array([1.0, -1.0, 0.5])
    _, raw, _ = mx.judgment_matrix(z, 0.4, eps, "stetig")
    assert np.log(raw) == pytest.approx(np.log([0.5 / 0.3, 0.5 / 0.2, 0.3 / 0.2]) + 0.4 * eps)


def test_judgment_matrix_rejects_unknown_scale(w3):
    with pytest.raises(ValueError):
        mx.judgment_matrix(w3, 0.1, np.zeros(3), "geometrisch")


def test_power_transform_keeps_reciprocity_and_identity(A4):
    B = mx.power_transform(A4, 2.0)
    assert np.allclose(B * B.T, 1.0) and B[0, 1] == pytest.approx(4.0) and B[0, 3] == pytest.approx(64.0)
    assert np.allclose(mx.power_transform(A4, 1.0), A4)


def test_format_entry():
    assert mx.format_entry(3.0, "saaty") == "3"
    assert mx.format_entry(1 / 3, "saaty") == "1/3"
    assert mx.format_entry(1.0, "saaty") == "1"
    assert mx.format_entry(2.5, "stetig") == "2,50"
    assert mx.format_entry(0.4, "stetig") == "0,40"
    assert mx.format_entry(0.4, "saaty") == "0,40"        # nach Verstärkung kein Skalenwert mehr
