"""Score-Helfer: Min-Max, Summe, Schwenk-Gewichte, Kendalls tau-b, Verschmälerung, Wertskalen."""

import numpy as np
import pytest

import ahp_score as sc


def test_minmax_hand_values_and_constant_column():
    S = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    assert sc.minmax(S).tolist() == [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]


def test_weighted_score_and_top1_tie_goes_to_the_first():
    N = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    s = sc.weighted_score(N, np.array([0.5, 0.5]))
    assert s.tolist() == [0.5, 0.5, 0.5] and sc.top1(s) == 0


def test_swing_weights_hand_values():
    w = sc.swing_weights(np.array([0.6, 0.3, 0.1]), np.array([0.8, 0.9, 0.5]))      # 0,48; 0,27; 0,05 -> Summe 0,80
    assert w == pytest.approx([0.6, 0.3375, 0.0625])


def test_kendall_tau_b_hand_values():
    assert sc.kendall_tau_b([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert sc.kendall_tau_b([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert sc.kendall_tau_b([1, 2, 3, 4], [1, 2, 4, 3]) == pytest.approx(4 / 6)
    assert sc.kendall_tau_b([1, 1, 1], [1, 2, 3]) == 0.0
    assert sc.kendall_tau_b([1, 1, 2], [1, 2, 3]) == pytest.approx(4 / np.sqrt(24))      # ein Gleichstand links: 2 konkordante, 1 gebundenes Paar


def test_narrow_criterion_shrinks_around_one_half_and_leaves_the_rest():
    g = np.array([[0.0, 0.3], [1.0, 0.7], [0.5, 0.9]])
    h = sc.narrow_criterion(g, 0.2)
    assert h[:, 0].tolist() == pytest.approx([0.4, 0.6, 0.5]) and (h[:, 1] == g[:, 1]).all()
    assert (sc.narrow_criterion(g, 1.0) == g).all()


def test_value_functions_on_both_scales():
    g = np.array([[0.2, 0.9], [0.8, 0.1]])
    S = np.array([[20.0, -1000.0], [80.0, -4000.0]])         # Kriterium 1 Punkte (↑ 0..100), Kriterium 2 Euro (↓ 800..5000, Vorzeichen)
    assert (sc.value_functions("guete", g, S, 2) == g).all()
    f = sc.value_functions("roh", g, S, 2)
    assert f[:, 0].tolist() == pytest.approx([0.2, 0.8])
    assert f[:, 1].tolist() == pytest.approx([(-1000 + 5000) / 4200, (-4000 + 5000) / 4200])
    with pytest.raises(ValueError):
        sc.value_functions("zehn", g, S, 2)


def test_utility_loss():
    assert sc.utility_loss(np.array([0.2, 0.9, 0.5]), 2) == pytest.approx(0.4)
    assert sc.utility_loss(np.array([0.2, 0.9, 0.5]), 1) == 0.0
