"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ahp_matrix as mx  # noqa: E402


@pytest.fixture
def A3():
    """Fast konsistente 3 x 3-Matrix: A12 = 3, A13 = 5, A23 = 2 (konsistent wäre A23 = 5/3).
    Geometrisches Mittel und Eigenvektor gleich (0,6483; 0,2297; 0,1220); lambda_max = 3,0037; CI = 0,0018; CR = 0,0036 (RI 0,52) bzw. 0,0032 (RI 0,58)."""
    return mx.reciprocal_matrix([3, 5, 2], 3)


@pytest.fixture
def A4():
    """Inkonsistente 4 x 4-Matrix: A12 = 2, A13 = 4, A14 = 8, A23 = 3, A24 = 4, A34 = 2.
    GM (0,5251; 0,2906; 0,1186; 0,0656); EV (0,5240; 0,2916; 0,1189; 0,0655); lambda_max = 4,0206; CI = 0,0069; CR = 0,0077 (RI 0,89); TV(EV, GM) = 0,0013."""
    return mx.reciprocal_matrix([2, 4, 8, 3, 4, 2], 4)


@pytest.fixture
def w3():
    """Gewichte der verdichteten Mini-Instanz: w = (0,6; 0,3; 0,1); z proportional w^0,5 = (0,4727; 0,3343; 0,1930); TV(z, w) = 0,1273; die Matrix aus z hat CR = 0."""
    return np.array([0.6, 0.3, 0.1])


@pytest.fixture
def counter():
    """Gegenbeispiel zur Ordnungsstabilität (4 x 4): EV(A) = (0,0866; 0,3846; 0,4013; 0,1275), Ordnung 3 > 2 > 4 > 1;
    EV(A^2) = (0,0203; 0,4717; 0,4625; 0,0455), Ordnung 2 > 3 > 4 > 1; GM(A) und GM(A^2) beide 3 > 2 > 4 > 1."""
    return mx.reciprocal_matrix([1 / 3, 1 / 3, 1 / 3, 1 / 2, 7.0, 3.0], 4)
