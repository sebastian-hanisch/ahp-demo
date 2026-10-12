"""Jede Zahl aus README.md und App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Experiment 1 und 4 laufen über SWEEP_SEEDS (200 feste Matrizen), die übrigen über feste Ziehungen (2 000 bis 5 000, deterministischer SplitMix64-Strom). Prozentwerte mit
Toleranz 0,5 Punkte, Gewichtsabstände mit 0,003 bis 0,005, damit Rundung und Bibliotheksversionen (CI installiert ungepinnt) nichts kippen. Der Standardfehler einer
Trefferquote über 200 Matrizen ist 1,3 bis 3,5 Punkte: Unterschiede unter etwa 6 Punkten sind NICHT belastbar und stehen nirgends als Aussage.
"""

import pytest

import ahp_constants as C
import ahp_evaluation as ev


@pytest.fixture(scope="module")
def ev_gm():
    return ev.experiment_ev_gm()


@pytest.fixture(scope="module")
def comparisons():
    return ev.experiment_comparisons()


@pytest.fixture(scope="module")
def intensify():
    return ev.experiment_intensify()


@pytest.fixture(scope="module")
def consistency():
    return ev.experiment_consistency()


@pytest.fixture(scope="module")
def compression():
    return ev.experiment_compression()


@pytest.fixture(scope="module")
def question():
    return ev.experiment_question()


@pytest.fixture(scope="module")
def scale():
    return ev.experiment_scale()


def cell(rows, **keys):
    hits = [r for r in rows if all(r[k] == v for k, v in keys.items())]
    assert len(hits) == 1, (keys, len(hits))
    return hits[0]


def pct(x):
    return 100 * x


# ----------------------------------------------------------------------------- Experiment 1: Eigenvektor gegen geometrisches Mittel

@pytest.mark.parametrize("m,sigma,tv_ev,tv_gm", [
    (3, 0.0, 0.042, 0.042), (3, 0.2, 0.051, 0.051), (3, 0.4, 0.071, 0.071), (3, 0.8, 0.112, 0.112),
    (5, 0.0, 0.051, 0.053), (5, 0.2, 0.060, 0.061), (5, 0.4, 0.075, 0.075), (5, 0.8, 0.123, 0.121),
    (7, 0.0, 0.050, 0.051), (7, 0.2, 0.056, 0.057), (7, 0.4, 0.072, 0.073), (7, 0.8, 0.113, 0.110),
])
def test_distance_to_the_true_weights(ev_gm, m, sigma, tv_ev, tv_gm):
    r = cell(ev_gm, m=m, sigma=sigma)
    assert r["tv_ev"] == pytest.approx(tv_ev, abs=0.003) and r["tv_gm"] == pytest.approx(tv_gm, abs=0.003)


@pytest.mark.parametrize("m,sigma,hit_ev,hit_gm,hit_wahr", [
    (3, 0.2, 86.0, 86.0, 94.0), (3, 0.8, 75.5, 75.5, 94.0),
    (5, 0.0, 93.5, 92.5, 96.5), (5, 0.2, 89.5, 90.0, 96.5), (5, 0.4, 90.0, 90.5, 96.5), (5, 0.8, 83.5, 84.5, 96.5),
    (7, 0.2, 89.5, 89.5, 93.5), (7, 0.8, 82.5, 83.5, 93.5),
])
def test_first_place_for_both_derivations(ev_gm, m, sigma, hit_ev, hit_gm, hit_wahr):
    r = cell(ev_gm, m=m, sigma=sigma)
    assert pct(r["hit_ev"]) == pytest.approx(hit_ev, abs=0.5) and pct(r["hit_gm"]) == pytest.approx(hit_gm, abs=0.5) and pct(r["hit_wahr"]) == pytest.approx(hit_wahr, abs=0.5)


def test_eigenvector_and_geometric_mean_are_level_in_every_cell(ev_gm):
    """Der Abstand zu den wahren Gewichten und die Trefferquote unterscheiden sich in keiner Zelle um mehr als 0,003 bzw. 1,5 Punkte."""
    for r in ev_gm:
        assert abs(r["tv_ev"] - r["tv_gm"]) <= 0.003 and abs(pct(r["hit_ev"] - r["hit_gm"])) <= 1.5 + 1e-9, (r["m"], r["sigma"])


@pytest.mark.parametrize("m,sigma,share", [
    (3, 0.0, 0.0), (3, 0.2, 0.0), (3, 0.4, 0.0), (3, 0.8, 0.0),
    (5, 0.0, 0.0), (5, 0.2, 0.5), (5, 0.4, 2.5), (5, 0.8, 15.0),
    (7, 0.0, 0.0), (7, 0.2, 2.5), (7, 0.4, 13.0), (7, 0.8, 35.0),
])
def test_share_of_matrices_where_the_two_derivations_order_a_pair_oppositely(ev_gm, m, sigma, share):
    assert pct(cell(ev_gm, m=m, sigma=sigma)["rev"]) == pytest.approx(share, abs=0.5)


def test_three_criteria_are_identical(ev_gm):
    for r in ev_gm:
        if r["m"] == 3:
            assert r["rev"] == 0.0 and r["tv_ev_gm"] < 1e-9


@pytest.mark.parametrize("m,sigma,share", [(5, 0.0, 10.5), (5, 0.2, 12.5), (5, 0.4, 18.5), (5, 0.8, 74.5), (3, 0.2, 14.0), (7, 0.2, 7.0)])
def test_share_of_matrices_with_a_consistency_ratio_over_the_threshold(ev_gm, m, sigma, share):
    assert pct(cell(ev_gm, m=m, sigma=sigma)["cr_over"]) == pytest.approx(share, abs=0.5)


@pytest.mark.parametrize("design,tv,comparisons_", [("star", 0.095, 4), ("chain", 0.108, 4), ("bwm", 0.066, 7), ("random", 0.070, 7), ("all", 0.057, 10)])
def test_fewer_comparisons(comparisons, design, tv, comparisons_):
    r = cell(comparisons, design=design)
    assert r["tv"] == pytest.approx(tv, abs=0.003) and r["comparisons"] == comparisons_


# ----------------------------------------------------------------------------- Experiment 2: Verstärkung

@pytest.mark.parametrize("gamma,ev_share", [(0.5, 2.6), (1.5, 2.7), (2.0, 5.4), (3.0, 7.9)])
def test_amplification_reverses_the_eigenvector_but_never_the_geometric_mean(intensify, gamma, ev_share):
    r = cell(intensify, gamma=gamma)
    assert pct(r["ev"]) == pytest.approx(ev_share, abs=0.4) and r["gm"] == 0.0


def test_counterexample_orders():
    ce = ev.counterexample()
    assert ce["order_ev1"] == [3, 2, 4, 1] and ce["order_ev2"] == [2, 3, 4, 1] and ce["order_gm1"] == ce["order_gm2"] == [3, 2, 4, 1]
    assert ce["ev1"].tolist() == pytest.approx([0.0866, 0.3846, 0.4013, 0.1275], abs=5e-5)
    assert ce["ev2"].tolist() == pytest.approx([0.0203, 0.4717, 0.4625, 0.0455], abs=5e-5)


# ----------------------------------------------------------------------------- Experiment 3: Konsistenz ist keine Richtigkeit

@pytest.mark.parametrize("sigma,cr_mean,over,tv_ok,tv_bad", [
    (0.0, 0.030, 7.2, 0.041, 0.107), (0.2, 0.038, 8.5, 0.050, 0.103), (0.4, 0.062, 15.7, 0.071, 0.094), (0.8, 0.158, 70.8, 0.117, 0.121),
])
def test_consistency_ratio_and_distance(consistency, sigma, cr_mean, over, tv_ok, tv_bad):
    r = cell(consistency, sigma=sigma)
    assert r["cr_mean"] == pytest.approx(cr_mean, abs=0.005) and pct(r["over"]) == pytest.approx(over, abs=1.0)
    assert r["tv_ok"] == pytest.approx(tv_ok, abs=0.004) and r["tv_bad"] == pytest.approx(tv_bad, abs=0.006)


def test_consistency_ratio_separates_weakly_at_large_noise(consistency):
    r = cell(consistency, sigma=0.8)
    assert abs(r["tv_bad"] - r["tv_ok"]) < 0.01


@pytest.mark.parametrize("beta,tv", [(1.0, 0.0), (0.5, 0.144), (0.25, 0.231)])
def test_consistent_but_wrong(compression, beta, tv):
    r = cell(compression, beta=beta)
    assert r["cr"] == pytest.approx(0.0, abs=1e-9) and r["tv"] == pytest.approx(tv, abs=0.004)


@pytest.mark.parametrize("m,sim", [(3, 0.529), (4, 0.888), (5, 1.107), (6, 1.248), (7, 1.343), (8, 1.405), (9, 1.451)])
def test_simulated_random_index(m, sim):
    ri, se = ev.ri_for(m)
    assert ri == pytest.approx(sim, abs=0.006) and se < 0.006


def test_no_literature_table_is_reproduced_in_every_row():
    """Keine der beiden Tabellen stimmt in allen Zeilen innerhalb von zwei Standardfehlern mit der Simulation überein."""
    rows = ev.random_index_table()
    for key in ("table_a", "table_b"):
        assert any(abs(r["sim"] - r[key]) > 2 * r["se"] for r in rows), key
    first = cell(rows, m=3)
    assert abs(first["sim"] - first["table_a"]) > 8 * first["se"]


# ----------------------------------------------------------------------------- Experiment 4: Frage und Skala

@pytest.mark.parametrize("world,delta,wahr,wicht,schwenk,ziel", [
    ("roh", 1.0, 96.5, 89.5, 92.0, 100.0), ("roh", 0.5, 86.5, 81.0, 91.0, 100.0), ("roh", 0.2, 76.0, 72.5, 93.0, 100.0), ("roh", 0.05, 67.0, 63.0, 90.0, 100.0),
    ("guete", 1.0, 67.0, 65.0, 64.0, 68.0), ("guete", 0.5, 66.0, 63.5, 70.0, 73.5), ("guete", 0.2, 55.5, 53.5, 67.5, 68.0), ("guete", 0.05, 50.0, 48.5, 66.5, 69.0),
])
def test_the_question_decides(question, world, delta, wahr, wicht, schwenk, ziel):
    got = {k: pct(cell(question, world=world, delta=delta, method=k)["hit"]) for k in ("wahr", "wichtigkeit", "schwenk", "ziel")}
    assert got["wahr"] == pytest.approx(wahr, abs=0.5) and got["wichtigkeit"] == pytest.approx(wicht, abs=0.5)
    assert got["schwenk"] == pytest.approx(schwenk, abs=0.5) and got["ziel"] == pytest.approx(ziel, abs=0.5)


def test_narrow_range_flips_the_ranking_of_the_two_questions(question):
    imp = cell(question, world="roh", delta=0.05, method="wichtigkeit")
    sw = cell(question, world="roh", delta=0.05, method="schwenk")
    assert pct(sw["hit"] - imp["hit"]) > 20 and sw["hit"] - imp["hit"] > 4 * max(imp["se"], sw["se"])


@pytest.mark.parametrize("m,over,tv_saaty,tv_stetig", [(3, 20.3, 0.051, 0.029), (5, 20.3, 0.056, 0.029), (8, 20.0, 0.056, 0.024)])
def test_scale_one_to_nine(scale, m, over, tv_saaty, tv_stetig):
    r = cell(scale, m=m)
    assert pct(r["over_nine"]) == pytest.approx(over, abs=1.0)
    assert r["tv_saaty"] == pytest.approx(tv_saaty, abs=0.003) and r["tv_stetig"] == pytest.approx(tv_stetig, abs=0.003)


def test_the_scale_roughly_doubles_the_error(scale):
    for r in scale:
        assert 1.5 < r["tv_saaty"] / r["tv_stetig"] < 2.6


def test_experiment_sizes_are_the_documented_ones():
    assert len(C.SWEEP_SEEDS) == 200 and C.EXP_N == 20 and C.EXP_GAMMA_DRAWS == 5000 and C.RI_DRAWS == 20000
