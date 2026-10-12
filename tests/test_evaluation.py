"""Auswertung: Fall der App (Invarianten), Urteilsrauschen unabhängig von Frage und Fehlerquellen, Experimente im Kleinen, Verdikte."""

import numpy as np
import pytest

import ahp_constants as C
import ahp_evaluation as ev
import ahp_matrix as mx
import ahp_weights as wt


def test_make_case_is_deterministic_and_uses_the_generators_weights():
    a, b = ev.make_case(20, 5, 7), ev.make_case(20, 5, 7)
    assert (a.S == b.S).all() and (a.w == b.w).all()
    assert a.w.sum() == pytest.approx(1.0) and a.utility == pytest.approx(a.f @ a.w)
    assert (a.value_range > 0).all()


def test_reference_weights_for_both_questions():
    case = ev.make_case(20, 5, 7, 1.0, "roh")
    assert (ev.reference_weights(case, "wichtigkeit") == case.w).all()
    s = case.w * case.value_range
    assert ev.reference_weights(case, "schwenk") == pytest.approx(s / s.sum())
    with pytest.raises(ValueError):
        ev.reference_weights(case, "bedeutung")


def test_pair_noise_depends_only_on_seed_and_m():
    assert (ev.pair_noise(5, 3) == ev.pair_noise(5, 3)).all()
    assert len(ev.pair_noise(5, 3)) == 10 and len(ev.pair_noise(8, 3)) == 28
    assert not np.allclose(ev.pair_noise(5, 3), ev.pair_noise(5, 4))


def test_noise_is_shared_between_questions_scales_and_fault_sources():
    base = ev.Settings(sigma=0.4, jscale="stetig", seed=11)
    for other in (ev.Settings(sigma=0.4, jscale="stetig", seed=11, question="schwenk"), ev.Settings(sigma=0.4, jscale="stetig", seed=11, beta=0.5, gamma=2.0)):
        a, b = ev.analyse(base), ev.analyse(other)
        ratio_a = np.log(a.A_base) - np.log(mx.consistent_matrix(a.z))
        ratio_b = np.log(b.A_base) - np.log(mx.consistent_matrix(b.z))
        assert ratio_a == pytest.approx(ratio_b)                          # derselbe Rauschterm sigma * eps in jedem Eintrag


def test_analyse_basic_invariants():
    a = ev.analyse(ev.Settings())
    assert a.ev.sum() == pytest.approx(1.0) and a.gm.sum() == pytest.approx(1.0)
    assert a.A.shape == (5, 5) and np.allclose(a.A * a.A.T, 1.0)
    assert a.ci == pytest.approx((a.lmax - 5) / 4) and a.cr == pytest.approx(a.ci / a.ri)
    assert a.ri == pytest.approx(ev.ri_for(5)[0])
    assert set(a.results) == {"wahr", "gleich", "ev", "gm"}
    assert a.results["wahr"].weights.tolist() == a.case.w.tolist() and a.best == int(np.argmax(a.case.utility))
    assert a.results["gleich"].weights == pytest.approx(np.full(5, 0.2))
    for r in a.results.values():
        assert r.hit == (r.top1 == a.best) and r.loss >= 0 and -1 <= r.tau <= 1 and 0 <= r.tv <= 1
    assert not a.reversal_ev and not a.reversal_gm                         # gamma = 1: keine Verstärkung, keine Umkehr


def test_geometric_mean_never_reverses_under_amplification_and_gamma_one_never_reverses():
    for seed in range(40):
        for gamma in (0.5, 2.0, 3.0):
            assert not ev.analyse(ev.Settings(sigma=0.4, gamma=gamma, seed=seed)).reversal_gm


def test_amplification_can_reverse_the_eigenvector_in_the_app_case():
    assert any(ev.analyse(ev.Settings(sigma=0.4, gamma=2.0, seed=s)).reversal_ev for s in range(100))


def test_matrix_reciprocal_and_clipping_on_the_saaty_scale():
    a = ev.analyse(ev.Settings(sigma=0.8, seed=5))
    assert np.allclose(a.A_base * a.A_base.T, 1.0)
    assert set(np.round(a.A_base[np.triu_indices(5, 1)], 10)) <= set(np.round(C.SAATY_VALUES, 10))
    assert a.clipped.shape == (10,) and (np.abs(np.log(a.A_base[np.triu_indices(5, 1)])) <= np.log(9) + 1e-9).all()


def test_beta_one_leaves_the_reference_and_beta_half_compresses_it():
    a = ev.analyse(ev.Settings(sigma=0.0, jscale="stetig", beta=0.5, seed=1))
    assert a.z == pytest.approx(a.reference ** 0.5 / (a.reference ** 0.5).sum())
    assert a.cr == pytest.approx(0.0, abs=1e-9)                            # konsistent, aber falsch
    assert a.results["ev"].tv > 0.05
    assert ev.analyse(ev.Settings(beta=1.0, seed=1)).z == pytest.approx(ev.analyse(ev.Settings(seed=1)).reference)


def test_ri_for_is_memoised_and_grows_with_m():
    assert ev.ri_for(4) is ev.ri_for(4)
    vals = [ev.ri_for(m)[0] for m in range(3, 9)]
    assert vals == sorted(vals)


def test_counterexample_values():
    ce = ev.counterexample()
    assert ce["order_ev1"] == [3, 2, 4, 1] and ce["order_ev2"] == [2, 3, 4, 1]
    assert ce["order_gm1"] == [3, 2, 4, 1] and ce["order_gm2"] == [3, 2, 4, 1]


def test_experiment_ev_gm_small_run():
    rows = ev.experiment_ev_gm(ms=(3,), sigmas=(0.4,), seeds=tuple(range(200000, 200010)))
    assert len(rows) == 1 and rows[0]["rev"] == 0.0 and rows[0]["tv_ev_gm"] < 1e-9 and rows[0]["tv_ev"] == pytest.approx(rows[0]["tv_gm"])
    assert 0.0 <= rows[0]["hit_ev"] <= 1.0 and 0.0 <= rows[0]["cr_over"] <= 1.0


def test_experiments_are_deterministic():
    assert ev.experiment_intensify(gammas=(2.0,), draws=200) == ev.experiment_intensify(gammas=(2.0,), draws=200)
    assert ev.experiment_scale(ms=(3,), draws=100) == ev.experiment_scale(ms=(3,), draws=100)


def test_experiment_comparisons_designs_and_counts():
    rows = {r["design"]: r for r in ev.experiment_comparisons(draws=50)}
    assert [rows[k]["comparisons"] for k in ("star", "chain", "bwm", "random", "all")] == [4.0, 4.0, 7.0, 7.0, 10.0]
    assert rows["all"]["tv"] < rows["star"]["tv"]


def test_experiment_compression_has_perfect_consistency():
    rows = ev.experiment_compression(draws=100)
    assert all(abs(r["cr"]) < 1e-9 for r in rows) and rows[0]["tv"] < 1e-9 and rows[1]["tv"] < rows[2]["tv"]


def test_experiment_question_target_hits_every_time_in_the_raw_world():
    rows = ev.experiment_question(deltas=(0.05,), worlds=("roh",), seeds=tuple(range(200000, 200030)))
    ziel = next(r for r in rows if r["method"] == "ziel")
    assert ziel["hit"] == 1.0 and ziel["se"] == 0.0


def test_random_index_table_has_both_literature_columns():
    t = ev.random_index_table(range(3, 6))
    assert [r["m"] for r in t] == [3, 4, 5] and t[0]["table_a"] == 0.58 and t[0]["table_b"] == 0.52 and t[0]["sim"] == ev.ri_for(3)[0]


def test_verdicts_read_their_numbers_from_the_rows():
    rows = [{"m": 5, "sigma": s, "tv_ev": 0.06, "tv_gm": 0.061, "hit_ev": 0.9, "hit_gm": 0.9, "hit_wahr": 0.95, "rev": 0.1 * s, "tv_ev_gm": 0.0, "cr_over": 0.1}
            for s in (0.0, 0.8)] + [{"m": 3, "sigma": 0.8, "tv_ev": 0.1, "tv_gm": 0.1, "hit_ev": 0.8, "hit_gm": 0.8, "hit_wahr": 0.9, "rev": 0.0, "tv_ev_gm": 0.0, "cr_over": 0.1}]
    kind, text = ev.verdict_ev_gm(rows)
    assert kind == "info" and "0,001" in text and "identisch" in text and "5 Kriterien" in text
    kind, text = ev.verdict_intensify([{"gamma": 2.0, "ev": 0.054, "gm": 0.0, "draws": 5000}])
    assert kind == "warning" and "5,4 %" in text and "0,0 %" in text
    kind, text = ev.verdict_consistency([{"sigma": 0.2, "tv_ok": 0.05, "tv_bad": 0.1, "cr_mean": 0.04, "over": 0.08, "draws": 10},
                                         {"sigma": 0.8, "tv_ok": 0.117, "tv_bad": 0.121, "cr_mean": 0.16, "over": 0.7, "draws": 10}], [{"beta": 0.5, "cr": 0.0, "tv": 0.144}])
    assert kind == "warning" and "0,144" in text and "0,100 gegen 0,050" in text and "0,121 gegen 0,117" in text and "σ = 0,8" in text
    kind, text = ev.verdict_question([{"world": "roh", "delta": 0.05, "method": "wichtigkeit", "hit": 0.63}, {"world": "roh", "delta": 0.05, "method": "schwenk", "hit": 0.9}])
    assert kind == "warning" and "63 %" in text and "90 %" in text and "5 %" in text


def test_de_formats_with_decimal_comma():
    assert ev.de(0.1234, 3) == "0,123" and ev.de(2.0, 0) == "2"
