"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset zeigt seinen Effekt."""

import ahp_constants as C
import ahp_evaluation as ev
import ahp_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.M_MIN <= p["m"] <= C.M_MAX and 0 <= p["seed"] <= C.SEED_MAX, name
        assert p["sigma"] in C.SIGMA_CHOICES and p["jscale"] in C.JUDGMENT_SCALES and p["question"] in C.QUESTIONS and p["world"] in C.WORLDS, name
        assert p["delta"] in C.DELTA_CHOICES and p["gamma"] in C.GAMMA_CHOICES and p["beta"] in C.BETA_CHOICES and p["threshold"] in C.THRESHOLD_CHOICES, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Fast konsistent (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_url_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_labels_cover_every_choice():
    assert set(C.JUDGMENT_SCALE_LABELS) == set(C.JUDGMENT_SCALES)
    assert set(C.QUESTION_LABELS) == set(C.QUESTIONS)
    assert set(C.WORLD_LABELS) == set(C.WORLDS)


def test_sweep_seeds_are_the_documented_fixed_set():
    assert len(C.SWEEP_SEEDS) == 200 and len(set(C.SWEEP_SEEDS)) == 200 and C.SWEEP_SEEDS[0] == 200000


def test_defaults_are_choices_and_default_m_is_not_three():
    assert C.DEFAULT_SIGMA in C.SIGMA_CHOICES and C.DEFAULT_GAMMA in C.GAMMA_CHOICES and C.DEFAULT_BETA in C.BETA_CHOICES
    assert C.DEFAULT_M >= 4        # bei m = 3 sind Eigenvektor und geometrisches Mittel identisch: die Voreinstellung zeigt sonst nichts


def _analysis(name):
    return ev.analyse(ev.Settings(**C.PRESETS[name]))


def test_every_preset_shows_the_effect_it_is_named_for():
    """Die Presets sind auf einen Seed abgestimmt, bei dem der Effekt sichtbar ist; ändert sich Generator oder Auswertung, soll das hier auffallen."""
    a = _analysis("Fast konsistent (Voreinstellung)")
    assert a.cr <= a.settings.threshold and a.results["ev"].hit and a.results["gm"].hit and a.results["wahr"].hit and not a.results["gleich"].hit
    a = _analysis("Stark verrauscht")
    assert a.settings.sigma == 0.8 and a.cr > a.settings.threshold
    assert a.results["wahr"].hit and not a.results["ev"].hit and not a.results["gm"].hit
    calm = ev.analyse(ev.Settings(**dict(C.PRESETS["Stark verrauscht"], sigma=0.0)))
    assert calm.results["ev"].hit
    a = _analysis("Verstärkung kippt den Eigenvektor")
    assert a.reversal_ev and not a.reversal_gm
    pairs = [(i, j) for i in range(5) for j in range(5) if a.ev_base[i] - a.ev_base[j] > 0.01 and a.ev[j] - a.ev[i] > 0.01]     # eindeutig geordnet, eindeutig umgekehrt
    assert pairs, "keine klar umgekehrte Eigenvektor-Ordnung"
    assert all(a.gm_base[i] - a.gm_base[j] > 0.02 and a.gm[i] - a.gm[j] > 0.02 for i, j in pairs)        # beim geometrischen Mittel dieselbe klare Ordnung
    a = _analysis("Konsistent, aber falsch")
    assert abs(a.cr) < 1e-9 and a.results["ev"].tv > 0.1 and a.results["wahr"].hit and not a.results["ev"].hit
