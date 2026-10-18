"""Auswertung: Fall der App, Experimente über feste Matrizen (SWEEP_SEEDS) und Verdikte. Nur numpy, kein Streamlit."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import ahp_constants as C
import ahp_matrix as mx
import ahp_score as sc
import ahp_weights as wt
from entscheidungsmatrix_generator import SplitMix64, make_matrix

JUDGMENT_OFFSET = 20011       # Strom der Urteile: SplitMix64(seed + JUDGMENT_OFFSET), unabhängig vom Strom der Matrix
RI_SEED = 7100                # Strom des simulierten Random Index: SplitMix64(RI_SEED + m)
DRAW_SEED = 31337             # Strom der Gewichts-Ziehungen in den Experimenten ohne Matrix


def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    m: int = C.DEFAULT_M
    sigma: float = C.DEFAULT_SIGMA
    jscale: str = C.DEFAULT_JUDGMENT_SCALE
    question: str = C.DEFAULT_QUESTION
    world: str = C.DEFAULT_WORLD
    delta: float = C.DEFAULT_DELTA
    gamma: float = C.DEFAULT_GAMMA
    beta: float = C.DEFAULT_BETA
    threshold: float = C.DEFAULT_THRESHOLD
    seed: int = C.DEFAULT_SEED


@lru_cache(maxsize=None)
def ri_for(m: int, draws: int = C.RI_DRAWS) -> tuple[float, float]:
    """Simulierter Random Index (Mittelwert, Standardfehler) für m Kriterien: fester Strom, je m berechnet und gemerkt."""
    return wt.random_index(m, SplitMix64(RI_SEED + m), draws)


def draw_weights(m: int, rng: SplitMix64) -> np.ndarray:
    """Gewichte gleichverteilt über dem Simplex (Dirichlet(1)): normierte Exponentialverteilte."""
    g = -np.log1p(-rng.uniform(m))
    return g / g.sum()


@dataclass
class Case:
    """Eine Matrix mit simuliertem Entscheider: Rohwerte nach Verschmälerung, Wert je Kriterium auf seiner Wertskala, wahre Gewichte, Nutzen."""
    matrix: object
    goodness: np.ndarray
    raw: np.ndarray
    S: np.ndarray
    f: np.ndarray
    w: np.ndarray
    utility: np.ndarray

    @property
    def value_range(self) -> np.ndarray:
        """Spannweite des Wertes je Kriterium in der Stichprobe (worst -> best)."""
        return self.f.max(axis=0) - self.f.min(axis=0)


def make_case(n: int, m: int, seed: int, delta: float = 1.0, world: str = "roh") -> Case:
    x = make_matrix(n, m, 0.0, SplitMix64(seed))
    g, raw, S, f = sc.build_case(x.goodness, x.directions, m, delta, world)
    return Case(x, g, raw, S, f, x.weights, f @ x.weights)


def pair_noise(m: int, seed: int) -> np.ndarray:
    """Die m(m-1)/2 Normalverteilten der Paarurteile: hängen nur an Seed und m, nicht an Frage, Skala, Verstärkung oder Verdichtung."""
    return SplitMix64(seed + JUDGMENT_OFFSET).normal(len(mx.upper_pairs(m)))


def reference_weights(case: Case, question: str) -> np.ndarray:
    """Die Gewichte, nach deren Verhältnissen gefragt wird: die Wichtigkeit w oder der Schwenk w_k mal Spannweite des Wertes."""
    if question == "wichtigkeit":
        return case.w
    if question == "schwenk":
        return sc.swing_weights(case.w, case.value_range)
    raise ValueError(f"unbekannte Frage: {question}")


def derive(A: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Eigenvektor, lambda_max und geometrisches Mittel einer Matrix."""
    v, lmax = wt.perron_vector(A)
    return v, lmax, wt.geometric_mean_weights(A)


@dataclass
class Result:
    weights: np.ndarray
    top1: int
    hit: bool
    tau: float
    loss: float
    tv: float


def _result(case: Case, N: np.ndarray, weights: np.ndarray, reference: np.ndarray, best: int) -> Result:
    score = sc.weighted_score(N, weights)
    t = sc.top1(score)
    return Result(weights, t, t == best, sc.kendall_tau_b(score, case.utility), sc.utility_loss(case.utility, t), wt.tv_distance(weights, reference))


@dataclass
class Analysis:
    settings: Settings
    case: Case
    reference: np.ndarray          # erfragte Gewichte vor der Verdichtung
    z: np.ndarray                  # erfragte Gewichte nach der Verdichtung (daraus urteilt der Entscheider)
    A_base: np.ndarray             # Urteilsmatrix ohne Verstärkung
    A: np.ndarray                  # Urteilsmatrix nach Verstärkung (gamma)
    raw_upper: np.ndarray
    clipped: np.ndarray            # Maske je Paar (Zeilenreihenfolge): Urteil wurde bei 9 abgeschnitten
    ev: np.ndarray
    gm: np.ndarray
    lmax: float
    ci: float
    ri: float
    ri_se: float
    cr: float
    best: int
    results: dict                  # "wahr", "gleich", "ev", "gm"
    ev_base: np.ndarray
    gm_base: np.ndarray
    reversal_ev: bool              # Eigenvektor von A gegen den von A^gamma: strikte Umkehr eines Paares
    reversal_gm: bool
    reversal_ev_gm: bool           # Eigenvektor gegen geometrisches Mittel derselben Matrix


def analyse(settings: Settings) -> Analysis:
    s = settings
    case = make_case(s.n, s.m, s.seed, s.delta, s.world)
    reference = reference_weights(case, s.question)
    z = mx.compress_weights(reference, s.beta)
    A_base, raw, clipped = mx.judgment_matrix(z, s.sigma, pair_noise(s.m, s.seed), s.jscale)
    A = mx.power_transform(A_base, s.gamma)
    ev, lmax, gm = derive(A)
    ev_base, _, gm_base = derive(A_base)
    ci = wt.consistency_index(lmax, s.m)
    ri, ri_se = ri_for(s.m)
    N = sc.minmax(case.S)
    best = sc.top1(case.utility)
    results = {
        "wahr": _result(case, N, case.w, reference, best),
        "gleich": _result(case, N, np.full(s.m, 1.0 / s.m), reference, best),
        "ev": _result(case, N, ev, reference, best),
        "gm": _result(case, N, gm, reference, best),
    }
    return Analysis(s, case, reference, z, A_base, A, raw, clipped, ev, gm, lmax, ci, ri, ri_se, wt.consistency_ratio(ci, ri), best, results, ev_base, gm_base,
                    wt.strict_reversal(ev_base, ev), wt.strict_reversal(gm_base, gm), wt.strict_reversal(ev, gm))


def counterexample() -> dict:
    """Das feste 4 x 4-Gegenbeispiel: Eigenvektor und geometrisches Mittel von A und von A^2."""
    A = mx.reciprocal_matrix(C.COUNTEREXAMPLE, 4)
    A2 = mx.power_transform(A, 2.0)
    ev1, _, gm1 = derive(A)
    ev2, _, gm2 = derive(A2)
    order = lambda v: [int(i) + 1 for i in np.argsort(-v, kind="stable")]  # noqa: E731
    return {"A": A, "ev1": ev1, "ev2": ev2, "gm1": gm1, "gm2": gm2, "order_ev1": order(ev1), "order_ev2": order(ev2), "order_gm1": order(gm1),
            "order_gm2": order(gm2)}


# ----------------------------------------------------------------------------- Experiment 1: Eigenvektor oder geometrisches Mittel?

def experiment_ev_gm(ms=C.EXP_M, sigmas=C.EXP_SIGMAS, n=C.EXP_N, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Je (m, sigma) über 200 feste Matrizen: Abstand zu w, Platz 1 (Welt Rohwerte, Saaty-Skala, Frage nach der Wichtigkeit), strikte Umkehr EV gegen GM."""
    rows = []
    for m in ms:
        ri, _ = ri_for(m)
        for sg in sigmas:
            acc = {k: [] for k in ("tv_ev", "tv_gm", "hit_ev", "hit_gm", "hit_wahr", "rev", "tv_ev_gm", "cr_over")}
            for s in seeds:
                case = make_case(n, m, s, 1.0, "roh")
                A, _, _ = mx.judgment_matrix(case.w, sg, pair_noise(m, s), "saaty")
                ev, lmax, gm = derive(A)
                N = sc.minmax(case.S)
                best = sc.top1(case.utility)
                acc["tv_ev"].append(wt.tv_distance(ev, case.w))
                acc["tv_gm"].append(wt.tv_distance(gm, case.w))
                acc["hit_ev"].append(sc.top1(sc.weighted_score(N, ev)) == best)
                acc["hit_gm"].append(sc.top1(sc.weighted_score(N, gm)) == best)
                acc["hit_wahr"].append(sc.top1(sc.weighted_score(N, case.w)) == best)
                acc["rev"].append(wt.strict_reversal(ev, gm))
                acc["tv_ev_gm"].append(wt.tv_distance(ev, gm))
                acc["cr_over"].append(wt.consistency_ratio(wt.consistency_index(lmax, m), ri) > C.CR_THRESHOLD)
            rows.append({"m": m, "sigma": sg, **{k: float(np.mean(v)) for k, v in acc.items()}})
    return rows


def experiment_comparisons(m=C.EXP_M_MAIN, sigma=C.EXP_QUESTION_SIGMA, draws=C.EXP_DESIGN_DRAWS) -> list[dict]:
    """Weniger Vergleiche: Log-Kleinste-Quadrate auf den vorhandenen Paaren (Saaty-Skala), Abstand zu w. Entwürfe: Stern um K1, Kette, Best-Worst-Stern
    (2m - 3 Paare), 2m - 3 zufällig gezogene Paare und alle Paare. Bei m = 5 lässt eine Auswahl von 7 aus 10 Paaren keinen Knoten isoliert."""
    pairs_all = mx.upper_pairs(m)
    npairs = len(pairs_all)
    rng = SplitMix64(DRAW_SEED + 4)
    k_random = 2 * m - 3
    designs = {"star": "Stern um K1", "chain": "Kette", "bwm": "Best-Worst-Stern", "random": "zufällig", "all": "alle Paare"}
    errs = {k: [] for k in designs}
    sizes = {k: [] for k in designs}
    for _ in range(draws):
        w = draw_weights(m, rng)
        eps = rng.normal(npairs)
        perm = rng.permutation(npairs)
        A, _, _ = mx.judgment_matrix(w, sigma, eps, "saaty")
        B, W = int(np.argmax(w)), int(np.argmin(w))
        chosen = {
            "star": [(0, j) for j in range(1, m)],
            "chain": [(i, i + 1) for i in range(m - 1)],
            "bwm": sorted({tuple(sorted((B, j))) for j in range(m) if j != B} | {tuple(sorted((j, W))) for j in range(m) if j != W}),
            "random": [pairs_all[t] for t in perm[:k_random]],
            "all": pairs_all,
        }
        for k, pr in chosen.items():
            sizes[k].append(len(pr))
            errs[k].append(wt.tv_distance(wt.log_least_squares(m, pr, [A[i, j] for i, j in pr]), w))
    return [{"design": k, "label": lab, "comparisons": float(np.mean(sizes[k])), "tv": float(np.mean(errs[k]))} for k, lab in designs.items()]


# ----------------------------------------------------------------------------- Experiment 2: Verstärkung der Urteile

def experiment_intensify(gammas=C.EXP_GAMMAS, m=C.EXP_M_MAIN, sigma=C.EXP_GAMMA_SIGMA, draws=C.EXP_GAMMA_DRAWS) -> list[dict]:
    """Anteil der Matrizen, bei denen die Verstärkung (elementweise Potenz gamma) ein Paar strikt umkehrt: Eigenvektor und geometrisches Mittel."""
    rng = SplitMix64(DRAW_SEED + 2)
    npairs = len(mx.upper_pairs(m))
    count = {g: [0, 0] for g in gammas}
    for _ in range(draws):
        w = draw_weights(m, rng)
        A, _, _ = mx.judgment_matrix(w, sigma, rng.normal(npairs), "saaty")
        v_ev, _, v_gm = derive(A)
        for g in gammas:
            e2, _, g2 = derive(mx.power_transform(A, g))
            count[g][0] += wt.strict_reversal(v_ev, e2)
            count[g][1] += wt.strict_reversal(v_gm, g2)
    return [{"gamma": g, "ev": count[g][0] / draws, "gm": count[g][1] / draws, "draws": draws} for g in gammas]


# ----------------------------------------------------------------------------- Experiment 3: Konsistenz ist keine Richtigkeit

def experiment_consistency(sigmas=C.EXP_SIGMAS, m=C.EXP_M_MAIN, draws=C.EXP_CONSISTENCY_DRAWS) -> list[dict]:
    """Je sigma: Anteil mit CR > 0,1 und Abstand des Eigenvektors zu w bei CR <= 0,1 gegen CR > 0,1 (Saaty-Skala, simulierter Random Index)."""
    ri, _ = ri_for(m)
    npairs = len(mx.upper_pairs(m))
    rows = []
    for sg in sigmas:
        rng = SplitMix64(DRAW_SEED + 3)
        crs, tvs = [], []
        for _ in range(draws):
            w = draw_weights(m, rng)
            A, _, _ = mx.judgment_matrix(w, sg, rng.normal(npairs), "saaty")
            v, lmax = wt.perron_vector(A)
            crs.append(wt.consistency_ratio(wt.consistency_index(lmax, m), ri))
            tvs.append(wt.tv_distance(v, w))
        crs, tvs = np.array(crs), np.array(tvs)
        ok = crs <= C.CR_THRESHOLD
        rows.append({"sigma": sg, "cr_mean": float(crs.mean()), "over": float((~ok).mean()), "tv_ok": float(tvs[ok].mean()) if ok.any() else float("nan"),
                     "tv_bad": float(tvs[~ok].mean()) if (~ok).any() else float("nan"), "draws": draws})
    return rows


def experiment_compression(betas=C.EXP_BETAS, m=C.EXP_M_MAIN, draws=C.EXP_BETA_DRAWS) -> list[dict]:
    """Konsistent, aber falsch: Gewichte z proportional w^beta ergeben eine vollständig konsistente Matrix (CR = 0); Abstand des Eigenvektors zu w."""
    ri, _ = ri_for(m)
    rng = SplitMix64(DRAW_SEED + 5)
    ws = [draw_weights(m, rng) for _ in range(draws)]
    rows = []
    for b in betas:
        crs, tvs = [], []
        for w in ws:
            v, lmax = wt.perron_vector(mx.consistent_matrix(mx.compress_weights(w, b)))
            crs.append(wt.consistency_ratio(max(wt.consistency_index(lmax, m), 0.0), ri))
            tvs.append(wt.tv_distance(v, w))
        rows.append({"beta": b, "cr": float(np.mean(crs)), "tv": float(np.mean(tvs))})
    return rows


# ----------------------------------------------------------------------------- Experiment 4: Die Frage entscheidet, und die Skala 1 bis 9

def experiment_question(deltas=C.DELTA_CHOICES, worlds=C.WORLDS, m=C.EXP_M_MAIN, n=C.EXP_N, sigma=C.EXP_QUESTION_SIGMA, seeds=C.SWEEP_SEEDS) -> list[dict]:
    """Platz 1 getroffen: wahre Gewichte, AHP (Eigenvektor) mit Wichtigkeitsurteilen, AHP mit Schwenkurteilen und die Schwenk-Gewichte selbst (ohne Rauschen)."""
    rows = []
    for world in worlds:
        for d in deltas:
            hits = {k: [] for k in ("wahr", "wichtigkeit", "schwenk", "ziel")}
            for s in seeds:
                case = make_case(n, m, s, d, world)
                best = sc.top1(case.utility)
                N = sc.minmax(case.S)
                eps = pair_noise(m, s)
                target = sc.swing_weights(case.w, case.value_range)
                v_imp = wt.perron_vector(mx.judgment_matrix(case.w, sigma, eps, "saaty")[0])[0]
                v_sw = wt.perron_vector(mx.judgment_matrix(target, sigma, eps, "saaty")[0])[0]
                for k, w in (("wahr", case.w), ("wichtigkeit", v_imp), ("schwenk", v_sw), ("ziel", target)):
                    hits[k].append(sc.top1(sc.weighted_score(N, w)) == best)
            for k, v in hits.items():
                p = float(np.mean(v))
                rows.append({"world": world, "delta": d, "method": k, "hit": p, "se": float(np.sqrt(p * (1 - p) / len(seeds)))})
    return rows


def experiment_scale(ms=C.EXP_SCALE_M, sigma=C.EXP_QUESTION_SIGMA, draws=C.EXP_SCALE_DRAWS) -> list[dict]:
    """Skala 1 bis 9 gegen stetige Urteile: Anteil der Paare mit wahrem Verhältnis über 9 und Abstand des Eigenvektors zu w (gleiche Rauschwerte)."""
    rows = []
    for m in ms:
        rng = SplitMix64(DRAW_SEED + 6 + m)
        npairs = len(mx.upper_pairs(m))
        share, t_saaty, t_cont = [], [], []
        for _ in range(draws):
            w = draw_weights(m, rng)
            eps = rng.normal(npairs)
            r = np.array([w[i] / w[j] for i, j in mx.upper_pairs(m)])
            share.append(float((np.maximum(r, 1 / r) > C.SAATY_MAX).mean()))
            t_saaty.append(wt.tv_distance(wt.perron_vector(mx.judgment_matrix(w, sigma, eps, "saaty")[0])[0], w))
            t_cont.append(wt.tv_distance(wt.perron_vector(mx.judgment_matrix(w, sigma, eps, "stetig")[0])[0], w))
        rows.append({"m": m, "over_nine": float(np.mean(share)), "tv_saaty": float(np.mean(t_saaty)), "tv_stetig": float(np.mean(t_cont))})
    return rows


def random_index_table(ms=range(3, 10)) -> list[dict]:
    """Die simulierte Tabelle neben den beiden Tabellen der Quellen."""
    return [{"m": m, "sim": ri_for(m)[0], "se": ri_for(m)[1], "table_a": C.RI_TABLE_A[m], "table_b": C.RI_TABLE_B[m]} for m in ms]


# ----------------------------------------------------------------------------- Verdikte (live aus den Zeilen)

def _row(rows, **keys):
    return next(r for r in rows if all(r[k] == v for k, v in keys.items()))


def verdict_ev_gm(rows: list[dict]) -> tuple[str, str]:
    main = [r for r in rows if r["m"] == C.EXP_M_MAIN]
    tv_gap = max(abs(r["tv_ev"] - r["tv_gm"]) for r in main)
    hit_gap = max(abs(r["hit_ev"] - r["hit_gm"]) for r in main)
    small = [r for r in rows if r["m"] == min(r_["m"] for r_ in rows)]
    top = max(rows, key=lambda r: (r["m"], r["sigma"]))
    return "info", (f"Eigenvektor und geometrisches Mittel liegen bei {C.EXP_M_MAIN} Kriterien gleichauf: Der Abstand zu den wahren Gewichten unterscheidet sich um höchstens "
                    f"{de(tv_gap, 3)}, die Trefferquote für Platz 1 um höchstens {de(100 * hit_gap, 1)} Punkte. Bei {top['m']} Kriterien und σ = {de(top['sigma'], 1)} "
                    f"ordnen sie in {de(100 * top['rev'], 0)} % der Matrizen mindestens ein Paar entgegengesetzt; bei {small[0]['m']} Kriterien sind sie identisch.")


def verdict_intensify(rows: list[dict]) -> tuple[str, str]:
    r = _row(rows, gamma=2.0)
    return "warning", (f"Werden alle Urteile mit γ = 2 verstärkt (die Richtung bleibt), kehrt der Eigenvektor in {de(100 * r['ev'], 1)} % der Matrizen mindestens ein Paar "
                       f"strikt um; das geometrische Mittel in {de(100 * r['gm'], 1)} %. Die Ordnung des Eigenvektors hängt damit davon ab, wie stark die Urteile ausgedrückt "
                       "werden; die des geometrischen Mittels nicht.")


def verdict_consistency(cons: list[dict], comp: list[dict]) -> tuple[str, str]:
    mid = _row(cons, sigma=C.EXP_QUESTION_SIGMA)
    top = _row(cons, sigma=max(r["sigma"] for r in cons))
    half = _row(comp, beta=0.5)
    return "warning", (f"Eine vollständig konsistente Matrix (CR = {de(half['cr'], 2)}), deren Urteile sich auf w hoch 0,5 statt auf w beziehen (β = 0,5), liegt im Mittel "
                       f"{de(half['tv'], 3)} von den wahren Gewichten entfernt. Bei σ = {de(mid['sigma'], 1)} ist der Abstand für Matrizen mit CR > 0,1 rund doppelt so groß "
                       f"({de(mid['tv_bad'], 3)} gegen {de(mid['tv_ok'], 3)}); bei σ = {de(top['sigma'], 1)} ist er kaum verschieden ({de(top['tv_bad'], 3)} gegen "
                       f"{de(top['tv_ok'], 3)}). Das Konsistenzverhältnis erkennt Widersprüche, aber keine systematisch verzerrten Urteile.")


def verdict_question(rows: list[dict]) -> tuple[str, str]:
    lo = min(r["delta"] for r in rows)
    imp = _row(rows, world="roh", delta=lo, method="wichtigkeit")["hit"]
    sw = _row(rows, world="roh", delta=lo, method="schwenk")["hit"]
    return "warning", (f"Streut das erste Kriterium nur noch zu {de(100 * lo, 0)} %, trifft AHP mit Wichtigkeitsurteilen Platz 1 in {de(100 * imp, 0)} % der Fälle, "
                       f"mit Schwenkurteilen in {de(100 * sw, 0)} %. Ob AHP bei schmaler Spannweite scheitert, entscheidet in der Welt der Rohwerte die gestellte Frage, nicht das Verfahren.")
