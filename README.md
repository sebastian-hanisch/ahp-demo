# AHP: Paarvergleiche, Eigenvektor, Konsistenz – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-ahp-demo.streamlit.app/)**

Viertes Stück der Linie **Mehrkriterien-Ranking** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Kind von
[Gewichte erheben](https://github.com/sebastian-hanisch/gewichte-erheben-demo). Dort gibt der Entscheider Punkte je Kriterium; hier **vergleicht er die Kriterien paarweise**
(wie viel wichtiger ist i als j?), und aus der Matrix dieser Urteile werden die Gewichte abgeleitet: über den **Eigenvektor** (Saaty) oder das **geometrische Mittel** der
Zeilen. Die Demo misst beides an der Entscheidung, die die gewichtete Summe damit trifft.

Vehikel ist dieselbe **neutrale, simulierte Entscheidungsmatrix** wie in den Vorgängern (Generator `entscheidungsmatrix_generator.py`, wortgleich). Der **Entscheider** kennt seine
Gewichte w; seine **Paarurteile** sind ein **angenommenes** Modell (lognormales Rauschen, Rundung auf die Saaty-Skala), nicht am Menschen gemessen.

## Warum dieses Problem

**Der Eigenvektor ist nicht das Problem; die Skala, die Frage und das Konsistenzmaß sind es.** Im Urteilsmodell dieser Demo sind Eigenvektor und geometrisches Mittel in Genauigkeit
und Entscheidung gleichauf (bei 3 Kriterien sogar identisch). Unterschiede liegen woanders: Die Skala 1 bis 9 macht den Fehler rund doppelt so groß wie stetige Urteile; ein
kleines Konsistenzverhältnis garantiert keine Richtigkeit; und ob AHP bei schmaler Spannweite scheitert, entscheidet die **gestellte Frage** (Wichtigkeit gegen Schwenk), nicht das
Verfahren. Das einzige, was den Eigenvektor trennt, ist die **Ordnungsstabilität**: Werden alle Urteile verstärkt, kann er ein Paar umkehren, das geometrische Mittel nie.

## Befunde und Korrekturen gegenüber dem Plan

- **Die Kachel versprach mehr, als die Messung hergibt.** „Eigenvektor kann bei gleich verstärkten Präferenzen die Ordnung ändern, geometrisches Mittel nicht“ stimmt (Experiment 2),
  ist aber der kleinste Teil des Befunds. Das geometrische Mittel ist nicht „besser“: Abstand zu den wahren Gewichten und Trefferquote liegen in jeder Zelle der Messung innerhalb
  von 0,003 bzw. 1,5 Punkten.
- **Bei 3 Kriterien gibt es keinen Unterschied.** Eigenvektor und geometrisches Mittel sind dort identisch (größte Abweichung 10⁻¹² über 5 000 Zufallsmatrizen); die
  Voreinstellung der Demo sind deshalb 5 Kriterien.
- **Ein hohes Konsistenzverhältnis zeigt Fehler an, ein niedriges garantiert keine Richtigkeit.** Bei σ = 0,2 ist der Abstand für CR > 0,1 doppelt so groß wie für CR ≤ 0,1;
  bei σ = 0,8 sind beide gleich. Eine vollständig konsistente, aber verdichtete Matrix hat CR = 0 und einen Abstand von 0,144.
- **Der Random Index ist keine Konstante.** Die Demo simuliert ihn (gleichverteilte Skalenwerte); bei 3 Kriterien 0,529 gegen 0,58 in der einen und 0,52 in der anderen Tabelle der
  Quellen (nach Suchzusammenfassungen, nicht im Original geprüft). Keine Tabelle wird in allen Zeilen innerhalb von zwei Standardfehlern getroffen.
- **Die Zahlen sind neu gemessen.** Die Phase 0 rechnete mit `numpy.default_rng`; die Demo nutzt den SplitMix64-Strom des Generators. Die Werte verschieben sich um etwa einen
  Standardfehler (Verstärkung γ = 2: 5,4 % statt 4,6 %; Random Index bei 3 Kriterien 0,529 statt 0,520).
- **Das Preset „Verstärkung kippt den Eigenvektor“ ist ein Zufallsfall,** kein festes Gegenbeispiel: Es ist auf einen Seed abgestimmt, bei dem ein klar geordnetes Paar des
  Eigenvektors umkehrt und das geometrische Mittel dieselbe klare Ordnung behält. Das feste 4 × 4-Gegenbeispiel steht im Experiment 2.
- **Die Experimente starten auf Klick** (rund 15 Sekunden), nicht beim Laden.

## Modell

- **Matrix:** n Alternativen (5 bis 40), m Kriterien (3 bis 8), Rohwerte mit Richtung und Einheit; optional ein Kriterium auf den Anteil δ seiner Streuung gestaucht.
- **Entscheider:** wahre Gewichte gleichverteilt über dem Simplex (Generator); Nutzen linear in der Güte oder linear in den Rohwerten (feste Nenn-Spannweite je Kriterium).
- **Urteile (angenommen):** Das Paarurteil ist das Verhältnis z_i / z_j der erfragten Gewichte mit lognormalem Rauschen exp(σε), auf die Saaty-Skala (ganze Zahlen 1 bis 9 und
  Kehrwerte, kaufmännisch gerundet, bei 9 abgeschnitten) oder stetig. Erfragt wird entweder die **Wichtigkeit** (z = w) oder der **Schwenk** worst → best in der Stichprobe
  (z ∝ w · Spannweite des Wertes). Fehlerquellen: **Verstärkung** γ (elementweise Potenz der Matrix) und **Verdichtung** β (z ∝ w^β).
- **Ableitung:** Eigenvektor (Perron-Vektor) und Zeilen-Geometrisches-Mittel; Konsistenz CI = (λmax − m)/(m − 1), CR = CI/RI mit **simuliertem** Random Index
  (20 000 Zufallsmatrizen je m, Erzeugung: Einträge oberhalb der Diagonalen unabhängig und gleichverteilt aus den 17 Skalenwerten).
- **Summe:** gewichtete Summe auf Min-Max-normierten Rohwerten (die Wahl der Normierung steht im zweiten Stück der Linie).
- **Fester Rahmen der Experimente:** 200 feste Matrizen (`SWEEP_SEEDS`), n = 20, unabhängig vom Regler n; Experimente ohne Matrix mit 2 000 bis 5 000 festen Ziehungen.

Formeln im 📐-Expander der App.

## Methodik

1. **Dieselben Rauschwerte für jede Variante:** Das Rauschen der Paarurteile hängt nur an Seed und m, nicht an Frage, Skala, Verstärkung oder Verdichtung; Vergleiche zwischen
   Varianten sind damit gepaart.
2. **Messung an der Entscheidung:** Platz 1 gegen den Besten des Entscheiders, Kendalls τ-b der Rangfolge, Nutzenverlust; dazu der Abstand der Gewichte (halber L1-Abstand).
3. **Strikte Umkehr:** Ein Paar zählt nur, wenn es in einer Ordnung um mehr als 10⁻⁹ größer und in der anderen um mehr als 10⁻⁹ kleiner ist. Gleichstände durch die Rundung auf die
   Skala sind keine Umkehr (eine erste Fassung der Messung zählte sie mit und meldete bis 15,5 % bei σ = 0).
4. **Zwei Welten:** Der Entscheider denkt in der Güte oder in den Rohwerten.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: `tests/test_claims.py`. Trefferquoten über 200 feste Matrizen, n = 20, Min-Max, Welt Rohwerte wenn nicht anders vermerkt. **Der Standardfehler einer Trefferquote ist
1,3 bis 3,5 Punkte; Unterschiede unter etwa 6 Punkten sind nicht belastbar und stehen nicht als Aussage.**

| Frage | Befund | Test |
|---|---|---|
| Eigenvektor gegen geometrisches Mittel: Abstand zu w (σ = 0 / 0,2 / 0,4 / 0,8) | 3 Kriterien identisch: 0,042 / 0,051 / 0,071 / 0,112. 5 Kriterien: EV 0,051 / 0,060 / 0,075 / 0,123, GM 0,053 / 0,061 / 0,075 / 0,121. 7 Kriterien: EV 0,050 / 0,056 / 0,072 / 0,113, GM 0,051 / 0,057 / 0,073 / 0,110. | `test_distance_to_the_true_weights` |
| … und Platz 1 (5 Kriterien) | σ = 0: EV 93,5 %, GM 92,5 %; σ = 0,2: 89,5 / 90,0 %; σ = 0,4: 90,0 / 90,5 %; σ = 0,8: 83,5 / 84,5 %; wahre Gewichte 96,5 %. In keiner Zelle mehr als 1,5 Punkte Unterschied. | `test_first_place_for_both_derivations`, `test_eigenvector_and_geometric_mean_are_level…` |
| Wie oft ordnen EV und GM ein Paar entgegengesetzt? (σ = 0 / 0,2 / 0,4 / 0,8) | 3 Kriterien: nie. 5 Kriterien: 0 / 0,5 / 2,5 / 15 %. 7 Kriterien: 0 / 2,5 / 13 / 35 %. | `test_share_of_matrices_where…` |
| Wie oft liegt CR über 0,1? (5 Kriterien, σ = 0 / 0,2 / 0,4 / 0,8) | 10,5 / 12,5 / 18,5 / 74,5 % (200 Matrizen, Standardfehler bis 3 Punkte); schon die Rundung auf die Skala allein macht Matrizen „inkonsistent“. | `test_share_of_matrices_with_a_consistency_ratio…` |
| Weniger Vergleiche (5 Kriterien, σ = 0,2, Log-Kleinste-Quadrate; Abstand zu w) | Stern um ein Kriterium (4 Vergleiche) 0,095; Kette (4) 0,108; Best-Worst-Stern (7) 0,066; zufällig 7 von 10 0,070; alle 10 0,057. | `test_fewer_comparisons` |
| Verstärkung γ = 0,5 / 1,5 / 2 / 3 (5 Kriterien, σ = 0,4, 5 000 Matrizen): strikte Umkehr | **Eigenvektor 2,6 / 2,7 / 5,4 / 7,9 %; geometrisches Mittel 0 %** (exakt invariant). | `test_amplification_reverses…` |
| Das feste 4 × 4-Gegenbeispiel (A12 = A13 = A14 = 1/3, A23 = 1/2, A24 = 7, A34 = 3) | EV(A): 3 > 2 > 4 > 1; EV(A²): 2 > 3 > 4 > 1; GM beide 3 > 2 > 4 > 1. | `test_counterexample_orders` |
| Konsistenzverhältnis und Richtigkeit (5 Kriterien; σ = 0 / 0,2 / 0,4 / 0,8; 5 000 Matrizen) | Abstand bei CR ≤ 0,1 / CR > 0,1: 0,041 / 0,107; 0,050 / 0,103; 0,071 / 0,094; **0,117 / 0,121**. Anteil CR > 0,1: 7,2 / 8,5 / 15,7 / 70,8 %. | `test_consistency_ratio_and_distance` |
| Konsistent, aber falsch (w hoch β; 5 Kriterien, 2 000 Ziehungen) | CR = 0; Abstand zu w bei β = 1 / 0,5 / 0,25: **0 / 0,144 / 0,231**. | `test_consistent_but_wrong` |
| Random Index, simuliert (m = 3 … 9; 20 000 Matrizen) | 0,529 / 0,888 / 1,107 / 1,248 / 1,343 / 1,405 / 1,451 (Standardfehler 0,002 bis 0,005). | `test_simulated_random_index` |
| Die Frage (Welt Rohwerte, 5 Kriterien, σ = 0,2; δ = 1 / 0,5 / 0,2 / 0,05) | wahre Gewichte 96,5 / 86,5 / 76,0 / 67,0 %; **AHP mit Wichtigkeitsurteilen 89,5 / 81,0 / 72,5 / 63,0 %; AHP mit Schwenkurteilen 92,0 / 91,0 / 93,0 / 90,0 %**; die Schwenk-Gewichte selbst 100 %. | `test_the_question_decides` |
| Dasselbe in der Welt Güte (δ = 1 / 0,05) | wahre Gewichte 67,0 / 50,0 %, Wichtigkeitsurteile 65,0 / 48,5 %, Schwenkurteile 64,0 / 66,5 % (die Güte ist bei den logarithmischen Kriterien nicht linear in den Rohwerten, daher bleiben auch die Schwenk-Gewichte unter 100 %). | `test_the_question_decides` |
| Die Skala 1 bis 9 (σ = 0,2; m = 3 / 5 / 8; 3 000 Ziehungen) | 20 % der Paare haben ein wahres Verhältnis über 9; Abstand mit Skala 0,051 / 0,056 / 0,056, mit stetigen Urteilen 0,029 / 0,029 / 0,024. | `test_scale_one_to_nine` |

## Ehrliche Grenzen

- **Das Urteilsmodell ist angenommen.** Lognormales Rauschen mit gleichem σ für alle Paare, keine systematischen Verzerrungen außer der einen gezielten (β); Menschen urteilen anders:
  Anker, runde Zahlen, Zyklen. Die Demo sagt nichts über die Konsistenz echter Urteile.
- **Nur die Saaty-Skala.** Andere Skalen (geometrisch, balanciert) ändern die Skalenkosten; nicht gemessen. Stetige Urteile sind nur die Gegenprobe.
- **Der Random Index hängt an der Erzeugung.** Andere Verfahren geben andere Werte; die Schwelle 0,1 ist eine Konvention.
- **Gewichte aus einer einzigen Matrix.** Die Hierarchie und die Synthese über die Alternativen (paarweise je Kriterium) sind nicht enthalten; die Rangumkehr durch eine hinzugefügte
  Alternative steht in einem eigenen Stück.
- **Kleine Unterschiede sind Rauschen.** Eigenvektor, geometrisches Mittel und wahre Gewichte liegen bei gleichmäßigen Kriterien innerhalb von zwei Standardfehlern, wo das erste
  Kriterium nicht verschmälert ist.
- **Ein Generator, eine Gewichtsverteilung.** n = 20, 3 bis 8 Kriterien, Gewichte gleichverteilt über dem Simplex; Quellenangaben nur aus Zusammenfassungen, nicht im Original geprüft.
- **Eine Normierung (Min-Max).** Mit anderen Normierungen verschieben sich die Zahlen; das steht im zweiten Stück.

## Verifikation

- **Kern:** Saaty-Rundung, reziproke Matrix, Eigenvektor, geometrisches Mittel, CI/CR, Verdichtung und Umkehr gegen Handrechnung auf den Mini-Instanzen (`tests/conftest.py`:
  3 × 3 mit A12 = 3, A13 = 5, A23 = 2; 4 × 4 mit A12 = 2, A13 = 4, A14 = 8, A23 = 3, A24 = 4, A34 = 2; w = (0,6; 0,3; 0,1) verdichtet; das 4 × 4-Gegenbeispiel).
- **Orakel** (`tests/test_oracle_ahp.py`): Eigenvektor gegen `numpy.linalg.eig`, `scipy.linalg.eig` und Potenzmethode; λmax ≥ m mit Gleichheit bei Konsistenz; geometrisches
  Mittel gegen die Definitionsschleife und als Minimierer der Kleinste-Quadrate-Funktion; **Potenzinvarianz des geometrischen Mittels exakt, des Eigenvektors nicht** (Suche findet
  Gegenbeispiele); **EV = GM bei 3 × 3** (Gegenprobe: bei 4 × 4 nicht); Random Index gegen eine unabhängige Schleife und ausdrücklich verschieden von 0,58 bei m = 3;
  unvollständige Matrizen gegen die Gleichungssystem-Lösung (Stern, Kette).
- **Generator** (`tests/test_generator.py`): SplitMix64 gegen unabhängige Python-Fassung, Randverteilung, Spearman-Korrelation der Copula, Richtungen und Bereiche.
- **Aussagen:** jede Zahl dieser README in `tests/test_claims.py`; Presets in `test_presets.py` (jedes zeigt seinen Effekt).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum), alle Presets, Permalink, PDF und Experimente erst nach Klick, Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `entscheidungsmatrix_generator.py` | Generator der Linie (wortgleich in allen Repos der Linie, kein Präfix): SplitMix64, Copula, Rohwerte, Entscheider |
| `ahp_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets, Gegenbeispiel |
| `ahp_matrix.py` | Saaty-Rundung, reziproke Matrix, Urteile mit Rauschen, Verstärkung, Verdichtung |
| `ahp_weights.py` | Eigenvektor, geometrisches Mittel, CI/CR, Random Index (Simulation), strikte Umkehr, Log-Kleinste-Quadrate |
| `ahp_score.py` | Min-Max, Score, Wertskalen des Entscheiders, Verschmälerung, Schwenk-Gewichte, Kendalls τ-b |
| `ahp_evaluation.py` | Fall der App, vier Experimente, Verdikte |
| `ahp_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `ahp_visualization.py` | Tabellen und Plotly-Abbildungen (Achsen gesperrt) |
| `ahp_pdf_export.py` | PDF-Export des aktuellen Falls (fpdf2, erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest und SciPy (nur für die Gegenproben) |
| `tests/` | Mini-Instanzen, Matrix, Gewichte, Orakel, Generator, Auswertung, Aussagen, Presets, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Die Hierarchie und die Synthese über Alternativen** (verteilend oder ideal): Die Demo leitet nur Kriteriengewichte ab; die Rangumkehr durch eine hinzugefügte Alternative hat ihr
  eigenes Stück.
- **Konsistenz menschlicher Urteile:** Die Simulation kann sie nicht prüfen.
- **Andere Skalen als Saaty:** geometrische und balancierte Skalen sind nicht gemessen.
- **Konsistenzverbesserung** (Urteile automatisch anpassen): nicht umgesetzt.
- **Gruppen:** Der Entscheider ist ein einzelnes, verborgenes Orakel; Aggregation mehrerer Matrizen hat ein eigenes Stück.
- **Unsicherheit der Urteile als Bereich:** Eine Matrix liefert genau einen Gewichtsvektor; die Weitergabe der Unsicherheit steht beim SMAA.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2 (SciPy nur in den Tests).

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Mehrkriterien-Ranking: von Pareto zum Vergleich](https://sebastianhanisch.net/konzepte-mehrkriterien-ranking.html).
