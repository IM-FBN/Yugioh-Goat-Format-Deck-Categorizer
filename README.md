# GOAT Deck Classifier 🐐

XGBoost-Modell zur automatischen Kategorisierung von Yu-Gi-Oh! **GOAT-Format**-Decks in Archetypen.

Das Modell ordnet eine Decklist anhand ihrer Kartenzusammensetzung einem von **35 Archetypen** zu (z. B. *Chaos Turbo*, *Goat Control*, *Warrior*, *Panda Burn*). Es funktioniert nicht nur auf vollständigen Decks, sondern liefert auch bei **unvollständigen Decks** (z. B. aus laufenden Replays) brauchbare Vorhersagen.

## Überblick

| | |
|---|---|
| **Algorithmus** | XGBoost (Multiclass-Klassifikation) |
| **Klassen** | 35 GOAT-Archetypen |
| **Test-Genauigkeit** | **91 %** (1.850 Samples) |
| **Cross-Validation-Genauigkeit** | 99,88 % |
| **Weighted F1** | 0,90 |
| **Macro F1** | 0,77 |

> Die hohe CV-Genauigkeit gegenüber der Test-Genauigkeit deutet auf leichtes Overfitting bzw. eine schwierigere Testverteilung hin – siehe [Einschränkungen](#einschränkungen).

## Genauigkeit bei Teil-Decks

Eine zentrale Eigenschaft des Modells: Es bleibt auch bei nur teilweise bekannten Decks robust. Das ist relevant, wenn ein Deck z. B. mitten in einem Match aus den gespielten Karten rekonstruiert wird.

| Deck-Anteil | Genauigkeit |
|---|---|
| 20 % | 0,76 |
| 40 % | 0,88 |
| 60 % | 0,95 |
| 80 % | 0,96 |
| 100 % | 0,97 |

Bereits ab ca. **60 % der Karten** erreicht das Modell über 95 % Genauigkeit.

## Performance je Archetyp

Alle 35 Archetypen aus dem Classification Report, absteigend nach Support sortiert:

| Archetyp | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Chaos Turbo | 0,96 | 0,97 | 0,96 | 710 |
| Warrior | 0,87 | 0,98 | 0,92 | 285 |
| Chaos Control | 0,84 | 0,86 | 0,85 | 175 |
| Goat Control | 0,92 | 0,94 | 0,93 | 135 |
| Panda Burn | 0,96 | 1,00 | 0,98 | 80 |
| Chaos Return | 0,90 | 0,80 | 0,85 | 70 |
| Chaos Warrior | 0,95 | 0,64 | 0,76 | 55 |
| Reasoning Gate Turbo | 1,00 | 0,89 | 0,94 | 45 |
| Flip Warrior | 0,83 | 0,69 | 0,75 | 35 |
| Earth Beat | 0,82 | 0,93 | 0,88 | 30 |
| Chaos Warrior Turbo | 0,83 | 0,80 | 0,82 | 25 |
| Cat Burn | 0,88 | 0,93 | 0,90 | 15 |
| Relinquished | 1,00 | 0,87 | 0,93 | 15 |
| Stein Gate Turbo | 0,75 | 1,00 | 0,86 | 15 |
| Zombie | 0,72 | 0,87 | 0,79 | 15 |
| Cat OTK | 1,00 | 0,60 | 0,75 | 10 |
| Chaos Recruiter | 0,78 | 0,70 | 0,74 | 10 |
| Dimension Fusion Turbo | 1,00 | 0,70 | 0,82 | 10 |
| Drain Burn | 1,00 | 0,30 | 0,46 | 10 |
| Flip Control | 1,00 | 0,90 | 0,95 | 10 |
| Gearfried | 1,00 | 0,70 | 0,82 | 10 |
| Gumo Warrior | 0,75 | 0,90 | 0,82 | 10 |
| SWAG | 1,00 | 0,80 | 0,89 | 10 |
| Tiger Stun | 0,73 | 0,80 | 0,76 | 10 |
| Bazoo Return | 0,83 | 1,00 | 0,91 | 5 |
| Drain Beat | 0,00 | 0,00 | 0,00 | 5 |
| Economics FTK | 1,00 | 0,80 | 0,89 | 5 |
| Gravekeeper | 1,00 | 0,80 | 0,89 | 5 |
| Library FTK | 1,00 | 1,00 | 1,00 | 5 |
| Mataza Rush | 0,80 | 0,80 | 0,80 | 5 |
| Monarch | 0,00 | 0,00 | 0,00 | 5 |
| Soul Control | 0,75 | 0,60 | 0,67 | 5 |
| Stall Burn | 1,00 | 0,80 | 0,89 | 5 |
| Stein Monarch | 0,67 | 0,80 | 0,73 | 5 |
| Tomato Monarch | 0,00 | 0,00 | 0,00 | 5 |
| **Accuracy** | | | **0,91** | **1850** |
| *Macro avg* | 0,82 | 0,75 | 0,77 | 1850 |
| *Weighted avg* | 0,91 | 0,91 | 0,90 | 1850 |

Die häufig vertretenen Archetypen werden sehr zuverlässig erkannt. Die größten Verwechslungen treten – wie erwartet – zwischen eng verwandten Decks mit großen Kartenüberschneidungen auf:

- **Chaos Turbo ↔ Chaos Control ↔ Chaos Return** (gemeinsamer Chaos-Kern)
- **Warrior ↔ Chaos Warrior ↔ Flip Warrior** (Warrior-Engine)

## Hyperparameter

Per GridSearch ermittelte beste Parameter:

```python
best_params = {
    "learning_rate":     0.1,
    "max_depth":         7,
    "min_child_weight":  1,
    "subsample":         0.7,
    "colsample_bytree":  0.7,
    "gamma":             0,
    "reg_alpha":         0,
    "reg_lambda":        0,
}
```

## Dateien

| Datei | Inhalt |
|---|---|
| `classification_report.txt` | Precision / Recall / F1 je Klasse + beste Hyperparameter |
| `confusion_matrix.png` / `.csv` | Konfusionsmatrix über alle 35 Klassen |
| `partial_deck_accuracy.png` / `.csv` | Genauigkeit in Abhängigkeit vom Deck-Anteil |

## Einschränkungen

- **Seltene Archetypen mit sehr wenig Daten** werden teils gar nicht erkannt (F1 = 0,00 bei *Drain Beat*, *Monarch*, *Tomato Monarch* – jeweils nur 5 Samples). Hier hilft primär mehr Trainingsdaten.
- Die Lücke zwischen CV-Genauigkeit (99,88 %) und Test-Genauigkeit (91 %) sollte beobachtet werden – ggf. Regularisierung erhöhen oder die Daten-Splits prüfen.
- Decks an der Grenze zwischen zwei verwandten Archetypen (z. B. Chaos-Varianten) sind inhärent mehrdeutig; ein gewisser Verwechslungsanteil ist unvermeidbar.

## Nächste Schritte

- Mehr Trainingsdaten für unterrepräsentierte Archetypen sammeln
- Regularisierung (`reg_alpha`, `reg_lambda`) gegen das Overfitting testen
- Konfidenzwerte der Vorhersage in der UI ausgeben, um mehrdeutige Decks sichtbar zu machen

---

*GOAT-Format Tooling*
