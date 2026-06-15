# GOAT Deck Classifier 🐐

XGBoost model for automatically categorizing Yu-Gi-Oh! **GOAT-format** decks into archetypes.

Given a decklist's card composition, the model assigns it to one of **35 archetypes** (e.g. *Chaos Turbo*, *Goat Control*, *Warrior*, *Panda Burn*). It works not only on complete decks but also produces usable predictions for **partial decks** (e.g. reconstructed from ongoing replays).

## Overview

| | |
|---|---|
| **Algorithm** | XGBoost (multiclass classification) |
| **Classes** | 35 GOAT archetypes |
| **Test accuracy** | **91%** (1,850 samples) |
| **Cross-validation accuracy** | 99.88% |
| **Weighted F1** | 0.90 |
| **Macro F1** | 0.77 |

> The high CV accuracy relative to the test accuracy points to mild overfitting or a harder test distribution — see [Limitations](#limitations).

## Accuracy on partial decks

A key property of the model: it stays robust even when only part of a deck is known. This matters when a deck is reconstructed mid-match from the cards that have been played.

| Deck fraction | Accuracy |
|---|---|
| 20% | 0.76 |
| 40% | 0.88 |
| 60% | 0.95 |
| 80% | 0.96 |
| 100% | 0.97 |

The model already exceeds 95% accuracy once roughly **60% of the cards** are known.

## Per-archetype performance

All 35 archetypes from the classification report, sorted by support (descending):

| Archetype | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Chaos Turbo | 0.96 | 0.97 | 0.96 | 710 |
| Warrior | 0.87 | 0.98 | 0.92 | 285 |
| Chaos Control | 0.84 | 0.86 | 0.85 | 175 |
| Goat Control | 0.92 | 0.94 | 0.93 | 135 |
| Panda Burn | 0.96 | 1.00 | 0.98 | 80 |
| Chaos Return | 0.90 | 0.80 | 0.85 | 70 |
| Chaos Warrior | 0.95 | 0.64 | 0.76 | 55 |
| Reasoning Gate Turbo | 1.00 | 0.89 | 0.94 | 45 |
| Flip Warrior | 0.83 | 0.69 | 0.75 | 35 |
| Earth Beat | 0.82 | 0.93 | 0.88 | 30 |
| Chaos Warrior Turbo | 0.83 | 0.80 | 0.82 | 25 |
| Cat Burn | 0.88 | 0.93 | 0.90 | 15 |
| Relinquished | 1.00 | 0.87 | 0.93 | 15 |
| Stein Gate Turbo | 0.75 | 1.00 | 0.86 | 15 |
| Zombie | 0.72 | 0.87 | 0.79 | 15 |
| Cat OTK | 1.00 | 0.60 | 0.75 | 10 |
| Chaos Recruiter | 0.78 | 0.70 | 0.74 | 10 |
| Dimension Fusion Turbo | 1.00 | 0.70 | 0.82 | 10 |
| Drain Burn | 1.00 | 0.30 | 0.46 | 10 |
| Flip Control | 1.00 | 0.90 | 0.95 | 10 |
| Gearfried | 1.00 | 0.70 | 0.82 | 10 |
| Gumo Warrior | 0.75 | 0.90 | 0.82 | 10 |
| SWAG | 1.00 | 0.80 | 0.89 | 10 |
| Tiger Stun | 0.73 | 0.80 | 0.76 | 10 |
| Bazoo Return | 0.83 | 1.00 | 0.91 | 5 |
| Drain Beat | 0.00 | 0.00 | 0.00 | 5 |
| Economics FTK | 1.00 | 0.80 | 0.89 | 5 |
| Gravekeeper | 1.00 | 0.80 | 0.89 | 5 |
| Library FTK | 1.00 | 1.00 | 1.00 | 5 |
| Mataza Rush | 0.80 | 0.80 | 0.80 | 5 |
| Monarch | 0.00 | 0.00 | 0.00 | 5 |
| Soul Control | 0.75 | 0.60 | 0.67 | 5 |
| Stall Burn | 1.00 | 0.80 | 0.89 | 5 |
| Stein Monarch | 0.67 | 0.80 | 0.73 | 5 |
| Tomato Monarch | 0.00 | 0.00 | 0.00 | 5 |
| **Accuracy** | | | **0.91** | **1850** |
| *Macro avg* | 0.82 | 0.75 | 0.77 | 1850 |
| *Weighted avg* | 0.91 | 0.91 | 0.90 | 1850 |

The frequently represented archetypes are recognized very reliably. The largest confusions occur — as expected — between closely related decks with heavy card overlap:

- **Chaos Turbo ↔ Chaos Control ↔ Chaos Return** (shared Chaos core)
- **Warrior ↔ Chaos Warrior ↔ Flip Warrior** (Warrior engine)

## Hyperparameters

Best parameters found via GridSearch:

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

## Files

| File | Contents |
|---|---|
| `classification_report.txt` | Precision / recall / F1 per class + best hyperparameters |
| `confusion_matrix.png` / `.csv` | Confusion matrix across all 35 classes |
| `partial_deck_accuracy.png` / `.csv` | Accuracy as a function of deck fraction |

## Limitations

- **Rare archetypes with very little data** are sometimes not recognized at all (F1 = 0.00 for *Drain Beat*, *Monarch*, *Tomato Monarch* — only 5 samples each). More training data is the primary fix here.
- The gap between CV accuracy (99.88%) and test accuracy (91%) is worth watching — consider increasing regularization or reviewing the data splits.
- Decks sitting on the boundary between two related archetypes (e.g. the Chaos variants) are inherently ambiguous; some amount of confusion is unavoidable.

## Next steps

- Collect more training data for underrepresented archetypes
- Test regularization (`reg_alpha`, `reg_lambda`) against the overfitting
- Surface prediction confidence scores in the UI to flag ambiguous decks

---

*GOAT-format tooling*
