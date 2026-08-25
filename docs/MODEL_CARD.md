# Model card — ExoDiscover binary classifier

**Version** 0.1.0 · **Family** CatBoost with isotonic calibration · **Task**
Confirmed planet vs false positive, from Kepler transit parameters.

## Start here: what it can actually do

On **new data from a different telescope**, this model reaches **0.762 ROC-AUC
and a Brier score of 0.222** — its probabilities stop being trustworthy
entirely. That is the number that describes its real-world use, and it is
reported before the flattering one on purpose.

On held-out Kepler stars it reaches 0.984 ROC-AUC. Both are true; they measure
different things.

## Intended use

Triage. Ranking a list of unvetted Kepler transit signals so that limited
follow-up time goes to the most promising ones first. Precision at the top of
the list is what matters, and precision@50 on the held-out set is 1.000.

**Not** intended to confirm a planet. Confirmation requires radial-velocity
measurement, transit-timing analysis, or statistical validation against the
local stellar population — none of which this model performs.

## Training data

NASA Exoplanet Archive `cumulative` table (Kepler Objects of Interest).

| | |
|---|---|
| Total catalog | 9,564 KOIs across 8,214 host stars |
| Used for training | 7,585 rows with a resolved disposition (2,746 CONFIRMED, 4,839 FALSE POSITIVE) |
| Fit on | 6,061 rows / 5,313 stars |
| Held out | 1,524 rows, from stars absent from training |
| Withheld entirely | 1,979 CANDIDATE rows — scored as the discovery set, never trained on |

Splits are grouped on `kepid` throughout. K2 is ingested but excluded: its
archive table lacks transit depth and duration, so merging it would introduce a
missingness pattern that identifies the mission.

## Features

17 features, listed in `ml/exodiscover/features/tabular.py::FEATURE_COLUMNS`.
Beyond the raw archive columns, three are physically derived:

- **`rho_star`** — mean stellar density from `logg` and radius.
- **`duration_ratio`** — observed transit duration over the duration implied by
  Kepler's third law at that period and stellar density. Eclipsing binaries
  deviate.
- **`depth_ratio`** — observed depth over `(Rp/R*)²`. Blends and giants deviate.
- **`n_kois_on_star`** — multiplicity, counted over the whole catalog before
  any label filtering. Multi-planet systems are rarely false positives.

Eight columns are permanently excluded as target-encoding, and three
uncertainty features were removed after measurement showed they encode
post-confirmation parameter refinement. See `LEAKAGE.md`.

## Performance

Held-out stars, n = 1,524, base rate 0.366:

| Metric | Value |
|---|---|
| ROC-AUC | 0.9844 |
| PR-AUC | 0.9701 |
| Brier | 0.0449 |
| Precision@50 | 1.000 |

Confusion matrix at threshold 0.5: 922 TN, 45 FP, 46 FN, 511 TP.

Model ladder, grouped 5-fold CV PR-AUC — every family measured identically:

| Model | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| CatBoost | 0.9686 | 0.9826 | 0.0463 |
| XGBoost | 0.9679 | 0.9823 | 0.0472 |
| LightGBM | 0.9662 | 0.9816 | 0.0503 |
| HistGradientBoosting | 0.9655 | 0.9815 | 0.0489 |
| RandomForest | 0.9613 | 0.9787 | 0.0539 |
| Logistic regression | 0.8523 | 0.9153 | 0.1124 |
| Dummy (prior) | 0.3620 | 0.5000 | 0.2310 |

The gradient-boosted families are separated by less than 0.003 PR-AUC — well
inside fold-to-fold noise. CatBoost ships because it came first, not because it
is meaningfully better.

## Calibration

Isotonic, fitted on a star-disjoint 25% slice the base model never saw. The
reliability curve is in `docs/metrics/reliability.png`. Calibration matters
because the discovery ranking consumes probabilities, not hard labels.

Calibration is **Kepler-specific** and does not transfer (Brier 0.067 → 0.222
on TESS).

**Isotonic saturates at the ends.** Being a step function, it maps everything in
its top bin to the same value — 15 of the first 50 ranked candidates came out at
exactly 1.0, which is both an unhelpful ordering and an overconfident number to
display. The discovery ranking therefore sorts on the base model's raw score,
which is finer grained and monotonically related, while still reporting the
calibrated probability. Read a displayed 1.0 as "at the top of the range the
calibration set could resolve", not as certainty.

## Limitations

1. **Cross-mission transfer is poor.** 0.762 ROC-AUC on TESS, with calibration
   destroyed. Do not use the probabilities off-domain.
2. **The task is easier than the headline suggests.** Kepler false positives are
   dominated by eclipsing binaries whose implied planet radius is physically
   impossible — 99.8% of objects above 40 R⊕ are false positives. The model is
   strong at *planet vs eclipsing binary*, which is not the same as being strong
   on genuinely marginal signals.
3. **Candidate labels reflect follow-up selection.** Which planets got confirmed
   depends on target brightness, period, and multiplicity — not only on physics.
   This is why the model is trained on resolved dispositions only.
4. **No light-curve model.** The cached window arrays record no star identity
   and their ordering was shuffled (1,172 label runs, not the ~15 expected), so
   a leakage-free evaluation is impossible with them. Rather than publish an
   unverifiable number, the track was dropped. The preprocessing pipeline
   remains as an explainer. Fixing this needs bulk MAST downloads with identity
   preserved.
5. **Not tuned.** Trained with `--fast`, which skips the Optuna search. The
   untuned CatBoost is within 0.003 PR-AUC of every other boosted family, so
   tuning was not expected to move the result beyond noise. `exo train
   --trials N` runs it.
6. **Single archive snapshot.** Dispositions change as vetting continues; the
   model reflects the snapshot recorded in `data/raw/koi.meta.json`.

## Reproducing

```bash
make install
exo ingest        # NASA archive -> data/raw/, with query + timestamp recorded
exo train         # ~20 min on a laptop CPU
```

Seed is 42 throughout, set once in `ml/exodiscover/config.py`.
