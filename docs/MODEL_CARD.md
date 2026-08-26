# Model card — ExoDiscover binary classifier

**Version** 0.1.0 · **Family** CatBoost, Optuna-tuned, isotonic calibration ·
**Task** Confirmed planet vs false positive, from Kepler transit parameters.

## Start here: what it can actually do

On **new data from a different telescope**, this model reaches **0.762 ROC-AUC
and a Brier score of 0.222** — its probabilities stop being trustworthy
entirely. That is the number that describes its real-world use, and it is
reported before the flattering one on purpose.

On held-out Kepler stars it reaches **0.983 ROC-AUC (95% CI 0.976–0.988)**.
Both are true; they measure different things.

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

Held-out stars, n = 1,524, base rate 0.366. Intervals are 95% bootstrap,
resampling **host stars** rather than rows — sibling KOIs are not independent
draws, so a row bootstrap would report an interval narrower than the data
supports.

| Metric | Value | 95% CI |
|---|---|---|
| ROC-AUC | 0.9827 | 0.9764 – 0.9881 |
| PR-AUC | 0.9699 | 0.9581 – 0.9793 |
| Brier | 0.0474 | — |
| Precision@50 | 1.000 | — |

Confusion matrix at threshold 0.5: 923 TN, 44 FP, 54 FN, 503 TP.

Model ladder, grouped 5-fold CV — every family measured identically, with the
fold-to-fold spread alongside:

| Model | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| Soft-vote ensemble | 0.9690 ± 0.0053 | 0.9829 | 0.0463 |
| CatBoost | 0.9686 ± 0.0048 | 0.9826 | 0.0463 |
| XGBoost | 0.9679 ± 0.0058 | 0.9823 | 0.0472 |
| LightGBM | 0.9662 ± 0.0056 | 0.9816 | 0.0503 |
| HistGradientBoosting | 0.9655 ± 0.0069 | 0.9815 | 0.0489 |
| RandomForest | 0.9613 ± 0.0069 | 0.9787 | 0.0539 |
| Logistic regression | 0.8523 ± 0.0226 | 0.9153 | 0.1124 |
| Dummy (prior) | 0.3620 ± 0.0107 | 0.5000 | 0.2310 |

**The spread is the point.** The five boosted rows span 0.0035 PR-AUC, which is
smaller than any one of their fold-to-fold standard deviations. Ranking them
against each other is not meaningful, and a project that reported "CatBoost is
our best model, 0.969" without that column would be over-reading its own table.

Optuna then searched CatBoost over 25 trials under the same grouped CV,
improving it 0.9686 → 0.9697 — enough to pass the untuned ensemble (0.9690), so
the tuned single model ships. That margin is also inside the noise band; the
tuned model is preferred for being simpler and smaller than the ensemble, not
for being detectably better.

The held-out ROC-AUC (0.9827) is marginally below the untuned run's (0.9844).
Tuning optimised cross-validated PR-AUC, and the test set is one draw; both
values sit inside the reported confidence interval.

## Calibration

Two methods were fitted and compared on a third, star-disjoint slice that
neither the base model nor the calibrator had seen:

| Method | Brier | Distinct probabilities emitted (n = 1,483) |
|---|---|---|
| **Isotonic** (selected) | **0.0537** | **42** |
| Sigmoid (Platt) | 0.0570 | 1,483 |

Isotonic is better calibrated and was selected on that basis. But the second
column is the trade it makes: as a step function it collapses its input into 42
levels, so it is nearly useless for fine-grained ordering — 15 of the first 50
ranked candidates came out at exactly 1.0. Sigmoid never ties but is measurably
worse calibrated.

Rather than pick one property and lose the other, the two jobs are separated:
**the calibrated probability is what gets displayed, and the base model's raw
score is what does the ranking.** Read a displayed 1.0 as "at the top of the
range the calibration set could resolve", not as certainty.

The reliability curve is in `docs/metrics/reliability.png`. Calibration is
**Kepler-specific** and does not transfer (Brier 0.067 → 0.222 on TESS).

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
5. **Model choice is inside the noise band.** Tuning moved CatBoost by 0.0011
   PR-AUC and the whole boosted group spans 0.0035, against fold standard
   deviations near 0.005. Do not read the ladder ordering as a finding.
6. **No nested cross-validation.** The honest estimate comes from a
   star-held-out test set plus a grouped bootstrap, not from nesting the Optuna
   search inside an outer CV loop. Nesting would multiply a ~20-minute run by
   the outer fold count, which the stated CPU budget does not allow. The
   selection is therefore mildly optimistic — tuning saw the CV folds — though
   the test set stayed untouched throughout.
7. **Single archive snapshot.** Dispositions change as vetting continues; the
   model reflects the snapshot in `data/raw/`, fetched 4 October 2025. The
   shortlist has not been checked against a later snapshot.

## Reproducing

```bash
make install
exo ingest              # NASA archive -> data/raw/, query + timestamp recorded
exo train --trials 25   # ~20 min on a laptop CPU
exo train --fast        # ~6 min, skips the Optuna search
```

Seed is 42 throughout, set once in `ml/exodiscover/config.py`.
