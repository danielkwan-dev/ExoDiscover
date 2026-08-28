# Model card — ExoDiscover binary classifier

**Version** 0.1.0 · **Family** Soft-vote ensemble (CatBoost + XGBoost +
LightGBM), isotonic calibration · **Task** Confirmed planet vs false positive,
from Kepler transit parameters.

## Start here: what it can actually do

On **new data from a different telescope**, this model reaches **0.838 ROC-AUC
and a Brier score of 0.177** — the ranking mostly survives, the probabilities
do not. That is the number that describes its real-world use, and it is
reported before the flattering one on purpose.

On held-out Kepler stars it reaches **0.984 ROC-AUC (95% CI 0.979–0.988)**.
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
| Fit on | 5,287 rows / 4,647 stars (70%) |
| Held out | 2,298 rows / 1,992 stars (30%), absent from training entirely |
| Withheld entirely | 1,979 CANDIDATE rows — scored as the discovery set, never trained on |

Splits are grouped on `kepid` throughout, and hold out whole stars at the exact
requested fraction rather than the nearest `1/k` a k-fold can express. K2 is
ingested but excluded: its
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

Held-out stars, n = 2,298, base rate 0.369. Intervals are 95% bootstrap,
resampling **host stars** rather than rows — sibling KOIs are not independent
draws, so a row bootstrap would report an interval narrower than the data
supports.

| Metric | Value | 95% CI |
|---|---|---|
| ROC-AUC | 0.9839 | 0.9792 – 0.9879 |
| PR-AUC | 0.9666 | 0.9551 – 0.9757 |
| Brier | 0.0440 | — |
| Accuracy | 0.9356 | — |
| Precision@50 | 1.000 | — |

Confusion matrix at threshold 0.5: 1,409 TN, 42 FP, 106 FN, 741 TP. Accuracy is
listed but not headlined: the classes are imbalanced 63/37, so always guessing
"false positive" already scores 0.631, and the operational question is ranking
rather than classification. The error profile is asymmetric on purpose — the
model misses 106 real planets to raise only 42 false alarms, which is the right
bias for a shortlist that costs telescope time.

### Is it overfitting?

No, and the evidence is recorded rather than asserted. Overfitting is a gap, not
a level:

| | ROC-AUC |
|---|---|
| On the rows it trained on | 0.9999 |
| On held-out stars | 0.9854 |
| **Gap** | **0.0145** |

The learning curve, measured against the same 1,992 held-out stars throughout,
is flat past half the data: 0.9764 from 519 rows, 0.9809 from 1,300, 0.9854 from
all 5,287. Both are written to `docs/metrics/metrics.json` by every training run
under the `overfitting` key.

Model ladder, grouped 5-fold CV — every family measured identically, with the
fold-to-fold spread alongside:

| Model | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| Soft-vote ensemble | 0.9694 ± 0.0056 | 0.9832 | 0.0457 |
| XGBoost | 0.9687 ± 0.0057 | 0.9826 | 0.0464 |
| CatBoost | 0.9686 ± 0.0053 | 0.9827 | 0.0463 |
| LightGBM | 0.9666 ± 0.0064 | 0.9818 | 0.0492 |
| HistGradientBoosting | 0.9664 ± 0.0061 | 0.9815 | 0.0483 |
| RandomForest | 0.9614 ± 0.0073 | 0.9786 | 0.0539 |
| Logistic regression | 0.8525 ± 0.0226 | 0.9154 | 0.1124 |
| Dummy (prior) | 0.3620 ± 0.0107 | 0.5000 | 0.2310 |

**The spread is the point.** The top five rows span 0.0030 PR-AUC, which is
smaller than any one of their fold-to-fold standard deviations. Ranking them
against each other is not meaningful, and a project that reported "our best
model, 0.969" without that column would be over-reading its own table.

Optuna then searched XGBoost — the best *tunable* family — over 40 trials under
the same grouped CV, improving it 0.9687 → 0.9689. That did not pass the untuned
ensemble (0.9694), so the soft-vote ensemble ships. Both margins are inside the
noise band, and the ensemble is preferred on its CV score alone, not because any
difference here is detectable.

Note that the ensemble fits its training data harder than a single model does —
0.9999 train ROC-AUC against CatBoost's 0.9947 — which widens the
generalisation gap from 0.008 to 0.015. Still well inside the threshold, but
worth stating rather than glossing.

## Calibration

Two methods were fitted and compared on a third, star-disjoint slice that
neither the base model nor the calibrator had seen:

| Method | Brier | Distinct probabilities emitted (n = 1,053) |
|---|---|---|
| **Isotonic** (selected) | **0.0576** | **30** |
| Sigmoid (Platt) | 0.0599 | 1,053 |

Isotonic is better calibrated and was selected on that basis. But the second
column is the trade it makes: as a step function it collapses its input into 30
levels, so it is nearly useless for fine-grained ordering — many of the first 50
ranked candidates come out at exactly 1.0. Sigmoid never ties but is measurably
worse calibrated.

Rather than pick one property and lose the other, the two jobs are separated:
**the calibrated probability is what gets displayed, and the base model's raw
score is what does the ranking.** Read a displayed 1.0 as "at the top of the
range the calibration set could resolve", not as certainty.

The reliability curve is in `docs/metrics/reliability.png`. Calibration is
**Kepler-specific** and does not transfer (Brier 0.069 → 0.177 on TESS).

## Limitations

1. **Cross-mission transfer degrades.** 0.838 ROC-AUC on TESS, down from 0.965
   in-domain, with the Brier score more than doubling. The ordering is still
   useful off-domain; the probabilities are not.
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
5. **Model choice is inside the noise band.** Tuning moved XGBoost by 0.0002
   PR-AUC and the top five families span 0.0030, against fold standard
   deviations near 0.006. Do not read the ladder ordering as a finding.
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
exo train --trials 40   # ~25 min on a laptop CPU
exo train --fast        # ~6 min, skips the Optuna search
```

The trained artifact is committed, so serving the API and UI needs none of the
above — see "Run it" in the README.

Seed is 42 throughout, set once in `ml/exodiscover/config.py`.
