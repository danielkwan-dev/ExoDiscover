# Leakage investigation

What was wrong with the original pipeline, what it cost, and what the honest
numbers are. Every figure here is reproduced by `exo train`, which writes
`docs/metrics/metrics.json`.

Archive snapshot: Kepler cumulative KOI table, 9,564 rows / 8,214 host stars.

---

## 1. Label leakage — the large one

`data/raw/koi.csv` ships five columns that encode the answer:

| Column | What it actually is |
|---|---|
| `koi_score` | The Robovetter's disposition confidence — its verdict, as a number |
| `koi_fpflag_nt` | "Not transit-like" vetting flag |
| `koi_fpflag_ss` | "Stellar eclipse" vetting flag |
| `koi_fpflag_co` | "Centroid offset" vetting flag |
| `koi_fpflag_ec` | "Ephemeris match / contamination" flag |

A model given these does not learn transit physics; it learns to read the
vetting pipeline's mind. The original `predict.py` and `train_xgboost.py`
selected feature columns by hand and never excluded them.

**Effect: ROC-AUC 0.9999 with them, 0.9865 without.** The leaky setup is
essentially a lookup table.

These are now quarantined in `ml/exodiscover/data/schema.py::LEAKY_COLUMNS`,
dropped on the way into `build_features` and asserted absent on the way out.
`tests/test_schema.py` and `tests/test_features.py` fail the build if one ever
reaches the feature matrix.

## 2. Split leakage — smaller than expected

Kepler lists 9,564 KOIs across only 8,214 stars, up to seven on one star.
Sibling KOIs share stellar parameters and photometry, so a random row split
puts related rows on both sides. `train_xgboost.py:110` split on rows; worse,
the light-curve pipeline split on 256-point *windows* cut from the same
continuous series.

**I expected this to be a major inflation source. It is not.**

| Split | ROC-AUC |
|---|---|
| Random rows | 0.9860 |
| Grouped by host star | 0.9865 |

The difference is within noise, and grouping actually scored marginally
*higher*. The reason is arithmetic: the binary training set holds 7,585 rows
across 6,639 stars, so only about 12% of rows have a sibling anywhere in the
data. There is not enough overlap for the contamination to matter.

Grouping is kept because it is the methodologically correct choice and it costs
nothing — but the honest finding is that on this dataset it was not the problem.
It would be the problem on the light-curve windows, where ~1,800 windows come
from a handful of stars (see §5).

## 3. Temporal leakage — the subtle one

Three engineered features were in the original design and have been **removed
after measurement**: `rel_err_period`, `rel_err_depth`, `rel_err_duration`
(relative parameter uncertainty, `err / value`).

They look like legitimate signal — a noisy measurement should be less
trustworthy. The Confirmed-vs-Candidate diagnostic showed what they really
encode:

| Feature | mean abs SHAP | Category |
|---|---|---|
| `rel_err_depth` | 0.930 | uncertainty |
| `koi_model_snr` | 0.769 | transit physics |
| `n_kois_on_star` | 0.614 | selection |
| `rel_err_period` | 0.454 | uncertainty |
| `log_period` | 0.391 | selection |
| `rel_err_duration` | 0.369 | uncertainty |

Selection proxies totalled 1.005 and transit physics 1.037 — near-tied — but
the three uncertainty terms together reached **1.75**, the largest group by a
wide margin.

The mechanism is temporal. A planet becomes CONFIRMED through follow-up
observation, and *that same follow-up refines its published parameters*. The
uncertainty is therefore a consequence of the label rather than a property of
the signal, and it would not be available at the moment a real classifier has
to make its call.

Removing them costs the binary model **0.003 ROC-AUC (0.9865 → 0.9835 at
measurement time)**. Cheap enough that keeping a feature I cannot defend was
not worth it. `tests/test_features.py::test_uncertainty_features_stay_excluded`
locks the decision in.

## 4. Why 0.98 is real here, not a fourth leak

The clean, grouped model scores 0.9827 ROC-AUC. That is high enough to be
suspicious, so it was checked rather than reported.

`log_prad` (planet radius) dominates the SHAP ranking at 1.63 mean abs — roughly
double the next feature (`koi_model_snr`, 0.86). It turns out to be genuine
physics:

| Disposition | median radius | 90th pct | max |
|---|---|---|---|
| CONFIRMED | 2.16 R⊕ | 4.42 R⊕ | 77.8 R⊕ |
| FALSE POSITIVE | 8.97 R⊕ | 82.97 R⊕ | **200,346 R⊕** |

99.8% of objects implying a radius above 40 R⊕ are false positives. Nothing
that large is a planet — those are eclipsing binaries and blended background
stars being interpreted through a planetary transit model. Separating them is
genuinely easy, and easy for the right reason.

Two checks confirm it is not a single load-bearing artefact:

- `log_prad` **alone** reaches only 0.8292 ROC-AUC.
- **Removing** `log_prad` entirely leaves 0.9886 — the signal is redundantly
  distributed across depth, duration consistency, and multiplicity.

**The honest caveat:** this model is good at *planet vs eclipsing binary*. That
is an easier question than the one the Robovetter faces on marginal signals,
and the cross-mission result in §6 shows where the limits actually are.

Because of this, the leakage guard in `cli.py` does **not** test accuracy
against a fixed threshold — a real number can legitimately be high. It checks
whether the clean pipeline has closed the gap on the deliberately-leaky one,
which is what leakage would actually look like.

## 5. The light-curve track cannot be evaluated

The original CNN reported strong accuracy on 1,811 windows. Two defects make
that number unrecoverable:

1. `create_training_data.py:21-23` selected targets with `.head(5)` per class —
   at most **15 stars** total, each contributing up to 100 overlapping windows.
2. Windows were split as independent rows, so overlapping segments of the same
   star's photometry appeared in both train and test.

The obvious fix is a star-grouped split. **It is not possible with the cached
data.** The arrays record no star identity, and reconstructing it from the label
ordering fails: the labels form **1,172 runs**, not the ~15 expected if each
target's windows were contiguous. The ordering was shuffled before saving, so
provenance is gone.

Rather than fabricate star IDs to produce a number, no light-curve model ships.
The preprocessing pipeline is retained as an explainer
(`ml/exodiscover/features/lightcurve.py`, tested offline). Fixing this properly
requires re-downloading light curves from MAST with identity preserved, which is
outside this project's stated CPU budget.

## 6. Task framing — settled by measurement

The original predicts three classes. The concern was that CANDIDATE is not a
physical category but a vetting state, so the CONFIRMED↔CANDIDATE boundary
would encode follow-up selection rather than physics.

All three framings, scored on the one decision they share — planet-like vs
false positive, collapsing the 3-class output at inference:

| Framing | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| Binary (Confirmed vs FP) | **0.9865** | **0.9778** | **0.0429** |
| Diagnostic (Confirmed vs Candidate) | 0.9349 | 0.9522 | 0.1017 |
| Flat 3-class | 0.9319 | 0.9318 | 0.1053 |

Binary wins decisively, so it ships. The diagnostic's SHAP breakdown (§3) is
the more interesting result: the third class is separable, but substantially on
parameter-refinement and selection terms rather than transit shape.

**My original hypothesis was only partly right.** I predicted brightness and
multiplicity would dominate that boundary. They contribute, but the dominant
signal was measurement refinement — which is why §3 exists at all.

## 7. Cross-mission generalisation — the real limit

Trained on Kepler, evaluated zero-shot on 2,562 resolved TESS objects using
only features both catalogs express:

| | n | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| Kepler (in-domain) | 1,524 | 0.9671 | 0.9409 | 0.0670 |
| TESS (zero-shot) | 2,562 | **0.7619** | 0.6906 | **0.2223** |

ROC-AUC falls by 0.205 and the Brier score more than triples. **The calibration
does not survive the domain shift at all** — probabilities that are trustworthy
on Kepler are not trustworthy on TESS. TESS has shorter baselines, a redder
bandpass, larger pixels and therefore more blending, and a different
false-positive population.

This is the number that says what the model can actually do on new data, and it
is much less flattering than 0.98. It is reported first in the model card for
that reason.

---

## 8. How certain are these numbers?

Every comparison above is between figures with real uncertainty, so the
uncertainty is measured rather than left implicit.

**Fold-to-fold spread** accompanies each cross-validated score. The five boosted
families span 0.0035 PR-AUC while their individual standard deviations run
0.0048–0.0069 — the ordering among them carries no information. Optuna's 25-trial
search moved the winner a further 0.0011, also inside the band.

**Confidence intervals** on the held-out metrics come from a bootstrap that
resamples **host stars, not rows**, for the same reason the splits are grouped:
sibling KOIs share stellar parameters and are not independent draws, so a row
bootstrap reports an interval narrower than the data supports.

| Metric | Value | 95% CI |
|---|---|---|
| ROC-AUC | 0.9827 | 0.9764 – 0.9881 |
| PR-AUC | 0.9699 | 0.9581 – 0.9793 |

**Calibration involved a real trade**, not a default. Isotonic and sigmoid were
both fitted and scored on a third, star-disjoint slice:

| Method | Brier | Distinct probabilities (n = 1,483) |
|---|---|---|
| Isotonic (selected) | 0.0537 | 42 |
| Sigmoid | 0.0570 | 1,483 |

Isotonic calibrates better; sigmoid never ties. Isotonic's 42 levels mean 15 of
the top 50 candidates land on exactly 1.0, which is useless for ordering. So the
two jobs are split: the calibrated probability is displayed, the raw model score
does the ranking.

**Not done: nested cross-validation.** The Optuna search saw the CV folds, so
model selection is mildly optimistic. The held-out test set was never touched by
tuning, and the bootstrap interval is computed on it — but nesting the search
inside an outer loop would multiply a 20-minute run by the outer fold count,
beyond the CPU budget. Recorded here rather than passed over.

## Summary

| Issue | Where | Effect |
|---|---|---|
| Robovetter verdict columns as features | `predict.py`, `train_xgboost.py` | 0.9999 → 0.9865 |
| Row-level rather than star-level splits | `train_xgboost.py:110` | negligible here; fatal for light curves |
| Uncertainty features encoding follow-up | new design, caught and removed | −0.003, removed anyway |
| Windows from ≤15 stars, identity lost | `create_training_data.py:21` | track dropped |
| TESS `KP` mapped to Candidate | `preprocess_merge.py:149` | corrected; `APC`/`FA` no longer dropped |
| Three-class framing | throughout | replaced by binary, 0.9319 → 0.9865 |
