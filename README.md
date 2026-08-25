# ExoDiscover

Finding exoplanets in NASA's Kepler catalog — and being honest about how well it
works.

Built for [NASA Space Apps 2025, "A World Away: Hunting for Exoplanets with
AI"](https://www.spaceappschallenge.org/2025/challenges/a-world-away-hunting-for-exoplanets-with-ai/),
then rebuilt with the methodology the hackathon version skipped.

<!-- Add once deployed: **[Live demo](https://…)** · **[API docs](https://…/docs)** -->

```
ROC-AUC 0.984 on held-out stars          ← the flattering number
ROC-AUC 0.762 zero-shot on TESS          ← what it does on genuinely new data
```

Both are reported, in that order, everywhere in this repo.

---

## The short version

The hackathon submission reported strong accuracy. While rebuilding it I found
the number was measuring the wrong thing, and the interesting part of this
project is the correction.

**The Kepler catalog ships the answer key.** `koi_score` is the automated
vetting pipeline's own confidence in its verdict, and four `koi_fpflag_*`
columns are its individual false-positive decisions. Feed them to a model and it
reaches **ROC-AUC 0.9999** — it has learned to read the label, not the physics.
Remove them and the honest figure is 0.9865.

They are now quarantined in one place, dropped on the way into feature
construction and asserted absent on the way out, with a test that fails the
build if one ever gets through.

**Two more problems, found by measuring rather than assuming:**

- Three uncertainty features I designed in myself turned out to encode *when a
  planet got confirmed* rather than what it is — a confirmed planet's parameters
  were tightened by the very follow-up that confirmed it. Removing them cost
  0.003 ROC-AUC. [Removed anyway.](docs/LEAKAGE.md#3-temporal-leakage--the-subtle-one)
- The light-curve CNN cannot be honestly evaluated at all: its training windows
  come from at most 15 stars and the cached arrays lost star identity, so no
  leakage-free split exists. Rather than publish an unverifiable number, [that
  model was dropped](docs/LEAKAGE.md#5-the-light-curve-track-cannot-be-evaluated).

**And one thing I got wrong.** I expected star-level split leakage to be a major
inflation source — sibling KOIs sharing a host star landing on both sides of a
random split. Measured, it moves the result by less than noise (0.9860 → 0.9865),
because only ~12% of rows have a sibling anywhere in the data. Grouping is kept
because it is correct, not because it rescued the number.

## The ablation

| Setup | ROC-AUC | What changed |
|---|---|---|
| Robovetter columns present, random split | **0.9999** | reproduces the original methodology |
| Leakage removed, random split | 0.9860 | the label columns quarantined |
| Leakage removed, stars held out | **0.9865** | the honest number |

## Is 0.98 too good?

It is high enough to be suspicious, so it was checked rather than reported.

`log_prad` — implied planet radius — dominates the model. That turns out to be
real physics: Kepler's false positives are mostly eclipsing binaries, whose
median implied radius is 8.97 R⊕ against 2.16 R⊕ for confirmed planets, topping
out at **200,346 R⊕**. Nothing that large is a planet. 99.8% of objects above
40 R⊕ are false positives.

Two checks confirm no single feature is load-bearing: `log_prad` alone reaches
only 0.829, and *removing* it entirely still leaves 0.989 — the signal is spread
across depth, duration consistency, and multiplicity.

So the score is real, but the task is easier than it sounds. This model is good
at *planet vs eclipsing binary*. The TESS result below is where the limits show.

Because a real score can legitimately be high, the leakage guard in CI does not
test accuracy against a fixed threshold. It checks whether the clean pipeline
has closed the gap on the deliberately-leaky one — which is what leakage would
actually look like.

## Cross-mission: the number that matters

Trained on Kepler, evaluated zero-shot on 2,562 resolved TESS objects using only
features both catalogs express:

| | n | ROC-AUC | Brier |
|---|---|---|---|
| Kepler (in-domain) | 1,524 | 0.9671 | 0.0670 |
| TESS (zero-shot) | 2,562 | **0.7619** | **0.2223** |

ROC-AUC drops 0.205 and the Brier score more than triples — the calibration does
not survive the domain shift at all. TESS has shorter baselines, a redder
bandpass, and larger pixels, so its false-positive population is different.

This is what the challenge actually asked for — analyse *new* data — and it is
the honest answer.

## The output: a candidate shortlist

The classifier trains only on resolved dispositions, so the catalog's **1,979
unvetted CANDIDATE rows are genuinely unseen**. Scoring and ranking them is the
product: `docs/metrics/top_candidates.csv`, each row carrying the SHAP
contributions that produced it. Precision@50 on the held-out set is 1.000.

`validate_against_snapshot()` closes the loop — re-fetch the archive later and
count how many of the shortlist have since been officially confirmed.

## Physics, not just columns

Three features are derived rather than read off:

- **`duration_ratio`** — observed transit duration over the duration Kepler's
  third law predicts from the period and the star's density. Eclipsing binaries
  break it.
- **`depth_ratio`** — observed depth over `(Rp/R*)²`. Blends and giants break it.
- **`rho_star`** — mean stellar density from surface gravity and radius.

These are unit-tested against Kepler-10 b's published values: the duration
formula returns 1.804 h against a measured 1.811 h. A wrong constant fails the
build rather than quietly degrading the model.

A useful sanity signal falls out of it — median `duration_ratio` is 1.045 for
confirmed planets and 1.090 for false positives. Real planets sit closest to
what the physics predicts.

## Run it

```bash
make install
exo ingest      # NASA archive -> data/raw/, recording query + timestamp
exo train       # ~20 min, CPU only
make serve      # API on :8000, docs at /docs
make web        # UI on :5173
```

Or `docker compose up --build`.

## Layout

```
ml/exodiscover/    ingest · leakage firewall · physics features · training · evaluation
api/               FastAPI: typed prediction, batch CSV, SHAP, metrics
web/               React UI — every screen calls the API, no mock data
docs/              LEAKAGE.md · MODEL_CARD.md · metrics/
tests/             80 tests, offline against committed fixtures
```

**Start with [`docs/LEAKAGE.md`](docs/LEAKAGE.md)** — it is the substance of the
project. [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) records intended use and
limitations.

## Stack

Python 3.11 · scikit-learn, CatBoost/XGBoost/LightGBM, Optuna, SHAP · FastAPI ·
React, TypeScript, Vite, Tailwind, Recharts · pytest, vitest, ruff, mypy,
GitHub Actions.

CPU only, start to finish.

## What I'd do next

- **Fix the light-curve track properly** — re-download from MAST preserving star
  identity, then train a 1D-CNN on phase-folded local/global views with grouped
  splits. The current data cannot support it.
- **Close the TESS gap** — domain adaptation, or per-mission calibration, so the
  probabilities mean something off-domain.
- **Validate the shortlist against a fresh snapshot** and report the hit rate.
