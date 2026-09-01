# ExoDiscover

Finding exoplanets in NASA's Kepler catalog — and being honest about how well it
works.

Built for [NASA Space Apps 2025, "A World Away: Hunting for Exoplanets with
AI"](https://www.spaceappschallenge.org/2025/challenges/a-world-away-hunting-for-exoplanets-with-ai/),
then rebuilt with the methodology the hackathon version skipped.

**Trained on Kepler. Tested on TESS.** The model learns from one telescope's
catalog and is then scored on a different telescope's — objects it has never
seen, from a mission with a different bandpass, shorter baselines, and larger
pixels. That is the challenge's "analyse new data", and it is the number this
project leads with.

```
77% accuracy zero-shot on TESS      (majority-class baseline 51%)   ← the honest number
90% on held-out Kepler stars        (majority-class baseline 63%)   ← same features, same model
ROC-AUC 0.965 → 0.838                                               ← the cost of changing telescope
```

Accuracy alone would overstate that gap — Kepler is 63% false positives while
TESS is nearly balanced, so the baselines differ by 12 points. ROC-AUC is
base-rate free, and it puts the real degradation at 0.13. The ranking survives
the domain shift; the calibration does not, with the Brier score more than
doubling.

Both rows above use the **11 features the two catalogs share**, so the
comparison is like-for-like. The model that actually ships uses all 17 and
scores **0.984 ROC-AUC / 93.6% accuracy** on held-out Kepler stars — reported
in full below, because a cross-mission result means nothing without the
in-domain one beside it.

Runs locally, CPU only — [setup below](#run-it). Nothing is hosted: the API and
the UI each start with one command.

---

## The short version

The hackathon submission reported strong accuracy. While rebuilding it I found
the number was measuring the wrong thing, and the interesting part of this
project is the correction.

**The Kepler catalog ships the answer key.** `koi_score` is the automated
vetting pipeline's own confidence in its verdict, and four `koi_fpflag_*`
columns are its individual false-positive decisions. Feed them to a model and it
reaches **ROC-AUC 0.9999** — it has learned to read the label, not the physics.
Remove them and the honest figure is 0.9840.

They are now quarantined in one place, dropped on the way into feature
construction and asserted absent on the way out, with a test that fails the
build if one ever gets through.

That check compares column *names*, which cannot catch a leaky column
reintroduced under a different one — so a second guard reads the values,
flagging any feature whose rank correlation with a Robovetter column exceeds
0.80. The threshold is measured, not chosen: the strongest legitimate pairing
in the catalog is 0.553, while `koi_score` renamed scores 1.000 and a diluted
copy still scores 0.885.

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
random split. Measured, it moves the result by less than noise (0.9846 random →
0.9840 grouped) — and in the direction that makes the honest number *lower*, by
0.0006. 21% of rows have a sibling anywhere in the data. Grouping is kept
because it is correct, not because it rescued the number.

**Model choice turned out not to matter either.** Every family was scored under
the same grouped CV, with the fold-to-fold spread reported next to the mean:

| Model | PR-AUC |
|---|---|
| Soft-vote ensemble | 0.9694 ± 0.0056 |
| XGBoost | 0.9687 ± 0.0057 |
| CatBoost | 0.9686 ± 0.0053 |
| LightGBM | 0.9666 ± 0.0064 |
| HistGradientBoosting | 0.9664 ± 0.0061 |
| Random forest | 0.9614 ± 0.0073 |
| Logistic regression | 0.8525 ± 0.0226 |
| Dummy (prior) | 0.3620 ± 0.0107 |

The six tree-based rows span 0.0080 — **about one standard deviation**, and the
top five span 0.0030. Optuna moved XGBoost another 0.0002 over 40 trials, well
inside the band. The honest reading is that this problem is won by the features
and the evaluation protocol, not by the model; reporting "soft-vote, 0.969"
without that ± column would be over-reading the table.

## The ablation

| Setup | ROC-AUC | What changed |
|---|---|---|
| Robovetter columns present, random split | **0.9999** | reproduces the original methodology |
| Leakage removed, random split | 0.9846 | the label columns quarantined |
| Leakage removed, stars held out | **0.9840** | the honest number |

## Is 0.98 too good?

It is high enough to be suspicious, so it was checked rather than reported.

`log_prad` — implied planet radius — dominates the model. That turns out to be
real physics: Kepler's false positives are mostly eclipsing binaries, whose
median implied radius is 8.97 R⊕ against 2.16 R⊕ for confirmed planets, topping
out at **200,346 R⊕**. Nothing that large is a planet. 99.8% of objects above
40 R⊕ are false positives.

Two checks confirm no single feature is load-bearing: `log_prad` alone reaches
only 0.838, and *removing* it entirely still leaves 0.985 — the signal is spread
across depth, duration consistency, and multiplicity.

**And it is not overfitting**, which is the other thing a high score usually
means. Overfitting is a gap, not a level, so every run records one:

| | ROC-AUC |
|---|---|
| On the rows it trained on | 0.9999 |
| On 1,992 held-out stars | 0.9854 |
| **Generalisation gap** | **0.0145** |

A model that memorised its training set scores ~1.0 on the left and falls away
sharply on the right. This one also barely notices being starved: trained on 10%
of the stars — 519 rows — it still reaches 0.9764 against 0.9854 on all 5,287,
and the curve is flat past half the data. The ceiling belongs to the task, not
to the model. Both the gap and the full learning curve are written to
`docs/metrics/metrics.json` on every run.

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
| Kepler (in-domain) | 2,298 | 0.9653 | 0.0694 |
| TESS (zero-shot) | 2,562 | **0.8376** | **0.1771** |

ROC-AUC drops 0.128 and the Brier score more than doubles: the *ranking* largely
survives the domain shift, the *calibration* does not. TESS has shorter
baselines, a redder bandpass, and larger pixels, so its false-positive
population is different.

This number is also the clearest evidence for one of the fixes above. TESS
frames do not carry several KOI columns at all, and an earlier version of the
feature builder filled those gaps with Kepler medians — fabricated values that
actively misled the model on another mission's data. Letting them stay missing,
for the boosted models to route down a learned branch, moved zero-shot ROC-AUC
from 0.762 to 0.838.

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

A useful sanity signal falls out of it — median `duration_ratio` is 1.042 for
confirmed planets and 1.429 for false positives. Real planets sit closest to
what the physics predicts.

## Run it

Everything runs on localhost; nothing is deployed. The trained model and its
metrics are committed, so **you do not need to download a catalog or train
anything** to see it working.

**Prerequisites:** Python 3.11 or 3.12, Node 20+.

```bash
make install     # pip install -e ".[dev,api]", then npm install in web/
```

Then two terminals:

```bash
make serve       # FastAPI on :8000  — interactive docs at http://localhost:8000/docs
make web         # React UI on :5173 — points at :8000 by default
```

Open <http://localhost:5173>. Set `VITE_API_URL` if you need the UI to reach the
API somewhere other than `localhost:8000`.

Without `make` (Windows, or no GNU make installed):

```bash
pip install -e ".[dev,api]"
cd web && npm install && cd ..
uvicorn api.main:app --reload --port 8000     # terminal 1
cd web && npm run dev                         # terminal 2
```

Confirm it is up without touching the UI:

```bash
curl http://localhost:8000/health      # {"status":"ok","model_loaded":true}
exo eval                               # prints the stored held-out metrics
```

Or run the whole stack in containers with `docker compose up --build`.

### Regenerating the model

Only needed to reproduce the artifacts rather than use the committed ones:

```bash
exo ingest              # NASA archive -> data/raw/, recording query + timestamp
exo train --trials 40   # ~25 min, CPU only  (exo train --fast skips Optuna, ~6 min)
```

`exo train` rewrites `models/production/model.joblib`,
`docs/metrics/metrics.json`, the three figures, and `top_candidates.csv`, and
prints the generalisation gap and leakage-guard verdict as it goes.

### Checks

```bash
make test        # pytest (offline, against committed fixtures) + vitest
make lint        # ruff, mypy, eslint
```

## Layout

```
ml/exodiscover/    ingest · leakage firewall · physics features · training · evaluation
api/               FastAPI: typed prediction, batch CSV, SHAP, metrics
web/               React UI — every screen calls the API, no mock data
docs/              LEAKAGE.md · MODEL_CARD.md · metrics/
tests/             115 tests, offline against committed fixtures
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
