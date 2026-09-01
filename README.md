# ExoDiscover

Training a planet classifier on one telescope and testing it on another — and
being honest about what survives the trip.

Built for [NASA Space Apps 2025, "A World Away: Hunting for Exoplanets with
AI"](https://www.spaceappschallenge.org/2025/challenges/a-world-away-hunting-for-exoplanets-with-ai/),
then rebuilt with the methodology the hackathon version skipped.

**Trained on Kepler. Tested on TESS.** The model learns from one telescope's
catalog and is scored on a different telescope's — objects it has never seen,
from a mission with a different bandpass, shorter baselines, and larger pixels.
That is the challenge's "analyse new data", and it is what this project reports.

```
77% accuracy zero-shot on TESS      (majority-class baseline 51%)
ROC-AUC 0.838 · Brier 0.177 · n = 2,562 resolved TESS objects
```

The ranking survives the change of telescope; the calibration does not — the
Brier score more than doubles off-domain, so the ordering stays useful while the
probabilities stop being trustworthy.

Runs locally, CPU only — [setup below](#run-it). Nothing is hosted: the API and
the UI each start with one command.

---

## Getting the training data honest first

A cross-mission result is only worth reporting if the model behind it was built
cleanly. The hackathon submission reported strong accuracy; while rebuilding it
I found the number was measuring the wrong thing, and the correction is the
substance of this project.

**The Kepler catalog ships the answer key.** `koi_score` is the automated
vetting pipeline's own confidence in its verdict, and four `koi_fpflag_*`
columns are its individual false-positive decisions. Feed them to a model and it
reaches **ROC-AUC 0.9999** — it has learned to read the label, not the physics.

They are now quarantined in one place, dropped on the way into feature
construction and asserted absent on the way out, with a test that fails the
build if one ever gets through.

That check compares column *names*, which cannot catch a leaky column
reintroduced under a different one — so a second guard reads the values,
flagging any feature whose rank correlation with a Robovetter column exceeds
0.80. The threshold is measured, not chosen: the strongest legitimate pairing in
the catalog is 0.553, while `koi_score` renamed scores 1.000 and a diluted copy
still scores 0.885.

| Setup | ROC-AUC | What changed |
|---|---|---|
| Robovetter columns present, random split | **0.9999** | reproduces the original methodology |
| Leakage removed, random split | 0.9846 | the label columns quarantined |
| Leakage removed, stars held out | 0.9840 | the training protocol used here |

**Two more problems, found by measuring rather than assuming:**

- Three uncertainty features I designed in myself turned out to encode *when a
  planet got confirmed* rather than what it is — a confirmed planet's parameters
  were tightened by the very follow-up that confirmed it. Removing them cost
  0.003 ROC-AUC. [Removed anyway.](docs/LEAKAGE.md#3-temporal-leakage--the-subtle-one)
- The light-curve CNN cannot be honestly evaluated at all: its training windows
  come from at most 15 stars and the cached arrays lost star identity, so no
  leakage-free split exists. Rather than publish an unverifiable number, [that
  model was dropped](docs/LEAKAGE.md#4-the-light-curve-track-cannot-be-evaluated).

**And one thing I got wrong.** I expected star-level split leakage to be a major
inflation source — sibling KOIs sharing a host star landing on both sides of a
random split. Measured, it moves the result by less than noise (0.9846 random →
0.9840 grouped), and in the direction that makes the honest number *lower*, by
0.0006. 21% of rows have a sibling anywhere in the data. Every split in the
project is grouped on `kepid` because it is correct, not because it rescued the
number.

## Cross-mission: the result

Trained on 7,585 resolved Kepler KOIs, evaluated zero-shot on 2,562 resolved
TESS objects, using only the 11 features both catalogs express. TESS has no
equivalent for Kepler-only quantities like signal-to-noise or KOI multiplicity,
so those are dropped from both sides and the comparison is like-for-like.

| | n | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| Kepler (in-domain reference) | 2,298 | 0.9653 | 0.9345 | 0.0694 |
| **TESS (zero-shot)** | 2,562 | **0.8376** | 0.8031 | **0.1771** |

**Zero-shot accuracy is 77.0%, against a majority-class baseline of 50.6%.**

That is the one accuracy this project quotes, and deliberately so: TESS is close
to class-balanced, so the number carries information. Kepler's held-out slice is
63% false positives, where predicting the majority class alone scores 0.631 —
an in-domain accuracy would flatter without informing, and the two could not be
compared. ROC-AUC is base-rate free, which is why the table above uses it.

```
confusion, threshold 0.5      predicted FP   predicted planet
  actual false positive             975              320
  actual planet                     270              997
```

ROC-AUC falls 0.128 and the Brier score more than doubles. **The ranking largely
survives the domain shift; the calibration does not.** Probabilities trustworthy
on Kepler are not trustworthy on TESS, which has shorter baselines, a redder
bandpass, larger pixels and therefore more blending — a genuinely different
false-positive population.

This number is also the clearest evidence for one of the fixes above. TESS
frames do not carry several KOI columns at all, and an earlier version of the
feature builder filled those gaps with *Kepler* medians — fabricated values
asserting measurements TESS never made. Letting them stay missing, for the
boosted models to route down a learned branch, moved zero-shot ROC-AUC from
**0.762 to 0.838** with no change to the model itself. Imputing across a domain
boundary was quietly costing 0.076.

## Physics, not just columns

What transfers between missions is physics; what does not is instrument
character. Three features are derived rather than read off:

- **`duration_ratio`** — observed transit duration over the duration Kepler's
  third law predicts from the period and the star's density. Eclipsing binaries
  break it.
- **`depth_ratio`** — observed depth over `(Rp/R*)²`. Blends and giants break it.
- **`rho_star`** — mean stellar density from surface gravity and radius.

These are unit-tested against Kepler-10 b's published values: the duration
formula returns 1.804 h against a measured 1.811 h. A wrong constant fails the
build rather than quietly degrading the model.

Two sanity signals fall out. Median `duration_ratio` is 1.042 for confirmed
planets against 1.429 for false positives — real planets sit closest to what the
physics predicts. And the false-positive population is dominated by eclipsing
binaries whose implied planet radius is impossible: median 8.97 R⊕ against
2.16 R⊕ for confirmed planets, topping out at 200,346 R⊕, with 99.8% of objects
above 40 R⊕ being false positives. That is why the in-domain task is easier than
the cross-mission one, and why the drop above is the number worth reporting.

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
exo eval                               # prints the stored metrics
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
prints the leakage-guard verdict as it goes.

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
tests/             116 tests, offline against committed fixtures
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

- **Close the TESS gap** — domain adaptation, or per-mission calibration, so the
  probabilities mean something off-domain. The ranking already transfers; the
  calibration is what breaks.
- **Fix the light-curve track properly** — re-download from MAST preserving star
  identity, then train a 1D-CNN on phase-folded local/global views with grouped
  splits. The current data cannot support it.
- **Test against a third mission (K2)** to see whether the drop is TESS-specific
  or a general cost of changing instrument.
