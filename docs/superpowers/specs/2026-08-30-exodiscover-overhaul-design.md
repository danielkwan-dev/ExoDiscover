# ExoDiscover Overhaul — Design

**Date:** 2026-08-30
**Status:** Approved, pending implementation plan
**Context:** NASA Space Apps 2025 — "A World Away: Hunting for Exoplanets with AI"

## Goal

Turn the hackathon submission into a portfolio piece that demonstrates ML/MLE
depth to a hiring reviewer. The emphasis is methodological rigor: honest
evaluation, leakage control, calibrated probabilities, and reproducible
training. Engineering quality supports that story rather than competing with it.

Constraints agreed with the project owner:

- **Compute:** CPU only, training must complete in minutes to about an hour. No
  bulk light-curve downloads from MAST.
- **Repo strategy:** restructure in place, preserve git history. No history
  rewrite (a collaborator has clones).
- **Deployment:** free-tier hosting, publicly linkable.

## Problems in the current project

Established by reading the repository, not assumed.

### Correctness and credibility

1. **The demo does not use the model.** `frontend/src/components/ExoplanetExplorer.tsx`
   imports `mockData.ts` and renders the strings "Live predictions" and
   "real-time predictions from both models." The frontend never calls the
   backend. This is the single most damaging defect.
2. **Label leakage.** `nasa_data/kepler.csv` contains `koi_score` and
   `koi_fpflag_nt/ss/co/ec`. These are the Robovetter's disposition score and
   its false-positive vetting flags — they encode the target. Any model given
   them scores near-perfectly and has learned nothing.
3. **Split leakage (tabular).** Kepler has 9,564 KOI rows spanning only 8,214
   unique stars, with up to 7 KOIs on a single star. A random row split places
   sibling KOIs from the same star on both sides of the split.
4. **Split leakage (light curve).** `train_xgboost.py:110` splits on *windows*.
   `create_training_data.py:21-23` selects targets with `.head(5)` per class, so
   all 1,500 windows originate from at most **15 stars**, ~100 overlapping
   windows each. The reported accuracy measures memorized stellar noise.
5. **Labeling bug.** `preprocess_merge.py:149` maps TESS `'KP': 1`, assigning
   *Known Planet* to the Candidate class; KP denotes a confirmed planet. The
   same map omits `APC` (462) and `FA` (98), which become NaN and are silently
   dropped.
6. **Questionable features.** `predict.py:12-15` feeds `koi_period_err1/err2`,
   `koi_time0bk`, and `koi_quarters` — measurement metadata and an observation
   bitmask, not physics.
7. **Mission-dependent missingness.** `preprocess_merge.py` populates
   `transit_depth`/`transit_duration` for Kepler and TESS but not K2 (the K2
   table lacks `pl_trandep`). The missingness pattern identifies the source
   mission, which correlates with the label distribution.

### Reproducibility

8. **`requirements.txt` cannot run the code.** It is a `pip freeze` dump. It
   omits `tensorflow` (imported by `predict.py:3`) and `lightkurve` (imported by
   `create_training_data.py:9`), while including `pygame`, `opencv-python`, and
   `torch==2.8`.
9. **README describes a repo that does not exist.** It references
   `exodiscover-backend/` (renamed to `backend/`) and `train_final_model.py`
   (absent). No metrics, screenshots, or live link.
10. **Windows-only hardcoded paths**, e.g. `predict.py:9`
    (`r"model\lighkurve_KOI_dataset.csv"`). Breaks on Linux and in CI.
11. **No tests, no CI, no packaging, no container, no deployment.** `app.py`
    serves 20 lines with `debug=True`, no validation, no error handling, and
    reloads nothing at startup.

### Repository hygiene

12. **Committed data and model binaries** total ~24.6 MB of git history: a 28 MB
    `exoTest.csv`, a 10 MB `xgb_exo_model.pkl`, 3.6 MB `.npy` arrays, and four
    raw NASA catalogs.
13. **`exoTest.csv` is unusable.** 570 stars with **5 positives**, and it is the
    Kaggle derivative, not a NASA source. The companion `exoTrain` split is
    absent.
14. **Duplicate files.** `App.jsx`+`App.tsx`, `main.jsx`+`main.tsx`,
    `vite.config.js`+`vite.config.ts`, `bun.lockb`+`package-lock.json`,
    `nasa_data/` at both root and `backend/`, `label_encoder.pkl` in two
    locations, `lighkurve_KOI_dataset.csv` in two locations (with the typo baked
    into the filename), and five model artifacts with no clear owner.
15. **Unmodified Lovable export.** `lovable-tagger` in devDependencies, package
    named `vite_react_shadcn_ts`, ~40 unused shadcn components.
16. **Branch sprawl.** No `main`; `origin/HEAD` points at `plot/graph`; both
    `ML_Pipeline` and `ML_Pipelines` exist.
17. **Fabricated social proof.** `ExoplanetExplorer.tsx:188-214` renders a
    "Community Validation" panel with invented agree/disagree vote counts.

### Worth preserving

- The four-stage light-curve preprocessing (raw → normalized → flattened →
  windowed) with its rendered plots is real signal-processing work.
- The multi-mission Kepler/K2/TESS framing is a strong narrative spine.

## Dataset facts

Measured, for reference during implementation.

| Source | Rows | Unique targets | Label distribution |
|---|---|---|---|
| Kepler KOI (`cumulative`) | 9,564 | 8,214 `kepid` | FALSE POSITIVE 4,839 · CONFIRMED 2,746 · CANDIDATE 1,979 |
| TESS TOI (`toi`) | 7,703 | 7,408 `tid` | PC 4,679 · FP 1,197 · CP 684 · KP 583 · APC 462 · FA 98 |
| K2 (`k2pandc`) | 4,004 | — | CONFIRMED 2,315 · CANDIDATE 1,374 · FALSE POSITIVE 293 · REFUTED 22 |

`koi_score` is null for 15.8% of Kepler rows.

## ML design

### Task framing — resolved by experiment, not assertion

The original trains three classes. The concern is that the CONFIRMED↔CANDIDATE
boundary is not physical: in the cumulative KOI table, CANDIDATE means the
signal *passed* Robovetter without a false-positive flag, so the difference from
CONFIRMED is whether follow-up confirmation happened. Confirmation is driven by
target brightness (radial-velocity follow-up needs bright stars), short period
(more observed transits), planet size, and multiplicity (multi-planet systems
are statistically validated in batches). Those are selection effects, and
`koi_kepmag`, period, and multiplicity are all in the feature set.

Rather than assume, train three framings and report them side by side:

- **A — Flat 3-class.** The original framing, with leakage removed and
  star-grouped splits. Provides the apples-to-apples restatement of the original
  headline number.
- **B — Binary CONFIRMED vs FALSE POSITIVE** (7,585 rows). The 1,979 CANDIDATEs
  are withheld as an unlabeled inference set and scored, producing the discovery
  feature.
- **C — Diagnostic: CONFIRMED vs CANDIDATE** among vetted signals. Inspect SHAP.
  If `koi_kepmag`, multiplicity, and period dominate, the third class
  demonstrably encodes follow-up selection. If transit-shape physics dominates
  instead, the 3-class framing is vindicated and A becomes the headline.

**Comparability.** A (3-class) and B (binary) do not share an output space, so
they are compared on the sub-decision they have in common: *planet-like vs false
positive*, obtained by collapsing A's CONFIRMED and CANDIDATE predictions into a
single class at inference time. The deciding metric is **PR-AUC on the
star-held-out test set** for that collapsed decision, with Brier score as the
tiebreaker since calibration is what the discovery ranking depends on.

**The UI presents three classes regardless.** Whichever model wins is surfaced
in the Confirmed / Candidate / False Positive vocabulary users expect — B
recovers the third class by thresholding its calibrated probability into a
"needs vetting" band, with the band edges set from the candidate score
distribution. No frontend work depends on this resolving one way.

### Leakage firewall

A `LEAKY_COLUMNS` constant in `ml/exodiscover/data/schema.py`:

```
koi_disposition, koi_pdisposition, koi_score,
koi_fpflag_nt, koi_fpflag_ss, koi_fpflag_co, koi_fpflag_ec,
koi_tce_delivname
```

Enforced two ways: a schema check that raises when any listed column reaches the
feature matrix, and a pytest that fails CI on the same condition. The guard is
part of the deliverable, not scaffolding.

### Features (~35)

Raw physical columns plus engineered terms chosen for discriminative power
against known false-positive modes:

- **Log transforms:** `log_period`, `log_depth`, `log_prad`, `log_insol`.
- **Duration consistency:** observed `koi_duration` against the duration implied
  by period and stellar density via Kepler's third law. Eclipsing binaries
  deviate.
- **Depth–radius consistency:** residual of `koi_depth` against `(Rp/R*)²`.
  Blended and giant-star false positives deviate.
- **Geometry:** `a_over_rstar`, `koi_impact` interactions.
- **Relative uncertainty:** `err1/value` ratios for period, depth, and duration.
  Uncertainty *magnitude* is legitimate signal; the raw `err` columns the
  original consumed are not.
- **Multiplicity:** count of KOIs sharing a `kepid`. Multi-planet systems are
  rarely false positives.
- **Retained raw:** `koi_model_snr`, `koi_teq`, `koi_steff`, `koi_slogg`,
  `koi_srad`, `koi_kepmag`, `koi_tce_plnt_num`.

### Models and evaluation

Ladder, each evaluated identically: `DummyClassifier` → `LogisticRegression` →
`RandomForest` → `HistGradientBoosting` → `XGBoost` → `LightGBM` → `CatBoost` →
soft-voting and stacked ensembles. Optuna tuning under `StratifiedGroupKFold(5)`
grouped on `kepid`. All CPU-bound and within budget.

Evaluation: nested cross-validation for the honest estimate, plus a star-held-out
test set. Metrics are ROC-AUC, PR-AUC, Brier score, per-class precision/recall/F1,
and precision@50. Isotonic calibration with a reliability diagram.

### The ablation table

The primary README artifact.

| Setup | Metric |
|---|---|
| With `koi_score` + fpflags, random split (original-style) | expected ≈0.99, meaningless |
| Leakage removed, random row split | inflated |
| Leakage removed, star-grouped split | **the honest number** |
| Task framing A / B / C | reported side by side |

### Cross-mission generalization

Train on Kepler physical features, evaluate zero-shot on the 2,462 labelled TESS
TOIs (CP + KP = planet; FP + FA = false positive). Report the degradation and
discuss domain shift. This directly answers the challenge's "analyze new data."

### Discovery output

Score the 1,979 CANDIDATEs with model B, publish `top_candidates.csv` ranked by
calibrated probability with per-row SHAP reasons, then pull a fresh archive
snapshot and count how many of the top-ranked have since been officially
confirmed. A checkable result rather than a leaderboard number.

### Light curve track — deliberately demoted

The CPU budget cannot support real MAST downloads, so the project will not claim
a second strong model. Instead: retain the four-stage preprocessing pipeline as
an interactive explainer, retrain the window model with **star-grouped** splits,
and publish the resulting (substantially lower) honest number alongside an
explicit statement of the 15-star limitation and what more compute would enable.
A self-identified limitation reads better than an unexplained 97%.

## Architecture

```
ExoDiscover/
├─ README.md                    rewritten: live link, honest metrics, diagram
├─ pyproject.toml               ml + api package; ruff, mypy, pytest config
├─ Makefile                     data | features | train | eval | serve | test
├─ docker-compose.yml
├─ .github/workflows/ci.yml
├─ docs/
│  ├─ MODEL_CARD.md
│  ├─ LEAKAGE.md                the writeup: leaks found, before/after numbers
│  └─ ARCHITECTURE.md
├─ data/                        gitignored; populated by the ingest script
├─ ml/exodiscover/
│  ├─ config.py                 pydantic-settings; seeds, paths
│  ├─ data/ingest.py            NASA Exoplanet Archive TAP client
│  ├─ data/schema.py            column contracts + LEAKY_COLUMNS
│  ├─ features/tabular.py
│  ├─ features/lightcurve.py    preserved pipeline, cleaned
│  ├─ models/train.py           ladder, Optuna, StratifiedGroupKFold
│  ├─ models/calibrate.py
│  ├─ evaluate.py               emits metrics.json + plots
│  ├─ explain.py                SHAP
│  └─ cli.py                    typer: exo ingest | train | eval | predict
├─ api/                         FastAPI + Pydantic
├─ web/                         cleaned Vite app (renamed from frontend/)
└─ tests/
```

`backend/` splits into `ml/` and `api/`; `frontend/` becomes `web/`. All moves
via `git mv` to preserve history.

### Data ingestion

One `ingest.py` against the NASA Exoplanet Archive TAP API (`cumulative`, `toi`,
`k2pandc`), caching to `data/raw/` and recording the query text and fetch
timestamp for reproducibility. Replaces `download_data.py`,
`download_training_data.py`, `preprocess_merge.py`, and the committed CSVs.

### API

Replaces the 20-line Flask app. FastAPI with Pydantic models, the classifier
loaded once via a lifespan handler, CORS from config, structured logging, real
error handling, and auto-generated OpenAPI docs.

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness |
| `GET /model-info` | model version, training date, headline metrics |
| `POST /predict` | one KOI → class, calibrated probability, SHAP contributions |
| `POST /predict/batch` | CSV upload (size-capped) → ranked results |
| `GET /metrics` | serves `metrics.json` for the dashboard |
| `POST /lightcurve/preprocess` | four-stage arrays for the explainer |

### Web

`mockData.ts` is deleted and every screen calls the API.

- **Predict** — parameter form → predicted class, calibrated probability, SHAP
  waterfall.
- **Batch** — CSV upload → ranked table.
- **Discoveries** — the top-ranked unvetted candidates with SHAP reasons.
- **Model** — dashboard rendered from `metrics.json`: confusion matrix, PR
  curve, reliability diagram, ablation table, feature importance.
- **How it works** — the light-curve preprocessing stages.

The fabricated "Community Validation" panel is removed. Unused shadcn components
and `lovable-tagger` are removed; the package is renamed.

### Testing and CI

- **pytest:** leakage guard (fails if a leaky column reaches the matrix), schema
  contract, physics feature math unit-tested against known Kepler-10 values,
  group-split correctness (asserts no `kepid` spans train and test), and API
  contract tests via `TestClient`.
- **vitest:** a small set of component and API-client tests.
- **GitHub Actions:** ruff, mypy, pytest, vitest, web build, and a smoke train.
  Because `data/` is gitignored, the smoke train and the feature tests run
  against a committed fixture — `tests/fixtures/koi_sample.csv`, roughly 300 KOI
  rows sampled to preserve both the class balance and the multi-KOI-per-star
  structure the group-split test depends on. CI never contacts the NASA archive.

### Deployment

API to Hugging Face Spaces (Docker SDK — free, and unlike Render's free tier it
is not evicted). Web to Vercel. The production model artifact (~2–5 MB) and
`metrics.json` are committed as the single versioned deliverable; all other
artifacts are gitignored.

## Deletions

`test_data/exoTest.csv` (28 MB, unusable); `xgb_exo_model.pkl`,
`xgb_wen_model.pkl`, `trained_cnn.pth`, `lightcurve_model.h5`, both
`label_encoder.pkl`, `koi_catalog.csv`, both `lighkurve_KOI_dataset.csv`; the
stale root `nasa_data/`; `merged_exoplanets.csv`; `App.jsx`, `main.jsx`,
`vite.config.js`, `bun.lockb`; ~40 unused shadcn components; `mockData.ts`; and
the superseded scripts `download_data.py`, `download_training_data.py`,
`create_lightcurve_features.py`, `train_lightcurve_xgboost.py`,
`predict_tabular.py`, `test_lightcurve_accuracy.py`, `preprocess_merge.py`.

The rendered preprocessing plots in `lightcurve_data/*.png` are regenerated into
`docs/assets/` rather than deleted.

### Retained and rewritten

For the avoidance of doubt, these are not deleted — they move and are rewritten:

| Current | Becomes |
|---|---|
| `lightcurve_preprocessing.py` | `ml/exodiscover/features/lightcurve.py` |
| `create_training_data.py` | `ml/exodiscover/data/lightcurve_windows.py` — the `.head(5)` target selection is replaced by sampling across as many distinct stars as the cached window set supports, and star IDs are persisted alongside the windows so the grouped split is possible at all |
| `train_xgboost.py`, `train_cnn.py` | folded into `ml/exodiscover/models/train.py` |
| `model/cnn_model.py`, `model/lstm_model.py` | `ml/exodiscover/models/architectures.py`, retained only if the demoted light-curve track uses them; dropped otherwise |
| `predict.py`, `predict_tabular.py` | `ml/exodiscover/cli.py` + `api/service.py` |
| `app.py` | `api/main.py` |
| `test_api.py` | real pytest cases under `tests/api/` |

`training_data/windows.npy` and `labels.npy` are retained locally as the cached
light-curve inputs (the CPU budget forbids re-downloading them) but are removed
from tracking and documented in `data/README.md`.

Because history is preserved by agreement, the existing blobs remain in git.
`.gitignore` is extended to cover `data/`, `*.npy`, and `artifacts/` so nothing
further accumulates.

### Git hygiene

Create `main` and set it as the default branch. Delete the stale remote branches
`ML_Pipeline`, `ML_Pipelines`, `frontEnd`, `plot/graph`, and `gh-pages` after
confirming with the owner, since a collaborator holds clones.

## Explicitly out of scope

MLflow and Weights & Biases (a versioned `metrics.json` plus an `artifacts/`
directory is sufficient and simpler to defend); DVC; authentication; any
database. The community-voting feature is removed rather than implemented.

## Risks

- **Cross-mission transfer may perform poorly.** That is a reportable result,
  not a failure; the writeup covers domain shift either way.
- **The honest accuracy will be materially lower than the current claim.** This
  is the intended outcome and is the substance of `docs/LEAKAGE.md`.
- **Free-tier cold starts.** Mitigated with a warm-up ping on page load and
  honest loading states.
- **Branch deletion affects a collaborator.** Gated on explicit confirmation
  before any remote deletion.
