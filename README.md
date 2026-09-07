# MANGAI — AI + Geospatial Intelligence Platform

**SIH Problem Statement ID:** 26009
**Organization:** Ministry of Steel / MOIL Ltd.

MANGAI is a modular mining decision-support platform for manganese reserve discovery, production forecasting, and mine decision support. It combines geological, borehole, terrain, and satellite data with machine learning to deliver actionable intelligence.

---

## ⚠️ Important Notice

**This prototype uses synthetic/demo data.** All metrics, predictions, and recommendations are generated from synthetic datasets and are **NOT MOIL field-validated**. MANGAI is a decision-support prototype until validated with real MOIL data, domain experts, applicable mining regulations, and operational systems.

---

## Architecture

```
Frontend (React + TypeScript + Vite + Tailwind)
    ↓
FastAPI Backend
    ↓
Service Layer
    ├── Reserve Intelligence (Prospectivity, Grade, Thickness, Resource Potential)
    ├── Production Intelligence (Forecasting, Shortfall Risk)
    ├── Operations Analytics (Equipment, Weather, Blasting)
    ├── Recommendation Engine (Evidence-backed actions)
    └── ML Inference (XGBoost models)
    ↓
PostgreSQL Database + Model Registry
```

---

## Features

### Reserve Intelligence
- **Manganese Prospectivity**: Spatial probability mapping using XGBoost with geological + satellite features
- **Mn-Grade Prediction**: Regression model for manganese percentage estimation
- **Ore Thickness Prediction**: Regression model for ore thickness in meters
- **Prototype Resource Potential**: Monte Carlo-based tonnage estimation with P10/P50/P90 uncertainty
- **SHAP Explanations**: Feature contribution analysis for model interpretability

### Production Intelligence
- **Production Forecasting**: XGBoost-based daily production forecasting with configurable horizons (1/7/30 days)
- **Shortfall Probability**: Calibrated probability of production falling below target
- **Risk Severity**: Classification into LOW/MEDIUM/HIGH/CRITICAL categories
- **Top Driver Attribution**: SHAP-based identification of key production drivers
- **Prediction Intervals**: P10/P50/P90 confidence intervals

### Operations Analytics
- **Equipment Health**: Availability, utilization, downtime ranking, maintenance trends
- **Weather Impact**: Rainfall, soil moisture, temperature monitoring with risk assessment
- **Blasting Analysis**: Delay tracking, trend analysis, weather overlap risk

### Recommendation Engine
- **Evidence-Backed Actions**: Structured recommendations with confidence scores
- **Impact Estimation**: Quantified production recovery estimates
- **Simulation**: What-if analysis for downtime reduction and blast scheduling
- **Safety Boundaries**: All recommendations require human approval

---

## Tech Stack

### Backend
- Python 3.11+
- FastAPI + Pydantic v2
- SQLAlchemy 2 + PostgreSQL + Alembic
- XGBoost + scikit-learn + SHAP
- pandas + numpy

### Frontend
- React + TypeScript + Vite
- Tailwind CSS
- MapLibre GL JS (GIS)
- Recharts (charts)
- Lucide React (icons)

### Infrastructure
- Docker Compose
- pytest + httpx (testing)
- ruff (linting)

---

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL 16+ (only required for `DATA_MODE=live`; not needed for demo mode)

### Local Development

These steps assume you are in the project root (`MANGAI/`).

#### 1. Create virtual environment

```bash
python -m venv backend/.venv
source backend/.venv/bin/activate        # Linux / macOS
# Windows (PowerShell):
# backend\.venv\Scripts\Activate.ps1
```

#### 2. Install dependencies

```bash
pip install -r backend/requirements.txt
```

#### 3. Configure environment

```bash
cp .env.example .env
```

By default `DATA_MODE=demo`, which requires no external services. See [Configuration](#configuration) for `DATA_MODE=live`.

#### 4. Seed demo data

```bash
python scripts/seed_demo.py --skip-train
```

This generates the offline synthetic datasets and initializes the database. Use `--skip-train` to avoid training ML models (models are optional in demo mode).

#### 5. Run migrations

```bash
alembic upgrade head
```

Note: `scripts/seed_demo.py` already runs migrations. Run this step directly when applying migrations without reseeding.

#### 6. Start backend

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is available at:
- API base: http://localhost:8000
- API docs (Swagger): http://localhost:8000/docs
- Health: http://localhost:8000/health
- Readiness: http://localhost:8000/ready

#### 7. Start frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend: http://localhost:5173

#### 8. Run tests

```bash
pytest
```

Run a single suite:

```bash
pytest tests/backend -q
pytest tests/ml -q
pytest tests/integration -q
```

Lint:

```bash
ruff check .
```

---

## Docker

Build and start all services (PostgreSQL, backend, frontend):

```bash
docker compose up --build
```

Services:
- PostgreSQL: port 5432
- Backend API: port 8000
- Frontend: port 8080

The backend container seeds demo data on startup, so `DATA_MODE=demo` works without internet access. To rebuild from scratch:

```bash
docker compose down -v
docker compose up --build
```

### Overview
- `GET /api/v1/overview` - Executive KPIs

### Reserve Intelligence
- `GET /api/v1/reserves/prospectivity` - Prospectivity cells (supports bbox filtering)
- `GET /api/v1/reserves/summary` - Reserve summary with resource potential
- `GET /api/v1/reserves/{reserve_id}` - Cell detail with explanations
- `GET /api/v1/reserves/boreholes` - Borehole data
- `POST /api/v1/predictions/reserve` - On-demand reserve prediction

### Production Intelligence
- `GET /api/v1/production/forecast` - Production forecast with horizon
- `GET /api/v1/production/risk` - Shortfall risk assessment
- `GET /api/v1/production/history` - Historical production data
- `POST /api/v1/predictions/production` - On-demand production prediction

### Operations
- `GET /api/v1/equipment` - Fleet status and analytics
- `GET /api/v1/weather` - Weather observations and risk
- `GET /api/v1/blasting` - Blasting schedule and delays

### Recommendations
- `GET /api/v1/recommendations` - Ranked corrective actions
- `POST /api/v1/recommendations/simulate` - What-if simulation

### MLOps
- `GET /api/v1/models` - Model registry
- `GET /api/v1/data-quality` - Data quality report

---

## Machine Learning

### Reserve Models
- **Prospectivity**: XGBoost classifier with spatial block validation
- **Grade**: XGBoost regressor for Mn percentage
- **Thickness**: XGBoost regressor for ore thickness
- **Resource Potential**: Monte Carlo simulation with uncertainty

### Production Models
- **Forecast**: XGBoost regressor with chronological validation
- **Shortfall**: Calibrated probability model

### Validation Strategy
- **Geological tasks**: Spatial block holdout (prevents spatial leakage)
- **Production tasks**: Chronological train/validation/test split
- **Metrics**: ROC-AUC, PR-AUC, F1, MAE, RMSE, R²

---

## Database Schema

Key entities:
- `mine_sites` - Mine locations and boundaries
- `geological_samples` - Surface/subsurface samples
- `boreholes` - Borehole intervals and assays
- `satellite_observations` - Remote sensing data
- `weather_observations` - Weather measurements
- `equipment` - Fleet assets
- `equipment_events` - Equipment telemetry
- `blasting_events` - Blasting schedule and delays
- `production_records` - Daily production data
- `model_versions` - ML model registry
- `predictions` - Model predictions
- `recommendations` - Generated recommendations
- `data_quality_runs` - Data quality reports

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/backend/ -v
python -m pytest tests/ml/ -v
python -m pytest tests/integration/ -v

# Run smoke tests
python scripts/run_smoke_tests.py
```

---

## Project Structure

```
MANGAI/
├── backend/
│   └── app/
│       ├── main.py              # FastAPI application
│       ├── core/                # Config, logging, security
│       ├── db/                  # Database models and session
│       ├── schemas/             # Pydantic schemas
│       ├── api/v1/              # API routes
│       ├── services/            # Business logic
│       ├── repositories/        # Data access
│       └── adapters/            # External data providers
├── ml/
│   ├── common/                  # Shared ML utilities
│   ├── reserve/                 # Reserve intelligence models
│   ├── production/              # Production intelligence models
│   └── risk/                    # Risk models
├── frontend/
│   └── src/
│       ├── api/                 # API client
│       ├── components/          # Reusable components
│       ├── pages/               # Page components
│       ├── hooks/               # React hooks
│       ├── types/               # TypeScript types
│       └── utils/               # Utilities
├── data/
│   ├── synthetic/               # Demo datasets
│   ├── processed/               # Processed features
│   └── schemas/                 # Data contracts
├── scripts/                     # Utility scripts
├── tests/                       # Test suites
├── models/                      # Trained model artifacts
├── alembic/                     # Database migrations
└── docs/                        # Documentation
---

## Configuration

Key environment variables (see `.env.example` for full list):

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_ENV` | Environment (development/production) | development |
| `DATABASE_URL` | Database connection string | sqlite:///./mangai_dev.db |
| `DATA_MODE` | Data mode: `demo` or `live` | demo |
| `MODEL_DIR` | Path to model artifacts | models |
| `DATA_DIR` | Path to data directory | data |
| `CORS_ORIGINS` | Comma-separated allowed CORS origins | http://localhost:5173,http://127.0.0.1:5173 |
| `LOG_LEVEL` | Logging level (DEBUG/INFO/WARNING/ERROR) | INFO |

### DATA_MODE=demo

Default mode. Uses synthetic datasets shipped in `data/synthetic/`. No external services, internet connection, or ML model artifacts are required. The database defaults to a local SQLite file. This is the recommended mode for local development and demos.

### DATA_MODE=live

Production-intended mode. Requires:
- A reachable PostgreSQL database (`DATABASE_URL`)
- Trained ML model artifacts under `MODEL_DIR`
- Real operational data

In live mode, endpoints that need ML inference return `503 MODEL_UNAVAILABLE` when model artifacts are missing, instead of falling back to demo heuristics.

### Database configuration

- **Demo / local development:** SQLite (default). Set `DATABASE_URL=sqlite:///./mangai_dev.db`.
- **Docker / production:** PostgreSQL. Set `DATABASE_URL=postgresql+psycopg://mangai:<password>@postgres:5432/mangai`.

Apply migrations with:

```bash
alembic upgrade head
```

### Health & readiness endpoints

- `GET /health` — Lightweight liveness check. Returns `{"status": "healthy", "service": "mangai-api"}`.
- `GET /ready` — Readiness check verifying database and dependencies. Returns structured status:

```json
{
  "status": "ready",
  "database": true,
  "data_mode": "demo",
  "models": {
    "reserve_prospectivity": true,
    "production_forecast": true
  }
}
```

`status` is `"ready"` when all required dependencies are available, otherwise `"degraded"`. In demo mode model artifacts are not required; in live mode they are.

---

## Data Layer

### Data directory structure

```
data/
├── raw/                  # Real data placeholder (empty until Phase 3 live data adapters)
├── synthetic/            # Deterministic synthetic datasets (demo mode)
│   ├── geological.csv
│   ├── satellite_features.csv
│   ├── boreholes.csv
│   ├── weather.csv
│   ├── equipment.csv
│   ├── blasting.csv
│   └── production.csv
├── processed/            # Generated by the fusion + feature pipeline
│   ├── reserve_predictions.csv
│   └── prospectivity_map.png
└── schemas/
    └── datasets.json     # Column listing summary
```

All canonical schemas/contracts live in code at `ml/common/contracts.py`. The `data/schemas/datasets.json` file is a human-readable column summary that tracks the same datasets.

**⚠️ All data under `data/synthetic/` is synthetic/demo data.** It is clearly labeled at every layer (see `BOUNDARY_NOTICE` in `backend/app/services/demo_data.py`). It is **NOT MOIL field-validated data** and must not be presented as official mineral reserves or operational performance.

### Canonical data contracts

Every major dataset is described by a machine-readable `DatasetContract` in [`ml/common/contracts.py`](ml/common/contracts.py). Each contract defines:

- **Required vs. optional columns** (with nullability)
- **Data types** (`str`, `float`, `int`, `date`)
- **Units** (documented explicitly, e.g. `decimal degrees`, `meters`, `percent`, `metric tons`)
- **Allowed ranges** (e.g. latitude ∈ [-90, 90], probabilities ∈ [0, 1])
- **Categorical constraints** (e.g. formations, equipment types, delay reasons)
- **Primary keys** and uniqueness constraints

Validation utilities (`validate_dataset`, `assert_valid`, `validate_borehole_intervals`, `check_leakage`) live in [`ml/common/validation.py`](ml/common/validation.py).

### Dataset inventory

| Dataset | File | Primary Key | Description |
|---------|------|-------------|-------------|
| Geological samples | `geological.csv` | `sample_id` | Sample locations with grade (% Mn, % Fe, % SiO₂), formation, elevation, slope, depth |
| Satellite features | `satellite_features.csv` | `sample_id` | Spectral bands (B2–B12), NDVI, NDWI, SWIR ratio, bare soil index, LST |
| Boreholes | `boreholes.csv` | (composite) | Drilling intervals with lithology and grade per interval |
| Weather | `weather.csv` | `date` | Daily rainfall (mm), soil moisture (fraction), temperature (°C), vegetation index |
| Equipment fleet | `equipment.csv` | (composite) | Daily fleet operating hours, downtime, utilization, maintenance flag per equipment |
| Equipment events | (in-memory) | `equipment_id` + `event_date` | Derived per-equipment daily events (used for DB seeding) |
| Blasting events | `blasting.csv` | `date` | Planned blasts, delay hours, delay reason category |
| Production | `production.csv` | `date` | Daily production (metric tons), target, gap, with weather/fleet/blasting context |

### Geological + satellite fusion

The raw geological and satellite datasets remain **conceptually separate**. A deterministic fusion layer (`ml/reserve/fusion.py`) aligns them on shared keys (`sample_id`, `latitude`, `longitude`) to produce the reserve feature dataset:

```
geological observations    +   satellite observations   +   terrain/context
                    ↓
          spatial/context alignment  (fusion.py)
                    ↓
          reserve feature dataset
                    ↓
          feature engineering  (reserve/features.py)
                    ↓
                        ML
```

The fusion produces:
- `FusionResult` with matched/unmatched statistics and merge rate
- No duplicated alignment columns (suffixes applied only on collision)
- Supports `inner`, `left`, and `outer` join strategies

### Leakage protection

Reserve targets (`mn_pct`, `fe_pct`, `sio2_pct`, `ore_thickness_m`, `is_manganese`) and production targets (`production_mt`, `target_mt`, `production_gap_mt`, `shortfall`) are registered in `LEAKAGE_COLUMNS` (`ml/common/contracts.py`). The `check_leakage()` function verifies that these columns never appear in a feature matrix. Feature engineering modules exclude them via `LEAKAGE_EXCLUSIONS`.

### Data generation & validation

Regenerate synthetic data:

```bash
python -m ml.generate_data
```

Validate against contracts:

```bash
python -m ml.common.validation  # validates all synthetic CSVs against their contracts
```

### Seeding & idempotency

Database seeding is idempotent — running it multiple times produces no duplicates:

```bash
python scripts/seed_demo.py --skip-train
```

The `seed_demo_database()` function checks for existing rows before inserting. Use `--compact` to generate a smaller dataset for faster iteration.

---

## Reserve AI Baseline (Phase 3)

Prototype resource-intelligence pipeline for three related prediction tasks,
trained **only on synthetic demo data**. Every output is a decision-support
signal — **NOT official mineral reserves/resources and NOT field-validated**.

### Pipeline

```
geological.csv  +  satellite_features.csv
            ↓  canonical fusion (ml/reserve/fusion.py)
      fused reserve table
            ↓  curated feature matrix (ml/reserve/features.py)
            ↓  LEAKAGE ASSERTION  (ml/reserve/spatial.assert_no_target_leakage)
            ↓  model comparison   (ml/reserve/evaluate.py)
            ↓  spatial-block holdout + grouped CV  (ml/reserve/spatial.py)
      versioned artifacts + registry records
            ↓  inference (ml/reserve/inference.py) + SHAP (ml/reserve/explain.py)
            ↓  prototype resource potential (ml/reserve/resource_estimator.py)
      FastAPI /reserves/* + /predictions/reserve
```

### The three prediction tasks

| Task | Target | Kind | Primary metric | Candidates compared |
|------|--------|------|----------------|---------------------|
| Prospectivity | `is_manganese` | binary classification | ROC-AUC | LogisticRegression, RandomForest, XGBoost |
| Mn grade | `mn_pct` | regression | RMSE | Ridge, RandomForest, XGBoost |
| Ore thickness | `ore_thickness_m` | regression | RMSE | Ridge, RandomForest, XGBoost |

All candidates share the same spatial validation; the full metric set of every
candidate is retained in the registry. Selection uses the primary metric but
never ignores the others (PR-AUC, F1, precision, recall, confusion matrix for
classification; MAE, R² for regression).

### Feature groups (explicit, curated — never "all numeric columns")

- **Geological/terrain:** `elevation_m`, `slope_deg`, `aspect_deg`, `depth_m`
- **Satellite bands:** `blue_b2`, `green_b3`, `red_b4`, `nir_b8`, `swir_b11`, `swir_b12`
- **Spectral indices:** `ndvi`, `ndwi`, `swir_ratio`, `bare_soil_index`, `land_surface_temperature`
- **Categorical:** `formation` (one-hot encoded)

No production, future, or target-derived variables are used in reserve models.

### Leakage protection (mandatory, automated)

- `RESERVE_FORBIDDEN_COLUMNS` (`ml/reserve/features.py`) forbids, for **every**
  task: `mn_pct`, `fe_pct`, `sio2_pct`, `is_manganese`, `ore_thickness_m`,
  `production_mt`, `target_mt`, `production_gap_mt`, `shortfall`.
- `assert_no_target_leakage()` runs before training and raises `ValueError` if
  a forbidden column enters the feature matrix.
- `check_leakage()` (Phase 2 contract validation) re-checks the matrix and its
  result is recorded as `leakage_check_passed` in model metadata.

### Validation strategy

- **Primary:** spatial-block holdout — `GroupShuffleSplit` over 5×5
  latitude/longitude blocks (`spatial_holdout_indices`, seed 42). Samples from
  the same block never appear in both train and validation sets, so spatially
  clustered mineralization cannot leak.
- **Secondary:** `GroupKFold` spatial cross-validation (`grouped_cv_scores`),
  reported as `cv_*` metrics.
- **Diagnostic only:** random i.i.d. split (`random_split_diagnostic`). It
  shares spatial blocks across the split and is recorded solely to quantify
  the optimism of non-spatial validation. It is never the primary metric.

### Training, artifacts, registry

```bash
python -m scripts.train_reserve          # full candidate comparison
python -m scripts.train_reserve --quick  # cheap smoke-test configuration
```

- Versioned artifacts: `models/reserve/versions/<version>/` — never silently
  overwritten (versions are `YYYY.MM.NNN`, auto-incremented per model).
- Latest serving copies: `models/reserve/{prospectivity,grade,thickness}_*`.
- Legacy compatibility copies: `models/reserve_xgboost.json` et al.
- Per-model metadata sidecar (`*_meta.json`): model name, version, task,
  algorithm, target, prediction type, feature names, feature schema hash,
  training-data hash, validation strategy, full metrics, random seed,
  `synthetic_data: true`, boundary notice, `status: candidate`.
- Registry records: `models/registry/<model_name>-<version>.json`, surfaced by
  `GET /api/v1/models`.

### Inference & API

- `ml/reserve/inference.py` validates artifact availability, applies the exact
  training feature schema, and fails clearly (`FileNotFoundError`) when a model
  artifact is missing. `maybe_predict_regressor` returns `None` instead of
  guessing.
- `POST /api/v1/predictions/reserve` returns the prediction plus
  `model_version` resolved from the served artifact's training metadata
  (never hard-coded), and the demo envelope carries `data_mode`,
  `synthetic_data`, and the boundary notice.
- Demo mode (`DATA_MODE=demo`) may fall back to a labelled heuristic
  (`reserve-prototype-heuristic-001`). Live mode (`DATA_MODE=live`) raises
  `503 MODEL_UNAVAILABLE` rather than faking predictions.
- Explanations: SHAP feature contributions via `ml/reserve/explain.py` when
  available, with model-level gain importance as fallback. Feature importance
  is **not** geological causality.

### Prototype resource potential

Connected via `ml/reserve/resource_estimator.py`:

```
cell_area_m2 × predicted_thickness_m × density_t_per_m3 × prospectivity_probability
```

- Density is configurable (default assumption 3.6 t/m³ — an assumption, not an
  official MOIL parameter).
- Uncertainty is a real Monte Carlo simulation (P10/P50/P90), not fabricated.
- Always labelled `prototype resource potential`, never official reserves.

> ⚠️ All reserve metrics in this repository are computed on synthetic data.
> They demonstrate pipeline mechanics only — never real-world MOIL performance.

---

## Advanced Reserve Intelligence (Phase 4)

Phase 4 upgrades the Phase 3 baseline into a more rigorous spatial
resource-intelligence prototype. It preserves every Phase 1/2/3 component and
adds: ensemble prospectivity, probability calibration, conformal prediction
intervals, resource-uncertainty propagation, spatial prediction grids,
data-support/extrapolation indicators, model-lifecycle management, drift
monitoring, and a real-data ingestion skeleton.

### Architecture

```
Phase 3 baseline models
        ↓  Phase 4 ensemble (ml/reserve/ensemble.py)
  weighted soft-voting: LogReg + RF + XGB
  weights ∝ OOF spatial-CV ROC-AUC
  calibration fitted on OOF ensemble probabilities (isotonic|sigmoid, by Brier)
        ↓  probability calibration (ml/reserve/calibration.py)
  out-of-fold calibration → no calibration leakage
  reliability diagrams + Brier score → models/reserve/evaluation/
        ↓  conformal prediction intervals (ml/reserve/conformal.py)
  split-conformal from grouped-OOF residuals
  grade + ore-thickness: [pred − q, pred + q], 90% coverage
  thickness clipped at 0 (documented)
        ↓  spatial prediction grid (ml/reserve/grid.py)
  nearest-observation covariate context (not fabricated)
  per-cell probability/grade/thickness + intervals + resource potential
        ↓  resource-uncertainty propagation (resource_estimator.py)
  conformal thickness interval → Monte Carlo; configurable density assumption
        ↓  data support + extrapolation (ml/reserve/support.py)
  standardized feature distance + observation proximity + completeness
        ↓  model lifecycle (ml/common/registry.py)
  candidate → validated → champion (+ retired); deterministic criteria
        ↓  drift monitoring (ml/common/drift.py)
  PSI / mean-percentile shift / missingness / categorical TV; warn-only
        ↓  real-data ingestion skeleton (ml/ingestion/)
  CSV/Parquet → load → validate (Phase 2 contracts) → normalize → quality → process
```

### Ensemble & calibration

- **Ensemble**: `ReserveEnsemble` (weighted soft-voting). Weights ∝
  `max(oof_roc_auc − 0.5, 0.01)`, normalized, from out-of-fold spatial
  predictions. The ensemble is only preferred when its ROC-AUC on the same
  spatial holdout is at least as good as the best single baseline.
- **Calibration**: monotone calibrator (isotonic or Platt/sigmoid) fitted on
  pooled OOF ensemble probabilities; the method with the lower Brier score is
  selected. No calibration leakage (holdout never used for fitting).
- **Evaluation artifacts** saved to `models/reserve/evaluation/`: reliability
  diagram PNG + per-bin calibration JSON consuming the actual predictions.

### Conformal prediction intervals

Split-conformal intervals from grouped out-of-fold absolute residuals on the
training blocks (90% target → `ceil((n+1)·0.9)/n` rank). Interval for point
`ŷ` is `[ŷ − q, ŷ + q]`; empirical coverage reported on the untouched spatial
holdout. Ore-thickness lower bound clipped at zero (documented as conservative).
Coverage is empirical under grouped-exchangeability, not a field guarantee.

### Resource-uncertainty propagation

`estimate_resource_potential_with_intervals` propagates the conformal thickness
interval via a fitted normal (5th/95th percentiles → interval bounds) and
Monte Carlo. Density stays configurable; if no density standard deviation is
given, density is a fixed assumption (no invented geological density
distribution). Only model-derived uncertainty is propagated.

### Spatial prediction grid

`generate_prediction_grid(bbox, cells_per_side)` — deterministic grid (capped
40×40). Each cell reuses terrain + satellite covariates from the nearest
fused observation (available context, not fabricated geology). Cells beyond the
search radius return a `no_context` data-support state. Every scored cell
carries calibrated probability, grade/thickness + intervals, prototype
resource potential, data support, and extrapolation level.

### Data support vs model uncertainty (separate concepts)

- **Model uncertainty**: calibration reliability + prediction-interval width.
- **Data support**: standardized feature distance (mean |z|; thresholds are
  documented heuristics) → `well_supported` / `moderate_support` /
  `extrapolation_warning`; plus nearest-observation distance, observations
  within 5 km, feature completeness. A warning mechanism only.

### Model lifecycle

`candidate → validated → champion` (+ `retired`) in `ml/common/registry.py`.
Promotion requires: `leakage_check_passed`, spatial-block validation, required
metrics present, artifact file exists. Champion promotion demotes the previous
champion (recorded as `previous_champion`) and appends `promotion_history`.
Training always writes `candidate`; promotion is explicit. No overwrites.

### Drift monitoring foundation

`compute_feature_drift(reference, current)`: PSI, mean/median/percentile
shift, missingness delta, categorical total-variation. Configurable
thresholds → `warning` (never auto-invalidates). A foundation for later MLOps.

### Real-data ingestion skeleton

`ml/ingestion/` — `DataSource` protocol with `CsvSource`/`ParquetSource` and an
`IngestionPipeline`: load → validate against Phase 2 contracts → normalize →
quality-check → process. Demo stays offline; real MOIL data supplied later.

### API changes

- `GET /api/v1/reserves/grid` — spatial prediction grid.
- `GET /api/v1/models/compare` — model-comparison view.
- `POST /api/v1/predictions/reserve` adds `calibrated_probability`,
  `base_probabilities`, `grade_interval`, `thickness_interval`,
  `extrapolation`, `data_support_detail` (all optional → `None` if unavailable).
- Model-registry responses include lifecycle status, `promoted_at`,
  `previous_champion`, artifact status.

### Running Phase 4

```bash
python -m scripts.train_reserve_advanced          # full pipeline + ensemble
python -m scripts.train_reserve_advanced --quick  # cheap smoke configuration

python -c "from backend.app.services.model_registry import ModelRegistryService; ModelRegistryService().promote('reserve_prospectivity','<version>','champion')"
```

---

## SIH Demo Narrative

1. **Discover**: Open Reserve Intelligence and explore the prospectivity heatmap
2. **Investigate**: Click a high-prospectivity cell to view grade, thickness, confidence, and contributing factors
3. **Forecast**: View the next 7 days production forecast against target
4. **Diagnose**: Examine equipment, weather, and blasting drivers behind shortfall risk
5. **Act**: Review ranked corrective actions with evidence and estimated impact
6. **Trust**: Inspect data quality, model version, and validation metrics
7. **Scale**: MOIL data can replace demo adapters without rewriting the core application

---

## Scientific / Operational Boundary

MANGAI is a **decision-support prototype** until validated with:
- Real MOIL operational data
- Domain expert review
- Applicable mining regulations
- Operational system integration

**Official mineral-resource/reserve classification, mine design, blasting safety, equipment dispatch, and production commitments must remain under qualified human and organizational control.**

---

## License

This project is developed for SIH 2026 (Smart India Hackathon) under Ministry of Steel / MOIL Ltd.