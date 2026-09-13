# MANGAI — AI-Powered Mining Intelligence Platform

> **Smart India Hackathon (SIH) 2026 Prototype** for **MOIL Limited / Ministry of Steel**
>
> **SIH Problem Statement ID:** `26009`

MANGAI (**Manganese AI**) is an end-to-end mining decision-support platform designed to combine **geological intelligence, satellite/remote-sensing data, production forecasting, equipment analytics, blasting and weather signals, recommendations, and MLOps** into a single operational intelligence system.

The platform is designed around one goal: **turn heterogeneous mining data into explainable, actionable intelligence for mine planning and operational decision-making.**

---

## ⚠️ Prototype / Data Boundary

**MANGAI is currently a demonstration and research prototype.**

The repository contains deterministic synthetic/demo datasets and prototype integrations. Unless explicitly stated otherwise, dashboard metrics, predictions, risk scores, reserve estimates, and recommendations shown in demo mode are **not MOIL field measurements and are not validated mineral-reserve estimates**.

A production deployment would require:

- Real MOIL geological, borehole, production, fleet and blasting datasets
- Validated satellite/remote-sensing inputs for the selected mine areas
- Domain-expert review and calibration
- Mine-specific model training and validation
- Integration with operational systems and data pipelines
- Appropriate mining, safety, environmental and regulatory controls
- Human approval for operational decisions

MANGAI should therefore be treated as a **decision-support prototype, not an autonomous mining-control system**.

---

## 🎯 Problem MANGAI Addresses

Mining intelligence is often distributed across geological records, borehole logs, production systems, equipment data, weather observations, blasting schedules and remote-sensing sources.

This makes it difficult to answer questions such as:

- Where are the most prospective manganese zones?
- What geological and satellite signals are associated with manganese prospectivity?
- How much production is expected over the next few days or weeks?
- What is the probability of missing the production target?
- Which equipment or operational factors are driving production risk?
- How much do downtime, weather and blasting delays affect production?
- Which operational actions should be prioritized?
- Are deployed ML models healthy, drifting or degrading?

**MANGAI connects these signals into one intelligence layer.**

---

## 🧠 Core Capabilities

### 1. Reserve Intelligence

Spatial and geological intelligence for manganese prospectivity and resource assessment.

- Geological + satellite feature fusion
- Manganese prospectivity probability mapping
- Spatial ML inference
- Mn-grade prediction
- Ore-thickness prediction
- Prototype resource-potential estimation
- P10 / P50 / P90 uncertainty representation
- Borehole context
- Spatial support / extrapolation indicators
- Model confidence and validation signals
- Feature-level explanations / SHAP analysis
- Interactive reserve map
- Dedicated reserve views for map, 3D-style analysis, estimates and geological layers

### 2. Production Intelligence

Forecasting and shortfall-risk intelligence for daily mine production.

- Daily production forecasting
- Configurable forecast horizons
- Historical production analysis
- Production target attainment
- Shortfall probability
- Risk severity classification
- Prediction intervals
- Production-driver attribution
- Production analytics
- Downtime analysis
- Blasting-delay analysis
- Operational risk context

### 3. Operations Intelligence

Cross-domain operational signals from fleet, weather and blasting data.

- Fleet availability
- Equipment utilization
- Downtime ranking
- Critical equipment identification
- Weather exposure
- Rainfall and soil-moisture signals
- Temperature / environmental context
- Blasting schedule and delay analysis
- Delay-reason analysis
- Production associations
- Operational risk signals with LOW / MEDIUM / HIGH severity
- Evidence-backed operational context

### 4. Recommendation Engine

Turns detected risks into prioritized decision-support actions.

- Ranked corrective actions
- Evidence attached to recommendations
- Confidence scoring
- Expected impact estimates
- What-if / simulation support
- Downtime-reduction scenarios
- Blast-scheduling scenarios
- Human approval boundary

### 5. MLOps / Model Monitoring

Monitoring layer for ML model health and data quality.

- Model registry
- Model version metadata
- Model health indicators
- Performance metrics
- Drift / PSI monitoring
- Data-quality reporting
- Model availability / readiness checks
- Explainability dimensions
- Registry-level model comparison

### 6. MANGAI AI Assistant

A context-aware assistant interface connected to the MANGAI backend.

The assistant can answer supported questions about:

- Production risk and forecast
- Reserve prospectivity
- Equipment status
- Model health and drift
- Operations summary
- Satellite signals
- Recommendations
- Platform help

### 7. Reports

The dashboard provides a report workflow for turning live dashboard/API information into a downloadable report artifact.

---

## 🖥️ Dashboard Experience

The frontend is designed as a **premium dark mining-intelligence command centre**, with a consistent visual language across the application.

Main dashboard areas include:

| Module | Purpose |
|---|---|
| **Executive Overview** | High-level mine KPIs, reserve, production and operational insights |
| **Reserve Intelligence** | Spatial prospectivity, geological/satellite intelligence and resource estimates |
| **Production Intelligence** | Forecasts, historical analytics, shortfall risk and production drivers |
| **Operations Intelligence** | Fleet, weather, blasting and operational risk signals |
| **Model Monitoring** | Model health, registry, drift and explainability information |
| **MANGAI AI Assistant** | Natural-language interaction with supported backend intelligence |
| **Reports** | Report generation and export workflow |

The current frontend also contains targeted interaction improvements for reserve and production analysis, including separate analytical views rather than showing the same panel for every tab.

---

## 🏗️ System Architecture

```text
                           MANGAI
                              │
              ┌───────────────┴───────────────┐
              │                               │
       React + TypeScript                External / Data Sources
       Vite + Tailwind                    ├─ Geological data
       Recharts + MapLibre                ├─ Boreholes
       Lucide icons                       ├─ Satellite data
              │                            ├─ Weather
              │                            ├─ Equipment
              │                            ├─ Blasting
              │                            └─ Production
              │
              ▼
        FastAPI REST API
              │
      ┌───────┼───────────────────────────────────┐
      │       │             │          │           │
      ▼       ▼             ▼          ▼           ▼
   Reserve Production   Operations Recommendations MLOps
   Services Intelligence Analytics     Engine     / Monitoring
      │       │             │          │           │
      └───────┴─────────────┴──────────┴───────────┘
                          │
                          ▼
                 ML / Feature Layer
                          │
              ┌───────────┴───────────┐
              │                       │
        Reserve Models          Production Models
              │                       │
       XGBoost / ensemble       Regression + risk
       spatial validation       chronological validation
              │                       │
              └───────────┬───────────┘
                          ▼
                Model Registry / Artifacts
                          │
                          ▼
                 PostgreSQL / SQLite
```

---

## 🔬 AI / ML Pipeline

### Reserve Intelligence Pipeline

```text
Geological Data ──────┐
                      │
Borehole Data ────────┼──► Data Validation / Fusion
                      │              │
Satellite Data ──────┘              ▼
                              Feature Engineering
                                      │
                                      ▼
                           Spatial ML / Ensemble Models
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
              Prospectivity       Grade            Thickness
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      ▼
                         Resource Potential / Uncertainty
                                      │
                                      ▼
                         Map + Explainability + API
```

### Production Intelligence Pipeline

```text
Historical Production ─┐
Equipment / Downtime ──┤
Weather ───────────────┼──► Feature Engineering
Blasting ──────────────┘            │
                                    ▼
                         Chronological Train / Validation
                                    │
                                    ▼
                         Candidate Model Evaluation
                                    │
                      ┌─────────────┴─────────────┐
                      ▼                           ▼
                 Forecast Model              Risk Model
                      │                           │
                      ▼                           ▼
              Production Forecast        Shortfall Probability
                      │                           │
                      └─────────────┬─────────────┘
                                    ▼
                         Drivers + Recommendations
```

---

## 🧪 Validation & Leakage Protection

MANGAI does not use a single generic train/test strategy for every problem.

### Reserve / Geological ML

Spatial separation is used where appropriate to reduce the risk of spatial leakage between training and validation data.

### Production ML

Production models use **chronological train/validation/test splits** so future observations are not allowed to influence historical model training.

### Data Leakage Controls

Target columns such as production outcomes and geological assay targets are explicitly tracked by the common data-contract / validation layer. Feature matrices are checked to ensure target variables do not accidentally enter model inputs.

### Evaluation Metrics

Depending on the task, the project uses:

- ROC-AUC
- PR-AUC
- F1 score
- MAE
- RMSE
- R²
- Calibration / probability-quality checks
- Drift / PSI metrics
- Spatial validation statistics

---

## 📊 Data Layer

The demo system contains deterministic synthetic datasets representing the major domains of a mining operation.

### Demo Dataset Inventory

| Dataset | Example fields | Purpose |
|---|---|---|
| Geological | latitude, longitude, elevation, slope, Mn/Fe/SiO₂ | Geological intelligence |
| Satellite | spectral bands, NDVI, NDWI, SWIR, bare-soil index, LST | Remote-sensing context |
| Boreholes | interval, lithology, depth, assay values | Subsurface context |
| Weather | rainfall, soil moisture, temperature, vegetation index | Environmental risk |
| Equipment | operating hours, downtime, utilization, maintenance | Fleet intelligence |
| Blasting | planned blasts, delay hours, delay reason | Blast-delay analysis |
| Production | production, target, production gap | Forecasting / shortfall |

### Geological + Satellite Fusion

Geological and satellite observations are kept as conceptually separate sources and aligned through a deterministic fusion layer before reserve feature engineering.

The fusion layer provides:

- Key-based spatial/context alignment
- Match / unmatched statistics
- Merge-rate reporting
- `inner`, `left` and `outer` join strategies
- Collision-safe column handling

### Data Contracts

Canonical dataset contracts define:

- Required and optional fields
- Nullability
- Data types
- Units
- Valid ranges
- Categorical constraints
- Primary keys
- Uniqueness constraints
- Leakage-sensitive columns

---

## 🛰️ Satellite / Remote Sensing

MANGAI includes a satellite-data pathway for reserve intelligence and environmental context.

The system is designed to work with remote-sensing features such as:

- Multispectral bands
- NDVI
- NDWI
- SWIR-derived ratios
- Bare-soil indicators
- Land-surface temperature
- Spatial coordinates / AOI context

The repository also includes live satellite-related smoke/integration tooling. Demo mode remains deterministic and does not require external satellite access.

---

## 🗄️ Database Model

Key entities include:

- `mine_sites`
- `geological_samples`
- `boreholes`
- `satellite_observations`
- `weather_observations`
- `equipment`
- `equipment_events`
- `blasting_events`
- `production_records`
- `model_versions`
- `predictions`
- `recommendations`
- `data_quality_runs`

SQLite is convenient for local demo development; PostgreSQL is supported for the containerized / production-oriented configuration.

---

## 🔌 API Surface

The FastAPI backend exposes versioned endpoints under `/api/v1`.

### Overview

```text
GET  /api/v1/overview
```

Executive-level KPIs and dashboard summary data.

### Reserve Intelligence

```text
GET  /api/v1/reserves/prospectivity
GET  /api/v1/reserves/summary
GET  /api/v1/reserves/{reserve_id}
GET  /api/v1/reserves/boreholes
POST /api/v1/predictions/reserve
```

### Production Intelligence

```text
GET  /api/v1/production/forecast
GET  /api/v1/production/risk
GET  /api/v1/production/history
POST /api/v1/predictions/production
```

### Operations

```text
GET /api/v1/equipment
GET /api/v1/weather
GET /api/v1/blasting
GET /api/v1/operations/summary
```

### Recommendations

```text
GET  /api/v1/recommendations
POST /api/v1/recommendations/simulate
```

### MLOps / Data Quality

```text
GET /api/v1/models
GET /api/v1/data-quality
```

### AI Assistant

```text
POST /api/v1/chat
```

The chat endpoint accepts a message, optional conversation history and optional page context.

### Health / Readiness

```text
GET /health
GET /ready
```

`/health` provides a lightweight liveness response. `/ready` checks application dependencies and reports structured readiness information.

---

## 🛠️ Technology Stack

### Backend

- Python 3.11+
- FastAPI
- Pydantic v2
- SQLAlchemy 2
- Alembic
- PostgreSQL / SQLite
- pandas
- NumPy
- scikit-learn
- XGBoost
- SHAP
- matplotlib / geospatial processing utilities

### Frontend

- React 19
- TypeScript
- Vite
- Tailwind CSS
- MapLibre GL JS
- Recharts
- Lucide React

### Engineering / Infrastructure

- Docker Compose
- pytest
- httpx
- Ruff
- Git / GitHub

---

## 🚀 Getting Started

### Prerequisites

Install:

- Python 3.11 or newer
- Node.js 18 or newer
- npm
- Git
- Docker Desktop (optional)
- PostgreSQL 16+ only when using a PostgreSQL/live-style environment

### 1. Clone the repository

```bash
git clone https://github.com/Parshant5625/MANGAI.git
cd MANGAI
```

### 2. Create and activate a Python environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install backend dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Configure environment

Copy the example environment file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

The default configuration is intended for demo development.

### 5. Seed demo data

```bash
python scripts/seed_demo.py --skip-train
```

This prepares the deterministic demo datasets and initializes the local database state. Model training is optional for demo startup.

### 6. Apply migrations

```bash
alembic upgrade head
```

The demo seeding workflow also handles migrations; run this command directly when you only need to apply database migrations.

### 7. Start the backend

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend:

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`
- Readiness: `http://localhost:8000/ready`

### 8. Start the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:5173
```

### 9. Run the full test suite

From the repository root:

```bash
pytest
```

### 10. Run linting

```bash
ruff check .
```

---

## 🐳 Docker

MANGAI includes a Docker Compose configuration for PostgreSQL, FastAPI and the frontend.

```bash
docker compose up --build
```

Services:

| Service | Port |
|---|---:|
| PostgreSQL | `5432` |
| FastAPI backend | `8000` |
| Frontend | `8080` |

Open the dashboard at:

```text
http://localhost:8080
```

To rebuild the complete Docker database volume:

```bash
docker compose down -v
docker compose up --build
```

The compose configuration uses `DATA_MODE=demo` for the demonstration environment.

---

## ⚙️ Configuration

Important environment variables include:

| Variable | Purpose | Typical default |
|---|---|---|
| `APP_ENV` | Application environment | `development` |
| `DATABASE_URL` | Database connection | `sqlite:///./mangai_dev.db` |
| `DATA_MODE` | `demo` or `live` data mode | `demo` |
| `MODEL_DIR` | ML artifact directory | `models` |
| `DATA_DIR` | Data directory | `data` |
| `CORS_ORIGINS` | Allowed frontend origins | localhost frontend URLs |
| `LOG_LEVEL` | Application logging level | `INFO` |

### Demo Mode

`DATA_MODE=demo` is the recommended mode for evaluation and local demonstration.

It uses deterministic synthetic data and does not require a live MOIL data connection.

### Live Mode

`DATA_MODE=live` is intended as the production-oriented path. A real deployment requires validated data sources, trained artifacts, a supported database configuration and appropriate external-service credentials/configuration.

When required model artifacts are unavailable in live mode, ML-dependent endpoints should report model unavailability rather than silently presenting demo fallback results.

---

## 🩺 Health & Readiness

### Health

```http
GET /health
```

Lightweight application liveness check.

### Readiness

```http
GET /ready
```

Reports structured readiness information, including database/data-mode/model availability where applicable.

Example shape:

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

The exact model availability depends on the configured environment and available artifacts.

---

## 🧪 Testing

MANGAI has backend, ML and integration test coverage.

Run everything:

```bash
pytest
```

Backend tests:

```bash
pytest tests/backend -q
```

ML tests:

```bash
pytest tests/ml -q
```

Integration tests:

```bash
pytest tests/integration -q
```

Verbose test run:

```bash
python -m pytest tests/ -v
```

Smoke tests:

```bash
python scripts/run_smoke_tests.py
```

Lint:

```bash
ruff check .
```

### Submission Checkpoint Validation

The SIH prototype checkpoint was validated with:

- **207 tests passing**
- **Ruff checks passing**
- Clean Git working tree at the checkpoint

These numbers describe the validated prototype checkpoint and may change with subsequent development commits.

---

## 📁 Project Structure

```text
MANGAI/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/              # FastAPI routes
│   │   ├── core/             # Configuration / security / logging
│   │   ├── db/               # SQLAlchemy models / sessions
│   │   ├── schemas/          # Pydantic schemas
│   │   ├── services/         # Business logic
│   │   ├── repositories/     # Data access
│   │   └── adapters/         # External data providers
│   ├── requirements.txt
│   └── Dockerfile
│
├── ml/
│   ├── common/               # Contracts, validation, monitoring
│   ├── reserve/              # Reserve intelligence / spatial ML
│   ├── production/           # Production forecasting / risk
│   └── risk/                 # Risk modelling utilities
│
├── frontend/
│   ├── src/
│   │   ├── api/              # API client
│   │   ├── components/        # Dashboard components
│   │   ├── hooks/             # React hooks
│   │   ├── pages/             # Page-level UI
│   │   ├── styles/            # Dashboard styling
│   │   ├── types/             # TypeScript API/domain types
│   │   └── utils/             # Frontend utilities
│   ├── package.json
│   └── Dockerfile
│
├── data/
│   ├── raw/                   # Real-data placeholder
│   ├── synthetic/             # Demo datasets
│   ├── processed/             # Generated outputs
│   └── schemas/               # Dataset summaries/contracts
│
├── models/                    # Model artifacts / registry metadata
├── scripts/                   # Training, seeding and smoke-test scripts
├── tests/                     # Backend / ML / integration tests
├── alembic/                   # Database migrations
├── docs/                      # Project documentation
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 🔐 Safety, Governance & Decision Boundary

MANGAI is designed as a **human-in-the-loop decision-support platform**.

The system should not independently:

- Control mining equipment
- Execute blasting operations
- Declare an official mineral reserve
- Override mine safety procedures
- Replace qualified geologists, mining engineers or operational decision-makers

Recommendations are intended to provide **evidence and prioritization**, while final operational decisions remain with authorized personnel.

---

## 🗺️ Roadmap

The prototype architecture is designed to support future deployment with real mine data.

### Near-term

- Connect validated MOIL datasets
- Add mine-specific calibration
- Improve real satellite/AOI ingestion
- Expand operational telemetry ingestion
- Strengthen model monitoring and alerting
- Add role-based access and audit trails

### Production-scale direction

- Multi-mine / multi-site support
- Streaming equipment telemetry
- Automated data-quality pipelines
- Feature store / centralized model-serving architecture
- Continuous model evaluation and drift monitoring
- GIS layers from validated mine-survey sources
- Integration with enterprise mining systems
- Role-specific dashboards for geology, production and operations

---

## 🏆 SIH Demonstration Narrative

A recommended demonstration flow is:

```text
1. Executive Overview
        ↓
2. Reserve Intelligence
        ↓
3. Production Forecast + Shortfall Risk
        ↓
4. Operations Intelligence
        ↓
5. Recommendations / What-if Analysis
        ↓
6. Model Monitoring
        ↓
7. MANGAI AI Assistant
        ↓
8. Reports
```

This flow demonstrates the complete decision chain:

**Observe → Predict → Explain → Assess Risk → Recommend → Monitor → Report**

---

## 📌 Submission Checkpoint

The repository contains a protected SIH prototype checkpoint tagged as:

```text
v1.0-submission
```

The tag represents the frozen prototype state before subsequent documentation/presentation work. The active branch may contain documentation commits made after that tag.

To inspect the exact tagged checkpoint:

```bash
git fetch --tags
git checkout v1.0-submission
```

To return to the active branch:

```bash
git checkout ui-targeted-fixes-2026-09-13
```

---

## 👨‍💻 Project

**MANGAI — Manganese AI**

Built as a Smart India Hackathon prototype for **MOIL Limited / Ministry of Steel**.

Repository: `https://github.com/Parshant5625/MANGAI`

---

## 📄 License

This repository is a project prototype. Add the appropriate project/open-source license here if required by the final submission or deployment context.
