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
- PostgreSQL 16+ (or use Docker)

### Quick Start

#### 1. Clone and Setup

```bash
git clone <repository-url>
cd MANGAI
```

#### 2. Backend Setup

```bash
# Create virtual environment
python -m venv backend/.venv
source backend/.venv/bin/activate  # On Windows: backend\.venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt

# Copy environment configuration
cp .env.example .env
# Edit .env with your settings

# Initialize database and seed demo data
python scripts/seed_demo.py
```

#### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

#### 4. Access the Application

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## Docker Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services:
- PostgreSQL: port 5432
- Backend API: port 8000
- Frontend: port 8080

---

## API Endpoints

### Health & Status
- `GET /health` - Health check
- `GET /ready` - Readiness check (database, models)

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
| `DATABASE_URL` | PostgreSQL connection string | sqlite:///./mangai_dev.db |
| `DATA_MODE` | Data mode (demo/file/external) | demo |
| `MODEL_DIR` | Path to model artifacts | models |
| `DATA_DIR` | Path to data directory | data |
| `CORS_ORIGINS` | Allowed CORS origins | http://localhost:5173 |
| `LOG_LEVEL` | Logging level | INFO |

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