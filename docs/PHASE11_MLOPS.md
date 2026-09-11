# Phase 11 MLOps Architecture

## Registry

`ml/common/registry.py` remains the source of truth for model lifecycle:

`candidate -> validated -> champion -> retired`

Promotion is deterministic and auditable. Existing artifacts are never overwritten.

## Monitoring

`ml/common/monitoring.py` provides:

- registry integrity checks
- missing-artifact detection
- duplicate champion detection
- champion leakage-check detection
- population stability index (PSI)
- deterministic drift classification

PSI thresholds:

- `< 0.10`: STABLE
- `0.10–<0.25`: WARNING
- `>= 0.25`: CRITICAL

These are monitoring conventions, not scientific claims about model correctness.

## API

`GET /api/v1/models/monitoring`

Returns registry health, computed drift where baseline/current populations exist, and actionable alerts.

## Governance

Monitoring is read-only. Drift never silently changes model predictions and does not trigger automatic retraining. Model promotion remains explicit.
