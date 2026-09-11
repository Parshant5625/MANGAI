# Phase 11 — MLOps, Model Governance & Monitoring

Phase 11 adds production-oriented model governance without changing inference behavior.

## Scope

- Registry health and lifecycle visibility.
- Deterministic model promotion checks.
- Artifact and feature-schema integrity checks.
- Training-data hash/provenance verification.
- Prediction/inference audit events.
- Feature drift monitoring using PSI-compatible statistics.
- Model freshness and champion-age monitoring.
- Fail-closed alerts for missing artifacts or invalid champion metadata.

## Safety rules

- Monitoring never silently changes predictions.
- A drift alert is diagnostic; automatic retraining is not performed.
- Promotion remains explicit and auditable.
- Synthetic/demo models remain clearly labelled as not field-validated.
- Real satellite inference remains fail-closed when required QA contracts are not satisfied.
