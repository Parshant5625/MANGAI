# Phase 12 — SIH Demo Polish & Integration

Phase 12 is the demo-readiness phase after MLOps hardening.

## Objectives

- Connect Data/Model Health dashboard widgets to `/api/v1/models/monitoring`.
- Surface model version, registry status, drift alerts, data-quality score, and satellite QA state in one operator view.
- Add a deterministic end-to-end SIH demo flow using demo data only.
- Add explicit mixed-data/live-data labels everywhere.
- Add a demo checklist and one-command smoke verification.
- Keep live satellite inference fail-closed and keep prototype resource potential clearly distinct from official mineral reserves.

## Phase 12 exit criteria

- Backend tests and Ruff pass.
- Frontend lint and production build pass.
- Demo smoke verifies overview → reserve → production → operations → recommendations → model monitoring.
- No synthetic data is silently presented as field-validated MOIL data.
- Live satellite QA failures are visible as data-quality/availability states rather than converted into fabricated values.
