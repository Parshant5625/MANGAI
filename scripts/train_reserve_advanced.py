"""Phase 4 advanced reserve training pipeline.

Runs the Phase 3 baselines (idempotent — produces new versions), then trains
the calibrated prospectivity ensemble with evaluation artifacts, and writes a
model-comparison summary. Models are left as ``candidate``; promotion is a
separate, explicit step.

Usage:
    python -m scripts.train_reserve_advanced          # full
    python -m scripts.train_reserve_advanced --quick  # cheap smoke config
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.reserve.train_prospectivity import train_grade, train_prospectivity, train_thickness


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 advanced reserve training (synthetic data)")
    parser.add_argument("--quick", action="store_true", help="Cheap configuration for smoke tests")
    args = parser.parse_args()

    base = {
        "prospectivity": train_prospectivity(quick=args.quick),
        "grade": train_grade(quick=args.quick),
        "thickness": train_thickness(quick=args.quick),
    }

    from ml.reserve.ensemble import train_prospectivity_ensemble

    ensemble = train_prospectivity_ensemble(quick=args.quick)

    summary_path = Path("models") / "reserve" / "evaluation" / "phase4_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "phase": 4,
        "baseline_versions": {name: result["version"] for name, result in base.items()},
        "baseline_algorithms": {name: result["algorithm"] for name, result in base.items()},
        "ensemble": {
            "version": ensemble["version"],
            "weights": ensemble["weights"],
            "calibration_method": ensemble["calibration_method"],
            "metrics": ensemble["metrics"],
        },
        "evaluation_artifacts": ensemble["evaluation_artifacts"],
        "synthetic_data": True,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("Phase 4 baselines:")
    for name, result in base.items():
        print(f"  [{name}] version={result['version']} algorithm={result['algorithm']}")
    print(
        f"[ensemble] version={ensemble['version']} calibration={ensemble['calibration_method']}"
    )
    print(f"  ensemble vs baseline: {ensemble['metrics'].get('ensemble_vs_baseline', {}).get('ensemble_preferred')}")
    print(f"Summary written to: {summary_path}")
    print("PHASE 4 ADVANCED TRAINING COMPLETE")


if __name__ == "__main__":
    main()
