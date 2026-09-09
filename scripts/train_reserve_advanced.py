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

    # Validation-integrity audit summary (Phase 4.1). Every value is copied
    # from the points the pipelines actually wrote — nothing is fabricated.
    summary_path = Path("models") / "reserve" / "evaluation" / "phase4_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    probe = base["prospectivity"]["metrics"]
    group_report = {
        "development_spatial_groups": probe.get("development_spatial_groups"),
        "final_test_spatial_groups": probe.get("final_test_spatial_groups"),
        "group_overlap": probe.get("group_overlap"),
        "leakage_check_passed": probe.get("leakage_check_passed"),
    }
    summary = {
        "phase": 4,
        "phase_4_1_validation_integrity_audit": True,
        "baseline_versions": {name: result["version"] for name, result in base.items()},
        "baseline_algorithms": {name: result["algorithm"] for name, result in base.items()},
        "ensemble": {
            "version": ensemble["version"],
            "weights": ensemble["weights"],
            "calibration_method": ensemble["calibration_method"],
            "metrics": ensemble["metrics"],
        },
        "validation_integrity": {
            "validation_strategy": "spatial_block_holdout_with_isolated_final_test",
            "development_sample_count": probe.get("development_sample_count"),
            "final_test_sample_count": probe.get("final_test_sample_count"),
            "development_spatial_groups": group_report["development_spatial_groups"],
            "final_test_spatial_groups": group_report["final_test_spatial_groups"],
            "group_overlap": group_report["group_overlap"],
            "leakage_check_passed": group_report["leakage_check_passed"],
            "ensemble_weight_selection_method": "GroupKFold OOF ROC-AUC on development set only",
            "ensemble_calibration_method": ensemble["calibration_method"],
            "conformal_calibration_method": (
                "split_conformal_quantile_on_development_oof_residuals"
            ),
            "final_evaluation_isolation": all(
                result["metrics"].get("final_evaluation_isolation") for result in base.values()
            )
            and ensemble["metrics"].get("final_evaluation_isolation") is True,
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
    print("Validation integrity audit (Phase 4.1):")
    integrity = summary["validation_integrity"]
    for key in (
        "development_sample_count",
        "final_test_sample_count",
        "group_overlap",
        "leakage_check_passed",
        "final_evaluation_isolation",
    ):
        print(f"  {key}={integrity[key]}")
    print(f"Summary written to: {summary_path}")
    print("PHASE 4 ADVANCED TRAINING COMPLETE")


if __name__ == "__main__":
    main()
