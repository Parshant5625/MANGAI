"""CLI: train the three MANGAI reserve baselines on the synthetic dataset.

Usage:
    python scripts/train_reserve.py            # full candidate comparison
    python scripts/train_reserve.py --quick    # cheap configuration (tests)

Outputs versioned artifacts under models/reserve/versions/<version>/,
"latest" serving copies under models/reserve/, and one registry record per
model under models/registry/. All metrics are synthetic-data metrics.
"""

from __future__ import annotations

import argparse
import json

from ml.reserve.train_prospectivity import train_grade, train_prospectivity, train_thickness


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MANGAI reserve baselines (synthetic data)")
    parser.add_argument("--quick", action="store_true", help="Small model configuration for smoke tests")
    args = parser.parse_args()

    results = {
        "prospectivity": train_prospectivity(quick=args.quick),
        "grade": train_grade(quick=args.quick),
        "thickness": train_thickness(quick=args.quick),
    }
    for name, result in results.items():
        metrics = result["metrics"]
        print(f"[{name}] version={result['version']} algorithm={result['algorithm']}")
        print(f"  candidates compared: {result['selected_from']}")
        if "roc_auc" in metrics:
            print(
                "  spatial holdout: roc_auc={roc_auc:.4f} pr_auc={pr_auc:.4f} f1={f1:.4f}".format(**metrics)
            )
            random = metrics["random_split_diagnostic"]["metrics"]
            print(
                "  random-split diagnostic (NOT primary): roc_auc={:.4f}".format(random["roc_auc"])
            )
        else:
            print(
                "  spatial holdout: mae={mae:.4f} rmse={rmse:.4f} r2={r2:.4f}".format(**metrics)
            )
            random = metrics["random_split_diagnostic"]["metrics"]
            print("  random-split diagnostic (NOT primary): rmse={:.4f}".format(random["rmse"]))
    print(json.dumps({name: {"version": res["version"], "artifact": res["artifact_path"]} for name, res in results.items()}, indent=2))
    print("RESERVE BASELINE TRAINING COMPLETE")


if __name__ == "__main__":
    main()
