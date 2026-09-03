from __future__ import annotations

from backend.app.services.data_quality import DataQualityService


def main() -> None:
    report = DataQualityService().run()
    print(f"overall_score={report['overall_score']}")
    for run in report["runs"]:
        print(f"{run['dataset_name']}: score={run['quality_score']} rows={run['row_count']} schema={run['schema_valid']}")
    if report["overall_score"] < 0.7:
        raise SystemExit("Data quality below demo threshold")


if __name__ == "__main__":
    main()
