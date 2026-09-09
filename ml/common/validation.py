"""Data validation utilities for MANGAI datasets.

Validates DataFrames against canonical contracts defined in
``ml.common.contracts``. Used by generation, seeding, and the data-quality
pipeline to keep synthetic data internally consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ml.common.contracts import DatasetContract


@dataclass
class ValidationResult:
    """Result of validating a DataFrame against a contract."""

    name: str
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def raise_for_errors(self) -> None:
        if not self.valid:
            raise ValueError(f"Dataset {self.name!r} failed validation: {'; '.join(self.errors)}")


def validate_dataset(df: pd.DataFrame, contract: DatasetContract) -> ValidationResult:
    """Validate a DataFrame against its contract."""
    result = ValidationResult(name=contract.name)

    # Schema check: required columns present
    required = set(contract.required_columns())
    present = set(df.columns)
    missing = required - present
    if missing:
        result.add_error(f"Missing required columns: {sorted(missing)}")

    # Extra columns are a warning, not an error
    extra = present - set(contract.column_names())
    if extra:
        result.add_warning(f"Extra columns not in contract: {sorted(extra)}")

    if df.empty:
        result.add_error("Dataset is empty.")
        return result

    for column in contract.columns:
        if column.name not in df.columns:
            continue
        series = df[column.name]
        non_null = series.dropna()

        # Nullability
        if not column.nullable and series.isna().any():
            result.add_error(f"Column {column.name!r} has {series.isna().sum()} null values but is non-nullable.")

        # Uniqueness
        if column.unique and non_null.duplicated().any():
            duplicates = int(non_null.duplicated().sum())
            result.add_error(f"Column {column.name!r} has {duplicates} duplicate values but must be unique.")

        # Range checks (numeric only)
        if column.dtype in ("float", "int") and not non_null.empty:
            if column.min_value is not None and (non_null < column.min_value).any():
                result.add_error(
                    f"Column {column.name!r} has values below minimum {column.min_value}."
                )
            if column.max_value is not None and (non_null > column.max_value).any():
                result.add_error(
                    f"Column {column.name!r} has values above maximum {column.max_value}."
                )

        # Categorical constraints
        if column.categorical_values:
            invalid = set(non_null.unique()) - set(column.categorical_values)
            if invalid:
                result.add_error(
                    f"Column {column.name!r} has invalid categories: {sorted(invalid)}. "
                    f"Allowed: {list(column.categorical_values)}"
                )

    return result


def assert_valid(df: pd.DataFrame, contract: DatasetContract) -> ValidationResult:
    """Validate and raise ValueError on failure."""
    result = validate_dataset(df, contract)
    result.raise_for_errors()
    return result


def validate_borehole_intervals(df: pd.DataFrame) -> ValidationResult:
    """Dataset-specific check: intervals must satisfy from_depth_m < to_depth_m."""
    result = ValidationResult(name="borehole_intervals")
    if "from_depth_m" not in df.columns or "to_depth_m" not in df.columns:
        result.add_error("Borehole intervals require from_depth_m and to_depth_m columns.")
        return result
    inverted = df["to_depth_m"] <= df["from_depth_m"]
    if inverted.any():
        result.add_error(f"{int(inverted.sum())} borehole intervals have to_depth_m <= from_depth_m.")
    negative_depth = (df["from_depth_m"] < 0) | (df["to_depth_m"] < 0)
    if negative_depth.any():
        result.add_error(f"{int(negative_depth.sum())} borehole intervals have negative depths.")
    return result


def check_leakage(
    feature_matrix: pd.DataFrame,
    targets: list[str] | frozenset[str],
    *,
    context: str = "",
) -> ValidationResult:
    """Verify that none of the target columns leaked into a feature matrix.

    Targets are prediction labels (e.g. mn_pct for reserve grade models,
    production_mt for forecasting). They must never be present as features.
    """
    from ml.common.contracts import LEAKAGE_COLUMNS

    result = ValidationResult(name=f"leakage{':' + context if context else ''}")
    unknown = set(targets) - set(LEAKAGE_COLUMNS)
    if unknown:
        result.add_warning(
            f"Targets not registered in LEAKAGE_COLUMNS: {sorted(unknown)}"
        )
    for target in sorted(set(targets)):
        if target in feature_matrix.columns:
            result.add_error(f"Leakage: target column {target!r} present in feature matrix.")
    return result


def validate_all_synthetic() -> None:
    """Validate all demo/synthetic CSV files against their contracts.

    Run with:  python -m ml.common.validation
    """
    from backend.app.services.demo_data import DemoDataStore

    store = DemoDataStore()
    store.ensure_demo_data()

    contracts = {
        "geological": "geological.csv",
        "satellite_features": "satellite_features.csv",
        "boreholes": "boreholes.csv",
        "weather": "weather.csv",
        "equipment": "equipment.csv",
        "blasting": "blasting.csv",
        "production": "production.csv",
    }

    all_valid = True
    for name, filename in contracts.items():
        from ml.common.contracts import ALL_CONTRACTS

        contract = ALL_CONTRACTS[name]
        path = store.synthetic_dir / filename
        if not path.exists():
            print(f"  [MISSING] {filename}")
            all_valid = False
            continue
        df = pd.read_csv(path)
        result = validate_dataset(df, contract)
        if name == "boreholes":
            result = validate_borehole_intervals(df)
        if result.valid:
            print(f"  [OK] {filename}: {len(df)} rows valid")
        else:
            print(f"  [FAIL] {filename}: {len(df)} rows, {len(result.errors)} errors")
            for error in result.errors:
                print(f"    - {error}")
            all_valid = False

    if all_valid:
        print("\nAll synthetic datasets pass contract validation.")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    validate_all_synthetic()
