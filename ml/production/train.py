from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from ml.common.metrics import mae, r2, rmse
from ml.common.registry import ModelVersionRecord, hash_file, hash_schema, write_registry_record
from ml.production.features import FEATURE_COLUMNS, build_daily_features
from ml.production.pipeline import HORIZONS, add_future_targets, assert_no_target_leakage, chronological_splits, load_fused_production_data


def _candidate_regressors(seed: int = 42) -> dict[str, object]:
    return {
        "naive_28": None,
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=seed, n_jobs=-1),
        "xgboost": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.9, objective="reg:squarederror", random_state=seed, n_jobs=-1),
    }


def _candidate_classifiers(seed: int = 42) -> dict[str, object]:
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.9, eval_metric="logloss", random_state=seed, n_jobs=-1),
    }


def _train_horizon(root: Path, fused, horizon: int) -> dict:
    supervised = add_future_targets(fused, horizon)
    features = build_daily_features(supervised)
    assert_no_target_leakage(FEATURE_COLUMNS)
    train, validation, test = chronological_splits(features)
    x_train, y_train = train[FEATURE_COLUMNS], train["future_production_mt"]
    x_val, y_val = validation[FEATURE_COLUMNS], validation["future_production_mt"]
    x_test, y_test = test[FEATURE_COLUMNS], test["future_production_mt"]

    regressors = _candidate_regressors()
    reg_scores = {}
    for name, model in regressors.items():
        if model is None:
            pred = np.repeat(float(train["production_mt"].tail(28).mean()), len(validation))
        else:
            pred = clone(model).fit(x_train, y_train).predict(x_val)
        reg_scores[name] = rmse(y_val, pred)
    best_reg_name = min(reg_scores, key=reg_scores.get)
    if best_reg_name == "naive_28":
        best_reg = None
        val_pred = np.repeat(float(train["production_mt"].tail(28).mean()), len(validation))
        test_pred = np.repeat(float(pd.concat([train, validation])["production_mt"].tail(28).mean()), len(test))
    else:
        best_reg = clone(regressors[best_reg_name]).fit(x_train, y_train)
        val_pred = best_reg.predict(x_val)
        test_pred = best_reg.predict(x_test)

    conformal_q = float(np.quantile(np.abs(y_val.to_numpy() - np.asarray(val_pred)), 0.90, method="higher"))

    y_all = features["shortfall_label"].astype(int)
    classifiers = _candidate_classifiers()
    clf_scores = {}
    for name, model in classifiers.items():
        fitted = clone(model).fit(train[FEATURE_COLUMNS], train["shortfall_label"])
        clf_scores[name] = average_precision_score(validation["shortfall_label"], fitted.predict_proba(validation[FEATURE_COLUMNS])[:, 1])
    best_clf_name = max(clf_scores, key=clf_scores.get)
    classifier = clone(classifiers[best_clf_name]).fit(train[FEATURE_COLUMNS], train["shortfall_label"])
    calibrated = CalibratedClassifierCV(classifier, method="isotonic", cv="prefit")
    calibrated.fit(validation[FEATURE_COLUMNS], validation["shortfall_label"])
    test_prob = calibrated.predict_proba(test[FEATURE_COLUMNS])[:, 1]

    artifact_dir = root / "models" / "production"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if best_reg is not None:
        joblib.dump(best_reg, artifact_dir / f"forecast_{horizon}d.pkl")
    else:
        (artifact_dir / f"forecast_{horizon}d.pkl").unlink(missing_ok=True)
    joblib.dump(calibrated, artifact_dir / f"shortfall_{horizon}d.pkl")
    joblib.dump(FEATURE_COLUMNS, artifact_dir / f"features_{horizon}d.pkl")

    metrics = {
        "horizon_days": horizon,
        "validation": "chronological_70_15_15",
        "walk_forward_ready": True,
        "leakage_check_passed": True,
        "candidate_regression_rmse": {k: round(float(v), 4) for k, v in reg_scores.items()},
        "selected_regressor": best_reg_name,
        "candidate_shortfall_pr_auc": {k: round(float(v), 4) for k, v in clf_scores.items()},
        "selected_classifier": best_clf_name,
        "test_mae": round(mae(y_test, test_pred), 4),
        "test_rmse": round(rmse(y_test, test_pred), 4),
        "test_r2": round(r2(y_test, test_pred), 4),
        "test_shortfall_pr_auc": round(float(average_precision_score(test["shortfall_label"], test_prob)), 4),
        "test_shortfall_brier": round(float(brier_score_loss(test["shortfall_label"], test_prob)), 4),
        "conformal_quantile_90": round(conformal_q, 4),
        "synthetic_data": True,
    }
    version = f"2026.09.{horizon + 1:03d}"
    training_hash = hash_file(root / "data/synthetic/production.csv")[:16]
    schema_hash = hash_schema({"features": FEATURE_COLUMNS, "horizon": horizon})[:16]
    for name, task, algorithm, artifact, target in (
        (f"production_forecast_{horizon}d", "future_production_regression", best_reg_name, f"models/production/forecast_{horizon}d.pkl" if best_reg is not None else "", "future_production_mt"),
        (f"production_shortfall_{horizon}d", "shortfall_classification", f"{best_clf_name}+isotonic", f"models/production/shortfall_{horizon}d.pkl", "shortfall_label"),
    ):
        write_registry_record(ModelVersionRecord(model_name=name, version=version, task=task, algorithm=algorithm, training_data_hash=training_hash, feature_schema_hash=schema_hash, metrics=metrics, artifact_path=artifact, status="candidate", target=target, feature_names=FEATURE_COLUMNS, validation_strategy="chronological_70_15_15"), root / "models" / "registry")
    return metrics


def train_production_models(root: Path | None = None) -> dict:
    root = root or Path(__file__).resolve().parents[2]
    fused = load_fused_production_data(root)
    results = {str(h): _train_horizon(root, fused, h) for h in HORIZONS}
    report = {"models": results, "horizons": list(HORIZONS), "synthetic_data": True}
    (root / "models" / "production").mkdir(parents=True, exist_ok=True)
    (root / "models" / "production" / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(train_production_models(), indent=2))
