"""Reproduce B's fixed configuration and export forecasts without modifying B/C."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from Part_A.data import market_day
from Part_A.features import FeatureConfig, get_features, get_inference_features
from .profiles import validate_half_hours


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fit_b_ridge(X, y):
    """Exactly B's declared scaler/Ridge configuration; fit through 2014 only."""
    days = market_day(X.index)
    selected = (days >= pd.Timestamp("2010-01-01")) & (days <= pd.Timestamp("2014-12-31"))
    if not selected.any():
        raise ValueError("Ridge reconstruction needs training rows within 2010--2014")
    if not X.index.equals(y.index):
        raise ValueError("Ridge feature/target indices must match")
    model = make_pipeline(StandardScaler(), Ridge(alpha=1000.0))
    model.fit(X.loc[selected], y.loc[selected])
    return model


def baseline_predictions(X, truth, meta):
    """Both official baselines; last-week price uses yesterday's mean threshold."""
    frames = []
    for name, lag in (("yesterday_baseline", 1), ("last_week_baseline", 7)):
        for task, target in (("demand", "nsw_demand_mw"), ("price", "nsw_price_up")):
            values = (X[f"nsw_demand_lag_{lag}d"].to_numpy() if task == "demand" else
                      (X[f"nsw_price_lag_{lag}d"].to_numpy() >
                       meta.price_reference_prev_day_mean.to_numpy()).astype(int))
            frames.append(pd.DataFrame({"timestamp": X.index, "task": task, "model": name,
                                        "strategy": "baseline", "y_true": truth[target].to_numpy(),
                                        "y_pred": values}))
    return pd.concat(frames, ignore_index=True)


def export_demand_and_baselines(observed, rf_path, output):
    """Daily frozen forecasts using B's saved RF and a reconstructed Ridge.

    No model selection or fitting uses 2015+ data. Building all causal features
    together is equivalent to A's day-ahead inference interface; representative
    days are checked explicitly before forecasting. All predictions are then
    generated in daily chronological order with immutable fitted parameters.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rf_path = Path(rf_path)
    if rf_path.stat().st_size < 1000:
        raise ValueError("Demand RF is still a Git LFS pointer; run git lfs pull")
    config = FeatureConfig()
    train_X, train_y, train_meta = get_features("2010-01-01", "2014-12-31", config=config, data=observed)
    ridge = fit_b_ridge(train_X, train_y.nsw_demand_mw)
    rf = joblib.load(rf_path)
    expected = {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 2, "random_state": 42}
    if any(rf.get_params().get(k) != v for k, v in expected.items()):
        raise ValueError("Saved RF configuration differs from B's published configuration")
    if list(rf.feature_names_in_) != list(train_X.columns):
        raise ValueError("B's RF feature schema differs from A's default interface")
    root_counts = [int(tree.tree_.weighted_n_node_samples[0]) for tree in rf.estimators_]
    if any(n != len(train_X) for n in root_counts):
        raise ValueError("RF root sample counts do not match the declared 2010--2014 fit")
    rf.set_params(n_jobs=1)  # Scheduling only; learned trees and forecasts are unchanged.
    artifact_dir = output / "artifacts"
    artifact_dir.mkdir()
    ridge_path = artifact_dir / "demand_ridge_rebuilt.joblib"
    joblib.dump(ridge, ridge_path, compress=3)
    X, truth, meta = get_features("2015-01-01", "2026-10-31", config=config, data=observed)
    validate_half_hours(X.index)
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Nonfinite forecasting features")
    observed_days = market_day(observed.index)
    check_days = ["2015-01-01", "2019-07-01", "2023-01-01", "2026-10-31"]
    for value in check_days:
        day = pd.Timestamp(value)
        history = observed.loc[observed_days < day]
        weather = observed.loc[observed_days == day, list(config.weather_columns)]
        inference = get_inference_features(history, weather, config=config)
        pd.testing.assert_frame_equal(X.loc[inference.index], inference)
    predictions = {"ridge_regression": np.empty(len(X)), "random_forest": np.empty(len(X))}
    for day, positions in meta.groupby("market_day", sort=True).indices.items():
        if len(positions) != 48:
            raise ValueError(f"Incomplete evaluation day: {day}")
        day_X = X.iloc[positions]
        predictions["ridge_regression"][positions] = ridge.predict(day_X)
        predictions["random_forest"][positions] = rf.predict(day_X)
        if day.month == 1 and day.day == 1:
            print(f"Frozen demand evaluation: {day.year}", flush=True)
    frames = [pd.DataFrame({"timestamp": X.index, "task": "demand", "model": name,
                            "strategy": "frozen", "y_true": truth.nsw_demand_mw.to_numpy(),
                            "y_pred": values}) for name, values in predictions.items()]
    demand = pd.concat(frames, ignore_index=True)
    baselines = baseline_predictions(X, truth, meta)
    predictions_dir = output / "predictions"
    predictions_dir.mkdir()
    demand.to_csv(predictions_dir / "demand_predictions_for_part_d.csv.gz", index=False,
                  compression={"method": "gzip", "mtime": 0})
    baselines.to_csv(predictions_dir / "baseline_predictions.csv.gz", index=False,
                     compression={"method": "gzip", "mtime": 0})
    metadata = {
        "ridge": {"source": "Part_B_Demand/train_demand.py: StandardScaler + Ridge(alpha=1000)",
                  "fit_start": str(train_meta.market_day.min().date()),
                  "fit_end": str(train_meta.market_day.max().date()), "fit_rows": len(train_X),
                  "artifact_sha256": file_sha256(ridge_path), "hyperparameters_reselected": False},
        "random_forest": {"source": str(rf_path.resolve()), "sha256": file_sha256(rf_path),
                          "declared_fit_period": ["2010-01-01", "2014-12-31"],
                          "provenance_basis": "B's training source and matching schema/parameters/root sample counts; timestamps are not stored in the estimator",
                          "parameters_checked": expected, "root_sample_count": root_counts[0],
                          "execution_n_jobs": 1, "refitted": False},
        "feature_columns": list(X.columns), "inference_equivalence_days": check_days,
        "evaluation_days": int(meta.market_day.nunique()), "intervals_per_model": len(X),
        "prediction_order": "chronological, one market day at a time", "retraining_during_simulation": False,
    }
    return demand, baselines, truth, metadata


def main():
    import argparse
    from threadpoolctl import threadpool_limits
    from Part_A.features import get_data
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--demand-rf", default="Part_B_Demand/artifacts/rf_model.joblib")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory")
    with threadpool_limits(limits=2):
        _, _, _, metadata = export_demand_and_baselines(get_data(path=args.data), args.demand_rf, output)
    metadata["data_sha256"] = file_sha256(args.data)
    metadata["source_sha256"] = file_sha256(__file__)
    metadata["generated_files_sha256"] = {str(p.relative_to(output)): file_sha256(p)
                                         for p in sorted(output.rglob("*")) if p.is_file()}
    (output / "demand_handover.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"evaluation_days": metadata["evaluation_days"],
                      "intervals_per_model": metadata["intervals_per_model"]}), flush=True)


if __name__ == "__main__":
    main()
