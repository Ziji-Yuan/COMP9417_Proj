"""Check saved-model integration against the official day-ahead harness.

No models are fitted or modified on disk. Representative boundary days are
diagnostics, not a strategy-selection or generalization evaluation.
"""

import argparse
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from Part_A.features import FeatureConfig, get_features, get_inference_features
from Part_C_Price_Direction.predict_price import predict_price


def load_harness(kit):
    path = kit / "harness" / "dayahead.py"
    spec = importlib.util.spec_from_file_location("official_dayahead", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/part_e_smoke"))
    args = parser.parse_args()
    kit = args.kit.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use an absent or empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    official = load_harness(kit)
    csv = kit / "data/nem_weather_halfhourly_2010_2026.csv"
    raw = official.load_dataset(csv)
    config = FeatureConfig()
    root = Path(__file__).resolve().parents[1]
    files = {
        "demand_random_forest": root / "Part_B_Demand/artifacts/rf_model.joblib",
        "price_random_forest": root / "Part_C_Price_Direction/artifacts/random_forest.joblib",
        "price_logistic_regression": root / "Part_C_Price_Direction/artifacts/logistic_regression.joblib",
    }
    models = {}
    load_seconds = {}
    for name, path in files.items():
        start = time.perf_counter()
        model = joblib.load(path)
        if hasattr(model, "n_jobs"):
            model.set_params(n_jobs=2)
        models[name] = model
        load_seconds[name] = time.perf_counter() - start

    # Exercise deployment start, leap day, year-end midnight and data boundary.
    days = ["2015-01-01", "2016-02-29", "2021-12-31", "2026-10-31"]
    records, forecasts = [], []
    price_reference = pd.read_csv(
        root / "Part_C_Price_Direction/results/price_predictions_for_part_d.csv",
        parse_dates=["timestamp"],
    )
    for day in days:
        history, weather = official.make_inputs(raw, day)
        truth = official.targets(raw, day)
        start = time.perf_counter()
        inference = get_inference_features(history, weather, config)
        feature_seconds = time.perf_counter() - start
        bulk, target, _ = get_features(day, day, config, data=raw)
        pd.testing.assert_frame_equal(inference, bulk)
        pd.testing.assert_frame_equal(target, truth, check_dtype=False)
        if inference.shape != (48, 47) or not np.isfinite(inference.to_numpy()).all():
            raise AssertionError("Expected finite 48x47 features")
        predictions = {}
        prediction_seconds = {}
        with threadpool_limits(limits=2):
            for name, model in models.items():
                if list(model.feature_names_in_) != list(inference.columns):
                    raise AssertionError(f"Feature schema mismatch: {name}")
                start = time.perf_counter()
                pred = (model.predict(inference) if name.startswith("demand")
                        else predict_price(inference, model=model))
                prediction_seconds[name] = time.perf_counter() - start
                if len(pred) != 48 or not np.isfinite(pred).all():
                    raise AssertionError(f"Invalid output: {name}")
                if name.startswith("price") and not np.isin(pred, [0, 1]).all():
                    raise AssertionError("Price predictions must be hard labels")
                if name.startswith("demand") and not ((pred >= 0) & (pred <= 20000)).all():
                    raise AssertionError("Demand prediction outside harness limits")
                predictions[name] = pred
                if name.startswith("price"):
                    model_name = name.removeprefix("price_")
                    reference = price_reference.loc[
                        price_reference.timestamp.isin(weather.index)
                        & price_reference.model.eq(model_name)
                    ].sort_values("timestamp")
                    if len(reference) != 48:
                        raise AssertionError("C export lacks complete diagnostic day")
                    np.testing.assert_array_equal(reference.timestamp.to_numpy(), weather.index.to_numpy())
                    np.testing.assert_array_equal(reference.y_true.to_numpy(), truth.nsw_price_up.to_numpy())
                    np.testing.assert_array_equal(reference.y_pred.to_numpy(), pred)
                forecasts.extend({
                    "timestamp": stamp, "task": "demand" if name.startswith("demand") else "price",
                    "model": name, "strategy": "frozen", "y_true": float(actual),
                    "y_pred": float(value),
                } for stamp, actual, value in zip(
                    weather.index, truth.nsw_demand_mw if name.startswith("demand") else truth.nsw_price_up, pred
                ))
        joint = pd.DataFrame({"nsw_demand_mw": predictions["demand_random_forest"],
                              "nsw_price_up": predictions["price_random_forest"]}, index=weather.index)
        for name, pred in [("selected_frozen", joint),
                           ("yesterday", official.baseline(history, weather, 1)),
                           ("last_week", official.baseline(history, weather, 7))]:
            records.append({"market_day": day, "method": name, **official.score(truth, pred)})
        records[-3].update(feature_seconds=feature_seconds,
                           prediction_seconds=sum(prediction_seconds.values()))
        print(f"PASS {day}: features, official targets, model outputs, C exported predictions", flush=True)
    pd.DataFrame(records).to_csv(args.output / "daily_metrics.csv", index=False)
    pd.DataFrame(forecasts).to_csv(args.output / "diagnostic_predictions.csv", index=False)
    summary = {
        "purpose": "Boundary integration diagnostics only; no fitting or strategy selection",
        "days": days, "checks_passed": True, "features": 47,
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "data_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
        "model_load_seconds": load_seconds,
        "model_bytes": {name: path.stat().st_size for name, path in files.items()},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
