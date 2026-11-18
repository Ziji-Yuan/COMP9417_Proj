"""Generate genuinely out-of-sample 2014 losses for error-monitor calibration.

Uses B/C's selected RF configurations, never their 2010--2014 fitted parameters.
No 2015+ outcomes select thresholds. Calibration is not a prospective detector
evaluation, nor an independent evaluation of the configurations selected in 2014.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from Part_A.features import FeatureConfig, get_features
from .policy import ErrorDriftDetector, daily_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/part_e_calibration"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use an empty or absent output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    X, y, meta = get_features("2010-01-01", "2014-12-31", FeatureConfig(), path=args.data)
    days = pd.DatetimeIndex(meta.market_day)
    training = days <= pd.Timestamp("2013-12-31")
    validation = (days >= pd.Timestamp("2014-01-01")) & (days <= pd.Timestamp("2014-12-31"))
    val_days = days[validation]
    root = Path(__file__).resolve().parents[1]
    sources = {"demand": root / "Part_B_Demand/artifacts/rf_model.joblib",
               "price": root / "Part_C_Price_Direction/artifacts/random_forest.joblib"}
    metadata, losses_out, predictions = {}, [], []
    for task, path in sources.items():
        target = "nsw_demand_mw" if task == "demand" else "nsw_price_up"
        # clone discards all fitted parameters; only the published configuration survives.
        model = clone(joblib.load(path)).set_params(n_jobs=2)
        start = time.perf_counter()
        print(f"Training {task} calibration forecaster: {training.sum()} rows through 2013", flush=True)
        with threadpool_limits(limits=2):
            model.fit(X.loc[training], y.loc[training, target])
            pred = model.predict(X.loc[validation])
        truth = y.loc[validation, target].to_numpy()
        records = []
        for day in val_days.unique():
            selected = val_days == day
            records.append((day, daily_loss(task, truth[selected], pred[selected])))
        losses = pd.Series(dict(records), name="loss").sort_index()
        monitor = ErrorDriftDetector(task).fit(losses, predictor_training_end="2013-12-31")
        joblib.dump(monitor, args.output / f"{task}_error_detector.joblib")
        losses_out.extend(dict(market_day=day, task=task, loss=loss) for day, loss in records)
        predictions.extend(dict(timestamp=stamp, task=task, y_true=float(actual), y_pred=float(value))
                           for stamp, actual, value in zip(X.index[validation], truth, pred))
        metadata[task] = {"config": asdict(monitor.config), "threshold": monitor.threshold_,
                          "loss_units": "percent", "calibration_mean_loss": float(losses.mean()),
                          "training_start": str(days[training].min().date()),
                          "training_end": str(days[training].max().date()),
                          "calibration_start": str(losses.index.min().date()),
                          "calibration_end": str(losses.index.max().date()),
                          "training_rows": int(training.sum()), "validation_rows": int(validation.sum()),
                          "model_parameters": model.get_params(),
                          "seconds": time.perf_counter() - start,
                          "configuration_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"PASS {task}: 2014 loss={losses.mean():.4f}%, threshold={monitor.threshold_:.4f}", flush=True)
    pd.DataFrame(losses_out).to_csv(args.output / "calibration_daily_losses.csv", index=False)
    pd.DataFrame(predictions).to_csv(args.output / "calibration_predictions.csv", index=False)
    summary = {"tasks": metadata, "python": platform.python_version(), "sklearn": sklearn.__version__,
               "numpy": np.__version__, "pandas": pd.__version__,
               "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
               "feature_columns": list(X.columns),
               "limitations": ["2014 also selected model configurations, so threshold calibration is not independent validation",
                               "Deployment models refit through 2014 differ from calibration predictors",
                               "Empirical quantile does not guarantee a future false-alarm rate",
                               "Fixed threshold after retraining; errors must come from each deployed strategy"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
