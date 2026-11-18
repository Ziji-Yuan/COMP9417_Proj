"""Run chronological selected-model adaptation; output one durable series at a time."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from Part_A.features import FeatureConfig, get_features
from Part_D_Drift.analysis import prepare_predictions
from Part_D_Drift.evaluation import aggregate_metrics, monthly_confidence_intervals
from Part_D_Drift.run import environment_info
from .engine import simulate
from .policy import AdaptationConfig


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=Path("outputs/part_e_calibration"))
    parser.add_argument("--input-reference", type=Path, default=Path("outputs/part_d_complete"))
    parser.add_argument("--output", type=Path, default=Path("outputs/part_e_full"))
    parser.add_argument("--end", default="2026-10-31")
    parser.add_argument("--tasks", nargs="+", choices=["demand", "price"], default=["demand", "price"])
    parser.add_argument("--strategies", nargs="+", choices=["frozen", "periodic", "input_triggered", "error_triggered"],
                        default=["frozen", "periodic", "input_triggered", "error_triggered"])
    args = parser.parse_args()
    environment = environment_info()
    if not environment["matches_official_core"]:
        raise ValueError(f"Use the official Python/library versions: {environment}")
    end = pd.Timestamp(args.end)
    if end != end.normalize() or end < pd.Timestamp("2015-01-01") or end > pd.Timestamp("2026-10-31"):
        raise ValueError("End must be a simulation date from 2015 through October 2026")
    root = Path(__file__).resolve().parents[1]
    paths = {"demand": root / "Part_B_Demand/artifacts/rf_model.joblib",
             "price": root / "Part_C_Price_Direction/artifacts/random_forest.joblib"}
    data_hash = sha(args.data)
    calibration_summary = json.loads((args.calibration / "summary.json").read_text())
    input_summary = json.loads((args.input_reference / "run_summary.json").read_text())
    if data_hash != calibration_summary["data_sha256"] or data_hash != input_summary["data_sha256"]:
        raise ValueError("Calibration/input reference and simulation must use identical released data")
    signals_path = args.input_reference / "tables/input_drift_daily.csv"
    input_signals = pd.read_csv(signals_path, parse_dates=["market_day", "effective_day"]).set_index("market_day")
    input_signals = input_signals.loc["2015-01-01":end]
    declaration = {
        "data_sha256": data_hash, "end": str(end.date()), "start": "2015-01-01",
        "tasks": sorted(set(args.tasks)), "strategies": sorted(set(args.strategies)),
        "training_years": 3, "budget": 6, "n_jobs": 2,
        "model_sha256": {task: sha(paths[task]) for task in args.tasks},
        "detector_sha256": {task: sha(args.calibration / f"{task}_error_detector.joblib") for task in args.tasks},
        "input_signals_sha256": sha(signals_path),
        "source_sha256": {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "shared_source_sha256": {str(p.relative_to(root)): sha(p)
                                  for folder in ("Part_A", "Part_D_Drift")
                                  for p in sorted((root / folder).glob("*.py"))},
        "environment": environment,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "declaration.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != declaration:
            raise ValueError("Existing output belongs to a different configuration/source; use new directory")
    elif any(args.output.iterdir()):
        raise ValueError("Nonempty output without declaration")
    else:
        manifest.write_text(json.dumps(declaration, indent=2) + "\n")
    print("Building causal feature table once (no learned preprocessing)...", flush=True)
    X, y, meta = get_features("2010-01-01", end, FeatureConfig(), path=args.data)
    all_days = pd.DatetimeIndex(meta.market_day)
    active = all_days >= pd.Timestamp("2015-01-01")
    simulation_X = X.loc[active]
    for task in dict.fromkeys(args.tasks):
        target = "nsw_demand_mw" if task == "demand" else "nsw_price_up"
        for strategy in dict.fromkeys(args.strategies):
            output = args.output / f"{task}_{strategy}"
            complete = output / "summary.json"
            if complete.exists():
                completed = json.loads(complete.read_text())
                if sha(output / "predictions.csv.gz") != completed["prediction_sha256"]:
                    raise ValueError("Completed predictions were altered; use a fresh output directory")
                print(f"Skip completed {task}/{strategy}", flush=True)
                continue
            output.mkdir(exist_ok=True)
            started = time.perf_counter()
            print(f"START {task}/{strategy} through {end.date()}", flush=True)
            initial = joblib.load(paths[task]).set_params(n_jobs=2)
            if list(initial.feature_names_in_) != list(X.columns):
                raise ValueError("Initial model feature schema differs from A")
            monitor = joblib.load(args.calibration / f"{task}_error_detector.joblib").reset()
            training_records = []

            def fit(request):
                selected = (all_days >= request.train_start) & (all_days <= request.train_end)
                if not selected.any() or all_days[selected].max() > request.observed_day:
                    raise ValueError("Invalid past-only training window")
                replacement = clone(initial)
                replacement.fit(X.loc[selected], y.loc[selected, target])
                training_records.append({"ordinal": request.ordinal, "n_training_rows": int(selected.sum()),
                                         "actual_train_start": all_days[selected].min(),
                                         "actual_train_end": all_days[selected].max()})
                return replacement

            config = AdaptationConfig(strategy, end_day=str(end.date()))
            with threadpool_limits(limits=2):
                result = simulate(simulation_X, y.loc[active, target], all_days[active], initial, monitor, config,
                                  input_signals=input_signals, fit_model=fit, progress=lambda s: print(s, flush=True))
            predictions = pd.DataFrame({"timestamp": simulation_X.index, "task": task, "model": "random_forest",
                                        "strategy": strategy, "y_true": y.loc[active, target].to_numpy(),
                                        "y_pred": result["prediction"], "model_id": result["model_ids"]})
            validated = prepare_predictions(predictions, truth=y)
            validated.to_csv(output / "predictions.csv.gz", index=False)
            result["signals"].to_csv(output / "error_signals.csv", index=False)
            retrains = result["retrains"]
            if len(retrains):
                retrains = retrains.merge(pd.DataFrame(training_records), on="ordinal", validate="one_to_one")
            if retrains.empty:
                retrains = pd.DataFrame(columns=["observed_day", "effective_day", "train_start", "train_end",
                                                "ordinal", "reason", "model_id", "fit_seconds", "n_training_rows"])
            retrains.to_csv(output / "retrains.csv", index=False)
            audit = result["audit"]
            if audit.empty:
                audit = pd.DataFrame(columns=["observed_day", "status"])
            audit.to_csv(output / "budget_audit.csv", index=False)
            summary = {"task": task, "strategy": strategy, "config": asdict(config),
                       "n_days": int(active.sum() // 48), "n_predictions": int(active.sum()),
                       "retrain_count": result["retrain_count"], "seconds": time.perf_counter() - started,
                       "prediction_sha256": sha(output / "predictions.csv.gz"),
                       "status": "complete"}
            # Completion marker written last; incomplete series can be rerun safely.
            complete.write_text(json.dumps(summary, indent=2) + "\n")
            print(f"DONE {task}/{strategy}: retrains={result['retrain_count']}", flush=True)
            del result, initial, validated
    tables = []
    for task in dict.fromkeys(args.tasks):
        target = "nsw_demand_mw" if task == "demand" else "nsw_price_up"
        for strategy in dict.fromkeys(args.strategies):
            tables.append(pd.read_csv(args.output / f"{task}_{strategy}/predictions.csv.gz"))
        for lag, name in [(1, "yesterday_baseline"), (7, "last_week_baseline")]:
            if task == "demand":
                pred = simulation_X[f"nsw_demand_lag_{lag}d"].to_numpy()
            else:
                pred = (simulation_X[f"nsw_price_lag_{lag}d"] > simulation_X.nsw_price_prev_day_mean).to_numpy(dtype=int)
            tables.append(pd.DataFrame({"timestamp": simulation_X.index, "task": task, "model": name,
                                        "strategy": "frozen", "y_true": y.loc[active, target].to_numpy(), "y_pred": pred}))
    combined = prepare_predictions(pd.concat(tables, ignore_index=True), truth=y)
    combined["year"] = pd.DatetimeIndex(combined.market_day).year
    for name, by in [("overall", ()), ("yearly", ("year",)), ("monthly", ("month",))]:
        aggregate_metrics(combined, by=by).to_csv(args.output / f"{name}_performance.csv", index=False)
    monthly_confidence_intervals(combined).to_csv(args.output / "monthly_confidence_intervals.csv", index=False)
    print("COMPLETE: pooled metrics and both monthly confidence intervals exported", flush=True)


if __name__ == "__main__":
    main()
