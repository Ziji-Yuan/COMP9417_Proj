"""Historical deployment analogs supporting a group-authored live-week forecast.

Six annual fits per task, each using the latest3years ending October31, frozen
while forecasting the nearby November/December weeks. All models see only
historically permitted features. Results are post-selection scenario evidence,
not an untouched test or a guarantee of prediction-interval coverage.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from Part_A.features import FeatureConfig, get_features
from Part_D_Drift.analysis import prepare_predictions
from Part_D_Drift.evaluation import aggregate_metrics
from .final_package.train import DATA_SHA256, create_models, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/part_e_live_forecast"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new/empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    with args.data.open("rb") as stream:
        data_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if data_hash != DATA_SHA256:
        raise ValueError("Use the fixed released CSV")
    env = environment()
    X, y, meta = get_features("2010-01-01", "2025-12-31", FeatureConfig(), path=args.data)
    all_days = pd.DatetimeIndex(meta.market_day)
    fit_records, exact_metrics, scenario_metrics = [], [], []
    for year in range(2020, 2026):
        cutoff = pd.Timestamp(year, 10, 31)
        train_start = cutoff + pd.Timedelta(days=1) - pd.DateOffset(years=3)
        training = (all_days >= train_start) & (all_days <= cutoff)
        forecast = (all_days >= pd.Timestamp(year, 11, 15)) & (all_days <= pd.Timestamp(year, 12, 19))
        model_pair = create_models()
        annual_predictions = []
        with threadpool_limits(limits=2):
            for task, target in [("demand", "nsw_demand_mw"), ("price", "nsw_price_up")]:
                started = time.perf_counter()
                model = model_pair[task]
                print(f"Analog {year}: fit {task} on {train_start.date()}--{cutoff.date()}", flush=True)
                model.fit(X.loc[training], y.loc[training, target])
                values = model.predict(X.loc[forecast])
                fit_records.append(dict(year=year, task=task, training_start=str(train_start.date()),
                                        training_end=str(cutoff.date()), n_training_rows=int(training.sum()),
                                        seconds=time.perf_counter() - started))
                annual_predictions.append(pd.DataFrame({"timestamp": X.index[forecast], "task": task,
                    "model": "random_forest", "strategy": "october_refit_three_years",
                    "y_true": y.loc[forecast, target].to_numpy(), "y_pred": values}))
                for lag, name in [(1, "yesterday_baseline"), (7, "last_week_baseline")]:
                    features = X.loc[forecast]
                    base = (features[f"nsw_demand_lag_{lag}d"].to_numpy() if task == "demand" else
                            (features[f"nsw_price_lag_{lag}d"] > features.nsw_price_prev_day_mean).to_numpy(dtype=int))
                    annual_predictions.append(pd.DataFrame({"timestamp": X.index[forecast], "task": task,
                        "model": name, "strategy": "frozen", "y_true": y.loc[forecast, target].to_numpy(), "y_pred": base}))
        long = prepare_predictions(pd.concat(annual_predictions, ignore_index=True), truth=y)
        long.to_csv(args.output / f"analog_{year}_predictions.csv.gz", index=False)
        day = pd.DatetimeIndex(long.market_day)
        exact = long[(day >= pd.Timestamp(year, 12, 7)) & (day <= pd.Timestamp(year, 12, 13))]
        exact = aggregate_metrics(exact)
        exact["year"] = year
        exact_metrics.append(exact)
        # Five nonoverlapping7day windows, excluding Christmas: horizons15--49days
        # after the October cutoff, surrounding the actual live-week37--43day gap.
        for i in range(5):
            start = pd.Timestamp(year, 11, 15) + pd.Timedelta(days=i * 7)
            finish = start + pd.Timedelta(days=6)
            selected = long[(day >= start) & (day <= finish)]
            scores = aggregate_metrics(selected)
            scores["year"], scores["week_start"], scores["week_end"] = year, start, finish
            scenario_metrics.append(scores)
        print(f"PASS analog {year}: December7--13 and five nearby complete weeks", flush=True)
        pd.DataFrame(fit_records).to_csv(args.output / "training_log.csv", index=False)
    exact = pd.concat(exact_metrics, ignore_index=True)
    scenario = pd.concat(scenario_metrics, ignore_index=True)
    exact.to_csv(args.output / "same_date_week_metrics.csv", index=False)
    scenario.to_csv(args.output / "nearby_week_scenarios.csv", index=False)
    records = []
    for task, metric in [("demand", "MAPE"), ("price", "BalancedAccuracy")]:
        direct = exact[(exact.task == task) & (exact.model == "random_forest")]
        nearby = scenario[(scenario.task == task) & (scenario.model == "random_forest")]
        assert len(direct) == 6 and len(nearby) == 30 and nearby.n_days.eq(7).all()
        lower, upper = np.quantile(nearby[metric], [.025, .975])
        records.append({"task": task, "metric": metric, "units": "percent",
                        "forecast_point_mean_of_six_same_date_weeks": float(direct[metric].mean()),
                        "recent_three_year_sensitivity_mean": float(direct.loc[direct.year >= 2023, metric].mean()),
                        "historical_scenario_p025": float(lower), "historical_scenario_p975": float(upper),
                        "historical_scenario_min": float(nearby[metric].min()),
                        "historical_scenario_max": float(nearby[metric].max()),
                        "n_exact_weeks": 6, "n_scenario_weeks": 30, "n_source_years": 6})
    result = pd.DataFrame(records)
    result.to_csv(args.output / "forecast_evidence.csv", index=False)
    summary = {"forecast_week": ["2026-12-07", "2026-12-13"], "analogs": list(range(2020, 2026)),
               "environment": env, "data_sha256": data_hash, "fits_per_task": 6,
               "point_rule": "Unweighted mean of six same-date analog week scores",
               "interval_rule": "Empirical2.5/97.5percentiles of30 nearby7day scenario scores; not a formal95percent prediction interval",
               "limitations": ["Only six source years; adjacent weeks and overlapping training sets are dependent",
                               "Historical simulation informed final strategy selection; these are retrospective post-selection diagnostics",
                               "Nearby weeks include different weather, price regimes and forecast horizons",
                               "Target-day weather is idealized reanalysis, as permitted by task",
                               "Unseen2026 changes/extremes can put actual live-week results outside this scenario range",
                               "Do not interpret these ranges as monthly-mean confidence intervals or guaranteed coverage"],
               "forecasts": records}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
