"""Audit completed adaptation series against independent D/B/C handovers."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Part_D_Drift.analysis import prepare_predictions
from Part_D_Drift.evaluation import aggregate_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("outputs/part_d_complete"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    declaration = json.loads((args.results / "declaration.json").read_text())
    days = pd.date_range(declaration["start"], declaration["end"])
    reference = pd.read_csv(args.reference / "joint/predictions/baseline_predictions.csv.gz", parse_dates=["timestamp"])
    reference = reference[reference.model.eq("yesterday_baseline")]
    frozen_demand = pd.read_csv(args.reference / "joint/predictions/demand_predictions_for_part_d.csv.gz", parse_dates=["timestamp"])
    frozen_demand = frozen_demand[frozen_demand.model.eq("random_forest")]
    frozen_price = pd.read_csv(root / "Part_C_Price_Direction/results/price_predictions_for_part_d.csv", parse_dates=["timestamp"])
    frozen_price = frozen_price[frozen_price.model.eq("random_forest")]
    records, matched = [], []
    for task in declaration["tasks"]:
        truth = reference[reference.task.eq(task)].set_index("timestamp").y_true
        for strategy in declaration["strategies"]:
            directory = args.results / f"{task}_{strategy}"
            summary = json.loads((directory / "summary.json").read_text())
            path = directory / "predictions.csv.gz"
            with path.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == summary["prediction_sha256"]
            frame = prepare_predictions(pd.read_csv(path, parse_dates=["timestamp"]))
            expected_index = pd.date_range(days[0] + pd.Timedelta(minutes=30), periods=48 * len(days), freq="30min")
            np.testing.assert_array_equal(frame.timestamp.to_numpy(), expected_index.to_numpy())
            np.testing.assert_allclose(frame.y_true, truth.reindex(expected_index), rtol=0, atol=1e-9)
            if strategy == "frozen":
                source = frozen_demand if task == "demand" else frozen_price
                source = source.set_index("timestamp").reindex(expected_index)
                if task == "price":
                    np.testing.assert_array_equal(frame.y_pred, source.y_pred)
                else:
                    np.testing.assert_allclose(frame.y_pred, source.y_pred, rtol=1e-12, atol=1e-9)
            try:
                retrains = pd.read_csv(directory / "retrains.csv")
            except pd.errors.EmptyDataError:
                retrains = pd.DataFrame()
            count = summary["retrain_count"]
            assert len(retrains) == count and count <= 6
            if count:
                for column in ["observed_day", "effective_day", "train_start", "train_end", "actual_train_end"]:
                    retrains[column] = pd.to_datetime(retrains[column])
                np.testing.assert_array_equal(retrains.ordinal, np.arange(1, count + 1))
                assert (retrains.actual_train_end <= retrains.observed_day).all()
                assert (retrains.train_end <= retrains.observed_day).all()
                assert ((retrains.effective_day - retrains.observed_day) == pd.Timedelta(days=1)).all()
                assert retrains.effective_day.is_monotonic_increasing
            expected_models = pd.Series("stage1_refit_2010_2014", index=days)
            if count:
                for row in retrains.itertuples():
                    expected_models.loc[row.effective_day:] = row.model_id
            np.testing.assert_array_equal(frame.model_id, expected_models.repeat(48).to_numpy())
            record = {"task": task, "strategy": strategy, "days": len(days), "predictions": len(frame),
                      "retrains": count, "targets_match_independent_handover": True,
                      "model_activation_and_training_cutoff_valid": True}
            records.append(record)
            frame = frame[pd.DatetimeIndex(frame.market_day).month <= 10]
            frame["year"] = pd.DatetimeIndex(frame.market_day).year
            matched.append(frame)
    ci = pd.read_csv(args.results / "monthly_confidence_intervals.csv")
    n_series = len(declaration["tasks"]) * (len(declaration["strategies"]) + 2)
    assert len(ci) == n_series * len(pd.period_range(days[0], days[-1], freq="M"))
    assert ci.n_bootstrap.ge(1000).all()
    assert np.isfinite(ci[["iid_normal_lower", "iid_normal_upper", "day_bootstrap_lower", "day_bootstrap_upper"]]).all().all()
    published = pd.read_csv(args.results / "overall_performance.csv")
    independent = pd.read_csv(args.reference / "joint/tables/overall_performance.csv")
    # Match common time coverage before comparing full-period baseline metrics.
    if declaration["start"] == "2015-01-01" and declaration["end"] == "2026-10-31":
        for task in declaration["tasks"]:
            for name in ["yesterday_baseline", "last_week_baseline"]:
                actual = published[(published.task == task) & (published.model == name)].iloc[0]
                expected = independent[(independent.task == task) & (independent.model == name)].iloc[0]
                for metric in (["MAPE", "MAE"] if task == "demand" else ["Accuracy", "BalancedAccuracy"]):
                    np.testing.assert_allclose(actual[metric], expected[metric], rtol=1e-12, atol=1e-9)
    aggregate_metrics(pd.concat(matched), by=("year",)).to_csv(args.results / "yearly_jan_oct_model_performance.csv", index=False)
    (args.results / "verification.json").write_text(json.dumps({"status": "passed", "series": records,
        "monthly_ci_rows": len(ci), "cross_reference": str(args.reference),
        "note": "Demand forest comparisons tolerate1e-9MW floating reduction rounding; classification checked exactly"}, indent=2) + "\n")
    print(f"PASS: {len(records)} series, {len(days)} days each, {len(ci)} monthly confidence-interval rows")


if __name__ == "__main__":
    main()
