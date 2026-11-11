"""Retrospective links to B/C predictions; never used to calibrate the detector."""

import numpy as np
import pandas as pd

from Part_A.data import market_day
from .profiles import validate_half_hours


GROUPS = ["task", "model", "strategy"]


def prepare_predictions(frame: pd.DataFrame, truth: pd.DataFrame | None = None) -> pd.DataFrame:
    """Validate the documented long-form CSV contract, including whole days.

    Columns: timestamp, task (demand/price), model, y_true, y_pred;
    optional strategy (defaults to frozen). Price predictions must be hard 0/1.
    If truth is provided, check it against the official targets produced by A.
    """
    frame = frame.copy()
    required = {"timestamp", "task", "model", "y_true", "y_pred"}
    if required.difference(frame.columns):
        raise ValueError(f"Missing prediction columns: {sorted(required.difference(frame.columns))}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Predictions must be nonempty with no missing required values")
    if "strategy" not in frame:
        frame["strategy"] = "frozen"
    if frame[GROUPS].isna().any().any() or (frame[GROUPS].astype(str).apply(lambda c: c.str.strip()) == "").any().any():
        raise ValueError("Task, model and strategy names cannot be empty")
    if not set(frame.task).issubset({"demand", "price"}):
        raise ValueError("task must be demand or price")
    frame["timestamp"] = pd.to_datetime(frame.timestamp)
    frame[["y_true", "y_pred"]] = frame[["y_true", "y_pred"]].apply(pd.to_numeric)
    if not np.isfinite(frame[["y_true", "y_pred"]].to_numpy()).all():
        raise ValueError("Prediction values must be finite")
    for (task, _, _), rows in frame.groupby(GROUPS):
        validate_half_hours(pd.DatetimeIndex(rows.sort_values("timestamp").timestamp))
        if task == "demand" and ((rows.y_true <= 0).any() or not rows.y_pred.between(0, 20000).all()):
            raise ValueError("Demand truth must be positive; predictions must be in [0, 20000] MW")
        if task == "price" and not rows[["y_true", "y_pred"]].isin([0, 1]).all().all():
            raise ValueError("Price targets and predictions must be hard labels 0 or 1")
        if truth is not None:
            column = "nsw_demand_mw" if task == "demand" else "nsw_price_up"
            expected = truth[column].reindex(pd.DatetimeIndex(rows.timestamp))
            if expected.isna().any() or not np.allclose(expected.to_numpy(dtype=float), rows.y_true, rtol=1e-8, atol=1e-8):
                raise ValueError(f"Provided {task} y_true disagrees with Part A's official targets")
    frame["market_day"] = market_day(pd.DatetimeIndex(frame.timestamp)).to_numpy()
    frame["month"] = pd.DatetimeIndex(frame.market_day).to_period("M")
    return frame.sort_values(GROUPS + ["timestamp"]).reset_index(drop=True)


def _metrics(rows, task):
    truth, pred = rows.y_true.to_numpy(), rows.y_pred.to_numpy()
    if task == "demand":
        return {"MAPE": float(100 * np.mean(np.abs(truth - pred) / truth)),
                "MAE": float(np.mean(np.abs(truth - pred)))}
    return {"Accuracy": float(100 * np.mean(truth == pred)),
            "BalancedAccuracy": float(100 * np.mean([
                np.mean(pred[truth == label] == label) for label in np.unique(truth)]))}


def _aggregate(predictions, by):
    keys = GROUPS + [by]
    records = []
    for values, rows in predictions.groupby(keys, observed=True):
        records.append({**dict(zip(keys, values)), "n_half_hours": len(rows),
                        "n_days": rows.market_day.nunique(), **_metrics(rows, values[0])})
    return pd.DataFrame(records)


def relate_predictions(predictions, assignments, signals):
    """Return descriptive daily/monthly/cluster comparisons and associations.

    Same-day observed clusters and subsequent errors are legitimate *analysis*
    outputs, not predictive inputs. Associations are not causal evidence or
    proof that P(Y|X) changed. This function has no access to detector.fit().
    """
    days = pd.DatetimeIndex(predictions.market_day.unique())
    if not days.isin(signals.index).all() or not days.isin(assignments.index).all():
        raise ValueError("Prediction dates must lie within the historical simulation's signal coverage")
    joined = predictions.join(assignments[["cluster", "novel", "nearest_profile_distance"]], on="market_day")
    daily = _aggregate(joined, "market_day").join(assignments, on="market_day").join(signals, on="market_day")
    monthly = _aggregate(joined, "month")
    covered_days = joined[["task", "model", "strategy", "market_day"]].drop_duplicates()
    covered_days = covered_days.join(assignments[["cluster", "novel", "nearest_profile_distance"]], on="market_day")
    covered_days = covered_days.join(signals[["score", "alarm"]], on="market_day")
    covered_days["month"] = pd.DatetimeIndex(covered_days.market_day).to_period("M")
    context = covered_days.groupby(GROUPS + ["month"], observed=True).agg(
        novel_share=("novel", "mean"), profile_distance=("nearest_profile_distance", "mean"),
        input_drift_score=("score", "mean"), alarms=("alarm", "sum")).reset_index()
    monthly = monthly.merge(context, on=GROUPS + ["month"], validate="one_to_one")
    # Match each model's actual coverage rather than joining full-month shares
    # when a model supplied predictions for only part of that month.
    type_counts = pd.crosstab([covered_days[key] for key in GROUPS + ["month"]],
                             covered_days.cluster).reindex(
                                 columns=sorted(assignments.cluster.unique()), fill_value=0)
    type_shares = type_counts.div(type_counts.sum(axis=1), axis=0)
    type_shares.columns = [f"cluster_{int(k)}_share" for k in type_shares.columns]
    share_columns = list(type_shares.columns)
    monthly = monthly.merge(type_shares.reset_index(), on=GROUPS + ["month"], validate="one_to_one")
    by_cluster = _aggregate(joined, "cluster")
    associations = []
    for values, rows in monthly.groupby(GROUPS):
        metric = "MAPE" if values[0] == "demand" else "Accuracy"
        for feature in ["novel_share", "profile_distance", "input_drift_score", *share_columns]:
            rho = rows[metric].corr(rows[feature], method="spearman") if (
                len(rows) >= 3 and rows[metric].nunique() > 1 and rows[feature].nunique() > 1) else np.nan
            associations.append({**dict(zip(GROUPS, values)), "metric": metric,
                                 "feature": feature, "spearman_rho": rho, "n_months": len(rows)})
    return {"daily_performance": daily, "monthly_performance": monthly,
            "performance_by_cluster": by_cluster,
            "descriptive_associations": pd.DataFrame(associations)}
