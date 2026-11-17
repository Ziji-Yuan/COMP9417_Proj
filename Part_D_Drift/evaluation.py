"""Retrospective frozen-model evidence and Appendix A monthly intervals.

No predictive model or detector is selected/refitted in this module. Correlations
and before/after comparisons are descriptive, not causal or change-point tests.
"""

import hashlib

import numpy as np
import pandas as pd

from .analysis import GROUPS, _metrics, relate_predictions
from .diagnostics import drift_episodes


BASELINE_NAMES = ("yesterday_baseline", "last_week_baseline")


def monthly_interval(values_by_day, task, n_boot=2000, seed=9417):
    """Return untruncated normal/Hoeffding bounds and whole-day percentiles.

    Values are per-half-hour absolute percentage errors for demand, or 0/1
    correctness for price. Resampling day means is algebraically identical to
    concatenating sampled complete 48-slot days for these two mean statistics.
    """
    values = np.asarray(values_by_day, dtype=float)
    if values.ndim != 2 or values.shape[1] != 48 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Intervals need finite complete days with exactly 48 half-hours")
    if n_boot < 1000:
        raise ValueError("Appendix A requires at least 1000 bootstrap replicates")
    if task not in ("demand", "price"):
        raise ValueError("task must be demand or price")
    if (values < 0).any() or (task == "price" and not np.isin(values, [0, 1]).all()):
        raise ValueError("Use nonnegative APEs or binary correctness indicators")
    n, k = values.size, len(values)
    mean = float(values.mean())
    factor = 100.0 if task == "price" else 1.0
    se = (np.sqrt(mean * (1 - mean) / n) if task == "price" else values.std(ddof=1) / np.sqrt(n))
    rng = np.random.default_rng(seed)
    means = values.mean(axis=1)
    boot = means[rng.integers(0, k, size=(n_boot, k))].mean(axis=1) * factor
    low, high = np.quantile(boot, [.025, .975])
    half = 1.96 * float(se) * factor
    result = {"estimate": mean * factor, "iid_normal_lower": mean * factor - half,
              "iid_normal_upper": mean * factor + half, "day_bootstrap_lower": float(low),
              "day_bootstrap_upper": float(high), "iid_normal_width": 2 * half,
              "day_bootstrap_width": float(high - low), "n_days": k, "n_half_hours": n,
              "n_bootstrap": n_boot, "bootstrap_seed": seed}
    result["day_to_iid_width_ratio"] = (high - low) / (2 * half) if half > 0 else np.nan
    if task == "price":
        hoeffding = np.sqrt(np.log(2 / .05) / (2 * n)) * 100
        result.update(hoeffding_lower=mean * 100 - hoeffding, hoeffding_upper=mean * 100 + hoeffding)
    return result


def monthly_confidence_intervals(predictions, n_boot=2000, seed=9417):
    records = []
    for keys, group in predictions.groupby(GROUPS + ["month"], observed=True, sort=True):
        group = group.sort_values("timestamp")
        counts = group.groupby("market_day", observed=True).size()
        if not (counts == 48).all():
            raise ValueError("Bootstrap must retain all 48 half-hours of every day")
        task = keys[0]
        values = (100 * np.abs(group.y_true.to_numpy() - group.y_pred.to_numpy()) / group.y_true.to_numpy()
                  if task == "demand" else (group.y_true.to_numpy() == group.y_pred.to_numpy()).astype(float))
        stable_seed = int.from_bytes(hashlib.sha256(f"{seed}:{keys}".encode()).digest()[:8], "little")
        records.append({**dict(zip(GROUPS + ["month"], keys)),
                        "metric": "MAPE" if task == "demand" else "Accuracy",
                        **monthly_interval(values.reshape(-1, 48), task, n_boot, stable_seed)})
    return pd.DataFrame(records)


def aggregate_metrics(predictions, by=()):
    rows = []
    keys = GROUPS + list(by)
    for values, group in predictions.groupby(keys, observed=True, sort=True):
        task = values[0]
        result = {**dict(zip(keys, values)), "n_days": group.market_day.nunique(), "n_half_hours": len(group),
                  **_metrics(group, task)}
        result["loss_pct"] = result["MAPE"] if task == "demand" else 100 - result["Accuracy"]
        if task == "price":
            result.update(class_1_share=100 * group.y_true.mean(), predicted_1_share=100 * group.y_pred.mean())
        rows.append(result)
    return pd.DataFrame(rows)


def add_baseline_context(table, period):
    table = table.copy()
    if "loss_pct" not in table:
        table["loss_pct"] = np.where(table.task == "demand", table.MAPE, 100 - table.Accuracy)
    reference = table[table.model == "yesterday_baseline"][["task", period, "loss_pct"]].rename(
        columns={"loss_pct": "yesterday_loss_pct"})
    if reference.duplicated(["task", period]).any():
        raise ValueError("Exactly one yesterday baseline is required per task and period")
    result = table.merge(reference, on=["task", period], how="left", validate="many_to_one")
    if result.yesterday_loss_pct.isna().any():
        raise ValueError("Yesterday baseline coverage must match every model")
    result["excess_loss_vs_yesterday_pp"] = result.loss_pct - result.yesterday_loss_pct
    return result


def _rho(x, y):
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 3 or valid.x.std() < 1e-10 or valid.y.std() < 1e-10:
        return np.nan
    return float(valid.x.corr(valid.y, method="spearman"))


def associations_with_calendar_checks(monthly):
    """Raw rank associations, year/month-adjusted ranks, and month-to-month changes."""
    records = []
    features = ["novel_share", "profile_distance", "input_drift_score", "cluster_0_share"]
    for keys, group in monthly.groupby(GROUPS, observed=True, sort=True):
        group = group.sort_values("month").reset_index(drop=True)
        months = pd.PeriodIndex(group.month, freq="M")
        calendar = pd.DataFrame({"year": months.year.astype(str), "month": months.month.astype(str)})
        design = np.column_stack([np.ones(len(group)), pd.get_dummies(calendar, drop_first=True, dtype=float)])
        y_rank = group.loss_pct.rank().to_numpy()
        y_residual = y_rank - design @ np.linalg.lstsq(design, y_rank, rcond=None)[0]
        consecutive = np.r_[False, np.diff(months.astype("int64")) == 1]
        for feature in features:
            x = group[feature]
            x_rank = x.rank().to_numpy()
            x_residual = x_rank - design @ np.linalg.lstsq(design, x_rank, rcond=None)[0]
            adjusted = (float(np.corrcoef(x_residual, y_residual)[0, 1])
                        if x_residual.std() > 1e-10 and y_residual.std() > 1e-10 else np.nan)
            for method, coefficient, count in (
                    ("raw_spearman", _rho(x, group.loss_pct), len(group)),
                    ("year_and_calendar_month_adjusted_ranks", adjusted, len(group)),
                    ("consecutive_month_first_differences", _rho(x.diff()[consecutive], group.loss_pct.diff()[consecutive]), int(consecutive.sum()))):
                records.append({**dict(zip(GROUPS, keys)), "feature": feature, "method": method,
                                "coefficient": coefficient, "n_observations": count,
                                "loss_definition": "MAPE_pct" if keys[0] == "demand" else "misclassification_pct"})
    return pd.DataFrame(records)


def within_month_novelty(daily, minimum_days=2):
    """Compare novel/ordinary days inside the same month; no causal adjustment claim."""
    daily = daily.copy()
    daily["month"] = pd.DatetimeIndex(daily.market_day).to_period("M")
    records = []
    for keys, group in daily.groupby(GROUPS + ["month"], observed=True, sort=True):
        novel, ordinary = group[group.novel], group[~group.novel]
        records.append({**dict(zip(GROUPS + ["month"], keys)), "novel_days": len(novel),
                        "ordinary_days": len(ordinary), "eligible": min(len(novel), len(ordinary)) >= minimum_days,
                        "novel_minus_ordinary_loss_pp": novel.loss_pct.mean() - ordinary.loss_pct.mean(),
                        "novel_minus_ordinary_excess_loss_pp": (novel.excess_loss_vs_yesterday_pp.mean() -
                                                                ordinary.excess_loss_vs_yesterday_pp.mean())})
    detail = pd.DataFrame(records)
    summary = []
    for keys, group in detail.groupby(GROUPS, observed=True, sort=True):
        eligible = group[group.eligible]
        summary.append({**dict(zip(GROUPS, keys)), "eligible_months": len(eligible),
                        "minimum_days_per_category_per_month": minimum_days,
                        "equal_month_mean_loss_gap_pp": eligible.novel_minus_ordinary_loss_pp.mean(),
                        "median_month_loss_gap_pp": eligible.novel_minus_ordinary_loss_pp.median(),
                        "equal_month_mean_excess_loss_gap_pp": eligible.novel_minus_ordinary_excess_loss_pp.mean(),
                        "share_eligible_months_novel_loss_higher": ((eligible.novel_minus_ordinary_loss_pp > 0).mean()
                                                                    if len(eligible) else np.nan)})
    return detail, pd.DataFrame(summary)


def _window_stats(group, start, end):
    subset = group.loc[start:end]
    return {"days": len(subset), "loss_pct": subset.loss_pct.mean(),
            "excess_loss_vs_yesterday_pp": subset.excess_loss_vs_yesterday_pp.mean()}


def alarm_performance_windows(daily, signals, window_days=28):
    """Pre = D-27..D; post = D+1..D+28. All models remain frozen."""
    alarms = drift_episodes(signals)["input_alarm_attribution"]
    records = []
    for keys, group in daily.groupby(GROUPS, observed=True, sort=True):
        group = group.set_index("market_day").sort_index()
        for alarm in alarms.itertuples():
            day = alarm.alarm_day
            before = _window_stats(group, day - pd.Timedelta(days=window_days - 1), day)
            after = _window_stats(group, day + pd.Timedelta(days=1), day + pd.Timedelta(days=window_days))
            records.append({**dict(zip(GROUPS, keys)), "alarm_day": day, "effective_day": alarm.effective_day,
                            "alarm_kind": alarm.alarm_kind, "episode_id": alarm.episode_id,
                            "dominant_feature": alarm.dominant_feature,
                            **{f"pre_{k}": v for k, v in before.items()},
                            **{f"post_{k}": v for k, v in after.items()},
                            "complete_windows": before["days"] == after["days"] == window_days,
                            "post_minus_pre_loss_pp": after["loss_pct"] - before["loss_pct"],
                            "post_minus_pre_excess_loss_pp": after["excess_loss_vs_yesterday_pp"] - before["excess_loss_vs_yesterday_pp"]})
    result = pd.DataFrame(records)
    summary = []
    if result.empty:
        return result, pd.DataFrame()
    for keys, group in result[result.complete_windows].groupby(GROUPS + ["alarm_kind"], observed=True, sort=True):
        summary.append({**dict(zip(GROUPS + ["alarm_kind"], keys)), "alarms": len(group),
                        "positive_loss_change_count": int((group.post_minus_pre_loss_pp > 0).sum()),
                        "mean_post_minus_pre_loss_pp": group.post_minus_pre_loss_pp.mean(),
                        "median_post_minus_pre_loss_pp": group.post_minus_pre_loss_pp.median(),
                        "mean_post_minus_pre_excess_loss_pp": group.post_minus_pre_excess_loss_pp.mean()})
    return result, pd.DataFrame(summary)


def market_event_performance(daily, events, window_days=28):
    """Event windows and matching previous-year windows; descriptive context only."""
    records = []
    for keys, group in daily.groupby(GROUPS, observed=True, sort=True):
        group = group.set_index("market_day").sort_index()
        for event in events:
            date = pd.Timestamp(event["date"])
            row = {**dict(zip(GROUPS, keys)), "event_id": event["id"], "event_date": date,
                   "event_title": event["title"], "source_url": event["url"]}
            for period, anchor in (("event", date), ("previous_year", date - pd.DateOffset(years=1))):
                pre = _window_stats(group, anchor - pd.Timedelta(days=window_days), anchor - pd.Timedelta(days=1))
                post = _window_stats(group, anchor, anchor + pd.Timedelta(days=window_days - 1))
                row.update({f"{period}_pre_{k}": v for k, v in pre.items()})
                row.update({f"{period}_post_{k}": v for k, v in post.items()})
                row[f"{period}_post_minus_pre_loss_pp"] = post["loss_pct"] - pre["loss_pct"]
            records.append(row)
    return pd.DataFrame(records)


def evaluate_joint(predictions, assignments, signals, events, n_boot=2000, seed=9417):
    tables = relate_predictions(predictions, assignments, signals)
    joined = predictions.join(assignments[["cluster", "novel"]], on="market_day")
    joined["year"] = pd.DatetimeIndex(joined.market_day).year
    tables["overall_performance"] = aggregate_metrics(joined)
    tables["yearly_performance"] = aggregate_metrics(joined, ["year"])
    tables["performance_by_novelty"] = aggregate_metrics(joined, ["novel"])
    tables["performance_by_year_and_cluster"] = aggregate_metrics(joined, ["year", "cluster"])
    tables["performance_by_year_and_novelty"] = aggregate_metrics(joined, ["year", "novel"])
    tables["performance_by_cluster"] = aggregate_metrics(joined, ["cluster"])
    monthly_classes = aggregate_metrics(joined, ["month"])[GROUPS + ["month", "class_1_share", "predicted_1_share"]]
    tables["monthly_performance"] = tables["monthly_performance"].merge(monthly_classes, on=GROUPS + ["month"], validate="one_to_one")
    tables["daily_performance"] = add_baseline_context(tables["daily_performance"], "market_day")
    tables["monthly_performance"] = add_baseline_context(tables["monthly_performance"], "month")
    tables["monthly_confidence_intervals"] = monthly_confidence_intervals(predictions, n_boot, seed)
    intervals = tables["monthly_confidence_intervals"]
    tables["confidence_interval_widths"] = intervals.groupby(GROUPS, observed=True).agg(
        months=("estimate", "size"), median_iid_width=("iid_normal_width", "median"),
        median_day_bootstrap_width=("day_bootstrap_width", "median"),
        median_day_to_iid_width_ratio=("day_to_iid_width_ratio", "median")).reset_index()
    tables["calendar_checked_associations"] = associations_with_calendar_checks(tables["monthly_performance"])
    tables["within_month_novelty"], tables["within_month_novelty_summary"] = within_month_novelty(tables["daily_performance"])
    tables["alarm_performance_windows"], tables["alarm_performance_summary"] = alarm_performance_windows(tables["daily_performance"], signals)
    tables["market_event_performance"] = market_event_performance(tables["daily_performance"], events)
    return tables
