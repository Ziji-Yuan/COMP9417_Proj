"""Past-only robustness checks and descriptive evidence, independent of B/C.

These functions never change the primary model/detector. Event comparisons and
post-2014 summaries are retrospective, not inputs to fitting or selection.
"""

from dataclasses import asdict, replace

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .drift import DriftConfig, InputDriftDetector
from .profiles import DayTypeModel, cluster_shares


def clustering_audit(profiles, primary):
    """PCA sensitivity and fixed-k seed stability, strictly on training days."""
    train = profiles.loc["2010-01-01":"2013-12-31"]
    primary_labels = primary.assign(train).cluster.to_numpy()
    rows = []
    for variance in (0.90, 0.95, 0.99):
        model = (primary if variance == primary.config.variance_retained else
                 DayTypeModel(replace(primary.config, variance_retained=variance)).fit(train))
        rows.append({"variance_target": variance, "pca_components": model.pca_.n_components_,
                     "retained_variance": model.pca_.explained_variance_ratio_.sum(),
                     "selected_k": model.n_clusters_,
                     "training_silhouette": model.selection_.loc[model.n_clusters_, "silhouette"],
                     "adjusted_rand_vs_primary": adjusted_rand_score(primary_labels, model.assign(train).cluster),
                     "training_days": len(train), "training_end": train.index.max()})
    values = train.to_numpy(dtype=float)
    if primary.config.normalize:
        values = values / values.mean(axis=1, keepdims=True)
    reduced = primary.pca_.transform(values)
    seeds = []
    for seed in sorted({0, 42, primary.config.random_state}):
        model = KMeans(n_clusters=primary.n_clusters_, random_state=seed,
                       n_init=primary.config.n_init).fit(reduced)
        seeds.append({"seed": seed, "k": primary.n_clusters_,
                      "training_silhouette": silhouette_score(reduced, model.labels_),
                      "adjusted_rand_vs_primary": adjusted_rand_score(primary_labels, model.labels_),
                      "min_cluster_days": np.bincount(model.labels_).min(),
                      "training_days": len(train), "training_end": train.index.max()})
    return {"pca_sensitivity_training": pd.DataFrame(rows),
            "cluster_seed_stability_training": pd.DataFrame(seeds)}


def profile_evidence(profiles, assignments, n_clusters):
    """Calendar interpretation, coverage-matched trends and observed shape change."""
    normalized = profiles.div(profiles.mean(axis=1), axis=0)
    stats = assignments.copy()
    stats["midday_ratio"] = normalized.loc[:, 22:29].mean(axis=1)
    stats["evening_ratio"] = normalized.loc[:, 34:41].mean(axis=1)
    stats["peak_interval_start_hour"] = normalized.to_numpy().argmax(axis=1) / 2
    train = stats.loc["2010-01-01":"2013-12-31"].copy()
    train["season"] = np.array(["summer", "autumn", "winter", "spring"])[train.index.month % 12 // 3]
    train["day_kind"] = np.where(train.index.day_of_week >= 5, "weekend", "weekday")
    calendar_rows = []
    for dimension in ("season", "day_kind"):
        counts = pd.crosstab(train[dimension], train.cluster).reindex(columns=range(n_clusters), fill_value=0)
        for category, row in counts.iterrows():
            for k in range(n_clusters):
                calendar_rows.append({"dimension": dimension, "category": category, "cluster": k,
                                      "days": int(row[k]), "category_days": int(row.sum()),
                                      "cluster_days": int(counts[k].sum()),
                                      "type_share_within_category": row[k] / row.sum(),
                                      "category_share_within_type": row[k] / counts[k].sum()})
    means = train.groupby("cluster").agg(
        n_days=("cluster", "size"), daily_mean_mw=("daily_mean_mw", "mean"),
        midday_ratio=("midday_ratio", "mean"), evening_ratio=("evening_ratio", "mean"),
        mean_daily_peak_hour=("peak_interval_start_hour", "mean"))
    matched = cluster_shares(assignments[assignments.index.month <= 10], n_clusters)
    matched["months_included"] = "January-October"
    # Complete year/month grid: unobserved November/December 2026 stay NA, not zero.
    periods = pd.period_range(f"{assignments.index.min().year}-01",
                             f"{assignments.index.max().year}-12", freq="M")
    monthly = cluster_shares(assignments, n_clusters, "month").reindex(periods).rename_axis("month")
    yearly = stats.groupby(stats.index.year).agg(
        n_days=("cluster", "size"), novel_share=("novel", "mean"),
        mean_profile_distance=("nearest_profile_distance", "mean"),
        mean_demand_mw=("daily_mean_mw", "mean"),
        midday_ratio=("midday_ratio", "mean"), evening_ratio=("evening_ratio", "mean"))
    yearly.index.name = "year"
    # Match January-October in all periods; these curves do not refit clusters.
    period_profiles = []
    for label, start, end in (("2010-2013", "2010", "2013"), ("2014-2019", "2014", "2019"),
                              ("2020-2023", "2020", "2023"), ("2024-2026", "2024", "2026")):
        selected = normalized.loc[start:end]
        selected = selected[selected.index.month <= 10]
        if selected.empty:
            continue
        for slot in range(48):
            period_profiles.append({"period": label, "slot": slot, "interval_start_hour": slot / 2,
                                    "mean_normalized_demand": selected[slot].mean(), "n_days": len(selected)})
    return {"training_calendar_composition": pd.DataFrame(calendar_rows),
            "training_shape_statistics": means.reset_index(),
            "yearly_shares_jan_oct": matched.reset_index(),
            "monthly_shares_calendar_grid": monthly.reset_index(),
            "yearly_profile_statistics": yearly.reset_index(),
            "observed_shape_by_period_jan_oct": pd.DataFrame(period_profiles)}


def detector_sensitivity(summaries):
    """One-factor diagnostics: fit through June 2014, assess July-December 2014.

    The primary 2014-calibrated detector is not reselected using this table.
    Exceedance rates are NOT false-positive rates: no stable-period labels exist.
    Post-2014 observations are excluded before any computation.
    """
    past = summaries.loc["2010-01-01":"2014-12-31"]
    reference = past.loc[:"2013-12-31"]
    calibration = past.loc["2014-01-01":"2014-06-30"]
    validation = past.loc["2014-07-01":"2014-12-31"]
    if calibration.empty or validation.empty or validation.index.max() != pd.Timestamp("2014-12-31"):
        raise ValueError("Sensitivity audit requires calibration Jan-Jun and validation Jul-Dec 2014")
    base = DriftConfig()
    choices = [("primary_config", base)]
    for parameter, values in (("window_days", (14, 56)), ("threshold_quantile", (.975, .995)),
                              ("persistence_days", (1, 7)), ("cooldown_days", (30, 180))):
        choices += [(f"{parameter}={value}", replace(base, **{parameter: value})) for value in values]
    rows, alarms, baseline_daily = [], [], None
    for label, config in choices:
        detector = InputDriftDetector(config).fit(reference, calibration)
        signals = detector.replay(validation)
        rows.append({"configuration": label, **asdict(config), "threshold": detector.threshold_,
                     "calibration_start": calibration.index.min(), "calibration_end": calibration.index.max(),
                     "validation_start": validation.index.min(), "validation_end": validation.index.max(),
                     "validation_days": len(signals), "exceedance_days": int(signals.exceeded.sum()),
                     "exceedance_share": signals.exceeded.mean(), "alarms": int(signals.alarm.sum()),
                     "max_score_over_threshold": signals.score.max() / detector.threshold_})
        for day in signals.index[signals.alarm]:
            alarms.append({"configuration": label, "alarm_day": day, "effective_day": day + pd.Timedelta(days=1)})
        if label == "primary_config":
            baseline_daily = signals.reset_index()
    return {"detector_sensitivity_2014": pd.DataFrame(rows),
            "detector_sensitivity_alarms_2014": pd.DataFrame(alarms, columns=["configuration", "alarm_day", "effective_day"]),
            "detector_holdout_daily_2014": baseline_daily}


def drift_episodes(signals):
    """Separate consecutive exceedance runs from cooldown-spaced alarm events."""
    columns = [c for c in signals if c.startswith("distance__")]
    episode_columns = ["episode_id", "start_day", "last_observed_day", "n_days", "right_censored",
                       "alarms", "first_alarm_day", "max_score", "dominant_feature"]
    daily = signals.copy()
    starts = daily.exceeded & ~daily.exceeded.shift(fill_value=False)
    daily["episode_id"] = starts.cumsum().where(daily.exceeded, 0).astype(int)
    episodes, alarm_rows = [], []
    for episode_id, group in daily[daily.exceeded].groupby("episode_id"):
        alarm_days = group.index[group.alarm]
        dominant = group[columns].mean().idxmax().removeprefix("distance__")
        episodes.append({"episode_id": episode_id, "start_day": group.index[0],
                         "last_observed_day": group.index[-1], "n_days": len(group),
                         "right_censored": group.index[-1] == daily.index[-1],
                         "alarms": len(alarm_days), "first_alarm_day": alarm_days[0] if len(alarm_days) else pd.NaT,
                         "max_score": group.score.max(), "dominant_feature": dominant})
        for j, day in enumerate(alarm_days):
            row = group.loc[day]
            ranked = row[columns].astype(float).sort_values(ascending=False)
            alarm_rows.append({"alarm_day": day, "effective_day": row.effective_day,
                               "episode_id": episode_id, "episode_start": group.index[0],
                               "days_since_episode_start": (day - group.index[0]).days,
                               "alarm_kind": "first_alarm_in_episode" if j == 0 else "repeat_within_episode",
                               "score": row.score, "threshold": row.threshold,
                               "dominant_feature": ranked.index[0].removeprefix("distance__"),
                               "dominant_distance": ranked.iloc[0],
                               "second_feature": ranked.index[1].removeprefix("distance__") if len(ranked) > 1 else None,
                               "second_distance": ranked.iloc[1] if len(ranked) > 1 else np.nan})
    alarm_columns = ["alarm_day", "effective_day", "episode_id", "episode_start", "days_since_episode_start",
                     "alarm_kind", "score", "threshold", "dominant_feature", "dominant_distance", "second_feature", "second_distance"]
    return {"input_drift_episodes": pd.DataFrame(episodes, columns=episode_columns),
            "input_alarm_attribution": pd.DataFrame(alarm_rows, columns=alarm_columns)}


def day_type_drift_context(assignments, signals, n_clusters):
    """Join input-day D to observed day-type D-1, matching the actual detector."""
    lagged = assignments[["cluster", "novel", "nearest_profile_distance"]].copy()
    lagged.index = lagged.index + pd.Timedelta(days=1)
    lagged = lagged.rename(columns={name: f"lag1_{name}" for name in lagged})
    daily = signals.join(lagged, how="left")
    if daily.lag1_cluster.isna().any():
        raise ValueError("Need the previous observed day's cluster for every input day")
    for k in range(n_clusters):
        daily[f"lag1_type_{k}_share"] = (daily.lag1_cluster == k).astype(float)
    components = [c for c in signals if c.startswith("distance__")]
    for group, names in {
        "demand": [c for c in components if any(s in c for s in ("demand", "ratio", "profile"))],
        "price": [c for c in components if "price" in c],
        "weather": [c for c in components if "syd_" in c],
    }.items():
        if names:
            daily[f"{group}_max_distance"] = daily[names].max(axis=1)
    metrics = {"input_score_mean": ("score", "mean"), "alarm_days": ("alarm", "sum"),
               "exceedance_share": ("exceeded", "mean"), "n_days": ("score", "size"),
               "lag1_novel_share": ("lag1_novel", "mean"),
               "lag1_profile_distance_mean": ("lag1_nearest_profile_distance", "mean")}
    for c in daily:
        if c.endswith("_max_distance") or c.startswith("lag1_type_"):
            metrics[c] = (c, "mean")
    monthly = daily.groupby(daily.index.to_period("M")).agg(**metrics)
    monthly.index.name = "month"
    daily["profile_source_day"] = daily.index - pd.Timedelta(days=1)
    return {"day_type_input_context_daily": daily.reset_index(),
            "day_type_input_context_monthly": monthly.reset_index()}


def event_comparisons(assignments, signals, events, window_days=28):
    """Descriptive windows around sourced dates, with signed nearest-alarm gaps.

    Previous-year windows are context, not a causal control group. The event
    catalogue is used only here, after all models, thresholds and alarms exist.
    """
    daily = signals.join(assignments[["novel", "nearest_profile_distance"]])
    component_cols = [c for c in signals if c.startswith("distance__")]
    alarm_days = signals.index[signals.alarm]
    rows, windows = [], []
    for event in events:
        date = pd.Timestamp(event["date"])
        nearest = (min(alarm_days, key=lambda day: abs((day - date).days)) if len(alarm_days) else pd.NaT)
        row = {"event_id": event["id"], "event_date": date, "event_title": event["title"],
               "source_url": event["url"], "nearest_alarm_day": nearest,
               "nearest_alarm_minus_event_days": (nearest - date).days if pd.notna(nearest) else np.nan}
        for label, anchor in (("event", date), ("previous_year", date - pd.DateOffset(years=1))):
            for side, lower, upper in (("pre", -window_days, -1), ("post", 0, window_days - 1)):
                selected = daily.loc[anchor + pd.Timedelta(days=lower):anchor + pd.Timedelta(days=upper)]
                prefix = f"{label}_{side}"
                row.update({f"{prefix}_days": len(selected), f"{prefix}_alarms": int(selected.alarm.sum()),
                            f"{prefix}_exceedance_share": selected.exceeded.mean(),
                            f"{prefix}_score_mean": selected.score.mean(),
                            f"{prefix}_novel_share": selected.novel.mean()})
                row[f"{prefix}_dominant_feature"] = (selected[component_cols].mean().idxmax().removeprefix("distance__")
                                                       if len(selected) else None)
        selected = daily.loc[date - pd.Timedelta(days=window_days):date + pd.Timedelta(days=window_days - 1)].copy()
        selected["event_id"] = event["id"]
        selected["offset_days"] = (selected.index - date).days
        windows.append(selected.reset_index())
        rows.append(row)
    return {"market_event_comparisons": pd.DataFrame(rows),
            "market_event_daily_windows": pd.concat(windows, ignore_index=True) if windows else pd.DataFrame()}


def reference_scale_table(detector):
    """Expose rare/low-variance features that can dominate standardized scores."""
    return pd.DataFrame({"feature": detector.columns_, "reference_scale": detector.scale_,
                         "reference_raw_median": detector.fallback_centre_,
                         "reference_standardized_std": detector.reference_residuals_.std(axis=0)})
