"""Standalone scientific figures; all times use interval-start market time."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
import pandas as pd


def _save(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figures(model, assignments, yearly, signals, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 12, "figure.constrained_layout.use": True})
    colours = plt.get_cmap("tab10")(np.arange(model.n_clusters_))
    hours = np.arange(48) / 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k, colour in enumerate(colours):
        count = model.cluster_summary_.loc[k, "training_days"]
        axes[0].plot(hours, model.centres_[k], color=colour, label=f"Type {k} (n={count})")
        axes[0].fill_between(hours, model.lower_profiles_[k], model.upper_profiles_[k], color=colour, alpha=0.12)
        axes[1].plot(hours, model.raw_centres_[k], color=colour, label=f"Type {k}")
    axes[0].set_ylabel("Demand / daily mean" if model.config.normalize else "Demand (MW)")
    axes[1].set_ylabel("Demand (MW)")
    axes[0].set_title("Training profiles: mean and interquartile band")
    axes[1].set_title("Training profiles: original demand scale")
    for ax in axes:
        ax.set_xlabel("Interval start hour (AEST)")
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.grid(alpha=0.15)
    axes[0].legend(fontsize=8)
    _save(fig, output / "01_typical_profiles.png")

    fig, ax = plt.subplots(figsize=(11, 4))
    bottom = np.zeros(len(yearly))
    for k, colour in enumerate(colours):
        values = yearly[f"cluster_{k}"].to_numpy() * 100
        ax.bar(np.arange(len(yearly)), values, bottom=bottom, label=f"Type {k}", color=colour)
        bottom += values
    labels = [f"{year}{'*' if year == 2026 else ''}\n(n={int(row.n_days)})" for year, row in yearly.iterrows()]
    ax.set_xticks(np.arange(len(yearly)), labels, rotation=45, ha="right", fontsize=8)
    ax.set(ylabel="Share of observed days (%)", xlabel="Market year (observed day count); *2026: January–October only",
           title="Day-type shares using the fixed 2010–2013 reference", ylim=(0, 100))
    ax.legend(ncol=min(model.n_clusters_, 4), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.19))
    _save(fig, output / "02_yearly_type_shares.png")

    monthly = assignments.groupby(assignments.index.to_period("M")).agg(
        distance=("nearest_profile_distance", "mean"), novel=("novel", "mean"))
    dates = monthly.index.to_timestamp()
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(dates, monthly.distance, color="#2166ac")
    axes[0].axhline(model.novelty_threshold_, color="#b2182b", ls="--", label="Training 99th percentile of daily distance")
    axes[0].set(ylabel="Mean profile RMSE\n(normalized units)" if model.config.normalize else "Mean profile RMSE (MW)",
                title="Distance to the nearest fixed training profile")
    axes[0].legend(fontsize=8)
    axes[1].plot(dates, 100 * monthly.novel, color="#2166ac")
    axes[1].set(ylabel="Novel days (%)", xlabel="Market month", ylim=(-2, 102))
    for ax in axes:
        ax.axvline(pd.Timestamp("2014-01-01"), color="grey", ls=":")
        ax.grid(alpha=0.15)
    _save(fig, output / "03_profile_novelty.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cumulative = np.cumsum(model.pca_.explained_variance_ratio_)
    axes[0].plot(np.arange(1, len(cumulative) + 1), cumulative * 100, "o-", color="#2166ac")
    axes[0].axhline(model.config.variance_retained * 100, ls="--", color="grey")
    axes[0].set(xlabel="Number of retained principal components", ylabel="Explained variance (%)",
                title="PCA fitted on 2010–2013")
    axes[1].plot(model.selection_.index, model.selection_.silhouette, "o-", color="#2166ac")
    axes[1].axvline(model.n_clusters_, ls="--", color="#b2182b", label=f"Selected k={model.n_clusters_}")
    axes[1].set(xlabel="Number of clusters k", ylabel="Training silhouette score", title="Cluster selection on training data only")
    axes[1].legend(fontsize=8)
    _save(fig, output / "04_model_selection.png")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(signals.index, signals.score, lw=0.8, color="#2166ac", label="Daily input-drift score")
    ax.axhline(signals.threshold.iloc[0], ls="--", color="#b2182b", label="Fixed threshold calibrated on 2014")
    alarms = signals[signals.alarm]
    ax.scatter(alarms.index, alarms.score, color="#b2182b", marker="v", s=20,
               label="Alarm (eligible from next day)", zorder=3)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set(xlabel="Observed market day", ylabel="Max standardized Wasserstein distance\n(symlog scale)",
           title="Input-only drift monitoring: 28-day windows")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    _save(fig, output / "05_input_drift.png")

    columns = [c for c in signals if c.startswith("distance__")]
    heat = signals[columns].groupby(signals.index.to_period("M")).mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(heat.to_numpy().T, aspect="auto", origin="upper", cmap="magma",
                   norm=SymLogNorm(linthresh=1.0, vmin=0, vmax=max(1.0, heat.to_numpy().max())))
    ax.set_yticks(np.arange(len(columns)), [c.removeprefix("distance__") for c in columns], fontsize=8)
    ticks = np.flatnonzero(heat.index.month == 1)
    ax.set_xticks(ticks, heat.index.year[ticks], rotation=45)
    ax.set(xlabel="Market month", title="Monthly mean distances for each monitored input")
    fig.colorbar(im, ax=ax, label="Standardized Wasserstein distance (symlog colors)")
    _save(fig, output / "06_drift_components.png")


def performance_figure(monthly, output):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for (task, model, strategy), rows in monthly.groupby(["task", "model", "strategy"]):
        row = 0 if task == "demand" else 1
        ax = axes[row, 0]
        rows = rows.sort_values("month")
        metric = "MAPE" if task == "demand" else "Accuracy"
        ax.plot(pd.PeriodIndex(rows.month, freq="M").to_timestamp(), rows[metric],
                label=f"{model} / {strategy}", lw=1, marker=".", markersize=3)
        axes[row, 1].scatter(rows.input_drift_score, rows[metric], s=12, alpha=0.7,
                             label=f"{model} / {strategy}")
    axes[0, 0].set(ylabel="Demand MAPE (%)", title="Supplied predictions: monthly metrics")
    axes[1, 0].set(ylabel="Price accuracy (%)", xlabel="Market month")
    axes[0, 1].set(ylabel="Demand MAPE (%)", title="Descriptive association, not causal evidence")
    axes[1, 1].set(ylabel="Price accuracy (%)")
    for ax in axes[:, 1]:
        ax.set_xlabel("Monthly mean input-drift score")
    for ax in axes.flat:
        if ax.lines or ax.collections:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
    _save(fig, Path(output) / "07_supplied_model_performance.png")


def diagnostic_figures(tables, model, output):
    """Additional independent evidence; no forecasts or model-error claims."""
    output = Path(output)
    colours = plt.get_cmap("tab10")(np.arange(model.n_clusters_))
    matched = tables["yearly_shares_jan_oct"]
    curves = tables["observed_shape_by_period_jan_oct"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, colour in enumerate(colours):
        axes[0].plot(matched.year, 100 * matched[f"cluster_{k}"], "o-", label=f"Type {k}", color=colour)
    axes[0].set(title="Matched calendar coverage: January–October", ylabel="Day-type share (%)", xlabel="Market year", ylim=(-2, 102))
    for period, group in curves.groupby("period", sort=False):
        axes[1].plot(group.interval_start_hour, group.mean_normalized_demand, label=period)
    axes[1].set(title="Observed shape change: January–October", ylabel="Demand / daily mean", xlabel="Interval start hour (AEST)")
    axes[1].set_xticks([0, 6, 12, 18, 24])
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    _save(fig, output / "08_matched_coverage_and_shape.png")

    monthly = tables["monthly_shares_calendar_grid"].copy()
    periods = pd.PeriodIndex(monthly.month, freq="M")
    monthly["year"], monthly["calendar_month"] = periods.year, periods.month
    fig, axes = plt.subplots(1, model.n_clusters_, figsize=(6 * model.n_clusters_, 6), squeeze=False)
    for k, ax in enumerate(axes.flat):
        heat = monthly.pivot(index="year", columns="calendar_month", values=f"cluster_{k}") * 100
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#d9d9d9")
        im = ax.imshow(heat, aspect="auto", vmin=0, vmax=100, cmap=cmap)
        ax.set_xticks(np.arange(12), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        ax.set_yticks(np.arange(len(heat)), heat.index)
        ax.set(title=f"Type {k}: monthly share", xlabel="Calendar month; gray = no observations", ylabel="Market year")
    fig.colorbar(im, ax=list(axes.flat), label="Share of days (%)", shrink=.8)
    _save(fig, output / "09_monthly_day_types.png")

    calendar = tables["training_calendar_composition"]
    labels = ["summer", "autumn", "winter", "spring", "weekday", "weekend"]
    fig, ax = plt.subplots(figsize=(8, 4))
    heat = calendar.pivot(index="category", columns="cluster", values="type_share_within_category").reindex(labels) * 100
    im = ax.imshow(heat, aspect="auto", vmin=0, vmax=100, cmap="Blues")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xticks(np.arange(model.n_clusters_), [f"Type {k}" for k in range(model.n_clusters_)])
    ax.set_title("Training interpretation: P(type | calendar category)")
    for i in range(len(labels)):
        for k in range(model.n_clusters_):
            value = heat.iloc[i, k]
            ax.text(k, i, f"{value:.1f}%", ha="center", va="center", color="white" if value > 55 else "black")
    fig.colorbar(im, ax=ax, label="Days assigned to type (%)")
    _save(fig, output / "10_training_calendar_interpretation.png")

    audit = tables["detector_sensitivity_2014"]
    holdout = tables["detector_holdout_daily_2014"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.1, 1]})
    positions = np.arange(len(audit))
    axes[0].barh(positions, audit.max_score_over_threshold, color="#2166ac")
    axes[0].set_yticks(positions, audit.configuration, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].axvline(1, color="#b2182b", ls="--", lw=.8)
    axes[0].set(title="One-factor sensitivity: held-out 2014 H2", xlabel="Maximum score / calibration threshold")
    for position, row in zip(positions, audit.itertuples()):
        axes[0].text(row.max_score_over_threshold + .02, position,
                     f"{row.exceedance_days} exceedances; {row.alarms} alarms", va="center", fontsize=7)
    axes[0].set_xlim(0, max(1, audit.max_score_over_threshold.max()) + .75)
    axes[1].plot(holdout.market_day, holdout.score, lw=1, label="Primary configuration, H1 threshold")
    axes[1].axhline(holdout.threshold.iloc[0], color="#b2182b", ls="--", label="Threshold calibrated Jan–Jun 2014")
    axes[1].set(title="Prospective check within 2014", ylabel="Input score", xlabel="Held-out day (July–December 2014)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(fontsize=8)
    _save(fig, output / "11_detector_past_only_sensitivity.png")

    context = tables["day_type_input_context_monthly"]
    dates = pd.PeriodIndex(context.month, freq="M").to_timestamp()
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for k, colour in enumerate(colours):
        axes[0].plot(dates, context[f"lag1_type_{k}_share"] * 100, label=f"Lag-one type {k}", color=colour, lw=1)
    axes[0].set(ylabel="Share of input days (%)", title="Day types and input drift: descriptive monthly alignment", ylim=(-2, 102))
    axes[1].plot(dates, context.lag1_novel_share * 100, color="#7570b3", label="Lag-one novel profiles")
    axes[1].set(ylabel="Novel profiles (%)", ylim=(-2, 102))
    for name, colour in (("demand", "#1b9e77"), ("price", "#d95f02"), ("weather", "#7570b3")):
        axes[2].plot(dates, context[f"{name}_max_distance"], label=f"{name.capitalize()} components", color=colour, lw=1)
    axes[2].set_yscale("symlog", linthresh=1)
    axes[2].set(ylabel="Mean daily max distance\n(symlog)", xlabel="Input market month; profile comes from preceding day")
    for ax in axes:
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=.15)
    _save(fig, output / "12_day_types_and_input_drift.png")

    events = tables["market_event_comparisons"]
    windows = tables["market_event_daily_windows"]
    if len(events):
        fig, axes = plt.subplots(len(events), 1, figsize=(10, 2.7 * len(events)), squeeze=False)
        for ax, event in zip(axes.flat, events.itertuples()):
            group = windows[windows.event_id == event.event_id]
            ax.plot(group.offset_days, group.score / group.threshold, color="#2166ac", label="Score / fixed threshold")
            alarm = group[group.alarm]
            ax.scatter(alarm.offset_days, alarm.score / alarm.threshold, color="#b2182b", marker="v", label="Alarm")
            ax.axhline(1, color="#b2182b", ls="--", lw=.8)
            ax.axvline(0, color="grey", ls=":")
            ax.set(title=f"{event.event_title} — {str(event.event_date.date())}", ylabel="Score / threshold", xlabel="Days from event; retrospective, no causal attribution")
            ax.set_yscale("symlog", linthresh=1)
            ax.legend(fontsize=8)
            ax.grid(alpha=.15)
        _save(fig, output / "13_market_event_windows.png")
