"""Report evidence from real frozen forecasts; all comparisons are retrospective."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import _save


MODELS = {"demand": ("ridge_regression", "random_forest"),
          "price": ("logistic_regression", "random_forest")}
LABELS = {"ridge_regression": "Ridge", "logistic_regression": "Logistic regression",
          "random_forest": "Random forest", "yesterday_baseline": "Yesterday baseline",
          "last_week_baseline": "Last-week baseline"}
COLORS = {"ridge_regression": "#1b9e77", "logistic_regression": "#1b9e77",
          "random_forest": "#d95f02", "yesterday_baseline": "#666666",
          "last_week_baseline": "#a6a6a6"}


def make_joint_figures(tables, assignments, signals, events, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 11, "figure.constrained_layout.use": True})
    monthly = tables["monthly_performance"]
    intervals = tables["monthly_confidence_intervals"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for row, task in enumerate(("demand", "price")):
        metric = "MAPE" if task == "demand" else "Accuracy"
        for col, model in enumerate(MODELS[task]):
            ax = axes[row, col]
            selected = intervals[(intervals.task == task) & (intervals.model == model)].sort_values("month")
            dates = pd.PeriodIndex(selected.month, freq="M").to_timestamp()
            ax.fill_between(dates, selected.day_bootstrap_lower, selected.day_bootstrap_upper,
                            color="#2166ac", alpha=.22, label="95% whole-day bootstrap")
            ax.fill_between(dates, selected.iid_normal_lower, selected.iid_normal_upper,
                            color="#2166ac", alpha=.5, label="95% independent half-hour normal")
            ax.plot(dates, selected.estimate, color="#08306b", lw=1, label="Frozen model")
            for baseline in ("yesterday_baseline", "last_week_baseline"):
                rows = monthly[(monthly.task == task) & (monthly.model == baseline)].sort_values("month")
                ax.plot(pd.PeriodIndex(rows.month, freq="M").to_timestamp(), rows[metric],
                        ls="--" if baseline == "yesterday_baseline" else ":", lw=.8,
                        color=COLORS[baseline], label=LABELS[baseline])
            ax.set(title=f"{task.capitalize()} — {LABELS[model]}", ylabel=f"{metric} (%)")
            ax.grid(alpha=.15)
            if row == 1:
                ax.set_xlabel("Market month")
                ax.set_ylim(0, 100)
    axes[0, 0].legend(fontsize=7, ncol=1)
    fig.suptitle("Frozen monthly performance: Appendix A confidence intervals", fontsize=13)
    _save(fig, output / "07_frozen_monthly_performance_ci.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for row, task in enumerate(("demand", "price")):
        metric = "MAPE" if task == "demand" else "Accuracy"
        for col, (table_name, group_key, group_labels) in enumerate((
                ("performance_by_cluster", "cluster", ["Type 0", "Type 1"]),
                ("performance_by_novelty", "novel", ["Ordinary", "Novel"]))):
            ax = axes[row, col]
            data = tables[table_name]
            for j, model in enumerate((*MODELS[task], "yesterday_baseline")):
                rows = data[(data.task == task) & (data.model == model)].sort_values(group_key)
                x = np.arange(len(rows)) + (j - 1) * .24
                ax.bar(x, rows[metric], width=.23, color=COLORS[model], label=LABELS[model])
            ax.set_xticks([0, 1], group_labels)
            ax.set(title=f"{task.capitalize()}: {'observed type' if col == 0 else 'profile novelty'}", ylabel=f"{metric} (%)")
            ax.grid(axis="y", alpha=.15)
            if row == 1:
                ax.set_ylim(0, 100)
            ax.legend(fontsize=8)
    fig.suptitle("Pooled comparisons: groups differ in year, season and class balance", fontsize=12)
    _save(fig, output / "14_errors_by_day_type_and_novelty.png")

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(signals.index, signals.score, color="#2166ac", lw=.7)
    axes[0].axhline(signals.threshold.iloc[0], color="#b2182b", ls="--", label="Fixed daily threshold")
    alarms = tables["alarm_performance_windows"].drop_duplicates("alarm_day")
    for kind, colour, label in (("first_alarm_in_episode", "#b2182b", "First alarm within exceedance run"),
                                ("repeat_within_episode", "#999999", "Repeat alarm within run")):
        days = pd.DatetimeIndex(alarms.loc[alarms.alarm_kind == kind, "alarm_day"])
        axes[0].scatter(days, signals.loc[days, "score"], s=18, marker="v", color=colour, label=label)
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set(ylabel="Daily input score\n(symlog)", title="Fixed input monitor aligned with frozen model performance")
    for row, task in ((1, "demand"), (2, "price")):
        metric = "MAPE" if task == "demand" else "Accuracy"
        for model in (*MODELS[task], "yesterday_baseline"):
            rows = monthly[(monthly.task == task) & (monthly.model == model)].sort_values("month")
            axes[row].plot(pd.PeriodIndex(rows.month, freq="M").to_timestamp(), rows[metric], color=COLORS[model],
                           lw=1, ls="--" if model == "yesterday_baseline" else "-", label=LABELS[model])
        axes[row].set_ylabel(f"Monthly {metric} (%)")
    context = monthly[(monthly.task == "price") & (monthly.model == "random_forest")].sort_values("month")
    dates = pd.PeriodIndex(context.month, freq="M").to_timestamp()
    axes[3].plot(dates, 100 * context.cluster_0_share, color="#888888", label="Observed Type 0 share")
    axes[3].plot(dates, 100 * context.novel_share, color="#7570b3", label="Observed novel-day share")
    axes[3].plot(dates, context.class_1_share, color="#1b9e77", label="True price class 1 share")
    axes[3].set(ylabel="Monthly share (%)", xlabel="Market time; shared trend does not establish causation", ylim=(0, 100))
    for ax in axes:
        ax.legend(fontsize=7, ncol=3)
        ax.grid(alpha=.15)
    _save(fig, output / "15_drift_and_frozen_error_alignment.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    windows = tables["alarm_performance_windows"]
    for row, task in enumerate(("demand", "price")):
        for col, model in enumerate(MODELS[task]):
            ax = axes[row, col]
            selected = windows[(windows.task == task) & (windows.model == model) & windows.complete_windows]
            for kind, colour, label in (("first_alarm_in_episode", "#b2182b", "First within run"),
                                        ("repeat_within_episode", "#888888", "Repeat within run")):
                rows = selected[selected.alarm_kind == kind]
                ax.scatter(rows.alarm_day, rows.post_minus_pre_loss_pp, s=30,
                           facecolors=colour if kind.startswith("first") else "none", edgecolors=colour, label=label)
            ax.axhline(0, color="black", lw=.7)
            ax.set(title=f"{task.capitalize()} — {LABELS[model]}", ylabel="Post minus pre loss\n(percentage points)")
            ax.legend(fontsize=8)
            ax.grid(alpha=.15)
            if row == 1:
                ax.set_xlabel("Alarm day D")
    fig.suptitle("28-day windows: pre ends on D; post begins D+1. Positive = worse; all models frozen.", fontsize=11)
    _save(fig, output / "16_alarm_window_error_changes.png")

    associations = tables["calendar_checked_associations"]
    methods = [("raw_spearman", "Raw rank association"),
               ("year_and_calendar_month_adjusted_ranks", "Ranks adjusted for year + calendar month"),
               ("consecutive_month_first_differences", "Month-to-month changes")]
    features = ["novel_share", "profile_distance", "input_drift_score", "cluster_0_share"]
    series = [(task, model) for task in ("demand", "price") for model in MODELS[task]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, (method, title) in zip(axes, methods):
        matrix = np.array([[associations[(associations.task == task) & (associations.model == model) &
                                          (associations.method == method) & (associations.feature == feature)].coefficient.iloc[0]
                            for feature in features] for task, model in series])
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(4), ["Novel share", "Profile distance", "Input score", "Type 0 share"], rotation=35, ha="right")
        ax.set_yticks(range(4), [f"{task}: {LABELS[model]}" for task, model in series], fontsize=8)
        ax.set_title(title, fontsize=10)
        for i in range(4):
            for j in range(4):
                value = matrix[i, j]
                ax.text(j, i, f"{value:.2f}" if np.isfinite(value) else "NA", ha="center", va="center",
                        color="white" if abs(value) > .55 else "black", fontsize=9)
    fig.colorbar(im, ax=list(axes), label="Association with loss: MAPE or misclassification rate", shrink=.8)
    fig.suptitle("Descriptive checks for shared time trends; no causal or significance claim", fontsize=12)
    _save(fig, output / "17_associations_and_calendar_checks.png")

    daily = tables["daily_performance"]
    fig, axes = plt.subplots(len(events), 2, figsize=(12, 3 * len(events)), squeeze=False)
    for row, event in enumerate(events):
        date = pd.Timestamp(event["date"])
        for col, task in enumerate(("demand", "price")):
            ax = axes[row, col]
            for model in (*MODELS[task], "yesterday_baseline"):
                rows = daily[(daily.task == task) & (daily.model == model)].sort_values("market_day").set_index("market_day")
                smooth = rows.loss_pct.rolling(7, min_periods=7).mean()
                selected = smooth.loc[date - pd.Timedelta(days=28):date + pd.Timedelta(days=27)]
                ax.plot((selected.index - date).days, selected, color=COLORS[model], lw=1,
                        ls="--" if model == "yesterday_baseline" else "-", label=LABELS[model])
            ax.axvline(0, color="black", ls=":")
            ax.set(title=f"{event['title']} — {task}", xlabel="Days relative to event", ylabel="Trailing 7-day mean loss (%)")
            ax.grid(alpha=.15)
            if row == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Retrospective event context: loss = demand APE or price misclassification; no causal attribution", fontsize=11)
    _save(fig, output / "18_market_events_and_forecast_errors.png")
