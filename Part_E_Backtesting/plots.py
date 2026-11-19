"""Standalone historical evidence figures, not report prose or strategy selection."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


COLORS = {"frozen": "#65758b", "periodic": "#2475ae", "input_triggered": "#b46b1b", "error_triggered": "#19826f"}
LABELS = {"frozen": "Frozen RF", "periodic": "Periodic", "input_triggered": "Input triggered", "error_triggered": "Error triggered"}


def save(fig, folder, name):
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png", dpi=180)
    fig.savefig(folder / f"{name}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--input-reference", type=Path, default=Path("outputs/part_d_complete"))
    args = parser.parse_args()
    root = args.results
    output = root / "figures"
    output.mkdir(exist_ok=True)
    monthly = pd.read_csv(root / "monthly_performance.csv")
    monthly["date"] = pd.to_datetime(monthly.month)
    intervals = pd.read_csv(root / "monthly_confidence_intervals.csv")
    intervals["date"] = pd.to_datetime(intervals.month)
    overall = pd.read_csv(root / "overall_performance.csv")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, task, metric in zip(axes, ["demand", "price"], ["MAPE", "Accuracy"]):
        for strategy, color in COLORS.items():
            rows = monthly[(monthly.task == task) & (monthly.model == "random_forest") & (monthly.strategy == strategy)].sort_values("date")
            ci = intervals[(intervals.task == task) & (intervals.model == "random_forest") & (intervals.strategy == strategy)].sort_values("date")
            ax.plot(rows.date, rows[metric], color=color, label=LABELS[strategy], lw=1.2,
                    marker="o" if len(rows) == 1 else None)
            ax.fill_between(ci.date, ci.day_bootstrap_lower, ci.day_bootstrap_upper, color=color, alpha=.08)
        for model, style in [("yesterday_baseline", "--"), ("last_week_baseline", ":")]:
            rows = monthly[(monthly.task == task) & (monthly.model == model)].sort_values("date")
            ax.plot(rows.date, rows[metric], style, label=model.replace("_", " "), color="#333333", alpha=.65, lw=.9,
                    marker="x" if len(rows) == 1 else None)
        ax.set_ylabel(f"{task.title()} {metric} (%)")
        ax.grid(alpha=.15)
        ax.legend(ncol=3, fontsize=8, loc="best")
    axes[0].set_title("Chronological adaptation: monthly performance and 95% whole-day bootstrap intervals")
    axes[1].set_xlabel(f"Target market month (AEST; evaluated through {monthly.month.max()})")
    if monthly.date.nunique() == 1:
        axes[1].set_xlim(monthly.date.min() - pd.Timedelta(days=16), monthly.date.max() + pd.Timedelta(days=16))
    save(fig, output, "01_monthly_performance")

    fig, ax = plt.subplots(figsize=(11, 3.7))
    for strategy, color in COLORS.items():
        rows = monthly[(monthly.task == "price") & (monthly.model == "random_forest") & (monthly.strategy == strategy)].sort_values("date")
        ax.plot(rows.date, rows.BalancedAccuracy, color=color, label=LABELS[strategy])
    for model, style in [("yesterday_baseline", "--"), ("last_week_baseline", ":")]:
        rows = monthly[(monthly.task == "price") & (monthly.model == model)].sort_values("date")
        ax.plot(rows.date, rows.BalancedAccuracy, style, color="#333333", alpha=.6, label=model.replace("_", " "))
    ax.set(ylabel="Balanced accuracy (%)", xlabel="Target market month (AEST)",
           title="Price direction: monthly pooled balanced accuracy (no Appendix A interval for this metric)")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=.15)
    save(fig, output, "02_price_balanced_accuracy")

    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    costs = []
    for ax, task in zip(axes, ["demand", "price"]):
        for row, strategy in enumerate(COLORS):
            folder = root / f"{task}_{strategy}"
            summary = json.loads((folder / "summary.json").read_text())
            try:
                log = pd.read_csv(folder / "retrains.csv")
            except pd.errors.EmptyDataError:
                log = pd.DataFrame()
            dates = pd.to_datetime(log.effective_day) if len(log) else []
            ax.scatter(dates, [row] * len(dates), c=COLORS[strategy], s=35)
            costs.append({"task": task, "strategy": strategy, "retrain_count": summary["retrain_count"],
                          "total_fit_seconds": float(log.fit_seconds.sum()) if len(log) else 0,
                          "simulation_seconds": summary["seconds"]})
        ax.set_yticks(range(4), [LABELS[s] for s in COLORS])
        ax.set_title(task.title() + ": actual replacement dates (six-retrain cap per strategy)")
        ax.grid(axis="x", alpha=.2)
    axes[1].set_xlabel("Effective deployment day (forecast starts after the triggering day)")
    save(fig, output, "03_retraining_dates")
    costs = pd.DataFrame(costs)
    merged = overall[overall.model == "random_forest"].merge(costs, on=["task", "strategy"], validate="one_to_one")
    for task in ["demand", "price"]:
        baseline = overall[(overall.task == task) & (overall.model == "yesterday_baseline")].iloc[0]
        selected = merged.task == task
        if task == "demand":
            merged.loc[selected, "mape_improvement_vs_yesterday_pct"] = 100 * (1 - merged.loc[selected, "MAPE"] / baseline.MAPE)
        else:
            merged.loc[selected, "balanced_accuracy_gain_vs_yesterday_pp"] = merged.loc[selected, "BalancedAccuracy"] - baseline.BalancedAccuracy
    merged.to_csv(root / "strategy_cost_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, task, metric in zip(axes, ["demand", "price"], ["MAPE", "BalancedAccuracy"]):
        for strategy, color in COLORS.items():
            point = merged[(merged.task == task) & (merged.strategy == strategy)].iloc[0]
            ax.scatter(point.retrain_count, point[metric], color=color, s=45)
            offset = -12 if strategy == "input_triggered" and task == "price" else 5
            ax.annotate(LABELS[strategy], (point.retrain_count, point[metric]), xytext=(5, offset), textcoords="offset points", fontsize=8)
        ax.set(xlabel="Actual model replacements (count)", ylabel=f"{metric} (%)", title=task.title(), xlim=(-.5, 7.5))
        ax.set_xticks(range(7))
        ax.grid(alpha=.15)
    save(fig, output, "04_budget_performance")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, task in zip(axes, ["demand", "price"]):
        for strategy, color in COLORS.items():
            rows = intervals[(intervals.task == task) & (intervals.model == "random_forest") & (intervals.strategy == strategy)]
            ax.scatter(rows.iid_normal_width, rows.day_bootstrap_width, s=7, color=color, alpha=.35, label=LABELS[strategy])
        bound = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, bound], [0, bound], "--", color="gray", lw=.8)
        ax.set(xlabel="Independent half-hour CI width (percentage points)",
               ylabel="Whole-day bootstrap CI width (percentage points)", title=task.title())
        ax.legend(fontsize=7)
    save(fig, output, "05_interval_widths")
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    input_signals = pd.read_csv(args.input_reference / "tables/input_drift_daily.csv", parse_dates=["market_day"])
    start, end = monthly.date.min(), monthly.date.max() + pd.offsets.MonthEnd(0)
    input_signals = input_signals[input_signals.market_day.between(start, end)]
    ratio = input_signals.score / input_signals.threshold
    axes[0].plot(input_signals.market_day, ratio, color=COLORS["input_triggered"], lw=.8)
    axes[0].scatter(input_signals.loc[input_signals.alarm, "market_day"], ratio[input_signals.alarm], s=8, c="#a53335", label="Input alarm")
    axes[0].set(ylabel="Input score / threshold (symlog)", title="Online alarm statistics (fixed pre-2015 thresholds)")
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    for ax, task in zip(axes[1:], ["demand", "price"]):
        for strategy, style in [("frozen", "--"), ("error_triggered", "-")]:
            signals = pd.read_csv(root / f"{task}_{strategy}/error_signals.csv", parse_dates=["market_day"])
            ax.plot(signals.market_day, signals.score / signals.threshold, style, color=COLORS[strategy], lw=.8, label=LABELS[strategy] + " error monitor")
            if strategy == "error_triggered":
                alarms = signals[signals.alarm]
                ax.scatter(alarms.market_day, alarms.score / alarms.threshold, s=8, c="#a53335", label="Error alarm (may exceed budget)")
        ax.set_ylabel(task.title() + " loss score / threshold")
        ax.legend(fontsize=8, ncol=2)
    for ax in axes:
        ax.axhline(1, color="#333333", lw=.7)
        ax.grid(alpha=.15)
    axes[-1].set_xlabel("Observed market day; alarms can affect forecasts only from the following day")
    save(fig, output, "06_detector_statistics")
    print(f"Exported six PNG/PDF figure pairs and strategy_cost_comparison.csv to {root}")


if __name__ == "__main__":
    main()
