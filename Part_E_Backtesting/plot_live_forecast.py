"""Render and validate historical live-week forecast evidence."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("outputs/part_e_live_forecast"))
    root = parser.parse_args().results
    log = pd.read_csv(root / "training_log.csv", parse_dates=["training_start", "training_end"])
    if len(log) != 12 or not log.groupby("task").size().eq(6).all():
        raise ValueError("Expected six fits per task")
    if not (log.training_end.dt.month.eq(10) & log.training_end.dt.day.eq(31)).all():
        raise ValueError("All analog training cutoffs must be October31")
    exact = pd.read_csv(root / "same_date_week_metrics.csv")
    nearby = pd.read_csv(root / "nearby_week_scenarios.csv")
    if not all(table.n_days.eq(7).all() and table.n_half_hours.eq(336).all() for table in [exact, nearby]):
        raise ValueError("Only complete seven-day weeks can enter the evidence")
    evidence = pd.read_csv(root / "forecast_evidence.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for ax, task, metric, title in zip(axes, ["demand", "price"], ["MAPE", "BalancedAccuracy"],
                                      ["Demand MAPE", "Price balanced accuracy"]):
        direct = exact[(exact.task == task) & (exact.model == "random_forest")]
        scenarios = nearby[(nearby.task == task) & (nearby.model == "random_forest")]
        point = evidence[evidence.task == task].iloc[0]
        if len(direct) != 6 or len(scenarios) != 30:
            raise ValueError("Expected six same-date and thirty nearby weeks")
        for year, group in scenarios.groupby("year"):
            ax.scatter(np.linspace(year - .16, year + .16, len(group)), group[metric], s=14, c="#b2bbc9", alpha=.75)
        ax.plot(direct.year, direct[metric], "o-", label="Exact Dec 7–13 analog", c="#2475ae", lw=1)
        ax.scatter([2026], [point.forecast_point_mean_of_six_same_date_weeks], marker="D", c="#19826f", s=45, label="2026 point forecast")
        ax.vlines(2026, point.historical_scenario_p025, point.historical_scenario_p975, color="#19826f", lw=2, label="Historical scenario range")
        ax.set(title=title, ylabel="Percent (%)", xlabel="Analog year / forecast year", xticks=range(2020, 2027))
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("October 31 refits, latest 3-year windows; scenario ranges do not guarantee future coverage", fontsize=10)
    fig.tight_layout()
    fig.savefig(root / "forecast_evidence.png", dpi=180)
    fig.savefig(root / "forecast_evidence.pdf")
    plt.close(fig)
    (root / "verification.json").write_text(json.dumps({"status": "passed", "fits_per_task": 6,
        "exact_weeks_per_model_per_task": 6, "scenario_weeks_per_model_per_task": 30,
        "observations_per_week": 336, "train_cutoff_month_day": "10-31",
        "no_future_coverage_guarantee": True}, indent=2) + "\n")
    print("PASS forecast evidence and figure export")


if __name__ == "__main__":
    main()
