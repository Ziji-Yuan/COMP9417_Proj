"""Reproduce Part D: python -m Part_D_Drift.run --data PATH --output DIR."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from Part_A.features import get_data, get_features
from .analysis import prepare_predictions, relate_predictions
from .diagnostics import (clustering_audit, day_type_drift_context, detector_sensitivity,
                          drift_episodes, event_comparisons, profile_evidence, reference_scale_table)
from .drift import DriftConfig, InputDriftDetector, input_summaries, monitor_config
from .profiles import DayTypeConfig, DayTypeModel, cluster_shares, daily_profiles


OFFICIAL_VERSIONS = {"numpy": "2.4.6", "pandas": "3.1.0", "scipy": "1.17.1",
                     "scikit-learn": "1.9.1", "joblib": "1.6.0"}


def environment_info():
    versions = {package: importlib.metadata.version(package)
                for package in [*OFFICIAL_VERSIONS, "matplotlib"]}
    return {"python": platform.python_version(), "packages": versions,
            "matches_official_core": platform.python_version_tuple()[:2] == ("3", "11")
            and all(versions[k] == v for k, v in OFFICIAL_VERSIONS.items())}


def run(args):
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new directory to avoid stale results")
    env = environment_info()
    if args.strict_versions and not env["matches_official_core"]:
        raise ValueError(f"Environment differs from official core versions: {env}")
    print("Loading A's validated data interface...", flush=True)
    observed = get_data(path=args.data)
    profiles = daily_profiles(observed)
    if profiles.index.min() > pd.Timestamp("2010-01-01") or profiles.index.max() < pd.Timestamp("2015-01-28"):
        raise ValueError("This experiment needs 2010--2014 and a post-2014 simulation period")
    if profiles.index.max() > pd.Timestamp("2026-10-31"):
        raise ValueError("Use only the released data, ending at market day 2026-10-31")
    print("Fitting fixed 2010--2013 PCA and clustering; comparing shape and raw demand...", flush=True)
    config = DayTypeConfig(normalize=args.normalization == "daily-mean", random_state=args.seed)
    model = DayTypeModel(config).fit(profiles)
    sensitivity = DayTypeModel(DayTypeConfig(normalize=not config.normalize, random_state=args.seed)).fit(profiles)
    assignments = model.assign(profiles)
    yearly = cluster_shares(assignments, model.n_clusters_, "year")
    monthly = cluster_shares(assignments, model.n_clusters_, "month")
    print("Building input-only daily summaries through A's shared features...", flush=True)
    feature_config = monitor_config()
    X, truth, _ = get_features(config=feature_config, data=observed)
    summaries = input_summaries(X, model)
    detector = InputDriftDetector(DriftConfig()).fit(
        summaries.loc[:"2013-12-31"], summaries.loc["2014-01-01":"2014-12-31"])
    print("Replaying the detector chronologically from 2015...", flush=True)
    signals = detector.replay(summaries.loc["2015-01-01":])
    print("Checking training-only clustering stability and pre-2015 detector sensitivity...", flush=True)
    diagnostics = clustering_audit(profiles, model)
    diagnostics.update(profile_evidence(profiles, assignments, model.n_clusters_))
    diagnostics.update(detector_sensitivity(summaries))
    diagnostics.update(drift_episodes(signals))
    diagnostics.update(day_type_drift_context(assignments, signals, model.n_clusters_))
    diagnostics["detector_reference_scales"] = reference_scale_table(detector)
    sources = json.loads(Path(__file__).with_name("market_sources.json").read_text(encoding="utf-8"))
    diagnostics.update(event_comparisons(assignments, signals, sources["events"]))
    comparisons = None
    if args.predictions:
        raw = pd.concat([pd.read_csv(path) for path in args.predictions], ignore_index=True)
        predictions = prepare_predictions(raw, truth=truth)
        comparisons = relate_predictions(predictions, assignments, signals)
    output.mkdir(parents=True, exist_ok=True)
    tables, figures, artifacts = [output / name for name in ("tables", "figures", "artifacts")]
    for folder in (tables, figures, artifacts):
        folder.mkdir()
    for name, table in diagnostics.items():
        table.to_csv(tables / f"{name}.csv", index=False)
    (output / "market_sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    assignments.to_csv(tables / "day_assignments.csv")
    yearly.to_csv(tables / "yearly_cluster_shares.csv")
    monthly.to_csv(tables / "monthly_cluster_shares.csv")
    model.selection_.to_csv(tables / "cluster_selection.csv")
    model.cluster_summary_.to_csv(tables / "training_cluster_summary.csv")
    pd.DataFrame(model.centres_, columns=range(48)).rename_axis("cluster").to_csv(tables / "training_profiles.csv")
    pd.DataFrame({"component": np.arange(1, model.pca_.n_components_ + 1),
                  "explained_variance_ratio": model.pca_.explained_variance_ratio_,
                  "cumulative_variance_ratio": np.cumsum(model.pca_.explained_variance_ratio_)}).to_csv(
                      tables / "pca_variance.csv", index=False)
    sensitivity.selection_.to_csv(tables / "alternative_normalization_selection.csv")
    alternative_assignments = sensitivity.assign(profiles)
    cluster_shares(alternative_assignments, sensitivity.n_clusters_).to_csv(
        tables / "alternative_normalization_yearly_shares.csv")
    pd.DataFrame([
        {"normalization": "daily-mean" if item.config.normalize else "raw", "selected_k": item.n_clusters_,
         "pca_components": item.pca_.n_components_,
         "selected_training_silhouette": item.selection_.loc[item.n_clusters_, "silhouette"],
         "role": "primary" if item is model else "sensitivity"}
        for item in (model, sensitivity)
    ]).to_csv(tables / "normalization_comparison.csv", index=False)
    summaries.to_csv(tables / "input_summaries.csv")
    detector.calibration_scores_.to_csv(tables / "calibration_scores_2014.csv")
    signals.to_csv(tables / "input_drift_daily.csv")
    signals.loc[signals.alarm].to_csv(tables / "input_drift_alarms.csv")
    signals.groupby(signals.index.year).agg(days=("alarm", "size"), alarms=("alarm", "sum"),
                                          exceedance_share=("exceeded", "mean"),
                                          mean_score=("score", "mean")).rename_axis("year").to_csv(
                                              tables / "yearly_drift_summary.csv")
    detector.reset()  # Export a pristine detector ready to process 2015-01-01.
    joblib.dump(model, artifacts / "day_types.joblib")
    joblib.dump(detector, artifacts / "input_detector.joblib")
    from .plots import diagnostic_figures, make_figures, performance_figure
    make_figures(model, assignments, yearly, signals, figures)
    diagnostic_figures(diagnostics, model, figures)
    if comparisons is not None:
        for name, table in comparisons.items():
            table.to_csv(tables / f"{name}.csv", index=False)
        performance_figure(comparisons["monthly_performance"], figures)
    with open(args.data, "rb") as data_file:
        data_hash = hashlib.file_digest(data_file, "sha256").hexdigest()
    summary = {
        "part_d_source_sha256": {str(path.relative_to(Path(__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sorted(Path(__file__).parent.rglob("*"))
                                 if path.is_file() and path.suffix in (".py", ".json")},
        "data_path": str(Path(args.data).resolve()),
        "data_sha256": data_hash,
        "market_day_start": str(profiles.index.min().date()),
        "market_day_end": str(profiles.index.max().date()), "days": len(profiles),
        "day_type_config": asdict(config), "training_days": model.training_days_,
        "training_end": str(model.train_end_.date()), "selected_k": model.n_clusters_,
        "pca_components": int(model.pca_.n_components_),
        "retained_variance": float(model.pca_.explained_variance_ratio_.sum()),
        "novelty_threshold": model.novelty_threshold_,
        "monitor_feature_config": asdict(feature_config), "detector_config": asdict(detector.config),
        "detector_reference": [str(detector.reference_start_.date()), str(detector.reference_end_.date())],
        "detector_calibration": [str(detector.calibration_start_.date()), str(detector.calibration_end_.date())],
        "input_threshold": detector.threshold_, "alarms": int(signals.alarm.sum()),
        "exceedance_episodes": len(diagnostics["input_drift_episodes"]),
        "repeat_alarms_within_episode": int((diagnostics["input_alarm_attribution"].alarm_kind == "repeat_within_episode").sum()),
        "first_alarm": str(signals.index[signals.alarm][0].date()) if signals.alarm.any() else None,
        "simulation_days": len(signals), "environment": env,
        "prediction_files": args.predictions,
        "performance_analysis_status": "completed" if comparisons is not None else "pending_B_C_predictions",
        "independent_D_evidence_status": "generated",
        "sensitivity_protocol": {"cluster_fit_end": "2013-12-31", "detector_calibration_end": "2014-06-30",
                                 "detector_holdout": ["2014-07-01", "2014-12-31"],
                                 "primary_configuration_reselected": False},
        "market_events_used_for_fitting": False,
        "limitations": [
            "Threshold is empirical; overlapping windows and temporal dependence invalidate a nominal false-alarm interpretation.",
            "Only marginal distribution changes in the chosen summaries are monitored; changes in dependence can be missed.",
            "A fixed old reference can remain different after a predictor is retrained; repeated cooldown alarms are possible.",
            "Rare training features can have small residual scales and dominate the standardized score; inspect the scale and attribution tables.",
            "2014 sensitivity exceedances are not labelled false positives, and these retrospective diagnostics did not reselect the primary rule.",
            "Input drift is not proof of concept drift or of reduced predictive accuracy.",
            "Observed day types are retrospective; only lag-one profiles enter the input detector.",
            "The six-retrain budget is enforced by Part E, not by suppressing D's diagnostic alarms.",
            "2026 contains January through October only; use the separate January-October table for matched seasonal coverage.",
            "Market-event windows and previous-year comparisons are descriptive, not causal estimates. Sources are interpretation only.",
            "No B/C model performance is inferred when their predictions are absent.",
        ],
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("selected_k", "pca_components", "retained_variance",
                                                  "input_threshold", "alarms", "performance_analysis_status")}, indent=2))
    print(f"Results saved to {output.resolve()}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the unchanged official CSV")
    parser.add_argument("--output", default="outputs/part_d_complete", help="A new or empty results directory")
    parser.add_argument("--predictions", action="append", default=[], help="B/C prediction CSV; repeat for multiple files")
    parser.add_argument("--normalization", choices=["daily-mean", "raw"], default="daily-mean")
    parser.add_argument("--seed", type=int, default=9417)
    parser.add_argument("--strict-versions", action="store_true", help="Require Python 3.11 and the official core pins")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args)


if __name__ == "__main__":
    main()
