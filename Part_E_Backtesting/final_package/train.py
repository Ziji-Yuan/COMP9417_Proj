"""Reproduce final artifacts: python train.py path/to/released.csv.

Published B/C RF configurations; latest three calendar years ending 2026-10-31.
No historical backtest, external data, detector or network is required.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from threadpoolctl import threadpool_limits

from Part_A.features import FeatureConfig, get_features


DATA_SHA256 = "c13711002906140eaf7cc67c1ce17a49facd359cf3d1a84722aaa3ccd647752c"
PINS = {"numpy": "2.4.6", "pandas": "3.1.0", "scipy": "1.17.1", "scikit-learn": "1.9.1", "joblib": "1.6.0"}


def create_models():
    return {
        "demand": RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=2,
                                        random_state=42, n_jobs=2),
        "price": RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=5,
                                        max_features="sqrt", random_state=8, n_jobs=2),
    }


def environment():
    versions = {package: importlib.metadata.version(package) for package in PINS}
    if platform.python_version_tuple()[:2] != ("3", "11") or versions != PINS:
        raise ValueError(f"Use Python3.11 and official pinned libraries: {versions}")
    return {"python": platform.python_version(), "libraries": versions}


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python train.py path/to/nem_weather_halfhourly_2010_2026.csv")
    started = time.perf_counter()
    env = environment()
    csv = Path(sys.argv[1])
    with csv.open("rb") as stream:
        data_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if data_hash != DATA_SHA256:
        raise ValueError("CSV does not match the fixed released dataset")
    end = pd.Timestamp("2026-10-31")
    start = end + pd.Timedelta(days=1) - pd.DateOffset(years=3)
    X, y, meta = get_features(start, end, FeatureConfig(), path=csv)
    if len(X) != len(pd.date_range(start, end)) * 48:
        raise ValueError("Incomplete final training window")
    models = create_models()
    fit_seconds = {}
    with threadpool_limits(limits=2):
        for task, target in [("demand", "nsw_demand_mw"), ("price", "nsw_price_up")]:
            before = time.perf_counter()
            print(f"Fitting final {task}: {start.date()}--{end.date()}, {len(X)} rows", flush=True)
            models[task].fit(X, y[target])
            fit_seconds[task] = time.perf_counter() - before
    artifacts = Path(__file__).resolve().parent / "artifacts"
    artifacts.mkdir(exist_ok=True)
    # Lossless compression changes storage, not fitted predictions.
    temporary = artifacts / "models.joblib.tmp"
    joblib.dump(models, temporary, compress=("xz", 3))
    temporary.replace(artifacts / "models.joblib")
    metadata = {
        "training_start": str(start.date()), "training_end": str(end.date()),
        "training_rows": len(X), "training_years": 3, "data_sha256": data_hash,
        "feature_columns": list(X.columns), "environment": env,
        "selection_basis": "Stage2 comparison: periodic replacements with trailing3year window performed best for both tasks",
        "stage3_rule": "Fresh fit of selected trailing3year RFs through last released day; freeze through live week",
        "model_parameters": {task: model.get_params() for task, model in models.items()},
        "fit_seconds": fit_seconds, "total_training_seconds": time.perf_counter() - started,
        "compression": ["xz", 3],
    }
    (artifacts / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    size = sum(p.stat().st_size for p in artifacts.iterdir() if p.is_file())
    if size > 100 * 2**20:
        raise ValueError(f"Artifacts exceed100MiB: {size} bytes; do not package this output")
    print(f"Final artifacts: {size / 2**20:.2f}MiB; total training {time.perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
