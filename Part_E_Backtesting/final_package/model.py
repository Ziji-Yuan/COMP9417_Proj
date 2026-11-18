"""Frozen next-day forecasts; initialization loads fitted files only."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from Part_A.features import FeatureConfig, get_inference_features


class Forecaster:
    def __init__(self, artifact_dir: str):
        directory = Path(artifact_dir)
        self.models = joblib.load(directory / "models.joblib")
        self.metadata = json.loads((directory / "metadata.json").read_text())

    def predict(self, history: pd.DataFrame, weather_next: pd.DataFrame) -> pd.DataFrame:
        features = get_inference_features(history, weather_next, FeatureConfig())
        if list(features.columns) != self.metadata["feature_columns"]:
            raise ValueError("Feature schema differs from fitted models")
        with threadpool_limits(limits=2):
            demand = self.models["demand"].predict(features)
            price = self.models["price"].predict(features)
        if not np.isfinite(demand).all() or not np.isin(price, [0, 1]).all():
            raise ValueError("Invalid fitted-model output")
        return pd.DataFrame({"nsw_demand_mw": np.asarray(demand, dtype=float),
                             "nsw_price_up": np.asarray(price, dtype=int)}, index=weather_next.index)
