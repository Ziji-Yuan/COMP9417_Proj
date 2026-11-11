"""Input-only, season-adjusted Wasserstein monitoring with past-only calibration.

No target, prediction or model error is accepted by the detector. Alarms emitted
after observing day D are available to Part E for forecasting D+1 at the earliest.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from Part_A.data import market_day
from Part_A.features import FeatureConfig
from .profiles import DayTypeModel, daily_profiles, validate_half_hours


def monitor_config() -> FeatureConfig:
    """A's minimal, explicit predictor schema used by this input detector."""
    return FeatureConfig(states=("nsw",), demand_lags=(1,), price_lags=(1,),
                         rolling_windows=(), include_calendar=False,
                         include_cyclical=False, weather_columns=("syd_temp", "syd_solar"),
                         include_prev_day_price_mean=False)


def input_summaries(X: pd.DataFrame, day_types: DayTypeModel) -> pd.DataFrame:
    """Aggregate A's safe features into one vector per target market day.

    A profile for target day D here contains *lag-one* demand, i.e. actual demand
    on D-1. Its cluster distance is safe. Observed demand on D is never accessed.
    Additional columns in X are ignored; the required schema is explicit.
    """
    required = ["nsw_demand_lag_1d", "nsw_price_lag_1d", "syd_temp", "syd_solar"]
    missing = sorted(set(required).difference(X.columns))
    if missing:
        raise ValueError(f"Missing monitor features: {missing}; use monitor_config()")
    validate_half_hours(X.index)
    if not np.isfinite(X[required].to_numpy(dtype=float)).all():
        raise ValueError("Monitor features must be finite")
    profiles = daily_profiles(X, "nsw_demand_lag_1d")
    # Relabel to the source day for the retrospective clustering interface.
    prior = profiles.copy()
    prior.index = prior.index - pd.Timedelta(days=1)
    assignments = day_types.assign(prior)
    means = profiles.mean(axis=1)
    grouped = X[required].groupby(market_day(X.index))
    result = pd.DataFrame({
        "lag1_demand_mean_mw": means,
        "lag1_midday_ratio": profiles.loc[:, 22:29].mean(axis=1) / means,
        "lag1_evening_ratio": profiles.loc[:, 34:41].mean(axis=1) / means,
        "lag1_profile_distance": assignments.nearest_profile_distance.to_numpy(),
        "lag1_price_median": grouped.nsw_price_lag_1d.median(),
        "lag1_negative_price_share": (X.nsw_price_lag_1d < 0).groupby(market_day(X.index)).mean(),
        "syd_temp_mean": grouped.syd_temp.mean(),
        "syd_temp_max": grouped.syd_temp.max(),
        "syd_solar_mean": grouped.syd_solar.mean(),
    }, index=profiles.index)
    result.index.name = "market_day"
    return result.astype(float)


@dataclass(frozen=True)
class DriftConfig:
    window_days: int = 28
    threshold_quantile: float = 0.99
    persistence_days: int = 3
    cooldown_days: int = 90

    def __post_init__(self):
        for name in ("window_days", "persistence_days", "cooldown_days"):
            value = getattr(self, name)
            if int(value) != value or value < (2 if name == "window_days" else 1):
                raise ValueError(f"Invalid {name}")
        if not 0 < self.threshold_quantile < 1:
            raise ValueError("threshold_quantile must lie between 0 and 1")


@dataclass(frozen=True)
class DriftSignal:
    day: pd.Timestamp
    effective_day: pd.Timestamp
    score: float
    threshold: float
    exceeded: bool
    alarm: bool
    consecutive_exceedances: int
    feature_distances: dict[str, float]

    def to_record(self):
        return {"market_day": self.day, "effective_day": self.effective_day,
                "score": self.score, "threshold": self.threshold,
                "exceeded": self.exceeded, "alarm": self.alarm,
                "consecutive_exceedances": self.consecutive_exceedances,
                **{f"distance__{key}": value for key, value in self.feature_distances.items()}}


def _check_daily(frame: pd.DataFrame):
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.empty:
        raise ValueError("Expected a nonempty frame indexed by market-day dates")
    if frame.index.tz is not None or not frame.index.equals(frame.index.normalize()):
        raise ValueError("Use naive, normalized market-day dates")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Daily summaries must be unique and chronological")
    if len(frame) > 1 and not np.all((frame.index[1:] - frame.index[:-1]) == pd.Timedelta(days=1)):
        raise ValueError("Daily summaries must be contiguous; do not silently skip days")
    if frame.columns.has_duplicates or not len(frame.columns):
        raise ValueError("Feature columns must be nonempty and unique")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Daily summaries must contain only finite numbers")


class InputDriftDetector:
    """Fixed reference and threshold; only the rolling buffer updates online.

    1. Fit month x weekend medians on the reference period (default 2010--2013).
    2. Subtract these medians, divide by a global reference residual scale.
    3. Compare each marginal's trailing 28-day distribution with the full
       reference residual distribution using Wasserstein-1 distance.
    4. Score = maximum feature distance; calibrate its 99th percentile using
       only the calibration period (default 2014), with identical rolling windows.
    5. Alarm after 3 consecutive exceedances and at least 90 days since last alarm.

    This is an empirical monitoring rule, not a hypothesis test with a guaranteed
    false-alarm probability. The cooldown does not enforce E's six-retrain budget.
    """

    def __init__(self, config: DriftConfig | None = None):
        self.config = config or DriftConfig()

    def fit(self, reference: pd.DataFrame, calibration: pd.DataFrame):
        _check_daily(reference)
        _check_daily(calibration)
        if list(reference.columns) != list(calibration.columns):
            raise ValueError("Reference and calibration must have identical feature columns")
        if reference.index.max() + pd.Timedelta(days=1) != calibration.index.min():
            raise ValueError("Calibration must immediately follow the reference period")
        if min(len(reference), len(calibration)) < self.config.window_days:
            raise ValueError("Each fitting period must contain at least window_days observations")
        self.columns_ = list(reference.columns)
        self.fallback_centre_ = reference.median().to_numpy()
        self.seasonal_centres_ = {}
        for month in range(1, 13):
            for weekend in (False, True):
                selected = reference[(reference.index.month == month)
                                     & ((reference.index.day_of_week >= 5) == weekend)]
                self.seasonal_centres_[(month, weekend)] = (
                    selected.median().to_numpy() if len(selected) else self.fallback_centre_.copy())
        residuals = np.stack([row.to_numpy(dtype=float) - self._centre(day)
                              for day, row in reference.iterrows()])
        iqr = np.quantile(residuals, 0.75, axis=0) - np.quantile(residuals, 0.25, axis=0)
        std = residuals.std(axis=0)
        # Degenerate variables (e.g. no negative prices in training) use one unit.
        self.scale_ = np.where(iqr > 1e-8, iqr, np.where(std > 1e-8, std, 1.0))
        self.reference_residuals_ = residuals / self.scale_
        self.reference_start_, self.reference_end_ = reference.index.min(), reference.index.max()
        self.calibration_start_, self.calibration_end_ = calibration.index.min(), calibration.index.max()
        window = deque(self.reference_residuals_[-(self.config.window_days - 1):],
                       maxlen=self.config.window_days)
        scores = []
        for day, row in calibration.iterrows():
            window.append(self._transform(day, row.to_numpy(dtype=float)))
            distances = self._distances(window)
            scores.append({"market_day": day, "score": float(distances.max()),
                           **{f"distance__{name}": float(value)
                              for name, value in zip(self.columns_, distances)}})
        self.calibration_scores_ = pd.DataFrame(scores).set_index("market_day")
        self.threshold_ = max(float(np.quantile(self.calibration_scores_.score,
                                                self.config.threshold_quantile)), 1e-8)
        self.seed_window_ = np.array(list(window)[-(self.config.window_days - 1):], copy=True)
        self.reset()
        return self

    def _centre(self, day):
        return self.seasonal_centres_[(day.month, day.day_of_week >= 5)]

    def _transform(self, day, values):
        return (values - self._centre(day)) / self.scale_

    def _distances(self, window):
        current = np.asarray(window)
        return np.array([wasserstein_distance(self.reference_residuals_[:, j], current[:, j])
                         for j in range(len(self.columns_))])

    def reset(self):
        """Restart at the end of calibration without changing fitted parameters."""
        if not hasattr(self, "threshold_"):
            raise ValueError("Fit the input detector first")
        self.window_ = deque((row.copy() for row in self.seed_window_), maxlen=self.config.window_days)
        self.last_day_ = self.calibration_end_
        self.last_alarm_ = None
        self.consecutive_ = 0
        return self

    def update(self, day, summary: pd.Series) -> DriftSignal:
        """Observe day D once, in order; return an alarm effective on D+1.

        E calls this after evaluating D, then decides whether budget and its
        retraining policy permit replacing the deployed model for D+1.
        """
        if not hasattr(self, "threshold_"):
            raise ValueError("Fit the input detector first")
        day = pd.Timestamp(day)
        if day.tz is not None or day != day.normalize():
            raise ValueError("day must be a naive market-day date")
        if day != self.last_day_ + pd.Timedelta(days=1):
            raise ValueError("Update exactly once per consecutive day after calibration")
        if not isinstance(summary, pd.Series) or list(summary.index) != self.columns_:
            raise ValueError("Summary must be a Series with the fitted feature order")
        values = summary.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Summary must be finite")
        self.window_.append(self._transform(day, values))
        distances = self._distances(self.window_)
        score = float(distances.max())
        exceeded = score > self.threshold_
        self.consecutive_ = self.consecutive_ + 1 if exceeded else 0
        cooled = self.last_alarm_ is None or (day - self.last_alarm_).days >= self.config.cooldown_days
        alarm = bool(exceeded and self.consecutive_ >= self.config.persistence_days and cooled)
        if alarm:
            self.last_alarm_ = day
        self.last_day_ = day
        return DriftSignal(day, day + pd.Timedelta(days=1), score, self.threshold_, exceeded,
                           alarm, self.consecutive_, dict(zip(self.columns_, map(float, distances))))

    def replay(self, summaries: pd.DataFrame) -> pd.DataFrame:
        """Replay a contiguous post-calibration period from a fresh detector state."""
        _check_daily(summaries)
        self.reset()
        return pd.DataFrame([self.update(day, row).to_record()
                             for day, row in summaries.iterrows()]).set_index("market_day")
