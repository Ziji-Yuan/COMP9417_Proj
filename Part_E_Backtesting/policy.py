"""Past-only error monitoring and deployment-budget bookkeeping.

These objects never fit forecasters. A caller evaluates the forecast already
issued for D, observes the resulting loss, requests a retrain, fits on data up
to D and commits its replacement for D+1. Failed fits do not spend the budget.
"""

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


def day_date(value):
    day = pd.Timestamp(value)
    if pd.isna(day) or day.tz is not None or day != day.normalize():
        raise ValueError("Use a finite naive market-day date")
    return day


def daily_loss(task, truth, prediction):
    """Demand MAPE in percent or price misclassification rate in percent."""
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != (48,) or prediction.shape != (48,):
        raise ValueError("Expected one complete market day of 48 intervals")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Truth and predictions must be finite")
    if task == "demand":
        if (truth <= 0).any():
            raise ValueError("Demand targets must be positive")
        return float(100 * np.mean(np.abs(truth - prediction) / truth))
    if task == "price":
        if not np.isin(truth, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
            raise ValueError("Price inputs must be hard binary labels")
        return float(100 * np.mean(truth != prediction))
    raise ValueError("Unknown task")


@dataclass(frozen=True)
class ErrorConfig:
    window_days: int = 28
    threshold_quantile: float = .99
    persistence_days: int = 3
    cooldown_days: int = 90

    def __post_init__(self):
        for name in ("window_days", "persistence_days", "cooldown_days"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Invalid {name}")
        if not 0 < self.threshold_quantile < 1:
            raise ValueError("Invalid quantile")


class ErrorDriftDetector:
    """Trailing mean loss above a fixed pre-deployment empirical threshold.

    Calibrate on *out-of-sample* daily losses (e.g. 2014 forecasts from a model
    trained through 2013). Provenance is supplied by the caller and recorded.
    Quantiles have no guaranteed false-alarm interpretation. Reference remains
    fixed after model replacement; only streaming loss state changes. This can
    leave a calibration/deployed-model mismatch after the optional 2014 refit.
    """

    def __init__(self, task, config=None):
        if task not in ("demand", "price"):
            raise ValueError("Unknown task")
        self.task = task
        self.config = config or ErrorConfig()

    def fit(self, losses, *, predictor_training_end, deployment_start="2015-01-01"):
        if not isinstance(losses, pd.Series) or not isinstance(losses.index, pd.DatetimeIndex):
            raise ValueError("Use a dated Series of out-of-sample daily losses")
        days = losses.index
        if len(days) < self.config.window_days or days.has_duplicates:
            raise ValueError("Insufficient or duplicate calibration days")
        for day in days:
            day_date(day)
        if not np.all((days[1:] - days[:-1]) == pd.Timedelta(days=1)):
            raise ValueError("Calibration must be chronological and contiguous")
        values = losses.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Losses must be finite and nonnegative")
        if self.task == "price" and (values > 100).any():
            raise ValueError("Price loss must be a percentage")
        training_end = day_date(predictor_training_end)
        deployment = day_date(deployment_start)
        if training_end >= days.min() or days.max() + pd.Timedelta(days=1) != deployment:
            raise ValueError("Calibration must be out-of-sample and end before deployment")
        scores = losses.rolling(self.config.window_days).mean().dropna()
        self.threshold_ = max(float(scores.quantile(self.config.threshold_quantile)), 1e-8)
        self.calibration_scores_ = scores.copy()
        self.calibration_start_, self.calibration_end_ = days.min(), days.max()
        self.predictor_training_end_ = training_end
        self.seed_ = values[-(self.config.window_days - 1):].copy() if self.config.window_days > 1 else []
        self.reset()
        return self

    def reset(self):
        if not hasattr(self, "threshold_"):
            raise ValueError("Fit detector first")
        self.window_ = deque(self.seed_, maxlen=self.config.window_days)
        self.last_day_ = self.calibration_end_
        self.last_alarm_ = None
        self.consecutive_ = 0
        return self

    def update(self, day, loss):
        if not hasattr(self, "threshold_"):
            raise ValueError("Fit detector first")
        day = day_date(day)
        if day != self.last_day_ + pd.Timedelta(days=1):
            raise ValueError("Observe every day exactly once in order")
        if not np.isfinite(loss) or loss < 0 or (self.task == "price" and loss > 100):
            raise ValueError("Invalid daily loss")
        self.window_.append(float(loss))
        score = float(np.mean(self.window_))
        exceeded = score > self.threshold_
        self.consecutive_ = self.consecutive_ + 1 if exceeded else 0
        cooled = self.last_alarm_ is None or (day - self.last_alarm_).days >= self.config.cooldown_days
        alarm = bool(exceeded and self.consecutive_ >= self.config.persistence_days and cooled)
        if alarm:
            self.last_alarm_ = day
        self.last_day_ = day
        return dict(market_day=day, effective_day=day + pd.Timedelta(days=1),
                    task=self.task, score=score, threshold=self.threshold_,
                    exceeded=exceeded, alarm=alarm, consecutive=self.consecutive_)


@dataclass(frozen=True)
class AdaptationConfig:
    strategy: str
    budget: int = 6
    training_years: int = 3
    start_day: str = "2015-01-01"
    end_day: str = "2026-10-31"

    def __post_init__(self):
        if self.strategy not in ("frozen", "periodic", "input_triggered", "error_triggered"):
            raise ValueError("Unknown strategy")
        if isinstance(self.budget, bool) or not isinstance(self.budget, int) or not 0 <= self.budget <= 6:
            raise ValueError("Budget must be an integer from 0 to 6")
        if isinstance(self.training_years, bool) or not isinstance(self.training_years, int) or self.training_years < 1:
            raise ValueError("training_years must be a positive integer")
        if day_date(self.start_day) > day_date(self.end_day):
            raise ValueError("Invalid simulation bounds")


@dataclass(frozen=True)
class RetrainRequest:
    observed_day: pd.Timestamp
    effective_day: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    ordinal: int
    reason: str


class BudgetedPolicy:
    """A separate instance per task/model/strategy; no future alarm lookup.

    Periodic: end of each odd calendar year (2015,17,19,21,23,25),
    independent of simulation losses. All non-frozen strategies use the same
    trailing three-calendar-year training rule by default. A request reserves
    no spent budget until caller confirms successful deployment replacement.
    """

    def __init__(self, config):
        self.config = config
        self.last_day = day_date(config.start_day) - pd.Timedelta(days=1)
        self.count = 0
        self.pending = None
        self.audit = []

    def observe(self, day, *, input_alarm=False, error_alarm=False):
        day = day_date(day)
        if self.pending is not None:
            raise ValueError("Commit or abort pending retraining before advancing")
        if day != self.last_day + pd.Timedelta(days=1) or day > day_date(self.config.end_day):
            raise ValueError("Observe each simulation day exactly once in order")
        if not isinstance(input_alarm, (bool, np.bool_)) or not isinstance(error_alarm, (bool, np.bool_)):
            raise ValueError("Alarm flags must be booleans")
        self.last_day = day
        trigger = {
            "frozen": False,
            "periodic": day.month == 12 and day.day == 31 and day.year % 2 == 1,
            "input_triggered": bool(input_alarm),
            "error_triggered": bool(error_alarm),
        }[self.config.strategy]
        if not trigger:
            return None
        effective = day + pd.Timedelta(days=1)
        if effective > day_date(self.config.end_day):
            self.audit.append(dict(observed_day=day, status="no_future_forecast"))
            return None
        if self.count >= self.config.budget:
            self.audit.append(dict(observed_day=day, status="budget_exhausted"))
            return None
        # Inclusive window: e.g. end 2015-12-31 => start 2013-01-01.
        start = max(pd.Timestamp("2010-01-01"), effective - pd.DateOffset(years=self.config.training_years))
        self.pending = RetrainRequest(day, effective, start, day, self.count + 1, self.config.strategy)
        return self.pending

    def commit(self, request, *, model_id):
        if request is not self.pending or self.pending is None:
            raise ValueError("Only commit the pending request once")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("Record deployed model identity")
        self.count += 1
        self.audit.append({**asdict(request), "status": "deployed", "model_id": model_id})
        self.pending = None

    def abort(self, request, *, reason):
        if request is not self.pending or self.pending is None:
            raise ValueError("Only abort the pending request")
        self.audit.append({**asdict(request), "status": "fit_failed", "failure": str(reason)})
        self.pending = None
