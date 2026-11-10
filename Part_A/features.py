"""Leakage-free shared feature engineering for COMP9417 day-ahead tasks.

Public interfaces
-----------------
``get_data`` returns aligned observed data for analysis.
``get_features`` builds supervised training rows and both official targets.
``get_inference_features`` builds the same features from the official
``history`` and ``weather_next`` inputs without reading a local file.

All historical features use explicit (market_day, slot) lookups.  A day lag
therefore means the same half-hour slot on an earlier market day and never a
fixed number of rows into an unvalidated table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .data import (
    EXPECTED_COLUMNS,
    MARKET_COLUMNS,
    WEATHER_COLUMNS,
    add_time_keys,
    half_hour_slot,
    load_data,
    market_day,
    select_market_days,
    validate_data,
)


_STATE_ORDER = ("nsw", "qld", "vic")
_ROLLING_STATS = ("mean", "std")


def _normalise_positive_ints(values: Iterable[int], name: str) -> tuple[int, ...]:
    result = tuple(sorted(set(int(value) for value in values)))
    if any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive market-day counts")
    return result


@dataclass(frozen=True)
class FeatureConfig:
    """Deterministic selection of safe day-ahead feature families.

    Rolling statistics are calculated for each selected state/market variable
    at the same half-hour slot across the previous ``window`` market days.
    ``std`` is the sample standard deviation (ddof=1).
    """

    states: Sequence[str] = ("nsw", "qld", "vic")
    demand_lags: Sequence[int] = (1, 7)
    price_lags: Sequence[int] = (1, 7)
    rolling_windows: Sequence[int] = (7,)
    rolling_stats: Sequence[str] = ("mean", "std")
    include_calendar: bool = True
    include_cyclical: bool = True
    include_weather: bool = True
    weather_columns: Sequence[str] | None = None
    include_prev_day_price_mean: bool = True
    drop_rows_with_missing_history: bool = True

    def __post_init__(self) -> None:
        states = tuple(state for state in _STATE_ORDER if state in set(self.states))
        invalid_states = sorted(set(self.states).difference(_STATE_ORDER))
        if invalid_states:
            raise ValueError(f"Unsupported states: {invalid_states}")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "demand_lags", _normalise_positive_ints(self.demand_lags, "demand_lags"))
        object.__setattr__(self, "price_lags", _normalise_positive_ints(self.price_lags, "price_lags"))
        object.__setattr__(self, "rolling_windows", _normalise_positive_ints(self.rolling_windows, "rolling_windows"))
        stats = tuple(stat for stat in _ROLLING_STATS if stat in set(self.rolling_stats))
        invalid_stats = sorted(set(self.rolling_stats).difference(_ROLLING_STATS))
        if invalid_stats:
            raise ValueError(f"Unsupported rolling statistics: {invalid_stats}")
        object.__setattr__(self, "rolling_stats", stats)
        if "std" in stats and any(window < 2 for window in self.rolling_windows):
            raise ValueError("rolling std requires every rolling window to be at least 2 days")
        selected_weather = tuple(WEATHER_COLUMNS if self.weather_columns is None else self.weather_columns)
        invalid_weather = sorted(set(selected_weather).difference(WEATHER_COLUMNS))
        if invalid_weather:
            raise ValueError(f"Unsupported weather columns: {invalid_weather}")
        selected_weather = tuple(column for column in WEATHER_COLUMNS if column in set(selected_weather))
        object.__setattr__(self, "weather_columns", selected_weather)

    @property
    def maximum_lookback_days(self) -> int:
        """Largest historical day count required by this configuration."""
        candidates = list(self.demand_lags) + list(self.price_lags) + list(self.rolling_windows)
        if self.include_prev_day_price_mean:
            candidates.append(1)
        return max(candidates, default=0)


def _source_data(data: pd.DataFrame | None, path: str | Path | None) -> pd.DataFrame:
    if data is not None and path is not None:
        raise ValueError("Pass either data or path, not both")
    if data is None:
        frame = load_data(path)
    else:
        frame = data
        if not isinstance(frame.index, pd.DatetimeIndex):
            if "timestamp" not in frame.columns:
                raise ValueError("Preloaded data needs a DatetimeIndex or timestamp column")
            frame = frame.set_index(pd.to_datetime(frame["timestamp"])).drop(columns="timestamp")
        frame = frame.sort_index()
    validate_data(frame.loc[:, EXPECTED_COLUMNS], require_complete_days=True)
    return frame.loc[:, EXPECTED_COLUMNS]


def get_data(
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    *,
    data: pd.DataFrame | None = None,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Return observed variables plus official ``market_day`` and ``slot``.

    ``start`` and ``end`` are inclusive market-day dates.  This interface is
    intended for analysis and may contain outcomes; it is not itself a
    leakage-free predictive feature matrix.
    """
    frame = _source_data(data, path)
    selected = select_market_days(frame, start=start, end=end)
    return add_time_keys(selected)


def _normalised_range(
    days: pd.DatetimeIndex,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_day = pd.Timestamp(start).normalize() if start is not None else days.min()
    end_day = pd.Timestamp(end).normalize() if end is not None else days.max()
    if start_day > end_day:
        raise ValueError("start must be on or before end")
    return start_day, end_day


def _market_series(frame: pd.DataFrame, column: str) -> pd.Series:
    keys = pd.MultiIndex.from_arrays(
        [market_day(frame.index), half_hour_slot(frame.index)],
        names=["market_day", "slot"],
    )
    series = pd.Series(frame[column].to_numpy(), index=keys, name=column)
    if series.index.has_duplicates:
        raise ValueError("Each (market_day, slot) pair must be unique")
    return series


def _same_slot_lookup(
    series: pd.Series,
    target_days: pd.DatetimeIndex,
    slots: pd.Index,
    lag_days: int,
) -> np.ndarray:
    lookup = pd.MultiIndex.from_arrays(
        [target_days - pd.Timedelta(days=lag_days), slots],
        names=["market_day", "slot"],
    )
    return series.reindex(lookup).to_numpy(dtype=float)


def _previous_day_price_mean(frame: pd.DataFrame, target_days: pd.DatetimeIndex) -> np.ndarray:
    source_days = market_day(frame.index)
    daily = frame["nsw_price"].groupby(source_days).mean()
    return daily.reindex(target_days - pd.Timedelta(days=1)).to_numpy(dtype=float)


def _calendar_features(target_days: pd.DatetimeIndex, slots: pd.Index, cyclical: bool) -> pd.DataFrame:
    result = pd.DataFrame({
        "slot": slots.to_numpy(dtype=int),
        "day_of_week": target_days.dayofweek,
        "month": target_days.month,
        "is_weekend": (target_days.dayofweek >= 5).astype(int),
    })
    if cyclical:
        result["slot_sin"] = np.sin(2 * np.pi * result["slot"] / 48)
        result["slot_cos"] = np.cos(2 * np.pi * result["slot"] / 48)
        result["dow_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
        result["dow_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
        result["month_sin"] = np.sin(2 * np.pi * (result["month"] - 1) / 12)
        result["month_cos"] = np.cos(2 * np.pi * (result["month"] - 1) / 12)
    return result


def _build_features(
    market_source: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    target_weather: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    """Shared training/inference implementation using safe historical lookups."""
    target_index = pd.DatetimeIndex(target_index)
    target_days = market_day(target_index)
    slots = half_hour_slot(target_index)
    pieces: list[pd.DataFrame] = []

    if config.include_calendar:
        pieces.append(_calendar_features(target_days, slots, config.include_cyclical))
    if config.include_weather:
        weather = target_weather.reindex(target_index).loc[:, list(config.weather_columns)].reset_index(drop=True)
        pieces.append(weather)

    for state in config.states:
        for kind, lags in (("demand", config.demand_lags), ("price", config.price_lags)):
            column = f"{state}_{kind}"
            series = _market_series(market_source, column)
            for lag in lags:
                pieces.append(pd.DataFrame({
                    f"{column}_lag_{lag}d": _same_slot_lookup(series, target_days, slots, lag)
                }))
            for window in config.rolling_windows:
                history = np.column_stack([
                    _same_slot_lookup(series, target_days, slots, lag)
                    for lag in range(1, window + 1)
                ])
                complete = np.isfinite(history).all(axis=1)
                if "mean" in config.rolling_stats:
                    values = np.nanmean(history, axis=1)
                    values[~complete] = np.nan
                    pieces.append(pd.DataFrame({f"{column}_roll_{window}d_mean": values}))
                if "std" in config.rolling_stats:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        values = np.nanstd(history, axis=1, ddof=1)
                    values[~complete] = np.nan
                    pieces.append(pd.DataFrame({f"{column}_roll_{window}d_std": values}))

    if config.include_prev_day_price_mean:
        pieces.append(pd.DataFrame({
            "nsw_price_prev_day_mean": _previous_day_price_mean(market_source, target_days)
        }))
    if not pieces:
        raise ValueError("FeatureConfig selects no features")
    result = pd.concat(pieces, axis=1)
    result.index = target_index
    result.index.name = target_index.name or "timestamp"
    if result.columns.duplicated().any():
        raise RuntimeError("Feature configuration produced duplicate column names")
    return result.astype(float)


def _targets(frame: pd.DataFrame, target_index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = frame.reindex(target_index)
    target_days = market_day(target_index)
    reference = _previous_day_price_mean(frame, target_days)
    direction = pd.Series(pd.NA, index=target_index, dtype="Int64")
    valid = target["nsw_price"].notna().to_numpy() & np.isfinite(reference)
    direction.iloc[np.flatnonzero(valid)] = (
        target["nsw_price"].to_numpy()[valid] > reference[valid]
    ).astype(int)
    y = pd.DataFrame({
        "nsw_demand_mw": target["nsw_demand"].to_numpy(dtype=float),
        "nsw_price_up": direction,
    }, index=target_index)
    y.index.name = target_index.name or "timestamp"
    meta = pd.DataFrame({
        "timestamp": target_index,
        "market_day": target_days,
        "slot": half_hour_slot(target_index).to_numpy(dtype=int),
        "price_reference_prev_day_mean": reference,
    }, index=target_index)
    meta.index.name = target_index.name or "timestamp"
    return y, meta


def get_features(
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    config: FeatureConfig | None = None,
    *,
    data: pd.DataFrame | None = None,
    path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build aligned supervised rows for an inclusive target market-day range.

    Historical context before ``start`` is retained for lag/rolling lookups.
    With the default configuration, rows lacking any required historical
    feature or either valid target are removed.  Set
    ``drop_rows_with_missing_history=False`` to retain feature ``NaN`` values;
    invalid target rows are always removed rather than labelled as zero.
    """
    config = config or FeatureConfig()
    frame = _source_data(data, path)
    days = market_day(frame.index)
    start_day, end_day = _normalised_range(days, start, end)
    target_mask = (days >= start_day) & (days <= end_day)
    target_index = frame.index[target_mask]
    if target_index.empty:
        raise ValueError("No observations fall in the requested target market-day range")
    target_weather = frame.loc[target_index, WEATHER_COLUMNS]
    X = _build_features(frame, target_index, target_weather, config)
    y, meta = _targets(frame, target_index)
    valid_targets = y["nsw_demand_mw"].notna() & y["nsw_price_up"].notna()
    keep = valid_targets
    if config.drop_rows_with_missing_history:
        keep &= X.notna().all(axis=1)
    return X.loc[keep], y.loc[keep], meta.loc[keep]


def get_inference_features(
    history: pd.DataFrame,
    weather_next: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Build exactly 48 next-day rows from official harness inputs only."""
    config = config or FeatureConfig()
    if not isinstance(history.index, pd.DatetimeIndex) or not isinstance(weather_next.index, pd.DatetimeIndex):
        raise ValueError("history and weather_next must use DatetimeIndex")
    history = history.sort_index()
    missing_market = sorted(set(MARKET_COLUMNS).difference(history.columns))
    missing_weather = sorted(set(WEATHER_COLUMNS).difference(weather_next.columns))
    if missing_market:
        raise ValueError(f"history is missing market columns: {missing_market}")
    if missing_weather:
        raise ValueError(f"weather_next is missing weather columns: {missing_weather}")
    if len(weather_next) != 48 or weather_next.index.has_duplicates:
        raise ValueError("weather_next must contain exactly 48 unique rows")
    target_days = market_day(weather_next.index)
    slots = half_hour_slot(weather_next.index)
    if target_days.nunique() != 1 or set(slots) != set(range(48)):
        raise ValueError("weather_next must be one complete ordered market day with slots 0..47")
    if not weather_next.index.is_monotonic_increasing:
        raise ValueError("weather_next must be ordered by timestamp")
    target_day = target_days[0]
    if len(history) and (market_day(history.index) >= target_day).any():
        raise ValueError("history contains target-day or future observations")
    X = _build_features(history, weather_next.index, weather_next, config)
    if config.drop_rows_with_missing_history and X.isna().any(axis=None):
        missing = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Insufficient history for configured inference features: {missing}")
    return X


__all__ = ["FeatureConfig", "get_data", "get_features", "get_inference_features"]
