"""Data loading and market-time alignment utilities.

Timestamps in the supplied data denote the *end* of each half-hour interval.
Consequently, the observation stamped at midnight belongs to the previous
market day.  The helpers here mirror ``harness/dayahead.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


MARKET_COLUMNS = [
    "nsw_demand", "nsw_price", "qld_demand", "qld_price",
    "vic_demand", "vic_price",
]
WEATHER_COLUMNS = [
    f"{city}_{variable}"
    for city in ("syd", "mel", "bne")
    for variable in ("temp", "rh", "cloud", "solar")
]
EXPECTED_COLUMNS = MARKET_COLUMNS + WEATHER_COLUMNS
N_PERIODS = 48


def default_data_path() -> Path:
    """Return the first available repository-relative dataset location.

    The preferred layout is ``<project>/data/<csv>``.  The second candidate
    supports the dataset currently stored beside this notebook.
    """
    part_dir = Path(__file__).resolve().parent
    candidates = (
        part_dir.parent / "data" / "nem_weather_halfhourly_2010_2026.csv",
        part_dir / "nem_weather_halfhourly_2010_2026.csv",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Dataset not found. Pass an explicit path to load_data(path=...)."
    )


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the fixed CSV without modifying it, indexed by sorted timestamp."""
    csv_path = Path(path) if path is not None else default_data_path()
    frame = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    missing = sorted(set(EXPECTED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return frame.sort_index()


def validate_data(frame: pd.DataFrame, require_complete_days: bool = True) -> None:
    """Validate the official schema, timestamps and optional day completeness.

    Raises ``ValueError`` instead of silently repairing or imputing source data.
    Extra columns are permitted so callers may pass a frame returned by
    :func:`add_time_keys`.
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex named timestamp.")
    if frame.index.has_duplicates:
        raise ValueError("Duplicate timestamps are not supported.")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Timestamps must be sorted in increasing order.")
    missing = sorted(set(EXPECTED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    if require_complete_days:
        counts = pd.Series(1, index=frame.index).groupby(market_day(frame.index)).sum()
        incomplete = counts[counts != N_PERIODS]
        if not incomplete.empty:
            sample = ", ".join(f"{day.date()} ({count})" for day, count in incomplete.head().items())
            raise ValueError(f"Incomplete market days found: {sample}")


def select_market_days(
    frame: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Select an inclusive range using official market days, not calendar dates."""
    if start is not None and end is not None and pd.Timestamp(start).normalize() > pd.Timestamp(end).normalize():
        raise ValueError("start must be on or before end")
    days = market_day(frame.index)
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= days >= pd.Timestamp(start).normalize()
    if end is not None:
        mask &= days <= pd.Timestamp(end).normalize()
    return frame.loc[mask.to_numpy()]


def market_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Map end-of-interval timestamps to their interval-start calendar day."""
    return (pd.DatetimeIndex(index) - pd.Timedelta(minutes=30)).normalize()


def half_hour_slot(index: pd.DatetimeIndex) -> pd.Index:
    """Return official within-market-day slots numbered 0 through 47."""
    starts = pd.DatetimeIndex(index) - pd.Timedelta(minutes=30)
    return pd.Index(starts.hour * 2 + starts.minute // 30, name="slot")


def add_time_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with market-day and slot columns for analysis."""
    result = frame.copy()
    result["market_day"] = market_day(result.index)
    result["slot"] = half_hour_slot(result.index).to_numpy()
    return result


def exploratory_period(frame: pd.DataFrame) -> pd.DataFrame:
    """Select market days from 2010 through 2013 inclusive."""
    days = market_day(frame.index)
    return frame[(days.year >= 2010) & (days.year <= 2013)].copy()
