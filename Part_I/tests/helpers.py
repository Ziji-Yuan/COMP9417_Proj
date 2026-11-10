from __future__ import annotations

import importlib
import os
import sys
import unittest
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


PART_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PART_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def synthetic_data(n_days: int = 20, start: str = "2010-01-01") -> pd.DataFrame:
    index = pd.date_range(pd.Timestamp(start) + pd.Timedelta(minutes=30), periods=n_days * 48, freq="30min", name="timestamp")
    position = np.arange(len(index))
    day = position // 48
    slot = position % 48
    frame = pd.DataFrame(index=index)
    for state_no, state in enumerate(("nsw", "qld", "vic"), start=1):
        frame[f"{state}_demand"] = state_no * 1000 + day * 100 + slot
        frame[f"{state}_price"] = state_no * 100 + day * 10 + slot / 10
    for city_no, city in enumerate(("syd", "mel", "bne"), start=1):
        for variable_no, variable in enumerate(("temp", "rh", "cloud", "solar"), start=1):
            frame[f"{city}_{variable}"] = city_no * 100 + variable_no * 10 + day + slot / 100
    return frame.astype(float)


def _kit_root() -> Path | None:
    """Locate optional official files without personal machine assumptions."""
    configured = os.environ.get("COMP9417_KIT_ROOT")
    candidates = [PROJECT_ROOT]
    if configured:
        candidates.insert(0, Path(configured).expanduser())
    for candidate in candidates:
        if (candidate / "harness" / "dayahead.py").exists():
            return candidate
    return None


@lru_cache(maxsize=1)
def load_official_dayahead():
    root = _kit_root()
    if root is None:
        raise unittest.SkipTest(
            "official harness unavailable; set COMP9417_KIT_ROOT to the project-kit directory"
        )
    candidates = [root / "harness"]
    for candidate in candidates:
        if (candidate / "dayahead.py").exists():
            sys.path.insert(0, str(candidate))
            return importlib.import_module("dayahead")
    raise AssertionError("unreachable")


@lru_cache(maxsize=1)
def load_official_data():
    from Part_I.data import load_data

    configured = os.environ.get("COMP9417_DATA_PATH")
    root = _kit_root()
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    if root is not None:
        candidates.append(root / "data" / "nem_weather_halfhourly_2010_2026.csv")
    candidates.append(PROJECT_ROOT / "data" / "nem_weather_halfhourly_2010_2026.csv")
    for candidate in candidates:
        if candidate.exists():
            return load_data(candidate)
    raise unittest.SkipTest(
        "official dataset unavailable; set COMP9417_DATA_PATH or COMP9417_KIT_ROOT"
    )
