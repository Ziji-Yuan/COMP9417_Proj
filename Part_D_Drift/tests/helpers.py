import numpy as np
import pandas as pd

from Part_A.tests.helpers import synthetic_data
from Part_D_Drift import DayTypeConfig, DayTypeModel, daily_profiles


def observations(n_days=80, start="2013-12-01"):
    frame = synthetic_data(n_days=n_days, start=start)
    rng = np.random.default_rng(9417)
    day = np.arange(len(frame)) // 48
    slot = np.arange(len(frame)) % 48
    frame["nsw_demand"] = (7000 + 800 * np.sin(2 * np.pi * slot / 48)
                           + 600 * ((day % 3) == 0) * np.cos(4 * np.pi * slot / 48)
                           + rng.normal(0, 80, len(frame)))
    return frame


def day_type_model(frame=None):
    frame = observations() if frame is None else frame
    return DayTypeModel(DayTypeConfig(candidate_k=(2, 3), n_init=3)).fit(daily_profiles(frame))


def monitor_frames():
    rng = np.random.default_rng(42)
    index = pd.date_range("2013-10-01", "2014-03-31", name="market_day")
    frame = pd.DataFrame(rng.normal(size=(len(index), 2)), index=index, columns=["x", "z"])
    return frame.loc[:"2013-12-31"], frame.loc["2014-01-01":"2014-01-31"], frame.loc["2014-02-01":]
