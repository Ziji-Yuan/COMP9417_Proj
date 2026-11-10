from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from Part_I.data import half_hour_slot, market_day
from Part_I.features import FeatureConfig, get_data, get_features
from Part_I.tests.helpers import synthetic_data


class FeatureCorrectnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = synthetic_data()

    def test_market_day_and_midnight_alignment(self):
        index = pd.DatetimeIndex(["2010-01-01 23:30", "2010-01-02 00:00", "2010-01-02 00:30"])
        self.assertEqual(list(market_day(index)), [pd.Timestamp("2010-01-01"), pd.Timestamp("2010-01-01"), pd.Timestamp("2010-01-02")])
        self.assertEqual(list(half_hour_slot(index)), [46, 47, 0])

    def test_get_data_uses_inclusive_market_days(self):
        selected = get_data("2010-01-02", "2010-01-03", data=self.data)
        self.assertEqual(len(selected), 96)
        self.assertEqual(selected.market_day.min(), pd.Timestamp("2010-01-02"))
        self.assertEqual(selected.market_day.max(), pd.Timestamp("2010-01-03"))
        self.assertEqual(set(selected.slot), set(range(48)))

    def test_lag_rolling_reference_and_targets(self):
        config = FeatureConfig(
            states=["nsw"], demand_lags=[1, 7], price_lags=[1],
            rolling_windows=[3], include_calendar=False,
            include_weather=False, include_prev_day_price_mean=True,
        )
        X, y, meta = get_features("2010-01-08", "2010-01-08", config, data=self.data)
        self.assertEqual(len(X), 48)
        first = X.iloc[0]
        self.assertAlmostEqual(first["nsw_demand_lag_1d"], 1600.0)
        self.assertAlmostEqual(first["nsw_demand_lag_7d"], 1000.0)
        self.assertAlmostEqual(first["nsw_demand_roll_3d_mean"], np.mean([1600, 1500, 1400]))
        self.assertAlmostEqual(first["nsw_demand_roll_3d_std"], np.std([1600, 1500, 1400], ddof=1))
        self.assertAlmostEqual(first["nsw_price_prev_day_mean"], np.mean(100 + 6 * 10 + np.arange(48) / 10))
        self.assertEqual(y.iloc[0]["nsw_demand_mw"], 1700.0)
        self.assertEqual(y.iloc[0]["nsw_price_up"], 1)
        self.assertEqual(meta.iloc[0]["slot"], 0)

    def test_official_target_definition(self):
        config = FeatureConfig(states=["nsw"], demand_lags=[1], price_lags=[1], rolling_windows=[], include_weather=False)
        _, y, meta = get_features("2010-01-05", "2010-01-05", config, data=self.data)
        target_rows = self.data.loc[y.index]
        expected = (target_rows.nsw_price.to_numpy() > meta.price_reference_prev_day_mean.to_numpy()).astype(int)
        np.testing.assert_array_equal(y.nsw_price_up.astype(int).to_numpy(), expected)

    def test_deterministic_column_order(self):
        config_a = FeatureConfig(states=["vic", "nsw"], demand_lags=[7, 1, 7], price_lags=[7, 1], rolling_windows=[3])
        config_b = FeatureConfig(states=["nsw", "vic"], demand_lags=[1, 7], price_lags=[1, 7], rolling_windows=[3])
        Xa, _, _ = get_features("2010-01-08", "2010-01-08", config_a, data=self.data)
        Xb, _, _ = get_features("2010-01-08", "2010-01-08", config_b, data=self.data)
        self.assertEqual(Xa.columns.tolist(), Xb.columns.tolist())

    def test_insufficient_history_policy(self):
        drop = FeatureConfig(states=["nsw"], demand_lags=[7], price_lags=[], rolling_windows=[], include_weather=False, include_calendar=False, include_prev_day_price_mean=False)
        keep = FeatureConfig(states=["nsw"], demand_lags=[7], price_lags=[], rolling_windows=[], include_weather=False, include_calendar=False, include_prev_day_price_mean=False, drop_rows_with_missing_history=False)
        X_drop, _, _ = get_features("2010-01-02", "2010-01-02", drop, data=self.data)
        X_keep, y_keep, _ = get_features("2010-01-02", "2010-01-02", keep, data=self.data)
        self.assertTrue(X_drop.empty)
        self.assertEqual(len(X_keep), 48)
        self.assertTrue(X_keep.isna().all(axis=None))
        self.assertFalse(y_keep.isna().any(axis=None))

    def test_invalid_one_day_rolling_std_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "at least 2 days"):
            FeatureConfig(rolling_windows=[1], rolling_stats=["std"])


if __name__ == "__main__":
    unittest.main()
