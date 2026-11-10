from __future__ import annotations

import unittest

import numpy as np
from pandas.testing import assert_frame_equal

from Part_I.data import market_day
from Part_I.features import FeatureConfig, get_features
from Part_I.tests.helpers import synthetic_data


class LeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = synthetic_data(n_days=62, start="2013-05-01")
        cls.target_day = "2013-06-15"
        cls.config = FeatureConfig(states=["nsw", "qld", "vic"], demand_lags=[1, 7], price_lags=[1, 7], rolling_windows=[7])

    def feature_matrix(self, data):
        return get_features(self.target_day, self.target_day, self.config, data=data)[0]

    def test_target_day_market_values_do_not_change_features(self):
        baseline = self.feature_matrix(self.data)
        changed = self.data.copy()
        mask = market_day(changed.index) == self.target_day
        changed.loc[mask, ["nsw_demand", "nsw_price", "qld_demand", "qld_price", "vic_demand", "vic_price"]] += 1_000_000
        assert_frame_equal(baseline, self.feature_matrix(changed))

    def test_future_market_values_do_not_change_features(self):
        baseline = self.feature_matrix(self.data)
        changed = self.data.copy()
        mask = market_day(changed.index) > self.target_day
        changed.loc[mask, ["nsw_demand", "nsw_price", "qld_demand", "qld_price", "vic_demand", "vic_price"]] *= -99
        assert_frame_equal(baseline, self.feature_matrix(changed))

    def test_relevant_historical_value_changes_expected_lag(self):
        baseline = self.feature_matrix(self.data)
        changed = self.data.copy()
        target_timestamp = baseline.index[10]
        source_timestamp = target_timestamp - np.timedelta64(1, "D")
        changed.loc[source_timestamp, "nsw_demand"] += 123.0
        revised = self.feature_matrix(changed)
        self.assertAlmostEqual(revised.loc[target_timestamp, "nsw_demand_lag_1d"] - baseline.loc[target_timestamp, "nsw_demand_lag_1d"], 123.0)
        unaffected = baseline.index.difference([target_timestamp])
        np.testing.assert_allclose(revised.loc[unaffected, "nsw_demand_lag_1d"], baseline.loc[unaffected, "nsw_demand_lag_1d"])

    def test_target_weather_is_permitted(self):
        baseline = self.feature_matrix(self.data)
        changed = self.data.copy()
        mask = market_day(changed.index) == self.target_day
        changed.loc[mask, "syd_temp"] += 5
        revised = self.feature_matrix(changed)
        np.testing.assert_allclose(revised["syd_temp"] - baseline["syd_temp"], 5)
        other = baseline.columns.difference(["syd_temp"])
        assert_frame_equal(baseline[other], revised[other])

    def test_rolling_excludes_target_day(self):
        baseline = self.feature_matrix(self.data)
        changed = self.data.copy()
        mask = market_day(changed.index) == self.target_day
        changed.loc[mask, "nsw_demand"] = -1e12
        revised = self.feature_matrix(changed)
        assert_frame_equal(
            baseline.filter(regex="nsw_demand_roll"),
            revised.filter(regex="nsw_demand_roll"),
        )


if __name__ == "__main__":
    unittest.main()
