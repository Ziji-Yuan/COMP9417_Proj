from __future__ import annotations

import unittest

from pandas.testing import assert_frame_equal

from Part_I.features import FeatureConfig, get_features, get_inference_features
from Part_I.tests.helpers import load_official_data, load_official_dayahead


class TrainingInferenceConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = [
            FeatureConfig(states=["nsw"], demand_lags=[1, 7], price_lags=[1], rolling_windows=[3], weather_columns=["syd_temp", "syd_solar"]),
            FeatureConfig(states=["nsw", "qld", "vic"], demand_lags=[1, 7], price_lags=[1, 7], rolling_windows=[7], include_cyclical=False),
        ]

    def setUp(self):
        self.da = load_official_dayahead()
        self.data = load_official_data()

    def test_training_equals_official_inference_inputs(self):
        for target_day in ("2011-01-01", "2012-02-29", "2014-01-01"):
            history, weather_next = self.da.make_inputs(self.data, target_day)
            for config in self.configs:
                with self.subTest(target_day=target_day, config=config):
                    X_train, _, _ = get_features(target_day, target_day, config, data=self.data)
                    X_next = get_inference_features(history, weather_next, config)
                    assert_frame_equal(X_train, X_next, check_exact=False, rtol=1e-12, atol=1e-12)

    def test_interface_shape_and_index(self):
        history, weather_next = self.da.make_inputs(self.data, "2013-12-31")
        X_next = get_inference_features(history, weather_next, self.configs[0])
        self.assertEqual(X_next.shape[0], 48)
        self.assertTrue(X_next.index.equals(weather_next.index))
        self.assertFalse(X_next.isna().any(axis=None))

    def test_targets_match_official_harness(self):
        for target_day in ("2011-01-01", "2012-02-29", "2014-01-01"):
            with self.subTest(target_day=target_day):
                _, y, _ = get_features(target_day, target_day, self.configs[0], data=self.data)
                official = self.da.targets(self.data, target_day)
                self.assertTrue(y.index.equals(official.index))
                assert_frame_equal(
                    y.astype({"nsw_price_up": int}),
                    official.astype({"nsw_demand_mw": float, "nsw_price_up": int}),
                    check_exact=True,
                )


if __name__ == "__main__":
    unittest.main()
