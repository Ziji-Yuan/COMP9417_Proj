import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from Part_A.data import MARKET_COLUMNS, WEATHER_COLUMNS, market_day
from Part_A.features import get_features, get_inference_features
from Part_D_Drift import DriftConfig, InputDriftDetector, input_summaries, monitor_config
from Part_D_Drift.tests.helpers import day_type_model, monitor_frames, observations


class InputContractTests(unittest.TestCase):
    def setUp(self):
        self.data = observations()
        self.model = day_type_model(self.data)
        self.target = pd.Timestamp("2014-01-15")

    def summarize(self, data):
        X, _, _ = get_features(self.target, self.target, monitor_config(), data=data)
        return input_summaries(X, self.model)

    def test_target_and_future_market_changes_do_not_change_inputs(self):
        original = self.summarize(self.data)
        changed = self.data.copy()
        changed.loc[market_day(changed.index) >= self.target, MARKET_COLUMNS] *= 9
        assert_frame_equal(original, self.summarize(changed))

    def test_inference_matches_offline_summary_and_lag_profile(self):
        history = self.data.loc[market_day(self.data.index) < self.target]
        weather = self.data.loc[market_day(self.data.index) == self.target, WEATHER_COLUMNS]
        inference = get_inference_features(history, weather, monitor_config())
        summary = input_summaries(inference, self.model)
        assert_frame_equal(self.summarize(self.data), summary)
        expected = history.iloc[-48:].nsw_demand.mean()
        self.assertAlmostEqual(summary.iloc[0].lag1_demand_mean_mw, expected)

    def test_target_weather_is_allowed_and_changes_summary(self):
        changed = self.data.copy()
        changed.loc[market_day(changed.index) == self.target, "syd_temp"] += 5
        a, b = self.summarize(self.data), self.summarize(changed)
        self.assertAlmostEqual(b.iloc[0].syd_temp_mean - a.iloc[0].syd_temp_mean, 5)


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.reference, self.calibration, self.live = monitor_frames()
        self.config = DriftConfig(window_days=7, persistence_days=3, cooldown_days=10)
        self.detector = InputDriftDetector(self.config).fit(self.reference, self.calibration)

    def test_future_perturbation_preserves_prefix_alarms(self):
        full = self.detector.replay(self.live)
        changed = self.live.copy()
        changed.loc["2014-03-01":] += 10000
        second = self.detector.replay(changed)
        assert_frame_equal(full.loc[:"2014-02-28"], second.loc[:"2014-02-28"])

    def test_streaming_equals_replay_and_serialization(self):
        expected = self.detector.replay(self.live)
        self.detector.reset()
        split = 15
        records = [self.detector.update(day, row).to_record() for day, row in self.live.iloc[:split].iterrows()]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.joblib"
            joblib.dump(self.detector, path)
            restored = joblib.load(path)
        records += [restored.update(day, row).to_record() for day, row in self.live.iloc[split:].iterrows()]
        assert_frame_equal(expected, pd.DataFrame(records).set_index("market_day"))

    def test_shifts_alarm_after_persistence_with_cooldown_and_next_day_effect(self):
        shifted = self.live + 10000
        results = self.detector.replay(shifted)
        alarms = results.loc[results.alarm]
        self.assertGreater(len(alarms), 1)
        self.assertGreaterEqual((alarms.index[0] - shifted.index[0]).days, 2)
        self.assertTrue(((alarms.index[1:] - alarms.index[:-1]) >= pd.Timedelta(days=10)).all())
        np.testing.assert_array_equal(results.effective_day.to_numpy(),
                                      (results.index + pd.Timedelta(days=1)).to_numpy())

    def test_reject_overlapping_calibration_and_nonchronological_updates(self):
        with self.assertRaisesRegex(ValueError, "immediately follow"):
            InputDriftDetector(self.config).fit(self.reference, self.reference.iloc[-20:])
        with self.assertRaisesRegex(ValueError, "consecutive"):
            self.detector.update(self.live.index[1], self.live.iloc[1])
        self.detector.update(self.live.index[0], self.live.iloc[0])
        with self.assertRaisesRegex(ValueError, "consecutive"):
            self.detector.update(self.live.index[0], self.live.iloc[0])

    def test_update_preserves_threshold_and_reference(self):
        threshold = self.detector.threshold_
        reference = self.detector.reference_residuals_.copy()
        self.detector.replay(self.live + 10000)
        self.assertEqual(threshold, self.detector.threshold_)
        np.testing.assert_array_equal(reference, self.detector.reference_residuals_)

    def test_reject_missing_days_and_nonfinite_features(self):
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.detector.replay(self.live.drop(self.live.index[2]))
        row = self.live.iloc[0].copy()
        row.iloc[0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            self.detector.update(self.live.index[0], row)


if __name__ == "__main__":
    unittest.main()
