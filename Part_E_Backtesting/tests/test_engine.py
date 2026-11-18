import unittest

import numpy as np
import pandas as pd

from Part_E_Backtesting.engine import simulate
from Part_E_Backtesting.policy import AdaptationConfig, ErrorConfig, ErrorDriftDetector


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


class EngineTests(unittest.TestCase):
    def data(self):
        index = pd.date_range("2015-01-01 00:30", periods=48 * 4, freq="30min")
        days = (index - pd.Timedelta(minutes=30)).normalize()
        X = pd.DataFrame({"past_feature": 1.0}, index=index)
        y = pd.Series(100.0, index=index)
        calibration = pd.Series(1.0, index=pd.date_range("2014-01-01", "2014-12-31"))
        monitor = ErrorDriftDetector("demand", ErrorConfig(window_days=1, persistence_days=1)).fit(
            calibration, predictor_training_end="2013-12-31")
        return X, y, days, monitor

    def test_real_engine_uses_replacement_only_next_day(self):
        X, y, days, monitor = self.data()
        fitted = []

        def fit(request):
            fitted.append(request)
            return ConstantModel(100)

        result = simulate(X, y, days, ConstantModel(50), monitor,
                          AdaptationConfig("error_triggered", end_day="2015-01-04"), fit_model=fit)
        np.testing.assert_array_equal(result["prediction"][:48], 50)
        np.testing.assert_array_equal(result["prediction"][48:], 100)
        self.assertEqual(fitted[0].train_end, pd.Timestamp("2015-01-01"))
        self.assertEqual(fitted[0].effective_day, pd.Timestamp("2015-01-02"))
        self.assertEqual(result["retrain_count"], 1)

    def test_future_targets_do_not_change_earlier_actions(self):
        X, y, days, monitor = self.data()
        config = AdaptationConfig("error_triggered", end_day="2015-01-04")
        first = simulate(X, y, days, ConstantModel(50), monitor, config,
                         fit_model=lambda _: ConstantModel(100))
        X, y, days, monitor = self.data()
        y.iloc[96:] = 200
        second = simulate(X, y, days, ConstantModel(50), monitor, config,
                          fit_model=lambda _: ConstantModel(100))
        np.testing.assert_array_equal(first["prediction"][:96], second["prediction"][:96])
        pd.testing.assert_frame_equal(first["signals"].iloc[:2], second["signals"].iloc[:2])

    def test_fit_failure_is_not_silently_tolerated(self):
        X, y, days, monitor = self.data()

        def fail(_):
            raise RuntimeError("training failed")

        with self.assertRaisesRegex(RuntimeError, "training failed"):
            simulate(X, y, days, ConstantModel(50), monitor,
                     AdaptationConfig("error_triggered", end_day="2015-01-04"), fit_model=fail)

    def test_invalid_input_effective_day_rejected(self):
        X, y, days, monitor = self.data()
        signals = pd.DataFrame({"alarm": True, "effective_day": pd.date_range("2015-01-01", periods=4)},
                               index=pd.date_range("2015-01-01", periods=4))
        with self.assertRaisesRegex(ValueError, "following day"):
            simulate(X, y, days, ConstantModel(50), monitor,
                     AdaptationConfig("input_triggered", end_day="2015-01-04"), input_signals=signals,
                     fit_model=lambda _: ConstantModel(100))
