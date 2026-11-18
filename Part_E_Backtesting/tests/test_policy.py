import pickle
import unittest

import numpy as np
import pandas as pd

from Part_E_Backtesting.policy import (
    AdaptationConfig, BudgetedPolicy, ErrorConfig, ErrorDriftDetector, daily_loss,
)


def detector():
    losses = pd.Series(np.ones(60), index=pd.date_range("2014-11-02", "2014-12-31"))
    return ErrorDriftDetector("demand", ErrorConfig(window_days=3, persistence_days=2, cooldown_days=3)).fit(
        losses, predictor_training_end="2013-12-31")


class DetectorTests(unittest.TestCase):
    def test_loss_uses_complete_days_and_correct_units(self):
        self.assertAlmostEqual(daily_loss("demand", np.full(48, 100), np.full(48, 90)), 10)
        self.assertEqual(daily_loss("price", np.zeros(48), np.r_[np.ones(12), np.zeros(36)]), 25)
        for truth, pred in [(np.ones(47), np.ones(47)), (np.zeros(48), np.ones(48))]:
            with self.assertRaises(ValueError):
                daily_loss("demand", truth, pred)
        with self.assertRaises(ValueError):
            daily_loss("price", np.zeros(48), np.full(48, .5))

    def test_calibration_rejects_future_and_in_sample(self):
        errors = pd.Series(np.ones(365), index=pd.date_range("2014-01-01", "2014-12-31"))
        for end in ["2014-01-01", "2014-12-31", "2015-01-01"]:
            with self.assertRaises(ValueError):
                ErrorDriftDetector("price").fit(errors, predictor_training_end=end)
        with self.assertRaises(ValueError):
            ErrorDriftDetector("price").fit(errors, predictor_training_end="2013-12-31", deployment_start="2016-01-01")
        with self.assertRaises(ValueError):
            ErrorDriftDetector("price").fit(errors.iloc[::-1], predictor_training_end="2013-12-31")

    def test_alarm_persistence_and_next_day(self):
        monitor = detector()
        first = monitor.update("2015-01-01", 9)
        second = monitor.update("2015-01-02", 9)
        self.assertFalse(first["alarm"])
        self.assertTrue(second["alarm"])
        self.assertEqual(second["effective_day"], pd.Timestamp("2015-01-03"))
        self.assertEqual(monitor.threshold_, 1)
        self.assertFalse(monitor.update("2015-01-03", 9)["alarm"])
        self.assertFalse(monitor.update("2015-01-04", 9)["alarm"])
        self.assertTrue(monitor.update("2015-01-05", 9)["alarm"])

    def test_duplicate_gap_and_invalid_loss_do_not_advance(self):
        monitor = detector()
        for day, loss in [("2015-01-02", 1), ("2014-12-31", 1), ("2015-01-01", np.nan)]:
            with self.assertRaises(ValueError):
                monitor.update(day, loss)
            self.assertEqual(monitor.last_day_, pd.Timestamp("2014-12-31"))

    def test_restart_and_prefix_invariance(self):
        monitor = detector()
        initial = pickle.dumps(monitor)
        monitor.update("2015-01-01", 9)
        restored = pickle.loads(pickle.dumps(monitor))
        self.assertEqual(monitor.update("2015-01-02", 9), restored.update("2015-01-02", 9))
        replay = pickle.loads(initial)
        self.assertFalse(replay.update("2015-01-01", 9)["alarm"])
        replay.reset()
        self.assertEqual(replay.last_day_, pd.Timestamp("2014-12-31"))
        self.assertEqual(replay.threshold_, 1)


class BudgetTests(unittest.TestCase):
    def test_six_successful_replacements_and_exhaustion(self):
        policy = BudgetedPolicy(AdaptationConfig("input_triggered"))
        for i, day in enumerate(pd.date_range("2015-01-01", periods=8)):
            request = policy.observe(day, input_alarm=True)
            if i < 6:
                self.assertEqual(request.train_end, day)
                self.assertEqual(request.effective_day, day + pd.Timedelta(days=1))
                policy.commit(request, model_id=f"fit-{i}")
            else:
                self.assertIsNone(request)
        self.assertEqual(policy.count, 6)
        self.assertEqual([r["status"] for r in policy.audit][-2:], ["budget_exhausted"] * 2)

    def test_periodic_schedule_is_fixed_and_window_is_past_only(self):
        policy = BudgetedPolicy(AdaptationConfig("periodic"))
        effective = []
        for day in pd.date_range("2015-01-01", "2026-10-31"):
            request = policy.observe(day)
            if request:
                self.assertEqual(request.train_end, day)
                self.assertEqual(request.train_start, pd.Timestamp(f"{day.year - 2}-01-01"))
                effective.append(str(request.effective_day.date()))
                policy.commit(request, model_id=str(day.date()))
        self.assertEqual(effective, [f"{y}-01-01" for y in [2016, 2018, 2020, 2022, 2024, 2026]])

    def test_pending_fit_failure_and_double_commit(self):
        policy = BudgetedPolicy(AdaptationConfig("error_triggered"))
        request = policy.observe("2015-01-01", error_alarm=True)
        self.assertEqual(policy.count, 0)
        with self.assertRaises(ValueError):
            policy.observe("2015-01-02")
        policy.abort(request, reason="test failure")
        self.assertEqual(policy.count, 0)
        request = policy.observe("2015-01-02", error_alarm=True)
        policy.commit(request, model_id="new-model")
        with self.assertRaises(ValueError):
            policy.commit(request, model_id="duplicate")
        self.assertEqual(policy.count, 1)

    def test_independent_policies_and_serialization(self):
        a = BudgetedPolicy(AdaptationConfig("input_triggered"))
        b = BudgetedPolicy(AdaptationConfig("input_triggered"))
        request = a.observe("2015-01-01", input_alarm=True)
        restored = pickle.loads(pickle.dumps(a))
        restored.commit(restored.pending, model_id="restored")
        a.commit(request, model_id="original")
        self.assertEqual(a.count, restored.count)
        self.assertEqual(b.count, 0)

    def test_frozen_and_last_day_never_retrain(self):
        frozen = BudgetedPolicy(AdaptationConfig("frozen"))
        self.assertIsNone(frozen.observe("2015-01-01", input_alarm=True, error_alarm=True))
        last = BudgetedPolicy(AdaptationConfig("error_triggered", start_day="2026-10-31"))
        self.assertIsNone(last.observe("2026-10-31", error_alarm=True))
        self.assertEqual(last.audit[-1]["status"], "no_future_forecast")

    def test_invalid_config_and_dates(self):
        for budget in [-1, 7, 1.5, True]:
            with self.assertRaises(ValueError):
                AdaptationConfig("periodic", budget=budget)
        policy = BudgetedPolicy(AdaptationConfig("periodic"))
        for day in ["2015-01-02", "2015-01-01 00:30", "2015-01-01T00:00Z"]:
            with self.assertRaises(ValueError):
                policy.observe(day)


if __name__ == "__main__":
    unittest.main()
