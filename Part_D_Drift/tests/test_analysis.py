import unittest

import numpy as np
import pandas as pd

from Part_D_Drift.analysis import prepare_predictions, relate_predictions


class PerformanceTests(unittest.TestCase):
    def frame(self, task="demand"):
        index = pd.date_range("2015-01-01 00:30", periods=96, freq="30min")
        truth = np.full(96, 1000.0) if task == "demand" else np.r_[np.zeros(72), np.ones(24)]
        pred = truth * 1.1 if task == "demand" else np.zeros(96)
        return pd.DataFrame({"timestamp": index, "task": task, "model": "test_model",
                             "y_true": truth, "y_pred": pred})

    def context(self):
        days = pd.date_range("2015-01-01", periods=2, name="market_day")
        assignments = pd.DataFrame({"cluster": [0, 1], "novel": [False, True],
                                    "nearest_profile_distance": [0.1, 0.9]}, index=days)
        signals = pd.DataFrame({"score": [0.2, 0.8], "alarm": [False, True]}, index=days)
        return assignments, signals

    def test_known_metrics_and_midnight_grouping(self):
        demand = self.frame()
        price = self.frame("price")
        prepared = prepare_predictions(pd.concat([demand, price], ignore_index=True))
        assignments, signals = self.context()
        result = relate_predictions(prepared, assignments, signals)
        monthly = result["monthly_performance"].set_index("task")
        self.assertAlmostEqual(monthly.loc["demand", "MAPE"], 10)
        self.assertAlmostEqual(monthly.loc["demand", "MAE"], 100)
        self.assertAlmostEqual(monthly.loc["price", "Accuracy"], 75)
        # Monthly balanced accuracy must not be the mean of the daily values.
        self.assertAlmostEqual(monthly.loc["price", "BalancedAccuracy"], 50)
        self.assertTrue((result["daily_performance"].n_half_hours == 48).all())

    def test_type_shares_match_each_models_prediction_coverage(self):
        predictions = prepare_predictions(self.frame().iloc[:48])
        assignments, signals = self.context()
        result = relate_predictions(predictions, assignments, signals)
        row = result["monthly_performance"].iloc[0]
        self.assertEqual(row.cluster_0_share, 1)
        self.assertEqual(row.cluster_1_share, 0)
        self.assertEqual(row.novel_share, 0)
        self.assertIn("cluster_0_share", set(result["descriptive_associations"].feature))

    def test_incomplete_duplicate_and_probability_predictions_rejected(self):
        with self.assertRaisesRegex(ValueError, "slots"):
            prepare_predictions(self.frame().iloc[1:])
        with self.assertRaisesRegex(ValueError, "unique"):
            prepare_predictions(pd.concat([self.frame(), self.frame()]))
        price = self.frame("price")
        price["y_pred"] = 0.7
        with self.assertRaisesRegex(ValueError, "hard labels"):
            prepare_predictions(price)

    def test_external_truth_is_verified(self):
        frame = self.frame()
        truth = pd.DataFrame({"nsw_demand_mw": frame.y_true.to_numpy()}, index=pd.DatetimeIndex(frame.timestamp))
        prepare_predictions(frame, truth)
        frame.loc[0, "y_true"] += 1
        with self.assertRaisesRegex(ValueError, "disagrees"):
            prepare_predictions(frame, truth)


if __name__ == "__main__":
    unittest.main()
