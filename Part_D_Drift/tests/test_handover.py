import unittest

import numpy as np
import pandas as pd

from Part_D_Drift.handover import baseline_predictions, fit_b_ridge


class HandoverTests(unittest.TestCase):
    def test_ridge_ignores_future_rows_including_midnight_boundary(self):
        index = pd.date_range("2014-12-31 00:30", periods=96, freq="30min")
        X = pd.DataFrame({"x": np.arange(96, dtype=float), "weather": np.sin(np.arange(96))}, index=index)
        y = pd.Series(1000 + 3 * X.x, index=index)
        first = fit_b_ridge(X, y)
        changed_X, changed_y = X.copy(), y.copy()
        changed_X.iloc[48:] = 1e8
        changed_y.iloc[48:] = 1e9
        second = fit_b_ridge(changed_X, changed_y)
        np.testing.assert_array_equal(first.predict(X.iloc[:48]), second.predict(X.iloc[:48]))
        self.assertEqual(first.named_steps["standardscaler"].n_samples_seen_, 48)
        self.assertEqual(first.named_steps["standardscaler"].mean_[0], 23.5)
        self.assertEqual(first.named_steps["ridge"].alpha, 1000.)

    def test_last_week_price_compares_with_yesterdays_threshold(self):
        index = pd.date_range("2015-01-01 00:30", periods=48, freq="30min")
        X = pd.DataFrame({"nsw_demand_lag_1d": 7000., "nsw_demand_lag_7d": 8000.,
                          "nsw_price_lag_1d": np.r_[np.zeros(24), np.full(24, 20.)],
                          "nsw_price_lag_7d": 15.}, index=index)
        truth = pd.DataFrame({"nsw_demand_mw": 7500., "nsw_price_up": 1}, index=index)
        meta = pd.DataFrame({"price_reference_prev_day_mean": 10.}, index=index)
        predictions = baseline_predictions(X, truth, meta)
        weekly_price = predictions[(predictions.task == "price") & (predictions.model == "last_week_baseline")]
        self.assertTrue((weekly_price.y_pred == 1).all())
        yesterday_price = predictions[(predictions.task == "price") & (predictions.model == "yesterday_baseline")]
        self.assertEqual(yesterday_price.y_pred.sum(), 24)
        self.assertEqual(len(predictions), 192)


if __name__ == "__main__":
    unittest.main()
