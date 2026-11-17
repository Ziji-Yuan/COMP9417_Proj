import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from Part_D_Drift.analysis import prepare_predictions
from Part_D_Drift.evaluation import (
    add_baseline_context, alarm_performance_windows, associations_with_calendar_checks,
    evaluate_joint, market_event_performance, monthly_confidence_intervals,
    monthly_interval, within_month_novelty,
)
from Part_D_Drift.tests.test_diagnostics import example_signals


class IntervalTests(unittest.TestCase):
    def test_appendix_a_demand_formula_and_day_resampling(self):
        values = np.repeat([[0.], [10.]], 48, axis=1)
        result = monthly_interval(values, "demand", seed=8)
        expected_half = 1.96 * values.std(ddof=1) / np.sqrt(96)
        self.assertEqual(result["estimate"], 5.)
        self.assertAlmostEqual(result["iid_normal_lower"], 5 - expected_half)
        # Resampling entire constant days can only produce 0, 5 or 10.
        self.assertEqual(result["day_bootstrap_lower"], 0.)
        self.assertEqual(result["day_bootstrap_upper"], 10.)
        self.assertGreater(result["day_to_iid_width_ratio"], 4)
        self.assertEqual(result, monthly_interval(values, "demand", seed=8))

    def test_price_binomial_and_hoeffding_formulas(self):
        values = np.tile(np.r_[np.zeros(12), np.ones(36)], (30, 1))
        result = monthly_interval(values, "price")
        half = 1.96 * np.sqrt(.75 * .25 / 1440) * 100
        self.assertAlmostEqual(result["estimate"], 75.)
        self.assertAlmostEqual(result["iid_normal_upper"], 75 + half)
        self.assertAlmostEqual(result["hoeffding_lower"], 75 - np.sqrt(np.log(40) / 2880) * 100)
        self.assertEqual(result["day_bootstrap_width"], 0.)

    def test_minimum_replicates_and_complete_days(self):
        for values, task, n in ((np.ones((3, 47)), "price", 2000),
                                (np.ones((3, 48)), "price", 999),
                                (np.full((3, 48), .5), "price", 2000)):
            with self.assertRaises(ValueError):
                monthly_interval(values, task, n)

    def test_month_uses_interval_end_market_day_and_seed_is_order_independent(self):
        index = pd.date_range("2015-01-31 00:30", periods=96, freq="30min")
        frame = pd.DataFrame({"timestamp": index, "task": "price", "model": "test",
                              "y_true": np.r_[np.ones(48), np.zeros(48)], "y_pred": 1})
        prepared = prepare_predictions(frame)
        first = monthly_confidence_intervals(prepared)
        second = monthly_confidence_intervals(prepared.sample(frac=1, random_state=2))
        assert_frame_equal(first, second)
        self.assertEqual(first.estimate.tolist(), [100., 0.])
        self.assertEqual(first.n_half_hours.tolist(), [48, 48])


class JointComparisonTests(unittest.TestCase):
    def daily(self, periods=400):
        return pd.DataFrame({"task": "demand", "model": "ridge_regression", "strategy": "frozen",
                             "market_day": pd.date_range("2014-01-01", periods=periods),
                             "loss_pct": np.arange(periods, dtype=float),
                             "excess_loss_vs_yesterday_pp": 1.})

    def test_alarm_pre_includes_d_post_begins_next_day_and_repeats_are_separate(self):
        signals = example_signals()
        daily = self.daily()
        detail, summary = alarm_performance_windows(daily, signals, window_days=2)
        self.assertTrue(detail.complete_windows.all())
        np.testing.assert_allclose(detail.post_minus_pre_loss_pp, 2.)
        np.testing.assert_allclose(detail.post_minus_pre_excess_loss_pp, 0.)
        self.assertEqual(summary.set_index("alarm_kind").alarms.to_dict(),
                         {"first_alarm_in_episode": 2, "repeat_within_episode": 1})
        self.assertTrue((detail.effective_day == detail.alarm_day + pd.Timedelta(days=1)).all())

    def test_event_windows_start_on_event_and_use_previous_year(self):
        daily = self.daily()
        events = [{"id": "event", "date": "2015-01-10", "title": "Example", "url": "https://example.test"}]
        row = market_event_performance(daily, events, window_days=2).iloc[0]
        self.assertEqual(row.event_pre_days, 2)
        self.assertEqual(row.event_post_days, 2)
        self.assertEqual(row.event_post_minus_pre_loss_pp, 2.)
        self.assertEqual(row.previous_year_post_minus_pre_loss_pp, 2.)
        self.assertEqual(row.event_post_loss_pct - row.previous_year_post_loss_pct, 365.)

    def test_within_month_comparison_removes_pure_calendar_composition_gap(self):
        daily = pd.DataFrame({"task": "demand", "model": "test", "strategy": "frozen",
                              "market_day": list(pd.date_range("2015-01-01", periods=10)) +
                                            list(pd.date_range("2015-02-01", periods=10)),
                              "novel": [True] * 2 + [False] * 8 + [True] * 8 + [False] * 2,
                              "loss_pct": [1.] * 10 + [10.] * 10,
                              "excess_loss_vs_yesterday_pp": 0.})
        self.assertGreater(daily[daily.novel].loss_pct.mean() - daily[~daily.novel].loss_pct.mean(), 0)
        detail, summary = within_month_novelty(daily)
        self.assertTrue(detail.eligible.all())
        self.assertEqual(summary.iloc[0].eligible_months, 2)
        self.assertEqual(summary.iloc[0].equal_month_mean_loss_gap_pp, 0.)

    def test_calendar_check_does_not_treat_year_only_trend_as_residual_evidence(self):
        months = pd.period_range("2015-01", "2016-12", freq="M")
        values = np.repeat([1., 2.], 12)
        frame = pd.DataFrame({"task": "demand", "model": "test", "strategy": "frozen",
                              "month": months, "loss_pct": values,
                              **{k: values for k in ("novel_share", "profile_distance", "input_drift_score", "cluster_0_share")}})
        result = associations_with_calendar_checks(frame)
        self.assertTrue((result[result.method == "raw_spearman"].coefficient == 1).all())
        self.assertTrue(result[result.method == "year_and_calendar_month_adjusted_ranks"].coefficient.isna().all())
        # January -> March is not a one-month difference.
        result = associations_with_calendar_checks(frame.drop(index=1))
        self.assertEqual(result[result.method == "consecutive_month_first_differences"].n_observations.unique().tolist(), [21])

    def test_missing_baseline_is_rejected(self):
        frame = pd.DataFrame({"task": ["demand"], "model": ["test"], "month": ["2015-01"],
                              "loss_pct": [2.]})
        with self.assertRaisesRegex(ValueError, "coverage"):
            add_baseline_context(frame, "month")

    def test_complete_evaluation_preserves_inputs_and_pools_balanced_accuracy(self):
        signals = example_signals()
        days = signals.index
        index = pd.date_range("2015-01-01 00:30", periods=len(days) * 48, freq="30min")
        assignments = pd.DataFrame({"cluster": np.arange(len(days)) % 2, "novel": np.arange(len(days)) % 3 == 0,
                                    "nearest_profile_distance": np.arange(len(days)) / 100}, index=days)
        frames = []
        for task in ("demand", "price"):
            truth = np.full(len(index), 1000.) if task == "demand" else np.tile(np.r_[np.zeros(36), np.ones(12)], len(days))
            pred = truth * 1.1 if task == "demand" else np.zeros(len(index))
            for model in ("test", "yesterday_baseline", "last_week_baseline"):
                frames.append(pd.DataFrame({"timestamp": index, "task": task, "model": model,
                                            "strategy": "frozen" if model == "test" else "baseline",
                                            "y_true": truth, "y_pred": pred}))
        predictions = prepare_predictions(pd.concat(frames))
        for name in ("task", "model", "strategy"):
            predictions[name] = predictions[name].astype("category")
        saved = signals.copy(deep=True)
        tables = evaluate_joint(predictions, assignments, signals, [], n_boot=1000)
        assert_frame_equal(signals, saved)
        monthly = tables["monthly_performance"]
        self.assertTrue((monthly[monthly.task == "price"].BalancedAccuracy == 50.).all())
        self.assertTrue((monthly[monthly.task == "price"].Accuracy == 75.).all())
        self.assertTrue((monthly.excess_loss_vs_yesterday_pp.abs() < 1e-10).all())
        self.assertEqual(len(tables["monthly_confidence_intervals"]), 6)


if __name__ == "__main__":
    unittest.main()
