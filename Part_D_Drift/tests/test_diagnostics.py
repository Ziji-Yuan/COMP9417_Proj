import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from threadpoolctl import threadpool_limits

from Part_D_Drift import daily_profiles
from Part_D_Drift.diagnostics import (clustering_audit, day_type_drift_context, detector_sensitivity,
                                     drift_episodes, event_comparisons, profile_evidence)
from Part_D_Drift.tests.helpers import day_type_model, observations


def example_signals():
    dates = pd.date_range("2015-01-01", periods=9, name="market_day")
    return pd.DataFrame({"score": [0, 2, 2, 2, 2, 0, 2, 2, 2], "threshold": 1.,
                         "exceeded": [False, True, True, True, True, False, True, True, True],
                         "alarm": [False, False, True, False, True, False, False, True, False],
                         "effective_day": dates + pd.Timedelta(days=1),
                         "distance__lag1_profile_distance": .5,
                         "distance__lag1_price_median": [0, 2, 2, 2, 2, 0, 2, 2, 2]}, index=dates)


class DiagnosticTests(unittest.TestCase):
    def test_clustering_robustness_ignores_future_and_preserves_primary(self):
        observed = observations()
        profiles, model = daily_profiles(observed), day_type_model(observed)
        components, centres = model.pca_.components_.copy(), model.centres_.copy()
        with threadpool_limits(limits=2):
            first = clustering_audit(profiles, model)
            changed = profiles.copy()
            changed.loc["2014":] *= np.linspace(1, 20, 48)
            second = clustering_audit(changed, model)
        for name in first:
            assert_frame_equal(first[name], second[name])
            self.assertTrue((first[name].training_end <= pd.Timestamp("2013-12-31")).all())
        np.testing.assert_array_equal(model.pca_.components_, components)
        np.testing.assert_array_equal(model.centres_, centres)

    def test_matched_months_and_missing_months_have_correct_denominators(self):
        dates = pd.DatetimeIndex(["2010-01-01", "2010-11-01", "2010-12-01", "2026-01-01", "2026-10-01"], name="market_day")
        profiles = pd.DataFrame(np.full((5, 48), 8000.), index=dates, columns=range(48))
        assignments = pd.DataFrame({"cluster": [0, 1, 1, 0, 1], "novel": False,
                                    "nearest_profile_distance": .01, "daily_mean_mw": 8000.}, index=dates)
        result = profile_evidence(profiles, assignments, 2)
        matched = result["yearly_shares_jan_oct"].set_index("year")
        self.assertEqual(matched.loc[2010, "n_days"], 1)
        self.assertEqual(matched.loc[2010, "cluster_0"], 1.)
        self.assertEqual(matched.loc[2026, "n_days"], 2)
        grid = result["monthly_shares_calendar_grid"].set_index("month")
        self.assertTrue(grid.loc[pd.Period("2026-11", freq="M")].isna().all())
        self.assertEqual(grid.loc[pd.Period("2026-10", freq="M"), "cluster_0"], 0.)
        composition = result["training_calendar_composition"]
        np.testing.assert_allclose(composition.groupby(["dimension", "category"]).type_share_within_category.sum(), 1.)
        np.testing.assert_allclose(composition.groupby(["dimension", "cluster"]).category_share_within_type.sum(), 1.)

    def test_detector_audit_uses_only_pre_2015_data(self):
        dates = pd.date_range("2010-01-01", "2015-01-15", name="market_day")
        rng = np.random.default_rng(9417)
        summaries = pd.DataFrame({"x": rng.normal(size=len(dates))}, index=dates)
        first = detector_sensitivity(summaries)
        changed = summaries.copy()
        changed.loc["2015":] = 1e9
        second = detector_sensitivity(changed)
        for name in first:
            assert_frame_equal(first[name], second[name])
        rows = first["detector_sensitivity_2014"]
        self.assertEqual(len(rows), 9)
        self.assertTrue((rows.calibration_end == pd.Timestamp("2014-06-30")).all())
        self.assertTrue((rows.validation_start == pd.Timestamp("2014-07-01")).all())
        self.assertTrue((rows.validation_days == 184).all())

    def test_episode_count_differs_from_repeated_alarm_count(self):
        result = drift_episodes(example_signals())
        episodes = result["input_drift_episodes"]
        alarms = result["input_alarm_attribution"]
        self.assertEqual(episodes.n_days.tolist(), [4, 3])
        self.assertEqual(episodes.alarms.tolist(), [2, 1])
        self.assertEqual(episodes.right_censored.tolist(), [False, True])
        self.assertEqual(alarms.alarm_kind.tolist(), ["first_alarm_in_episode", "repeat_within_episode", "first_alarm_in_episode"])
        self.assertEqual(alarms.dominant_feature.unique().tolist(), ["lag1_price_median"])

    def test_no_exceedances_produces_empty_tables_with_schema(self):
        signals = example_signals()
        signals[["alarm", "exceeded"]] = False
        result = drift_episodes(signals)
        self.assertEqual(len(result["input_drift_episodes"]), 0)
        self.assertIn("alarm_kind", result["input_alarm_attribution"])

    def test_context_joins_the_previous_profile_not_the_target_profile(self):
        signals = example_signals()
        days = pd.date_range("2014-12-31", periods=10, name="market_day")
        assignments = pd.DataFrame({"cluster": np.arange(10) % 2, "novel": False,
                                    "nearest_profile_distance": np.arange(10) / 100}, index=days)
        result = day_type_drift_context(assignments, signals, 2)
        daily = result["day_type_input_context_daily"]
        self.assertEqual(daily.iloc[0].lag1_cluster, 0)
        self.assertEqual(daily.iloc[0].profile_source_day, pd.Timestamp("2014-12-31"))
        self.assertEqual(result["day_type_input_context_monthly"].iloc[0].lag1_type_0_share, 5 / 9)
        with self.assertRaisesRegex(ValueError, "previous observed"):
            day_type_drift_context(assignments.iloc[1:], signals, 2)

    def test_event_windows_are_half_open_and_nearest_alarm_gap_is_signed(self):
        signals = example_signals()
        assignments = pd.DataFrame({"novel": False, "nearest_profile_distance": .01}, index=signals.index)
        events = [{"id": "example", "date": "2015-01-06", "title": "Example", "url": "https://example.test/event"}]
        result = event_comparisons(assignments, signals, events, window_days=2)
        row = result["market_event_comparisons"].iloc[0]
        self.assertEqual(row.nearest_alarm_minus_event_days, -1)
        self.assertEqual(row.event_pre_days, 2)
        self.assertEqual(row.event_post_days, 2)
        self.assertEqual(row.event_pre_alarms, 1)
        self.assertEqual(row.event_post_alarms, 0)
        self.assertTrue(pd.isna(row.previous_year_pre_score_mean))
        self.assertEqual(result["market_event_daily_windows"].offset_days.tolist(), [-2, -1, 0, 1])


if __name__ == "__main__":
    unittest.main()
