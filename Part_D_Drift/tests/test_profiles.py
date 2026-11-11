import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from Part_D_Drift import DayTypeConfig, DayTypeModel, daily_profiles
from Part_D_Drift.profiles import cluster_shares
from Part_D_Drift.tests.helpers import day_type_model, observations


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.observed = observations()
        self.profiles = daily_profiles(self.observed)

    def test_midnight_is_previous_days_last_slot(self):
        expected = self.observed.loc[pd.Timestamp("2013-12-02"), "nsw_demand"]
        self.assertEqual(self.profiles.loc["2013-12-01", 47], expected)
        self.assertEqual(self.profiles.shape, (80, 48))

    def test_reject_incomplete_and_off_grid_days(self):
        with self.assertRaisesRegex(ValueError, "slots"):
            daily_profiles(self.observed.iloc[1:])
        changed = self.observed.copy()
        changed.index = changed.index + pd.Timedelta(minutes=1)
        with self.assertRaisesRegex(ValueError, "grid"):
            daily_profiles(changed)

    def test_future_changes_do_not_change_fitted_reference_or_selection(self):
        first = day_type_model(self.observed)
        changed = self.profiles.copy()
        changed.loc["2014-01-01":] *= np.linspace(1, 20, 48)
        second = DayTypeModel(first.config).fit(changed)
        np.testing.assert_array_equal(first.centres_, second.centres_)
        np.testing.assert_array_equal(first.pca_.components_, second.pca_.components_)
        assert_frame_equal(first.selection_, second.selection_)
        self.assertEqual(first.training_days_, 31)
        self.assertLessEqual(first.train_end_, pd.Timestamp("2013-12-31"))

    def test_normalization_removes_daily_scale_but_raw_does_not(self):
        normal = day_type_model(self.observed)
        raw = DayTypeModel(DayTypeConfig(normalize=False, candidate_k=(2,), n_init=3)).fit(self.profiles)
        evaluation = self.profiles.loc["2014-01-01":"2014-01-04"]
        a, b = normal.assign(evaluation), normal.assign(evaluation * 2)
        np.testing.assert_array_equal(a.cluster, b.cluster)
        np.testing.assert_allclose(a.nearest_profile_distance, b.nearest_profile_distance, atol=1e-12)
        self.assertGreater(raw.assign(evaluation * 2).nearest_profile_distance.mean(),
                           raw.assign(evaluation).nearest_profile_distance.mean())

    def test_assignment_does_not_update_fitted_state(self):
        model = day_type_model(self.observed)
        before = model.centres_.copy(), model.pca_.components_.copy()
        model.assign(self.profiles.loc["2014-01-01":])
        np.testing.assert_array_equal(before[0], model.centres_)
        np.testing.assert_array_equal(before[1], model.pca_.components_)

    def test_novel_profile_and_yearly_denominators(self):
        model = day_type_model(self.observed)
        changed = self.profiles.copy()
        changed.loc["2014-01-01", 0:20] *= 8
        assignments = model.assign(changed)
        self.assertTrue(assignments.loc["2014-01-01", "novel"])
        shares = cluster_shares(assignments, model.n_clusters_)
        np.testing.assert_allclose(shares.filter(like="cluster_").sum(axis=1), 1)
        self.assertEqual(shares.loc[2013, "n_days"], 31)
        self.assertEqual(shares.loc[2014, "n_days"], 49)


if __name__ == "__main__":
    unittest.main()
