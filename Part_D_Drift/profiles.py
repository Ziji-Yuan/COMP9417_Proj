"""Daily demand profiles with a fixed 2010--2013 PCA/clustering reference."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from Part_A.data import half_hour_slot, market_day


def validate_half_hours(index: pd.DatetimeIndex) -> None:
    """Require complete, ordered market days on the exact half-hour grid."""
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        raise ValueError("A nonempty DatetimeIndex is required")
    if index.tz is not None:
        raise ValueError("Use naive fixed AEST market timestamps, without DST conversion")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique and increasing")
    if not index.equals(index.floor("30min")):
        raise ValueError("Timestamps must lie on the exact half-hour grid")
    slots = pd.Series(half_hour_slot(index).to_numpy(), index=market_day(index))
    for _, values in slots.groupby(level=0):
        if len(values) != 48 or not np.array_equal(values.to_numpy(), np.arange(48)):
            raise ValueError("Each market day must have exactly slots 0--47")


def daily_profiles(observed: pd.DataFrame, column: str = "nsw_demand") -> pd.DataFrame:
    """Pivot A's observations (or a lag feature) into days x 48 slots.

    For observed demand this is retrospective: a day's full profile is available
    only at the end of that day. Never use its cluster to forecast that same day.
    """
    validate_half_hours(observed.index)
    values = observed[column].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Demand profiles must contain finite, strictly positive values")
    rows = pd.DataFrame({"market_day": market_day(observed.index),
                         "slot": half_hour_slot(observed.index), "value": values})
    result = rows.pivot(index="market_day", columns="slot", values="value")
    return result.loc[:, list(range(48))].astype(float)


def validate_profiles(profiles: pd.DataFrame) -> None:
    if not isinstance(profiles.index, pd.DatetimeIndex) or profiles.empty:
        raise ValueError("Profiles require a nonempty daily DatetimeIndex")
    if profiles.index.tz is not None or not profiles.index.equals(profiles.index.normalize()):
        raise ValueError("Profile dates must be naive market-day dates")
    if profiles.index.has_duplicates or not profiles.index.is_monotonic_increasing:
        raise ValueError("Profile dates must be unique and increasing")
    if list(profiles.columns) != list(range(48)):
        raise ValueError("Profiles must have ordered slot columns 0--47")
    values = profiles.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Demand profiles must be finite and strictly positive")


@dataclass(frozen=True)
class DayTypeConfig:
    normalize: bool = True
    variance_retained: float = 0.95
    candidate_k: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)
    random_state: int = 9417
    n_init: int = 20
    novelty_quantile: float = 0.99

    def __post_init__(self):
        if not 0 < self.variance_retained < 1:
            raise ValueError("variance_retained must be between 0 and 1")
        if not self.candidate_k or any(k < 2 or int(k) != k for k in self.candidate_k):
            raise ValueError("candidate_k must contain integer cluster counts >= 2")
        if not 0 < self.novelty_quantile < 1 or self.n_init < 1:
            raise ValueError("Invalid novelty quantile or n_init")


class DayTypeModel:
    """PCA followed by K-means; fit always selects 2010--2013 only.

    Select k by the highest training silhouette, preferring smaller k on ties.
    This is a reproducible starting rule, not proof of the true number of types.
    Exported inertia, cluster sizes and profiles support human interpretation.
    """

    def __init__(self, config: DayTypeConfig | None = None):
        self.config = config or DayTypeConfig()

    def _representation(self, profiles):
        values = profiles.to_numpy(dtype=float)
        return values / values.mean(axis=1, keepdims=True) if self.config.normalize else values

    def fit(self, profiles: pd.DataFrame):
        validate_profiles(profiles)
        train = profiles.loc["2010-01-01":"2013-12-31"]
        if len(train) < 4:
            raise ValueError("Need at least four training days within 2010--2013")
        values = self._representation(train)
        if np.var(values, axis=0).sum() < 1e-15:
            raise ValueError("Training profiles have no variation to cluster")
        self.pca_ = PCA(n_components=self.config.variance_retained, svd_solver="full")
        reduced = self.pca_.fit_transform(values)
        candidates, fitted = [], {}
        for k in sorted(set(self.config.candidate_k)):
            if k >= len(train) or k > len(np.unique(reduced, axis=0)):
                continue
            model = KMeans(n_clusters=k, n_init=self.config.n_init,
                           random_state=self.config.random_state).fit(reduced)
            labels = model.labels_
            if len(np.unique(labels)) != k:
                continue
            silhouette = silhouette_score(reduced, labels)
            candidates.append({"k": k, "silhouette": float(silhouette),
                               "inertia": float(model.inertia_),
                               "min_cluster_days": int(np.bincount(labels).min())})
            fitted[k] = model
        if not candidates:
            raise ValueError("No valid candidate clustering; use more varied training days")
        self.selection_ = pd.DataFrame(candidates).set_index("k")
        self.n_clusters_ = int(self.selection_.silhouette.idxmax())
        self.kmeans_ = fitted[self.n_clusters_]
        labels = self.kmeans_.labels_
        self.centres_ = np.stack([values[labels == k].mean(axis=0)
                                  for k in range(self.n_clusters_)])
        self.raw_centres_ = np.stack([train.to_numpy()[labels == k].mean(axis=0)
                                      for k in range(self.n_clusters_)])
        self.lower_profiles_ = np.stack([np.quantile(values[labels == k], 0.25, axis=0)
                                         for k in range(self.n_clusters_)])
        self.upper_profiles_ = np.stack([np.quantile(values[labels == k], 0.75, axis=0)
                                         for k in range(self.n_clusters_)])
        # Full 48-dimensional distance also catches changes outside retained PCs.
        distances = np.sqrt(np.mean((values[:, None, :] - self.centres_[None, :, :]) ** 2, axis=2))
        self.novelty_threshold_ = float(np.quantile(distances.min(axis=1), self.config.novelty_quantile))
        self.train_start_, self.train_end_ = train.index.min(), train.index.max()
        self.training_days_ = len(train)
        self.cluster_summary_ = pd.DataFrame([
            {"cluster": k, "training_days": int((labels == k).sum()),
             "daily_mean_mw": float(train.to_numpy()[labels == k].mean()),
             "weekend_share": float(np.mean(train.index[labels == k].day_of_week >= 5)),
             "summer_share": float(np.mean(np.isin(train.index[labels == k].month, [12, 1, 2]))),
             "winter_share": float(np.mean(np.isin(train.index[labels == k].month, [6, 7, 8]))),
             "peak_interval_start_hour": float(self.centres_[k].argmax() / 2)}
            for k in range(self.n_clusters_)
        ]).set_index("cluster")
        return self

    def assign(self, profiles: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "kmeans_"):
            raise ValueError("Fit the day-type model first")
        validate_profiles(profiles)
        values = self._representation(profiles)
        reduced = self.pca_.transform(values)
        labels = self.kmeans_.predict(reduced)
        distances = np.sqrt(np.mean((values[:, None, :] - self.centres_[None, :, :]) ** 2, axis=2))
        nearest = distances.min(axis=1)
        reconstructed = self.pca_.inverse_transform(reduced)
        return pd.DataFrame({
            "cluster": labels,
            "nearest_profile_distance": nearest,
            "pca_distance": self.kmeans_.transform(reduced).min(axis=1),
            "reconstruction_rmse": np.sqrt(np.mean((values - reconstructed) ** 2, axis=1)),
            "novel": nearest > self.novelty_threshold_,
            "daily_mean_mw": profiles.mean(axis=1).to_numpy(),
        }, index=profiles.index.rename("market_day"))


def cluster_shares(assignments: pd.DataFrame, n_clusters: int, frequency="year") -> pd.DataFrame:
    """Return all cluster proportions and coverage; 2026 is a partial year."""
    if frequency not in ("year", "month"):
        raise ValueError("frequency must be year or month")
    groups = assignments.index.year if frequency == "year" else assignments.index.to_period("M")
    counts = pd.crosstab(groups, assignments.cluster).reindex(columns=range(n_clusters), fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0)
    shares.columns = [f"cluster_{k}" for k in shares.columns]
    shares["n_days"] = counts.sum(axis=1)
    shares.index.name = frequency
    return shares
