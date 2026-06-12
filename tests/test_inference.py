"""Tests for cluster-robust inference utilities.

The estimators target the mean of per-auction ACC scores when auctions
are sampled in clusters (markets, regions, time blocks).
"""

import numpy as np
import pytest

from acc_metrics import (
    cluster_robust_ci,
    cluster_robust_se,
    wild_cluster_bootstrap_ci,
)


class TestClusterRobustSE:
    def test_singleton_clusters_reduce_to_iid_se(self):
        """With one observation per cluster, CR SE equals s/sqrt(n) exactly."""
        rng = np.random.default_rng(7)
        x = rng.normal(size=40)
        clusters = np.arange(40)
        iid_se = x.std(ddof=1) / np.sqrt(len(x))
        assert cluster_robust_se(x, clusters) == pytest.approx(iid_se, rel=1e-12)

    def test_within_cluster_correlation_inflates_se(self):
        """Duplicating each draw within a cluster must raise the SE above naive."""
        rng = np.random.default_rng(11)
        base = rng.normal(size=30)
        x = np.repeat(base, 5)
        clusters = np.repeat(np.arange(30), 5)
        naive_se = x.std(ddof=1) / np.sqrt(len(x))
        assert cluster_robust_se(x, clusters) > 1.5 * naive_se

    def test_label_dtype_agnostic(self):
        """String labels and integer labels give the same answer."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert cluster_robust_se(x, np.array(["a", "a", "b", "b"])) == pytest.approx(
            cluster_robust_se(x, np.array([0, 0, 1, 1]))
        )

    def test_rejects_single_cluster_and_mismatch(self):
        with pytest.raises(ValueError, match="clusters"):
            cluster_robust_se([1.0, 2.0], [0, 0])
        with pytest.raises(ValueError, match="equal length"):
            cluster_robust_se([1.0, 2.0, 3.0], [0, 1])


class TestClusterRobustCI:
    def test_interval_brackets_mean_and_uses_t(self):
        rng = np.random.default_rng(3)
        x = rng.normal(loc=2.0, size=100)
        clusters = np.repeat(np.arange(10), 10)
        lo, hi = cluster_robust_ci(x, clusters)
        assert lo < x.mean() < hi
        # t_{9} critical value exceeds the normal one, so the interval is
        # wider than mean +/- 1.96 SE.
        se = cluster_robust_se(x, clusters)
        assert hi - lo > 2 * 1.96 * se


class TestWildClusterBootstrapCI:
    def test_reproducible_with_seed(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=60)
        clusters = np.repeat(np.arange(12), 5)
        ci1 = wild_cluster_bootstrap_ci(x, clusters, seed=123)
        ci2 = wild_cluster_bootstrap_ci(x, clusters, seed=123)
        assert ci1 == ci2

    def test_interval_brackets_mean(self):
        rng = np.random.default_rng(9)
        x = rng.normal(loc=1.0, size=80)
        clusters = np.repeat(np.arange(16), 5)
        lo, hi = wild_cluster_bootstrap_ci(x, clusters, seed=1)
        assert lo < x.mean() < hi

    def test_close_to_cr_interval_for_many_clusters(self):
        """With many clusters the wild interval approximates the CR-t one."""
        rng = np.random.default_rng(21)
        x = rng.normal(size=1000)
        clusters = np.repeat(np.arange(100), 10)
        wlo, whi = wild_cluster_bootstrap_ci(x, clusters, seed=2, n_bootstrap=4999)
        clo, chi = cluster_robust_ci(x, clusters)
        assert whi - wlo == pytest.approx(chi - clo, rel=0.15)
