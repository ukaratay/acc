"""Statistical inference utilities for ACC analysis.

This module provides:
- Bootstrap confidence intervals (percentile and BCa methods)
- Normal/asymptotic confidence intervals
- Monte Carlo standard errors for proportions

All bootstrap functions are numba-optimized for high performance.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange
from scipy import stats as scipy_stats
from scipy.special import ndtr, ndtri

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


# =============================================================================
# Numba-Optimized Bootstrap Internals
# =============================================================================


@njit(parallel=True, fastmath=True, cache=True)
def _bootstrap_means(data: np.ndarray, base_seed: int, n_bootstrap: int) -> np.ndarray:
    """Generate indices and compute bootstrap means in parallel.

    Uses per-iteration seeding for reproducibility with parallel execution.
    Generates indices on-the-fly to avoid allocating (n_bootstrap, n) array.

    Args:
        data: Original data array of shape (n,).
        base_seed: Base seed for reproducibility.
        n_bootstrap: Number of bootstrap samples.

    Returns:
        Array of bootstrap means, shape (n_bootstrap,).
    """
    n = len(data)
    means = np.empty(n_bootstrap)

    for i in prange(n_bootstrap):  # type: ignore[not-iterable]
        # Sequential seed (base + i) ensures reproducibility regardless of thread
        # scheduling order in prange. Each iteration is independent.
        np.random.seed(base_seed + i)  # noqa: NPY002
        total = 0.0
        for _ in range(n):
            idx = np.random.randint(0, n)  # noqa: NPY002
            total += data[idx]
        means[i] = total / n

    return means


@njit(fastmath=True, cache=True)
def _jackknife_means(data: np.ndarray) -> np.ndarray:
    """Compute jackknife (leave-one-out) means efficiently.

    Uses the identity: mean_{-i} = (sum - x_i) / (n - 1)

    This avoids O(n^2) recomputation and runs in O(n) time.

    Args:
        data: Original data array of shape (n,).

    Returns:
        Array of jackknife means, shape (n,).
    """
    n = len(data)
    total = 0.0
    for i in range(n):
        total += data[i]

    means = np.empty(n)
    for i in range(n):
        means[i] = (total - data[i]) / (n - 1)

    return means


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================


def bootstrap_percentile_ci(
    data: ArrayLike,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval.

    The percentile method uses the empirical quantiles of the bootstrap
    distribution directly. It is simple but can have poor coverage for
    skewed distributions.

    Args:
        data: Array of observations.
        statistic: Statistic to compute (default: mean). For mean, uses
            numba-optimized path; other statistics use Python loop.
        confidence: Confidence level (default: 0.95).
        n_bootstrap: Number of bootstrap samples (default: 1000).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (lower, upper) bounds.
    """
    arr = np.asarray(data, dtype=np.float64)

    rng = np.random.default_rng(42 if seed is None else seed)

    # Use numba-optimized path for mean statistic
    if statistic is np.mean or (
        hasattr(statistic, "__name__") and statistic.__name__ == "mean"
    ):
        base_seed = int(rng.integers(0, 2**31))
        bootstrap_stats = _bootstrap_means(arr, base_seed, n_bootstrap)
    else:
        # General path for arbitrary statistics
        n = len(arr)
        bootstrap_stats = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            indices = rng.integers(0, n, size=n)
            bootstrap_stats[i] = statistic(arr[indices])

    alpha = 1 - confidence
    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))
    return lower, upper


def bootstrap_bca_ci(
    data: ArrayLike,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Bias-corrected and accelerated (BCa) bootstrap confidence interval.

    The BCa method adjusts for bias and skewness in the bootstrap distribution,
    providing better coverage than the percentile method. It requires computing
    the jackknife values for the acceleration factor.

    References:
        Efron, B. (1987). "Better Bootstrap Confidence Intervals".
        Journal of the American Statistical Association.

    Args:
        data: Array of observations.
        statistic: Statistic to compute (default: mean). For mean, uses
            numba-optimized path for both bootstrap and jackknife.
        confidence: Confidence level (default: 0.95).
        n_bootstrap: Number of bootstrap samples (default: 1000).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (lower, upper) bounds.
    """
    arr = np.asarray(data, dtype=np.float64)

    if len(arr) < 2:
        val = float(statistic(arr)) if len(arr) == 1 else 0.0
        return val, val

    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")

    rng = np.random.default_rng(42 if seed is None else seed)

    theta_hat = float(statistic(arr))

    # Generate bootstrap distribution
    is_mean = statistic is np.mean or (
        hasattr(statistic, "__name__") and statistic.__name__ == "mean"
    )

    if is_mean:
        base_seed = int(rng.integers(0, 2**31))
        bootstrap_thetas = _bootstrap_means(arr, base_seed, n_bootstrap)
        jackknife_thetas = _jackknife_means(arr)
    else:
        n = len(arr)
        bootstrap_thetas = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            indices = rng.integers(0, n, size=n)
            bootstrap_thetas[i] = statistic(arr[indices])

        # Jackknife for acceleration
        jackknife_thetas = np.empty(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            jackknife_thetas[i] = statistic(arr[mask])

    # Bias correction factor (z0), clipped to avoid ndtri(0) or ndtri(1) = ±∞
    prop = np.mean(bootstrap_thetas < theta_hat)
    prop = np.clip(prop, 1 / (2 * n_bootstrap), 1 - 1 / (2 * n_bootstrap))
    z0 = ndtri(prop)

    # Acceleration factor (a)
    theta_dot = np.mean(jackknife_thetas)
    diff = theta_dot - jackknife_thetas
    num = np.sum(diff**3)
    denom = 6 * (np.sum(diff**2) ** 1.5)
    a = num / denom if denom != 0 else 0

    # BCa adjusted quantiles
    alpha = 1 - confidence
    z_alpha_lower = ndtri(alpha / 2)
    z_alpha_upper = ndtri(1 - alpha / 2)

    alpha1 = ndtr(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
    alpha2 = ndtr(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))

    lower = float(np.percentile(bootstrap_thetas, 100 * alpha1))
    upper = float(np.percentile(bootstrap_thetas, 100 * alpha2))

    return lower, upper


# =============================================================================
# Normal (Asymptotic) Confidence Intervals
# =============================================================================


def normal_ci(data: ArrayLike, confidence: float = 0.95) -> tuple[float, float, float]:
    """Compute normal (asymptotic) confidence interval for mean.

    Uses the Central Limit Theorem approximation:
        mean +/- z * (std / sqrt(n))

    Args:
        data: Array of observations.
        confidence: Confidence level (default: 0.95).

    Returns:
        Tuple of (mean, lower, upper).
    """
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    z = ndtri(1 - (1 - confidence) / 2)
    return mean, mean - z * se, mean + z * se


# =============================================================================
# Monte Carlo Standard Errors
# =============================================================================


def monte_carlo_se(p: float, n_reps: int) -> float:
    """Compute Monte Carlo standard error for a proportion.

    When estimating a proportion p from n_reps Monte Carlo replications,
    the standard error is sqrt(p(1-p)/n).

    Args:
        p: Estimated proportion (e.g., coverage rate).
        n_reps: Number of Monte Carlo replications.

    Returns:
        Standard error of the proportion estimate.
    """
    return float(np.sqrt(p * (1 - p) / n_reps))


def monte_carlo_ci(
    p: float, n_reps: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Compute Monte Carlo confidence interval for a proportion.

    Args:
        p: Estimated proportion.
        n_reps: Number of replications.
        confidence: Confidence level (default: 0.95).

    Returns:
        Tuple of (lower, upper) bounds.
    """
    se = monte_carlo_se(p, n_reps)
    z = ndtri(1 - (1 - confidence) / 2)
    return float(p - z * se), float(p + z * se)


# =============================================================================
# Cluster-Robust Inference
# =============================================================================


def _cluster_residual_sums(
    data: ArrayLike, clusters: ArrayLike
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (data array, within-cluster residual sums, n) for the mean."""
    x = np.asarray(data, dtype=np.float64)
    c = np.asarray(clusters)
    if x.ndim != 1 or c.ndim != 1 or len(x) != len(c):
        raise ValueError("data and clusters must be 1D arrays of equal length")
    if len(x) < 2:
        raise ValueError("need at least 2 observations")
    _, inverse = np.unique(c, return_inverse=True)
    resid = x - x.mean()
    sums = np.bincount(inverse, weights=resid)
    if len(sums) < 2:
        raise ValueError("need at least 2 clusters")
    return x, sums, len(x)


def cluster_robust_se(data: ArrayLike, clusters: ArrayLike) -> float:
    """Cluster-robust standard error of the sample mean.

    Treats observations within a cluster as arbitrarily dependent and
    clusters as independent (Liang-Zeger sandwich for the mean), with the
    small-sample factor M/(M-1). With singleton clusters it reduces exactly
    to the i.i.d. standard error s/sqrt(n) with s the ddof-1 deviation.

    Args:
        data: Array of observations (e.g., per-auction ACC values).
        clusters: Cluster labels, same length as data.

    Returns:
        Standard error of the mean under cluster sampling.
    """
    _, sums, n = _cluster_residual_sums(data, clusters)
    m = len(sums)
    var = (m / (m - 1)) * float(np.sum(sums**2)) / n**2
    return float(np.sqrt(var))


def cluster_robust_ci(
    data: ArrayLike, clusters: ArrayLike, confidence: float = 0.95
) -> tuple[float, float]:
    """Cluster-robust confidence interval for the mean.

    Uses the cluster-robust standard error with a t critical value on
    M - 1 degrees of freedom, the standard small-M recommendation.

    Args:
        data: Array of observations.
        clusters: Cluster labels, same length as data.
        confidence: Confidence level (default: 0.95).

    Returns:
        Tuple of (lower, upper) bounds.
    """
    x, sums, _ = _cluster_residual_sums(data, clusters)
    m = len(sums)
    se = cluster_robust_se(data, clusters)
    tcrit = float(scipy_stats.t.ppf(1 - (1 - confidence) / 2, df=m - 1))
    mean = float(x.mean())
    return mean - tcrit * se, mean + tcrit * se


def wild_cluster_bootstrap_ci(
    data: ArrayLike,
    clusters: ArrayLike,
    confidence: float = 0.95,
    n_bootstrap: int = 999,
    seed: int | None = None,
) -> tuple[float, float]:
    """Wild cluster bootstrap percentile interval for the mean.

    Perturbs within-cluster residual sums with Rademacher signs:
    mean*_b = mean + (1/n) sum_m w_{mb} S_m. Recommended when the number
    of clusters is small (Cameron, Gelbach, and Miller 2008).

    Args:
        data: Array of observations.
        clusters: Cluster labels, same length as data.
        confidence: Confidence level (default: 0.95).
        n_bootstrap: Number of bootstrap draws (default: 999).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (lower, upper) bounds.
    """
    x, sums, n = _cluster_residual_sums(data, clusters)
    rng = np.random.default_rng(42 if seed is None else seed)
    weights = rng.choice(np.array([-1.0, 1.0]), size=(n_bootstrap, len(sums)))
    boot_means = float(x.mean()) + (weights @ sums) / n
    alpha = 1 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper
