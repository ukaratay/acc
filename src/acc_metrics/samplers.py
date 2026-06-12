"""Auction value samplers for Monte Carlo simulations.

This module provides:
- Single-auction samplers for various distributions
- Vectorized batch sampler (~30x faster than loops)
- Copula-based affiliated value samplers

All samplers return values suitable for auction analysis.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import ndtr

if TYPE_CHECKING:
    from numpy.random import Generator


# =============================================================================
# Single-Auction Distribution Samplers
# =============================================================================


def sample_uniform(rng: Generator, n_bidders: int) -> np.ndarray:
    """Sample from Uniform[0,1] distribution.

    Args:
        rng: NumPy random generator.
        n_bidders: Number of bidders.

    Returns:
        Array of values in [0, 1].
    """
    return rng.uniform(0, 1, n_bidders)


def sample_exponential(rng: Generator, n_bidders: int) -> np.ndarray:
    """Sample from Exponential(1) distribution.

    Args:
        rng: NumPy random generator.
        n_bidders: Number of bidders.

    Returns:
        Array of positive values.
    """
    return rng.exponential(1.0, n_bidders)


def sample_lognormal(rng: Generator, n_bidders: int) -> np.ndarray:
    """Sample from LogNormal(0, 1) distribution.

    Args:
        rng: NumPy random generator.
        n_bidders: Number of bidders.

    Returns:
        Array of positive values.
    """
    return rng.lognormal(0, 1, n_bidders)


def sample_pareto(rng: Generator, n_bidders: int) -> np.ndarray:
    """Sample from Pareto(4, 1) distribution.

    Uses numpy's shifted Pareto: x = pareto(a) + 1 for x_min = 1.

    Args:
        rng: NumPy random generator.
        n_bidders: Number of bidders.

    Returns:
        Array of values >= 1.
    """
    return rng.pareto(4, n_bidders) + 1


def sample_beta(
    rng: Generator, n_bidders: int, a: float = 2.0, b: float = 2.0
) -> np.ndarray:
    """Sample from Beta(a, b) distribution.

    Args:
        rng: NumPy random generator.
        n_bidders: Number of bidders.
        a: Alpha parameter.
        b: Beta parameter.

    Returns:
        Array of values in [0, 1].
    """
    return rng.beta(a, b, n_bidders)


def sample_weibull(rng: Generator, n_bidders: int, k: float = 1.5) -> np.ndarray:
    """Sample from Weibull(k, 1) distribution.

    Args:
        rng: NumPy random generator.
        n_bidders: Number of bidders.
        k: Shape parameter.

    Returns:
        Array of positive values.
    """
    return rng.weibull(k, n_bidders)


def sample_gamma(
    rng: Generator, n_bidders: int, shape: float = 2.0, scale: float = 1.0
) -> np.ndarray:
    """Sample from Gamma(shape, scale) distribution.

    Args:
        rng: NumPy random generator.
        n_bidders: Number of bidders.
        shape: Shape parameter (k).
        scale: Scale parameter (theta).

    Returns:
        Array of positive values.
    """
    return rng.gamma(shape, scale, n_bidders)


# =============================================================================
# Vectorized Batch Sampler (~30x faster than loops)
# =============================================================================


def sample_auctions(
    rng: Generator, n_auctions: int, n_bidders: int, distribution: str = "uniform"
) -> np.ndarray:
    """Generate multiple auctions in one vectorized call.

    ~30x faster than calling single-auction samplers in a loop.

    Args:
        rng: NumPy random generator.
        n_auctions: Number of auctions to generate.
        n_bidders: Number of bidders per auction.
        distribution: Distribution name from DISTRIBUTIONS registry.

    Returns:
        Array of shape (n_auctions, n_bidders).

    Raises:
        ValueError: If distribution is unknown.
    """
    shape = (n_auctions, n_bidders)

    if distribution == "uniform":
        return rng.uniform(0, 1, shape)
    elif distribution == "exponential":
        return rng.exponential(1.0, shape)
    elif distribution == "lognormal":
        return rng.lognormal(0, 1, shape)
    elif distribution == "pareto":
        return rng.pareto(4, shape) + 1
    elif distribution == "beta_symmetric":
        return rng.beta(2.0, 2.0, shape)
    elif distribution == "beta_asymmetric":
        return rng.beta(2.0, 5.0, shape)
    elif distribution == "weibull":
        return rng.weibull(1.5, shape)
    elif distribution == "gamma":
        return rng.gamma(2.0, 1.0, shape)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


# =============================================================================
# Copula-Based Affiliated Value Samplers
# =============================================================================


def sample_gaussian_copula(
    rng: Generator, n_bidders: int, correlation: float
) -> np.ndarray:
    """Generate affiliated values via Gaussian copula.

    The Gaussian copula induces affiliation through correlated normal
    random variables transformed to uniforms via the normal CDF.

    Args:
        rng: NumPy random generator.
        n_bidders: Number of bidders.
        correlation: Correlation coefficient in [0, 1).

    Returns:
        Array of affiliated values in [0, 1].
    """
    cov_matrix = np.full((n_bidders, n_bidders), correlation)
    np.fill_diagonal(cov_matrix, 1.0)
    normals = rng.multivariate_normal(np.zeros(n_bidders), cov_matrix)
    return ndtr(normals)


def sample_clayton_copula(
    rng: Generator, n_bidders: int, kendall_tau: float
) -> np.ndarray:
    """Generate affiliated values via Clayton copula.

    Clayton copula exhibits lower tail dependence: extreme low values
    are more likely to occur together than extreme high values.

    Uses the Marshall-Olkin algorithm with gamma frailty.

    Args:
        rng: NumPy random generator.
        n_bidders: Number of bidders.
        kendall_tau: Kendall's tau correlation in [0, 1).

    Returns:
        Array of affiliated values in [0, 1].
    """
    if kendall_tau <= 0:
        return rng.uniform(0, 1, n_bidders)

    # Clayton parameter: theta = 2 * tau / (1 - tau)
    theta = 2.0 * kendall_tau / (1.0 - kendall_tau)

    # Marshall-Olkin algorithm:
    # 1. Generate V ~ Gamma(1/theta, 1)
    # 2. Generate E_i ~ Exp(1) independently
    # 3. U_i = (1 + E_i/V)^(-1/theta)
    v = rng.gamma(1.0 / theta, 1.0)
    e = rng.exponential(1.0, n_bidders)
    u = np.power(1.0 + e / v, -1.0 / theta)

    return u


def _sample_positive_stable(rng: Generator, alpha: float) -> float:
    """Sample from positive stable distribution S(alpha, 1, 0, 0).

    Uses the Chambers-Mallows-Stuck algorithm for totally skewed stable.

    Args:
        rng: NumPy random generator.
        alpha: Stability parameter in (0, 1].

    Returns:
        A positive stable random variate.
    """
    if alpha >= 1.0 - 1e-10:
        return 1.0

    # Chambers-Mallows-Stuck for positive (totally skewed) stable
    # Sample U uniform on (-pi/2, pi/2), W exponential(1)
    u = rng.uniform(-np.pi / 2 + 1e-10, np.pi / 2 - 1e-10)
    w = rng.exponential(1.0)

    # Positive stable formula (beta = 1 case)
    t = np.tan(np.pi * alpha / 2)
    b = np.arctan(t) / alpha
    s = np.power(1 + t * t, 1.0 / (2 * alpha))

    term1 = s * np.sin(alpha * (u + b)) / np.power(np.cos(u), 1.0 / alpha)
    term2 = np.power(np.cos(u - alpha * (u + b)) / w, (1.0 - alpha) / alpha)

    return np.maximum(term1 * term2, 1e-10)  # Ensure positive


def sample_gumbel_copula(
    rng: Generator, n_bidders: int, kendall_tau: float
) -> np.ndarray:
    """Generate affiliated values via Gumbel copula.

    Gumbel copula exhibits upper tail dependence: extreme high values
    are more likely to occur together than extreme low values.

    Uses the Marshall-Olkin algorithm with positive stable frailty.

    Args:
        rng: NumPy random generator.
        n_bidders: Number of bidders.
        kendall_tau: Kendall's tau correlation in [0, 1).

    Returns:
        Array of affiliated values in [0, 1].
    """
    if kendall_tau <= 0:
        return rng.uniform(0, 1, n_bidders)

    # Gumbel parameter: theta = 1 / (1 - tau), so theta >= 1
    theta = 1.0 / (1.0 - kendall_tau)
    alpha = 1.0 / theta  # alpha in (0, 1]

    # Marshall-Olkin algorithm:
    # 1. Generate S ~ PositiveStable(1/theta)
    # 2. Generate E_i ~ Exp(1) independently
    # 3. U_i = exp(-(E_i/S)^(1/theta))
    s = _sample_positive_stable(rng, alpha)
    e = rng.exponential(1.0, n_bidders)
    u = np.exp(-np.power(e / s, alpha))

    return u


def _debye1(theta: float) -> float:
    """Compute the first Debye function D_1(theta).

    D_1(theta) = (1/theta) * integral from 0 to theta of t/(e^t - 1) dt
    """
    if abs(theta) < 1e-10:
        return 1.0

    # Numerical integration using simple quadrature
    n_points = 100
    t = np.linspace(1e-10, abs(theta), n_points)
    integrand = t / (np.exp(t) - 1)
    integral = np.trapezoid(integrand, t)
    return integral / abs(theta)


def _frank_tau_to_theta(tau: float) -> float:
    """Convert Kendall's tau to Frank copula parameter theta.

    Uses bisection to solve: tau = 1 - 4/theta + 4*D_1(theta)/theta
    """
    if tau <= 0:
        return 0.0

    # Bisection search
    theta_low, theta_high = 0.01, 50.0

    for _ in range(50):  # Max iterations
        theta_mid = (theta_low + theta_high) / 2
        tau_mid = 1.0 - 4.0 / theta_mid + 4.0 * _debye1(theta_mid) / theta_mid

        if abs(tau_mid - tau) < 1e-6:
            return theta_mid
        elif tau_mid < tau:
            theta_low = theta_mid
        else:
            theta_high = theta_mid

    return theta_mid


def _sample_logarithmic(rng: Generator, p: float) -> int:
    """Sample from logarithmic series distribution.

    P(X = k) = -p^k / (k * log(1-p)) for k = 1, 2, 3, ...

    Uses the inverse CDF method.
    """
    if p <= 0 or p >= 1:
        return 1

    u = rng.uniform()
    log_1_minus_p = np.log(1 - p)

    # Inverse CDF via sequential search
    k = 1
    cumsum = -p / log_1_minus_p
    while u > cumsum and k < 1000:
        k += 1
        cumsum += -np.power(p, k) / (k * log_1_minus_p)

    return k


def sample_frank_copula(
    rng: Generator, n_bidders: int, kendall_tau: float
) -> np.ndarray:
    """Generate affiliated values via Frank copula.

    Frank copula exhibits symmetric tail dependence (none): unlike
    Clayton and Gumbel, extreme values are not preferentially
    correlated at either tail.

    Uses the logarithmic series distribution method.

    Args:
        rng: NumPy random generator.
        n_bidders: Number of bidders.
        kendall_tau: Kendall's tau correlation in [0, 1).

    Returns:
        Array of affiliated values in [0, 1].
    """
    if kendall_tau <= 0:
        return rng.uniform(0, 1, n_bidders)

    # Convert tau to Frank parameter theta
    theta = _frank_tau_to_theta(kendall_tau)

    if theta < 1e-6:
        return rng.uniform(0, 1, n_bidders)

    # Logarithmic series parameter
    p = 1.0 - np.exp(-theta)

    # Sample L from logarithmic series
    l_val = _sample_logarithmic(rng, p)

    # Generate E_i ~ Exp(1) independently
    e = rng.exponential(1.0, n_bidders)

    # Transform: U_i = -log(1 - (1 - e^(-theta)) * e^(-E_i/L)) / theta
    u = -np.log(1.0 - p * np.exp(-e / l_val)) / theta

    return u


# =============================================================================
# Registries
# =============================================================================

# Type alias for sampler functions
Sampler = Callable[["Generator", int], np.ndarray]
CopulaSampler = Callable[["Generator", int, float], np.ndarray]

# Distribution registry (single-auction samplers)
DISTRIBUTIONS: dict[str, Sampler] = {
    "uniform": sample_uniform,
    "exponential": sample_exponential,
    "lognormal": sample_lognormal,
    "pareto": sample_pareto,
    "beta_symmetric": lambda rng, n: sample_beta(rng, n, 2.0, 2.0),
    "beta_asymmetric": lambda rng, n: sample_beta(rng, n, 2.0, 5.0),
    "weibull": lambda rng, n: sample_weibull(rng, n, 1.5),
    "gamma": lambda rng, n: sample_gamma(rng, n, 2.0, 1.0),
}

# Copula registry (affiliated value samplers)
COPULAS: dict[str, CopulaSampler] = {
    "gaussian": sample_gaussian_copula,
    "clayton": sample_clayton_copula,
    "gumbel": sample_gumbel_copula,
    "frank": sample_frank_copula,
}
