"""First-price auction equilibrium bid functions.

This module provides equilibrium bid calculations for first-price sealed-bid
auctions under various value distributions.

For uniform F[0,1], the equilibrium bid has a closed form: b(v) = ((n-1)/n) * v.
For other distributions, numerical integration is required.

All functions are numba-optimized with scalar wrappers for convenience.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


# =============================================================================
# Gauss-Legendre Quadrature (Precomputed)
# =============================================================================

# 32-point Gauss-Legendre nodes and weights for numerical integration
_GL_NODES_32, _GL_WEIGHTS_32 = np.polynomial.legendre.leggauss(32)


# =============================================================================
# Uniform Distribution (Closed Form)
# =============================================================================


@njit(parallel=True, fastmath=True, cache=True)
def _first_price_bids_uniform_numba(values: np.ndarray, scale: float) -> np.ndarray:
    """Numba-compiled uniform equilibrium bids."""
    m = len(values)
    bids = np.empty(m)
    for i in prange(m):  # type: ignore[not-iterable]
        bids[i] = scale * values[i]
    return bids


def first_price_bids_uniform(values: ArrayLike, n: int) -> np.ndarray:
    """Equilibrium bids in first-price auction with uniform[0,1] values.

    In the symmetric first-price auction with i.i.d. uniform[0,1] values,
    the unique symmetric equilibrium strategy is:

        b(v) = ((n-1)/n) * v

    This follows from the first-order condition of the bidder's optimization
    problem. The bid shades the value by a factor that increases with n.

    Supports both 1D (single auction) and 2D (batch of auctions) input.

    Args:
        values: 1D array (single auction) or 2D array (n_auctions, n_bidders).
        n: Number of bidders.

    Returns:
        Array of equilibrium bids (same shape as input).
    """
    if n < 2:
        raise ValueError(f"Need n >= 2 bidders, got {n}")
    arr = np.asarray(values, dtype=np.float64)
    scale = (n - 1) / n

    if arr.ndim == 1:
        return _first_price_bids_uniform_numba(arr, scale)

    if arr.ndim == 2:
        # Flatten, process, reshape
        flat = arr.ravel()
        bids_flat = _first_price_bids_uniform_numba(flat, scale)
        return bids_flat.reshape(arr.shape)

    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")


# =============================================================================
# Log-Normal Distribution (Numerical Integration)
# =============================================================================


@njit(parallel=True, fastmath=True, cache=True)
def _first_price_bids_lognormal_numba(
    values: np.ndarray, n: int, sigma: float, nodes: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Numba-compiled first-price bid computation for log-normal F.

    Uses Gauss-Legendre quadrature with parallel execution.
    ~60x faster than scipy.integrate.quad in a loop.

    The equilibrium bid for value v is:
        b(v) = v - integral_0^v [F(t)/F(v)]^(n-1) dt

    Args:
        values: Array of bidder values.
        n: Number of bidders.
        sigma: Log-normal shape parameter.
        nodes: Gauss-Legendre nodes.
        weights: Gauss-Legendre weights.

    Returns:
        Array of equilibrium bids.
    """
    m = len(values)
    bids = np.empty(m)

    for i in prange(m):  # type: ignore[not-iterable]
        v = values[i]
        if v <= 0:
            bids[i] = 0.0
            continue

        log_v = np.log(v)

        # Abramowitz & Stegun normal CDF approximation (inline for numba)
        def norm_cdf(x: float) -> float:
            if x >= 0:  # noqa: SIM108
                t = 1.0 / (1.0 + 0.2316419 * x)
            else:
                t = 1.0 / (1.0 - 0.2316419 * x)
            d = 0.3989422804014327  # 1/sqrt(2*pi)
            p = (
                d
                * np.exp(-0.5 * x * x)
                * (
                    t
                    * (
                        0.319381530
                        + t
                        * (
                            -0.356563782
                            + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))
                        )
                    )
                )
            )
            return 1.0 - p if x >= 0 else p

        f_v = norm_cdf(log_v / sigma)
        if f_v <= 1e-10:
            bids[i] = 0.0
            continue

        # Gauss-Legendre quadrature over [0, v]
        integral = 0.0
        for j in range(len(nodes)):
            t = v * (nodes[j] + 1) / 2  # Transform from [-1,1] to [0,v]
            log_t = np.log(max(t, 1e-300))
            f_t = norm_cdf(log_t / sigma)
            ratio = f_t / f_v
            integral += weights[j] * (ratio ** (n - 1))

        bids[i] = v - (v / 2) * integral

    return bids


def first_price_bids_lognormal(
    values: ArrayLike, n: int, sigma: float = 1.0
) -> np.ndarray:
    """First-price equilibrium bids for log-normal distribution.

    Uses numba-compiled Gauss-Legendre quadrature with parallel execution.
    Supports both 1D (single auction) and 2D (batch of auctions) input.

    Args:
        values: 1D array (single auction) or 2D array (n_auctions, n_bidders).
        n: Number of bidders.
        sigma: Log-normal shape parameter (default 1.0).

    Returns:
        Array of equilibrium bids (same shape as input).
    """
    if n < 2:
        raise ValueError(f"Need n >= 2 bidders, got {n}")
    if sigma <= 0:
        raise ValueError(f"Need sigma > 0, got {sigma}")
    arr = np.asarray(values, dtype=np.float64)

    if arr.ndim == 1:
        return _first_price_bids_lognormal_numba(
            arr, n, sigma, _GL_NODES_32, _GL_WEIGHTS_32
        )

    if arr.ndim == 2:
        flat = arr.ravel()
        bids_flat = _first_price_bids_lognormal_numba(
            flat, n, sigma, _GL_NODES_32, _GL_WEIGHTS_32
        )
        return bids_flat.reshape(arr.shape)

    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")


# =============================================================================
# Exponential Distribution (Numerical Integration)
# =============================================================================


@njit(parallel=True, fastmath=True, cache=True)
def _first_price_bids_exponential_numba(
    values: np.ndarray, n: int, nodes: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Numba-compiled first-price bid computation for exponential F.

    Uses Gauss-Legendre quadrature with parallel execution.

    For exponential F(v) = 1 - exp(-v), the equilibrium bid is:
        b(v) = v - integral_0^v [F(t)/F(v)]^(n-1) dt

    Args:
        values: Array of bidder values.
        n: Number of bidders.
        nodes: Gauss-Legendre nodes.
        weights: Gauss-Legendre weights.

    Returns:
        Array of equilibrium bids.
    """
    m = len(values)
    bids = np.empty(m)

    for i in prange(m):  # type: ignore[not-iterable]
        v = values[i]
        if v <= 0:
            bids[i] = 0.0
            continue

        # Exponential CDF: F(v) = 1 - exp(-v)
        f_v = 1.0 - np.exp(-v)
        if f_v <= 1e-10:
            bids[i] = 0.0
            continue

        # Gauss-Legendre quadrature over [0, v]
        integral = 0.0
        for j in range(len(nodes)):
            t = v * (nodes[j] + 1) / 2  # Transform from [-1,1] to [0,v]
            f_t = 1.0 - np.exp(-t)
            ratio = f_t / f_v
            integral += weights[j] * (ratio ** (n - 1))

        bids[i] = v - (v / 2) * integral

    return bids


def first_price_bids_exponential(values: ArrayLike, n: int) -> np.ndarray:
    """First-price equilibrium bids for exponential distribution.

    Uses numba-compiled Gauss-Legendre quadrature with parallel execution.
    Supports both 1D (single auction) and 2D (batch of auctions) input.

    Args:
        values: 1D array (single auction) or 2D array (n_auctions, n_bidders).
        n: Number of bidders.

    Returns:
        Array of equilibrium bids (same shape as input).
    """
    if n < 2:
        raise ValueError(f"Need n >= 2 bidders, got {n}")
    arr = np.asarray(values, dtype=np.float64)

    if arr.ndim == 1:
        return _first_price_bids_exponential_numba(arr, n, _GL_NODES_32, _GL_WEIGHTS_32)

    if arr.ndim == 2:
        flat = arr.ravel()
        bids_flat = _first_price_bids_exponential_numba(
            flat, n, _GL_NODES_32, _GL_WEIGHTS_32
        )
        return bids_flat.reshape(arr.shape)

    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")


# =============================================================================
# Pareto Distribution (Numerical Integration)
# =============================================================================


@njit(parallel=True, fastmath=True, cache=True)
def _first_price_bids_pareto_numba(
    values: np.ndarray,
    n: int,
    alpha: float,
    x_m: float,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Numba-compiled first-price bid computation for Pareto F.

    Uses Gauss-Legendre quadrature with parallel execution.

    For Pareto F(v) = 1 - (x_m/v)^alpha for v >= x_m, the equilibrium bid is:
        b(v) = v - integral_{x_m}^v [F(t)/F(v)]^(n-1) dt

    Args:
        values: Array of bidder values.
        n: Number of bidders.
        alpha: Pareto shape parameter.
        x_m: Pareto scale parameter (minimum value).
        nodes: Gauss-Legendre nodes.
        weights: Gauss-Legendre weights.

    Returns:
        Array of equilibrium bids.
    """
    m = len(values)
    bids = np.empty(m)

    for i in prange(m):  # type: ignore[not-iterable]
        v = values[i]
        if v <= x_m:
            bids[i] = x_m
            continue

        # Pareto CDF: F(v) = 1 - (x_m/v)^alpha
        f_v = 1.0 - (x_m / v) ** alpha
        if f_v <= 1e-10:
            bids[i] = x_m
            continue

        # Gauss-Legendre quadrature over [x_m, v]
        # Transform from [-1,1] to [x_m, v]: t = x_m + (v - x_m) * (node + 1) / 2
        half_range = (v - x_m) / 2
        mid_point = (v + x_m) / 2

        integral = 0.0
        for j in range(len(nodes)):
            t = mid_point + half_range * nodes[j]
            f_t = 1.0 - (x_m / t) ** alpha
            ratio = f_t / f_v
            integral += weights[j] * (ratio ** (n - 1))

        bids[i] = v - half_range * integral

    return bids


def first_price_bids_pareto(
    values: ArrayLike, n: int, alpha: float = 2.0, x_m: float = 1.0
) -> np.ndarray:
    """First-price equilibrium bids for Pareto distribution.

    Uses numba-compiled Gauss-Legendre quadrature with parallel execution.
    Supports both 1D (single auction) and 2D (batch of auctions) input.

    Args:
        values: 1D array (single auction) or 2D array (n_auctions, n_bidders).
        n: Number of bidders.
        alpha: Pareto shape parameter (default 2.0).
        x_m: Pareto scale parameter, minimum value (default 1.0).

    Returns:
        Array of equilibrium bids (same shape as input).
    """
    if n < 2:
        raise ValueError(f"Need n >= 2 bidders, got {n}")
    if alpha <= 0:
        raise ValueError(f"Need alpha > 0, got {alpha}")
    if x_m <= 0:
        raise ValueError(f"Need x_m > 0, got {x_m}")
    arr = np.asarray(values, dtype=np.float64)

    if arr.ndim == 1:
        return _first_price_bids_pareto_numba(
            arr, n, alpha, x_m, _GL_NODES_32, _GL_WEIGHTS_32
        )

    if arr.ndim == 2:
        flat = arr.ravel()
        bids_flat = _first_price_bids_pareto_numba(
            flat, n, alpha, x_m, _GL_NODES_32, _GL_WEIGHTS_32
        )
        return bids_flat.reshape(arr.shape)

    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")
