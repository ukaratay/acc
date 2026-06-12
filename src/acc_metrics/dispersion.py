"""Alternative dispersion measures for auction competitiveness analysis.

These measures can be used as alternatives to the normalized Gini coefficient
in the ACC construction (i.e., replacing the dispersion term used to form
the adjusted ratio r = ρ / D). Each measures bid dispersion differently.

Note: The default and recommended dispersion measure is `normalized_gini` from
the core module, which satisfies Schur-convexity (axiom D3).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from ._sorting import _sort_small

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


# =============================================================================
# Numba-optimized batch functions (internal)
# =============================================================================


@njit(parallel=True, fastmath=True, cache=True)
def _gini_batch(values: np.ndarray) -> np.ndarray:
    """Compute Gini coefficient for multiple value arrays in parallel."""
    n_arrays, n_values = values.shape
    result = np.empty(n_arrays)

    # Hoist loop invariants
    use_network = n_values <= 9
    gini_offset = (n_values + 1) / n_values
    two_inv_n = 2.0 / n_values

    for i in prange(n_arrays):  # type: ignore[not-iterable]
        row = values[i].copy()
        if use_network:
            _sort_small(row)
        else:
            row.sort()

        total = 0.0
        weighted_sum = 0.0
        for j in range(n_values):
            val = row[j]
            total += val
            weighted_sum += (j + 1) * val

        if total <= 0:
            result[i] = 0.0
        else:
            # G = (2 * sum(i*x_i)) / (n * sum(x_i)) - (n+1)/n
            g = two_inv_n * weighted_sum / total - gini_offset
            result[i] = g if g > 0 else 0.0

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _mad_normalized_batch(values: np.ndarray) -> np.ndarray:
    """Compute normalized MAD for multiple value arrays in parallel."""
    n_arrays, n_values = values.shape
    result = np.empty(n_arrays)

    for i in prange(n_arrays):  # type: ignore[not-iterable]
        row = values[i].copy()
        row.sort()

        # Median
        if n_values % 2 == 0:
            median = (row[n_values // 2 - 1] + row[n_values // 2]) / 2
        else:
            median = row[n_values // 2]

        # MAD (median absolute deviation)
        abs_devs = np.empty(n_values)
        for j in range(n_values):
            abs_devs[j] = abs(row[j] - median)
        abs_devs.sort()

        if n_values % 2 == 0:
            mad = (abs_devs[n_values // 2 - 1] + abs_devs[n_values // 2]) / 2
        else:
            mad = abs_devs[n_values // 2]

        if median > 0:
            result[i] = min(max(mad / median, 0.01), 10.0)
        else:
            result[i] = 0.5

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _normalized_range_batch(values: np.ndarray) -> np.ndarray:
    """Compute normalized range for multiple value arrays in parallel."""
    n_arrays, n_values = values.shape
    result = np.empty(n_arrays)

    for i in prange(n_arrays):  # type: ignore[not-iterable]
        v_min = values[i, 0]
        v_max = values[i, 0]
        for j in range(1, n_values):
            val = values[i, j]
            if val < v_min:
                v_min = val
            if val > v_max:
                v_max = val

        denom = v_max + v_min
        if denom > 0:
            result[i] = (v_max - v_min) / denom
        else:
            result[i] = 0.5

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _coefficient_of_variation_batch(values: np.ndarray) -> np.ndarray:
    """Compute coefficient of variation for multiple value arrays in parallel."""
    n_arrays, n_values = values.shape
    result = np.empty(n_arrays)

    for i in prange(n_arrays):  # type: ignore[not-iterable]
        # Compute mean
        total = 0.0
        for j in range(n_values):
            total += values[i, j]
        mean = total / n_values

        # Compute variance
        var_sum = 0.0
        for j in range(n_values):
            diff = values[i, j] - mean
            var_sum += diff * diff
        std = np.sqrt(var_sum / n_values)

        if mean > 0:
            result[i] = max(std / mean, 0.01)
        else:
            result[i] = 1.0

    return result


@njit(parallel=True, fastmath=True, cache=True)
def _quartile_variation_batch(values: np.ndarray) -> np.ndarray:
    """Compute coefficient of quartile variation for multiple value arrays.

    V_b = (Q3 - Q1) / (Q3 + Q1)

    Note: This measure does NOT satisfy Schur-convexity (axiom D3).
    Included for comparison purposes only.
    """
    n_arrays, n_values = values.shape
    result = np.empty(n_arrays)

    for i in prange(n_arrays):  # type: ignore[not-iterable]
        row = values[i].copy()
        row.sort()

        # Handle small n specially
        if n_values == 2:
            # For n=2, use min and max
            q1 = row[0]
            q3 = row[1]
        else:
            # Linear interpolation for quartiles
            # Q1 at position (n-1) * 0.25, Q3 at position (n-1) * 0.75
            q1_pos = (n_values - 1) * 0.25
            q3_pos = (n_values - 1) * 0.75

            q1_lo = int(q1_pos)
            q1_hi = min(q1_lo + 1, n_values - 1)
            q1_frac = q1_pos - q1_lo
            q1 = row[q1_lo] * (1 - q1_frac) + row[q1_hi] * q1_frac

            q3_lo = int(q3_pos)
            q3_hi = min(q3_lo + 1, n_values - 1)
            q3_frac = q3_pos - q3_lo
            q3 = row[q3_lo] * (1 - q3_frac) + row[q3_hi] * q3_frac

        denom = q3 + q1
        if denom > 0:
            result[i] = (q3 - q1) / denom
        else:
            result[i] = 0.0

    return result


# =============================================================================
# Public API (scalar/batch transparent)
# =============================================================================


def gini(values: ArrayLike) -> float | np.ndarray:
    """Compute Gini coefficient as dispersion measure.

    The Gini coefficient is Schur-convex, making it the recommended
    dispersion measure for ACC (satisfies axiom D3).

    Note: Computes on the full vector including zeros, unlike
    ``core.normalized_gini`` which operates on S(b) = {i: b_i > 0}.

    Args:
        values: 1D array for single measurement, 2D array for batch.

    Returns:
        float for single measurement, ndarray for batch. Range [0, 1).
    """
    arr = np.asarray(values, dtype=np.float64)

    if arr.ndim == 1:
        if len(arr) < 2:
            return 0.0
        result = _gini_batch(arr.reshape(1, -1))
        return float(result[0])

    if arr.ndim == 2:
        return _gini_batch(arr)

    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")


def median_absolute_deviation(values: ArrayLike) -> float | np.ndarray:
    """Compute normalized median absolute deviation: MAD / median.

    Args:
        values: 1D array for single measurement, 2D array for batch.

    Returns:
        float for single measurement, ndarray for batch. Range [0.01, 10].
    """
    arr = np.asarray(values, dtype=np.float64)

    if arr.ndim == 1:
        if len(arr) < 2:
            return 0.5
        result = _mad_normalized_batch(arr.reshape(1, -1))
        return float(result[0])

    if arr.ndim == 2:
        return _mad_normalized_batch(arr)

    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")


def normalized_range(values: ArrayLike) -> float | np.ndarray:
    """Compute normalized range: (max - min) / (max + min).

    Args:
        values: 1D array for single measurement, 2D array for batch.

    Returns:
        float for single measurement, ndarray for batch. Range [0, 1].
    """
    arr = np.asarray(values, dtype=np.float64)

    if arr.ndim == 1:
        if len(arr) < 2:
            return 0.0
        result = _normalized_range_batch(arr.reshape(1, -1))
        return float(result[0])

    if arr.ndim == 2:
        return _normalized_range_batch(arr)

    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")


def coefficient_of_variation(values: ArrayLike) -> float | np.ndarray:
    """Compute coefficient of variation: std / mean.

    Args:
        values: 1D array for single measurement, 2D array for batch.

    Returns:
        float for single measurement, ndarray for batch. Range [0.01, inf).
    """
    arr = np.asarray(values, dtype=np.float64)

    if arr.ndim == 1:
        if len(arr) < 2:
            return 1.0
        result = _coefficient_of_variation_batch(arr.reshape(1, -1))
        return float(result[0])

    if arr.ndim == 2:
        return _coefficient_of_variation_batch(arr)

    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")


def quartile_variation(values: ArrayLike) -> float | np.ndarray:
    """Compute coefficient of quartile variation: V_b = (Q3 - Q1) / (Q3 + Q1).

    WARNING: This measure does NOT satisfy Schur-convexity (axiom D3).
    Included for comparison purposes only. Use `gini` for production.

    Args:
        values: 1D array for single measurement, 2D array for batch.

    Returns:
        float for single measurement, ndarray for batch. Range [0, 1).
    """
    arr = np.asarray(values, dtype=np.float64)

    if arr.ndim == 1:
        if len(arr) < 2:
            return 0.0
        result = _quartile_variation_batch(arr.reshape(1, -1))
        return float(result[0])

    if arr.ndim == 2:
        return _quartile_variation_batch(arr)

    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")


# Registry of dispersion measures
MEASURES: dict[str, Callable[[ArrayLike], float | np.ndarray]] = {
    "gini": gini,
    "mad": median_absolute_deviation,
    "range": normalized_range,
    "cv": coefficient_of_variation,
    "vb": quartile_variation,  # Does NOT satisfy Schur-convexity (D3)
}
