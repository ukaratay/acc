"""Core metric implementations for ACC (Auction Competitive Capacity).

The ACC formula is: ACC = H(p) · log(1 + r)  where r = ρ / G*

where:
    H(p) = -Σ p_i log(p_i) = Shannon entropy of bid shares p_i = b_i / Σb_j
    ρ = b_(2) / b_(1) (competitive ratio)
    G* = normalized Gini coefficient (dispersion measure)
    r = ρ / G* (adjusted ratio)

The evidence-factor index is: E(b) = (1 + r)^{H(p)}

Key properties:
    - ACC = log(E): Evidence factors multiply, so ACC values add
    - Concave in r: Diminishing marginal returns to competition
    - Derived from Evidence Composition axiom C6'

Higher ACC indicates stronger competitive pressure (lower information rent).

Reverse/Procurement Auctions:
    For auctions where the LOW bid wins (procurement, reverse auctions),
    use reverse=True. This computes ρ = b_(1)/b_(2) (lowest/second-lowest)
    instead of the standard ρ = b_(2)/b_(1) (second-highest/highest).

All functions handle both single auction (1D) and batch (2D) inputs.
Batch processing is numba-optimized for high performance.

Support convention:
    All statistics are computed on the support S(b) = {i : b_i > 0}.
    A "dummy bidder" is a bid of exactly 0, which is ignored.
    If |S(b)| < 2 then ACC(b) = 0 and E(b) = 1 by convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from ._sorting import _sort_small

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


# =============================================================================
# Constants
# =============================================================================

# All @njit kernels use fastmath=True for performance. This is safe here because:
# (1) all bid values are positive (no NaN/Inf by construction),
# (2) the epsilon floor on G* prevents division-by-zero before log1p, and
# (3) log1p(r) avoids catastrophic cancellation for small r.
_EPS = np.finfo(np.float64).eps  # ≈ 2.22e-16


# =============================================================================
# Numba-optimized batch functions (internal)
# =============================================================================


@njit(parallel=True, fastmath=True, cache=True)
def _acc_components_batch(
    auctions: np.ndarray,
    reverse: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute entropy, rho, Gini, ACC, and evidence for multiple auctions.

    ACC = H(p) · log(1 + r)  where r = ρ / G*
    E = (1 + r)^{H(p)}

    Args:
        auctions: 2D array of shape (n_auctions, n_bidders).
        reverse: If True, compute for reverse/procurement auctions (low bid wins).
            In reverse auctions, ρ = b_(1)/b_(2) (lowest/second-lowest).

    Returns:
        Tuple of (entropy, rho, gini, acc, evidence), each shape (n_auctions,).
    """
    n_auctions, n_bidders = auctions.shape
    entropy_arr = np.empty(n_auctions)
    rho_arr = np.empty(n_auctions)
    gini_arr = np.empty(n_auctions)
    acc_arr = np.empty(n_auctions)
    evidence_arr = np.empty(n_auctions)

    use_network = n_bidders <= 9

    for i in prange(n_auctions):  # type: ignore[not-iterable]
        row = auctions[i]

        # Entropy from original row
        total = 0.0
        sum_b_log_b = 0.0
        for j in range(n_bidders):
            val = row[j]
            if val > 0.0:
                total += val
                sum_b_log_b += val * np.log(val)

        h_i = np.log(total) - sum_b_log_b / total if total > 0.0 else 0.0
        entropy_arr[i] = h_i

        # Sort for rho and Gini
        row_sorted = row.copy()
        if use_network:
            _sort_small(row_sorted)
        else:
            row_sorted.sort()

        # Identify support S(b) = {i: b_i > 0} in sorted array.
        start = 0
        while start < n_bidders and row_sorted[start] <= 0.0:
            start += 1
        n_active = n_bidders - start

        # If fewer than 2 active bidders, ACC=0, evidence=1 by convention.
        if n_active < 2 or total <= 0.0:
            rho_arr[i] = 1.0
            gini_arr[i] = 0.0
            acc_arr[i] = 0.0
            evidence_arr[i] = 1.0
            continue

        # Compute rho based on auction type, on the active support.
        if reverse:
            # Reverse/procurement: low bid wins, ρ = b_(1)/b_(2) = lowest/second-lowest
            b1 = row_sorted[start]  # lowest active (winner)
            b2 = row_sorted[start + 1]  # second-lowest active (runner-up)
            rho_i = b1 / b2 if b2 > 0.0 else 1.0
        else:
            # Standard: high bid wins, ρ = b_(2)/b_(1) = second-highest/highest
            b1 = row_sorted[n_bidders - 1]  # highest (winner)
            b2 = row_sorted[n_bidders - 2]  # second-highest (runner-up)
            rho_i = b2 / b1 if b1 > 0.0 else 1.0
        rho_arr[i] = rho_i

        # Gini
        weighted_sum = 0.0
        for j in range(n_active):
            weighted_sum += (j + 1) * row_sorted[start + j]

        two_inv_n_active = 2.0 / n_active
        gini_offset_active = (n_active + 1) / n_active
        g_raw = two_inv_n_active * weighted_sum / total - gini_offset_active
        g_i = g_raw if g_raw > 0.0 else 0.0
        gini_arr[i] = g_i

        # ACC = H(p) · log(1 + r) with r = ρ / G*, using a small floor on G*.
        g_safe = g_i if g_i > _EPS else _EPS
        r_i = rho_i / g_safe
        acc_arr[i] = h_i * np.log1p(r_i)  # log1p for numerical stability
        evidence_arr[i] = (1.0 + r_i) ** h_i

    return entropy_arr, rho_arr, gini_arr, acc_arr, evidence_arr


@njit(parallel=True, fastmath=True, cache=True)
def _information_rent_batch(auctions: np.ndarray, reverse: bool = False) -> np.ndarray:
    """Compute normalized information rent = 1 - ρ.

    Args:
        auctions: 2D array of shape (n_auctions, n_bidders).
        reverse: If True, compute for reverse/procurement auctions (low bid wins).

    Returns:
        Array of information rents, shape (n_auctions,).
    """
    n_auctions, n_bidders = auctions.shape
    rents = np.empty(n_auctions)

    for i in prange(n_auctions):  # type: ignore[not-iterable]
        row = auctions[i]

        if reverse:
            # Reverse: find two lowest bids
            v1 = np.inf
            v2 = np.inf
            n_active = 0
            for j in range(n_bidders):
                val = row[j]
                if val <= 0.0:
                    continue
                n_active += 1
                if val < v1:
                    v2 = v1
                    v1 = val
                elif val < v2:
                    v2 = val

            if n_active < 2 or not np.isfinite(v2) or v2 <= 0.0:
                rents[i] = 0.0
            else:
                # rent = (v2 - v1) / v2 = 1 - v1/v2 = 1 - ρ_reverse
                rents[i] = (v2 - v1) / v2
        else:
            # Standard: find two highest bids
            v1 = 0.0
            v2 = 0.0
            n_active = 0
            for j in range(n_bidders):
                val = row[j]
                if val <= 0.0:
                    continue
                n_active += 1
                if val > v1:
                    v2 = v1
                    v1 = val
                elif val > v2:
                    v2 = val

            if n_active < 2 or v1 <= 0.0:
                rents[i] = 0.0
            else:
                # rent = (v1 - v2) / v1 = 1 - v2/v1 = 1 - ρ
                rents[i] = (v1 - v2) / v1

    return rents


@njit(parallel=True, fastmath=True, cache=True)
def _normalized_gini_batch(auctions: np.ndarray) -> np.ndarray:
    """Compute normalized Gini coefficient G*."""
    n_auctions, n_bidders = auctions.shape
    gini_arr = np.empty(n_auctions)

    use_network = n_bidders <= 9

    for i in prange(n_auctions):  # type: ignore[not-iterable]
        row = auctions[i].copy()
        if use_network:
            _sort_small(row)
        else:
            row.sort()

        total = 0.0
        weighted_sum = 0.0
        start = 0
        while start < n_bidders and row[start] <= 0.0:
            start += 1
        n_active = n_bidders - start

        for j in range(n_active):
            val = row[start + j]
            total += val
            weighted_sum += (j + 1) * val

        if n_active < 2 or total <= 0.0:
            gini_arr[i] = 0.0
        else:
            gini_offset_active = (n_active + 1) / n_active
            two_inv_n_active = 2.0 / n_active
            gini_arr[i] = max(
                0.0, two_inv_n_active * weighted_sum / total - gini_offset_active
            )

    return gini_arr


@njit(parallel=True, fastmath=True, cache=True)
def _competitive_ratio_batch(auctions: np.ndarray, reverse: bool = False) -> np.ndarray:
    """Compute competitive ratio ρ.

    Args:
        auctions: 2D array of shape (n_auctions, n_bidders).
        reverse: If True, compute for reverse/procurement auctions (low bid wins).
            Standard: ρ = b_(2)/b_(1) (second-highest/highest)
            Reverse: ρ = b_(1)/b_(2) (lowest/second-lowest)

    Returns:
        Array of competitive ratios, shape (n_auctions,).
    """
    n_auctions, n_bidders = auctions.shape
    rho_arr = np.empty(n_auctions)

    for i in prange(n_auctions):  # type: ignore[not-iterable]
        row = auctions[i]

        if reverse:
            # Reverse: find two lowest bids
            b1 = np.inf
            b2 = np.inf
            n_active = 0
            for j in range(n_bidders):
                val = row[j]
                if val <= 0.0:
                    continue
                n_active += 1
                if val < b1:
                    b2 = b1
                    b1 = val
                elif val < b2:
                    b2 = val

            if n_active < 2 or not np.isfinite(b2) or b2 <= 0.0:
                rho_arr[i] = 1.0
            else:
                # ρ_reverse = b1/b2 = lowest/second-lowest
                rho_arr[i] = b1 / b2
        else:
            # Standard: find two highest bids
            b1 = 0.0
            b2 = 0.0
            n_active = 0
            for j in range(n_bidders):
                val = row[j]
                if val <= 0.0:
                    continue
                n_active += 1
                if val > b1:
                    b2 = b1
                    b1 = val
                elif val > b2:
                    b2 = val

            if n_active < 2 or b1 <= 0.0:
                rho_arr[i] = 1.0
            else:
                # ρ = b2/b1 = second-highest/highest
                rho_arr[i] = b2 / b1

    return rho_arr


@njit(parallel=True, fastmath=True, cache=True)
def _entropy_batch(auctions: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy H(p) = -Σ p_i log(p_i)."""
    n_auctions, n_bidders = auctions.shape
    entropy_arr = np.empty(n_auctions)

    for i in prange(n_auctions):  # type: ignore[not-iterable]
        row = auctions[i]
        total = 0.0
        sum_b_log_b = 0.0
        for j in range(n_bidders):
            val = row[j]
            if val > 0.0:
                total += val
                sum_b_log_b += val * np.log(val)

        if total > 0.0:
            entropy_arr[i] = np.log(total) - sum_b_log_b / total
        else:
            entropy_arr[i] = 0.0

    return entropy_arr


# =============================================================================
# Public API
# =============================================================================


def normalized_gini(bids: ArrayLike) -> float | np.ndarray:
    """Calculate normalized Gini coefficient G* as dispersion measure."""
    arr = np.asarray(bids, dtype=np.float64)
    if arr.ndim == 1:
        return (
            0.0
            if len(arr) < 2
            else float(_normalized_gini_batch(arr.reshape(1, -1))[0])
        )
    if arr.ndim == 2:
        return _normalized_gini_batch(arr)
    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")


def competitive_ratio(bids: ArrayLike, *, reverse: bool = False) -> float | np.ndarray:
    """Calculate competitive ratio ρ.

    Args:
        bids: Bid vector (1D) or matrix of bid vectors (2D).
        reverse: If True, compute for reverse/procurement auctions (low bid wins).
            Standard: ρ = b_(2)/b_(1) (second-highest/highest)
            Reverse: ρ = b_(1)/b_(2) (lowest/second-lowest)

    Returns:
        Competitive ratio(s).
    """
    arr = np.asarray(bids, dtype=np.float64)
    if arr.ndim == 1:
        return (
            1.0
            if len(arr) < 2
            else float(_competitive_ratio_batch(arr.reshape(1, -1), reverse)[0])
        )
    if arr.ndim == 2:
        return _competitive_ratio_batch(arr, reverse)
    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")


def entropy(bids: ArrayLike) -> float | np.ndarray:
    """Calculate Shannon entropy of bid shares H(p)."""
    arr = np.asarray(bids, dtype=np.float64)
    if arr.ndim == 1:
        return 0.0 if len(arr) < 2 else float(_entropy_batch(arr.reshape(1, -1))[0])
    if arr.ndim == 2:
        return _entropy_batch(arr)
    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")


def effective_bidders(bids: ArrayLike) -> float | np.ndarray:
    """Calculate effective number of bidders n_eff = exp(H(p))."""
    arr = np.asarray(bids, dtype=np.float64)
    if arr.ndim == 1:
        return (
            1.0
            if len(arr) < 2
            else float(np.exp(_entropy_batch(arr.reshape(1, -1))[0]))
        )
    if arr.ndim == 2:
        return np.exp(_entropy_batch(arr))
    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")


def acc(bids: ArrayLike, *, reverse: bool = False) -> float | np.ndarray:
    """Calculate Auction Competitive Capacity.

    ACC = H(p) · log(1 + r)  where r = ρ / G*

    Higher ACC indicates stronger competitive pressure (lower information rent).

    Args:
        bids: Bid vector (1D) or matrix of bid vectors (2D).
        reverse: If True, compute for reverse/procurement auctions (low bid wins).

    Returns:
        ACC value(s).
    """
    arr = np.asarray(bids, dtype=np.float64)
    if arr.ndim == 1:
        if len(arr) < 2:
            return 0.0
        _, _, _, acc_arr, _ = _acc_components_batch(arr.reshape(1, -1), reverse)
        return float(acc_arr[0])
    if arr.ndim == 2:
        _, _, _, acc_arr, _ = _acc_components_batch(arr, reverse)
        return acc_arr
    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")


def evidence_factor(bids: ArrayLike, *, reverse: bool = False) -> float | np.ndarray:
    """Calculate evidence-factor index E(b) = (1 + r)^{H(p)}.

    Evidence factors multiply across auctions, so ACC values add.

    Args:
        bids: Bid vector (1D) or matrix of bid vectors (2D).
        reverse: If True, compute for reverse/procurement auctions (low bid wins).

    Returns:
        Evidence factor(s).
    """
    arr = np.asarray(bids, dtype=np.float64)
    if arr.ndim == 1:
        if len(arr) < 2:
            return 1.0
        _, _, _, _, evidence_arr = _acc_components_batch(arr.reshape(1, -1), reverse)
        return float(evidence_arr[0])
    if arr.ndim == 2:
        _, _, _, _, evidence_arr = _acc_components_batch(arr, reverse)
        return evidence_arr
    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")


def acc_components(
    bids: ArrayLike,
    *,
    reverse: bool = False,
) -> tuple[
    float | np.ndarray,
    float | np.ndarray,
    float | np.ndarray,
    float | np.ndarray,
    float | np.ndarray,
]:
    """Compute all ACC components: (entropy, rho, gini, acc, evidence).

    Args:
        bids: Bid vector (1D) or matrix of bid vectors (2D).
        reverse: If True, compute for reverse/procurement auctions (low bid wins).

    Returns:
        Tuple of (entropy, rho, gini, acc, evidence).
    """
    arr = np.asarray(bids, dtype=np.float64)
    if arr.ndim == 1:
        if len(arr) < 2:
            return 0.0, 1.0, 0.0, 0.0, 1.0
        h, rho, g, a, e = _acc_components_batch(arr.reshape(1, -1), reverse)
        return float(h[0]), float(rho[0]), float(g[0]), float(a[0]), float(e[0])
    if arr.ndim == 2:
        return _acc_components_batch(arr, reverse)
    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")


def information_rent(bids: ArrayLike, *, reverse: bool = False) -> float | np.ndarray:
    """Compute normalized information rent = 1 - ρ.

    Args:
        bids: Bid vector (1D) or matrix of bid vectors (2D).
        reverse: If True, compute for reverse/procurement auctions (low bid wins).

    Returns:
        Information rent(s).
    """
    arr = np.asarray(bids, dtype=np.float64)
    if arr.ndim == 1:
        return (
            0.0
            if len(arr) < 2
            else float(_information_rent_batch(arr.reshape(1, -1), reverse)[0])
        )
    if arr.ndim == 2:
        return _information_rent_batch(arr, reverse)
    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")


def _active_floor(row: np.ndarray, delta: float) -> float:
    """One-tick dispersion floor on the active support of a single bid vector.

    G_delta = (n - 1) * delta / (n^2 * b_bar), where n is the number of
    active (positive) bids and b_bar their mean. Returns 0.0 when fewer
    than two bidders are active (ACC is 0 by convention there).
    """
    active = row[row > 0.0]
    n_active = active.size
    if n_active < 2:
        return 0.0
    return (n_active - 1) * delta / (n_active**2 * float(active.mean()))


def dispersion_floor(bids: ArrayLike, *, delta: float) -> float | np.ndarray:
    """One-tick dispersion floor G_delta = (n-1)·delta/(n²·b_bar).

    The floor is the normalized-Gini level of a one-tick spread at the
    auction's observed price scale: dispersion below it is not measurable
    at bid-grid resolution delta.

    Args:
        bids: Bid vector (1D) or matrix of bid vectors (2D).
        delta: Bid-grid resolution (minimum tick), must be positive.

    Returns:
        Floor value(s).
    """
    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")
    arr = np.asarray(bids, dtype=np.float64)
    if arr.ndim == 1:
        return _active_floor(arr, delta)
    if arr.ndim == 2:
        return np.array([_active_floor(row, delta) for row in arr])
    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")


def acc_delta(
    bids: ArrayLike, *, delta: float, reverse: bool = False
) -> float | np.ndarray:
    """Finite-resolution ACC: ACC_delta = H(p) · log(1 + ρ / max(G*, G_delta)).

    Floors baseline dispersion at the one-tick level so the index is finite
    for every recorded auction, including exact ties. Equals acc() whenever
    G* exceeds the floor; differs only on the saturated set.

    Args:
        bids: Bid vector (1D) or matrix of bid vectors (2D).
        delta: Bid-grid resolution (minimum tick), must be positive.
        reverse: If True, compute for reverse/procurement auctions (low bid wins).

    Returns:
        Finite-resolution ACC value(s).
    """
    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")
    arr = np.asarray(bids, dtype=np.float64)
    if arr.ndim == 1:
        if len(arr) < 2:
            return 0.0
        h, rho, gini, _, _ = _acc_components_batch(arr.reshape(1, -1), reverse)
        floor = _active_floor(arr, delta)
        if floor <= 0.0:
            return 0.0
        g_eff = max(float(gini[0]), floor)
        return float(h[0] * np.log1p(rho[0] / g_eff))
    if arr.ndim == 2:
        h, rho, gini, _, _ = _acc_components_batch(arr, reverse)
        out = np.empty(arr.shape[0])
        for i in range(arr.shape[0]):
            floor = _active_floor(arr[i], delta)
            if floor <= 0.0:
                out[i] = 0.0
                continue
            g_eff = max(float(gini[i]), floor)
            out[i] = h[i] * np.log1p(rho[i] / g_eff)
        return out
    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")


# Relative tolerance for the knife-edge G* = floor: a one-tick two-bid spread
# sits exactly at the floor (n^2*bbar = n(nc+delta) identically), and float
# rounding must not flip that auction into the saturated set.
_SAT_RTOL = 1e-9


def saturation_flag(bids: ArrayLike, *, delta: float) -> bool | np.ndarray:
    """Saturation flag SAT_delta = 1{G* < G_delta}.

    Flags auctions whose dispersion lies strictly below the one-tick floor:
    exactly those where the floor binds and ACC_delta < ACC. At equality
    (e.g., two bids one tick apart) the estimands coincide and the auction
    is not flagged; the comparison uses a relative tolerance so floating
    point cannot resolve the knife-edge arbitrarily. Returns False when
    fewer than two bidders are active.

    Args:
        bids: Bid vector (1D) or matrix of bid vectors (2D).
        delta: Bid-grid resolution (minimum tick), must be positive.

    Returns:
        Boolean flag(s).
    """
    if delta <= 0.0:
        raise ValueError(f"delta must be positive, got {delta}")
    arr = np.asarray(bids, dtype=np.float64)
    if arr.ndim == 1:
        floor = _active_floor(arr, delta)
        if floor <= 0.0:
            return False
        gini = _normalized_gini_batch(arr.reshape(1, -1))[0]
        return bool(gini < floor * (1.0 - _SAT_RTOL))
    if arr.ndim == 2:
        gini = _normalized_gini_batch(arr)
        out = np.empty(arr.shape[0], dtype=np.bool_)
        for i in range(arr.shape[0]):
            floor = _active_floor(arr[i], delta)
            out[i] = bool(floor > 0.0 and gini[i] < floor * (1.0 - _SAT_RTOL))
        return out
    raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")
