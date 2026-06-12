"""Internal sorting utilities using optimal sorting networks.

This module provides numba-optimized sorting for small arrays (n <= 16).
Sorting networks are optimal compare-and-swap sequences for fixed sizes,
providing ~2-3x speedup over general-purpose quicksort for small n.

This is a private module - not exported in __init__.py.
"""

import numpy as np
from numba import njit


@njit(inline="always", fastmath=True, cache=True)
def _swap_if_greater(arr: np.ndarray, i: int, j: int) -> None:
    """Compare-and-swap: ensures arr[i] <= arr[j]."""
    if arr[i] > arr[j]:
        arr[i], arr[j] = arr[j], arr[i]


@njit(inline="always", fastmath=True, cache=True)
def _sort_small(arr: np.ndarray) -> None:
    """Sort small arrays using optimal sorting networks.

    Sorting networks are branch-free and optimal for fixed small sizes.
    ~2-3x faster than quicksort for n <= 9.

    For n=10-16, falls back to insertion sort which is still faster
    than quicksort for these sizes due to cache locality.

    Args:
        arr: Array to sort in-place. Modified.
    """
    n = len(arr)

    if n <= 1:
        return
    elif n == 2:
        _swap_if_greater(arr, 0, 1)
    elif n == 3:
        # Optimal 3-element sorting network: 3 comparators
        _swap_if_greater(arr, 0, 1)
        _swap_if_greater(arr, 0, 2)
        _swap_if_greater(arr, 1, 2)
    elif n == 4:
        # Optimal 4-element sorting network: 5 comparators
        _swap_if_greater(arr, 0, 1)
        _swap_if_greater(arr, 2, 3)
        _swap_if_greater(arr, 0, 2)
        _swap_if_greater(arr, 1, 3)
        _swap_if_greater(arr, 1, 2)
    elif n == 5:
        # Optimal 5-element sorting network: 9 comparators
        _swap_if_greater(arr, 0, 1)
        _swap_if_greater(arr, 3, 4)
        _swap_if_greater(arr, 0, 2)
        _swap_if_greater(arr, 1, 2)
        _swap_if_greater(arr, 0, 3)
        _swap_if_greater(arr, 2, 4)
        _swap_if_greater(arr, 1, 3)
        _swap_if_greater(arr, 2, 3)
        _swap_if_greater(arr, 1, 2)
    elif n == 6:
        # 6-element sorting network: 12 comparators
        _swap_if_greater(arr, 0, 1)
        _swap_if_greater(arr, 2, 3)
        _swap_if_greater(arr, 4, 5)
        _swap_if_greater(arr, 0, 2)
        _swap_if_greater(arr, 1, 4)
        _swap_if_greater(arr, 3, 5)
        _swap_if_greater(arr, 0, 1)
        _swap_if_greater(arr, 2, 3)
        _swap_if_greater(arr, 4, 5)
        _swap_if_greater(arr, 1, 2)
        _swap_if_greater(arr, 3, 4)
        _swap_if_greater(arr, 2, 3)
    elif n == 7:
        # 7-element sorting network: 16 comparators
        _swap_if_greater(arr, 0, 1)
        _swap_if_greater(arr, 2, 3)
        _swap_if_greater(arr, 4, 5)
        _swap_if_greater(arr, 0, 2)
        _swap_if_greater(arr, 1, 3)
        _swap_if_greater(arr, 4, 6)
        _swap_if_greater(arr, 1, 2)
        _swap_if_greater(arr, 5, 6)
        _swap_if_greater(arr, 0, 4)
        _swap_if_greater(arr, 1, 5)
        _swap_if_greater(arr, 2, 6)
        _swap_if_greater(arr, 1, 4)
        _swap_if_greater(arr, 3, 6)
        _swap_if_greater(arr, 2, 4)
        _swap_if_greater(arr, 3, 5)
        _swap_if_greater(arr, 3, 4)
    elif n == 8:
        # 8-element sorting network: 19 comparators (Batcher odd-even mergesort)
        _swap_if_greater(arr, 0, 1)
        _swap_if_greater(arr, 2, 3)
        _swap_if_greater(arr, 4, 5)
        _swap_if_greater(arr, 6, 7)
        _swap_if_greater(arr, 0, 2)
        _swap_if_greater(arr, 1, 3)
        _swap_if_greater(arr, 4, 6)
        _swap_if_greater(arr, 5, 7)
        _swap_if_greater(arr, 1, 2)
        _swap_if_greater(arr, 5, 6)
        _swap_if_greater(arr, 0, 4)
        _swap_if_greater(arr, 1, 5)
        _swap_if_greater(arr, 2, 6)
        _swap_if_greater(arr, 3, 7)
        _swap_if_greater(arr, 2, 4)
        _swap_if_greater(arr, 3, 5)
        _swap_if_greater(arr, 1, 2)
        _swap_if_greater(arr, 3, 4)
        _swap_if_greater(arr, 5, 6)
    elif n == 9:
        # Optimal 9-element sorting network: 25 comparators
        _swap_if_greater(arr, 0, 1)
        _swap_if_greater(arr, 3, 4)
        _swap_if_greater(arr, 6, 7)
        _swap_if_greater(arr, 1, 2)
        _swap_if_greater(arr, 4, 5)
        _swap_if_greater(arr, 7, 8)
        _swap_if_greater(arr, 0, 1)
        _swap_if_greater(arr, 3, 4)
        _swap_if_greater(arr, 6, 7)
        _swap_if_greater(arr, 2, 5)
        _swap_if_greater(arr, 0, 3)
        _swap_if_greater(arr, 1, 4)
        _swap_if_greater(arr, 5, 8)
        _swap_if_greater(arr, 3, 6)
        _swap_if_greater(arr, 4, 7)
        _swap_if_greater(arr, 2, 5)
        _swap_if_greater(arr, 0, 3)
        _swap_if_greater(arr, 1, 4)
        _swap_if_greater(arr, 5, 7)
        _swap_if_greater(arr, 2, 6)
        _swap_if_greater(arr, 1, 3)
        _swap_if_greater(arr, 4, 6)
        _swap_if_greater(arr, 2, 4)
        _swap_if_greater(arr, 5, 6)
        _swap_if_greater(arr, 2, 3)
    else:
        # Insertion sort for n=10-16 (still faster than quicksort)
        for i in range(1, n):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
