#!/usr/bin/env python3
"""Generate Figure 3: Depth Curves.

Shows how ACC changes as we include more bidders beyond top-2.
Three bid vectors with same ρ but different tail structure.

Key insight: Depth curve D(k) = ACC(top-k bids) reveals whether
competition is "deep" (strong bidders beyond runner-up) or
"shallow" (only top-2 matter).
"""

import matplotlib.pyplot as plt
import numpy as np
from style import (
    COLOR_1,
    COLOR_3,
    COLOR_5,
    FIGSIZE_WIDE,
    save_figure,
    setup_ax,
)

import acc_metrics as acc


def compute_depth_curve(bids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute depth curve for a bid vector.

    Args:
        bids: 1D array of bids

    Returns:
        k_values: array [2, 3, ..., n]
        acc_values: ACC computed on top-k bids for each k
    """
    sorted_bids = np.sort(bids)[::-1]  # Descending order
    n = len(sorted_bids)

    k_values = np.arange(2, n + 1)
    acc_values = []

    for k in k_values:
        top_k = sorted_bids[:k]
        acc_k = acc.acc(top_k)
        acc_values.append(acc_k)

    return k_values, np.array(acc_values)


def generate_figure() -> None:
    """Generate Figure 3: Depth curves for three bid vectors."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Three bid vectors with same top-2 ratio (ρ ≈ 0.9)
    # but different tail structure

    # Vector A: "Deep" competition - strong lower bidders
    bids_deep = np.array([100, 90, 85, 80, 75, 70, 65])

    # Vector B: "Shallow" competition - weak lower bidders
    bids_shallow = np.array([100, 90, 40, 35, 30, 25, 20])

    # Vector C: "Mixed" - moderate tail
    bids_mixed = np.array([100, 90, 65, 55, 45, 35, 25])

    # Panel (a): Raw depth curves
    # Labels clarified per referee: separate competition depth from tail behavior
    for bids, label, color, marker in [
        (bids_deep, "Deep (uniform tail)", COLOR_1, "o"),
        (bids_shallow, "Shallow (weak tail)", COLOR_5, "s"),
        (bids_mixed, "Moderate (tapered)", COLOR_3, "^"),
    ]:
        k_vals, acc_vals = compute_depth_curve(bids)
        ax1.plot(
            k_vals,
            acc_vals,
            color=color,
            marker=marker,
            markersize=6,
            linewidth=1.5,
            label=label,
        )

    setup_ax(
        ax1,
        xlabel=r"Number of bidders included $k$",
        ylabel=r"ACC$(b_{(1:k)})$",
        title="(a) Depth Curves",
    )
    # No legend in panel (a) - panel (b) has the legend
    ax1.set_xlim(1.8, 7.2)
    ax1.set_xticks([2, 3, 4, 5, 6, 7])

    # Panel (b): Marginal contribution (ACC increment)
    for bids, label, color, marker in [
        (bids_deep, "Deep (uniform tail)", COLOR_1, "o"),
        (bids_shallow, "Shallow (weak tail)", COLOR_5, "s"),
        (bids_mixed, "Moderate (tapered)", COLOR_3, "^"),
    ]:
        k_vals, acc_vals = compute_depth_curve(bids)
        # Marginal = ACC(k) - ACC(k-1)
        marginal = np.diff(acc_vals)
        k_marginal = k_vals[1:]  # k=3, 4, 5, ...

        ax2.plot(
            k_marginal,
            marginal,
            color=color,
            marker=marker,
            markersize=6,
            linewidth=1.5,
            label=label,
        )

    ax2.axhline(0, color="gray", linestyle=":", linewidth=0.8)

    setup_ax(
        ax2,
        xlabel=r"Bidder rank $k$",
        ylabel=r"$\Delta_k$ = ACC$(k)$ $-$ ACC$(k-1)$",
        title="(b) Marginal Contribution",
    )
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_xlim(2.8, 7.2)
    ax2.set_xticks([3, 4, 5, 6, 7])

    # Adjust layout
    fig.tight_layout()

    path = save_figure(fig, "fig_depth.png")
    print(f"Saved figure to {path}")


if __name__ == "__main__":
    generate_figure()
