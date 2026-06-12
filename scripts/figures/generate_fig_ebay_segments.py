#!/usr/bin/env python3
"""Generate Figure H.2: ACC by Market Segment.

Two-panel heatmap showing:
(a) ACC_delta mean (std) by category × duration
(b) ACE_delta mean (std) by category × duration

Uses the finite-resolution estimand (Section 8.1) so tied auctions enter
at the resolution cap instead of being dropped.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from style import CMAP_SEQ, FIGSIZE_WIDE, save_figure


def load_data() -> pl.DataFrame:
    """Load processed eBay auction data."""
    data_path = Path(__file__).parent.parent.parent / "data" / "empirical"
    return pl.read_parquet(data_path / "ebay" / "ebay_auctions.parquet")


def generate_figure() -> None:
    """Generate Figure H.2: ACC by market segment heatmaps."""
    auctions = load_data()

    # Retain all auctions; use ACC_delta so ties enter at the resolution cap
    n_sat = int(auctions["SAT"].sum())
    print(f"Retained all {auctions.height} auctions ({n_sat} saturated at $0.01 tick)")

    # Define orders
    cat_order = ["Xbox", "Cartier", "Palm"]
    dur_order = ["3-day", "5-day", "7-day"]

    # Build ACC matrix (3 categories × 3 durations)
    acc_means = np.zeros((3, 3))
    acc_stds = np.zeros((3, 3))
    acc_norm_means = np.zeros((3, 3))
    acc_norm_stds = np.zeros((3, 3))
    counts = np.zeros((3, 3), dtype=int)

    for i, cat in enumerate(cat_order):
        for j, dur in enumerate(dur_order):
            subset = auctions.filter(
                (pl.col("category") == cat) & (pl.col("duration") == dur)
            )
            if subset.height > 0:
                acc = subset["ACC_delta"].to_numpy()
                acc_norm = subset["ACE_delta"].to_numpy()
                acc_means[i, j] = acc.mean()
                acc_stds[i, j] = acc.std()
                acc_norm_means[i, j] = acc_norm.mean()
                acc_norm_stds[i, j] = acc_norm.std()
                counts[i, j] = len(acc)

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # =========================================================================
    # Panel (a): ACC heatmap
    # =========================================================================
    im1 = ax1.imshow(acc_means, cmap=CMAP_SEQ, aspect="auto")

    # Add text annotations
    for i in range(3):
        for j in range(3):
            # Choose text color based on background
            bg_val = (acc_means[i, j] - acc_means.min()) / (
                acc_means.max() - acc_means.min()
            )
            text_color = "white" if bg_val > 0.5 else "black"
            ax1.text(
                j,
                i,
                f"{acc_means[i, j]:.2f}\n({acc_stds[i, j]:.2f})",
                ha="center",
                va="center",
                fontsize=11,
                color=text_color,
            )

    ax1.set_xticks(range(3))
    ax1.set_xticklabels(dur_order)
    ax1.set_yticks(range(3))
    ax1.set_yticklabels(cat_order)
    ax1.set_xlabel("Auction Duration")
    ax1.set_ylabel("Category")
    ax1.set_title(r"(a) $\mathrm{ACC}_\delta$: mean (std)")

    # Add colorbar
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label(r"$\mathrm{ACC}_\delta$")

    # =========================================================================
    # Panel (b): ACE (Auction Competitive Efficiency) heatmap
    # =========================================================================
    im2 = ax2.imshow(acc_norm_means, cmap=CMAP_SEQ, aspect="auto")

    # Add text annotations
    for i in range(3):
        for j in range(3):
            bg_val = (acc_norm_means[i, j] - acc_norm_means.min()) / (
                acc_norm_means.max() - acc_norm_means.min()
            )
            text_color = "white" if bg_val > 0.5 else "black"
            ax2.text(
                j,
                i,
                f"{acc_norm_means[i, j]:.2f}\n({acc_norm_stds[i, j]:.2f})",
                ha="center",
                va="center",
                fontsize=11,
                color=text_color,
            )

    ax2.set_xticks(range(3))
    ax2.set_xticklabels(dur_order)
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(cat_order)
    ax2.set_xlabel("Auction Duration")
    ax2.set_ylabel("Category")
    ax2.set_title(r"(b) $\mathrm{ACE}_\delta$: mean (std)")

    # Add colorbar
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label(r"$\mathrm{ACE}_\delta$")

    fig.tight_layout()

    path = save_figure(fig, "fig_ebay_segments.png")
    print(f"Saved figure to {path}")

    # Print detailed statistics
    print("\n=== Cross-tabulation: Category × Duration ===")
    print("\nACC means:")
    print(f"{'':10s} " + " ".join(f"{d:>10s}" for d in dur_order))
    for i, cat in enumerate(cat_order):
        row = " ".join(f"{acc_means[i, j]:>10.2f}" for j in range(3))
        print(f"{cat:10s} {row}")

    print("\nACE means:")
    print(f"{'':10s} " + " ".join(f"{d:>10s}" for d in dur_order))
    for i, cat in enumerate(cat_order):
        row = " ".join(f"{acc_norm_means[i, j]:>10.2f}" for j in range(3))
        print(f"{cat:10s} {row}")

    print("\nCounts:")
    print(f"{'':10s} " + " ".join(f"{d:>10s}" for d in dur_order))
    for i, cat in enumerate(cat_order):
        row = " ".join(f"{counts[i, j]:>10d}" for j in range(3))
        print(f"{cat:10s} {row}")


if __name__ == "__main__":
    generate_figure()
