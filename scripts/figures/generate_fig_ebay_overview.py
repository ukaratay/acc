#!/usr/bin/env python3
"""Generate Figure H.1: Data Overview and ACC Components.

Four-panel figure showing:
(a) Distribution of n_bidders by category
(b) H(p) vs log(n) with ceiling line
(c) G* vs n with trend line
(d) ρ (competitive ratio) distributions by category
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from style import (
    COLOR_1,
    COLOR_3,
    COLOR_5,
    FIGSIZE_TALL,
    save_figure,
    setup_ax,
)


def load_data() -> pl.DataFrame:
    """Load processed eBay auction data."""
    data_path = Path(__file__).parent.parent.parent / "data" / "empirical"
    return pl.read_parquet(data_path / "ebay" / "ebay_auctions.parquet")


def generate_figure() -> None:
    """Generate Figure H.1: Data overview and ACC components."""
    auctions = load_data()

    # Retain all auctions; ties are flagged, not dropped (Section 7.1)
    n_sat = int(auctions["SAT"].sum())
    print(f"Retained all {auctions.height} auctions ({n_sat} saturated at $0.01 tick)")

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_TALL)
    (ax1, ax2), (ax3, ax4) = axes

    # Category colors, hatches, and markers
    cat_colors = {"Xbox": COLOR_1, "Cartier": COLOR_3, "Palm": COLOR_5}
    cat_hatches = {"Xbox": "ooo", "Cartier": "\\\\", "Palm": ""}  # dots for Xbox
    cat_markers = {"Xbox": "o", "Cartier": "s", "Palm": "^"}

    # Extract arrays
    n_arr = auctions["n_bidders"].to_numpy()
    log_n = auctions["log_n"].to_numpy()
    h_arr = auctions["H"].to_numpy()
    gstar_arr = auctions["G_star"].to_numpy()
    rho_arr = auctions["rho"].to_numpy()
    categories = auctions["category"].to_numpy()

    # =========================================================================
    # Panel (a): Distribution of n_bidders by category (side-by-side)
    # =========================================================================
    bins_a = np.arange(2, 26, 2)  # Bin edges
    bin_width = bins_a[1] - bins_a[0]
    bar_width = bin_width / 3.5  # Width for each category bar

    for i, (cat, color) in enumerate(cat_colors.items()):
        mask = categories == cat
        counts, _ = np.histogram(n_arr[mask], bins=bins_a)
        bin_centers = (bins_a[:-1] + bins_a[1:]) / 2
        offset = (i - 1) * bar_width  # Center the three bars
        ax1.bar(
            bin_centers + offset,
            counts,
            width=bar_width,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            label=cat,
            hatch=cat_hatches[cat],
            alpha=0.85,
        )

    setup_ax(
        ax1,
        xlabel="Number of bidders $n$",
        ylabel="Count",
        title="(a) Distribution of Bidder Count",
    )
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_xlim(0, 25)  # Align with panel (c)

    # =========================================================================
    # Panel (b): H(p) vs log(n) — entropy ceiling
    # =========================================================================
    for cat, color in cat_colors.items():
        mask = categories == cat
        ax2.scatter(
            log_n[mask],
            h_arr[mask],
            c=color,
            alpha=0.8,
            s=25,
            marker=cat_markers[cat],
            label=cat,
        )

    # Ceiling line: H = log(n)
    x_line = np.linspace(0.5, 3.5, 100)
    ax2.plot(x_line, x_line, "k--", linewidth=2, label=r"$H_{\max} = \log(n)$")

    setup_ax(
        ax2,
        xlabel=r"$\log(n)$",
        ylabel=r"Shannon entropy $H(p)$",
        title="(b) Entropy Near Ceiling",
    )
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_xlim(0.5, 3.5)
    ax2.set_ylim(0.5, 3.5)

    # =========================================================================
    # Panel (c): G* vs n — dispersion scaling
    # =========================================================================
    # Line for fit through origin
    n_line = np.linspace(0, 25, 100)

    for cat, color in cat_colors.items():
        mask = categories == cat
        n_cat = n_arr[mask]
        gstar_cat = gstar_arr[mask]

        # Fit G* = a*n through origin: a = Σ(n·G*) / Σ(n²)
        slope = np.sum(n_cat * gstar_cat) / np.sum(n_cat**2)

        ax3.scatter(
            n_cat,
            gstar_cat,
            c=color,
            alpha=0.8,
            s=25,
            marker=cat_markers[cat],
            label=f"{cat} ({slope:.3f}$n$)",
        )
        ax3.plot(n_line, slope * n_line, "--", color=color, linewidth=1.5)

    setup_ax(
        ax3,
        xlabel="Number of bidders $n$",
        ylabel=r"Normalized Gini $G^*$",
        title="(c) Dispersion Increases with $n$",
    )
    ax3.legend(loc="upper left", fontsize=8)
    ax3.set_xlim(0, 25)
    ax3.set_ylim(0, None)  # Origin at bottom left

    # =========================================================================
    # Panel (d): ρ distributions by category (side-by-side histogram)
    # =========================================================================
    # Use histogram bins from 0.4 to 1.0 (ρ ∈ (0, 1] by definition)
    bins_d = np.linspace(0.4, 1.0, 16)  # Fewer bins for grouped bars
    bin_width_d = bins_d[1] - bins_d[0]
    bar_width_d = bin_width_d / 3.5

    for i, (cat, color) in enumerate(cat_colors.items()):
        subset = auctions.filter(pl.col("category") == cat)
        rho_cat = subset["rho"].to_numpy()
        counts, _ = np.histogram(rho_cat, bins=bins_d)
        bin_centers = (bins_d[:-1] + bins_d[1:]) / 2
        offset = (i - 1) * bar_width_d
        ax4.bar(
            bin_centers + offset,
            counts,
            width=bar_width_d,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            label=f"{cat} (med={np.median(rho_cat):.2f})",
            hatch=cat_hatches[cat],
            alpha=0.85,
        )

    setup_ax(
        ax4,
        xlabel=r"Competitive ratio $\rho = b_2/b_1$",
        ylabel="Count (log scale)",
        title=r"(d) Competitive Ratio Distributions",
    )
    ax4.set_yscale("log")
    ax4.legend(loc="upper left", fontsize=9)
    ax4.set_xlim(0.4, 1.0)

    fig.tight_layout()

    path = save_figure(fig, "fig_ebay_overview.png")
    print(f"Saved figure to {path}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total auctions: {auctions.height}")
    print(f"Mean n: {n_arr.mean():.1f} (range: {n_arr.min()}-{n_arr.max()})")
    print(f"Mean H/log(n): {(h_arr / log_n).mean():.3f}")
    gstar_min, gstar_max = gstar_arr.min(), gstar_arr.max()
    print(f"Mean G*: {gstar_arr.mean():.3f} (range: {gstar_min:.3f}-{gstar_max:.3f})")
    print(f"Mean ρ: {rho_arr.mean():.3f} (median: {np.median(rho_arr):.3f})")


if __name__ == "__main__":
    generate_figure()
