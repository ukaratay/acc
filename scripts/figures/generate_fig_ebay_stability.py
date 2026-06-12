#!/usr/bin/env python3
"""Generate Figure H.3: Economic Validation of ACC.

Two-panel figure showing ACC predicts seller pricing behavior:
(a) ACC vs price premium scatter (per-category correlations)
(b) Mean price premium by ACC quartile (effect magnitude)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.linear_model import TheilSenRegressor
from style import (
    COLOR_1,
    COLOR_3,
    COLOR_5,
    FIGSIZE_WIDE,
    save_figure,
    setup_ax,
)


def load_data() -> pl.DataFrame:
    """Load processed eBay auction data."""
    data_path = Path(__file__).parent.parent.parent / "data" / "empirical"
    return pl.read_parquet(data_path / "ebay" / "ebay_auctions.parquet")


def generate_figure() -> None:
    """Generate Figure H.3: ACC and price outcomes (descriptive).

    Uses the full sample with the finite-resolution estimand ACC_delta;
    saturated auctions are retained and marked (Section 8.1).
    """
    auctions = load_data()

    # Retain all auctions; use ACC_delta so ties enter at the resolution cap
    n_sat = int(auctions["SAT"].sum())
    print(f"Retained all {auctions.height} auctions ({n_sat} saturated at $0.01 tick)")
    print(f"Sample size: N = {auctions.height}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Category styling (matching fig_ebay_overview)
    cat_colors = {"Xbox": COLOR_1, "Cartier": COLOR_3, "Palm": COLOR_5}
    cat_markers = {"Xbox": "o", "Cartier": "s", "Palm": "^"}
    cat_hatches = {"Xbox": "ooo", "Cartier": "\\\\", "Palm": ""}

    # Extract arrays and handle any invalid price_ratio values
    acc_arr = auctions["ACC_delta"].to_numpy()
    sat_arr = auctions["SAT"].to_numpy().astype(bool)
    price_ratio = auctions["price_ratio"].to_numpy()
    categories = auctions["category"].to_numpy()

    # Handle potential invalid values (null, infinite) by masking
    valid_pr = np.isfinite(price_ratio) & (price_ratio > 0)
    log_price_ratio = np.full_like(price_ratio, np.nan)
    log_price_ratio[valid_pr] = np.log10(price_ratio[valid_pr])

    # =========================================================================
    # Panel (a): ACC vs Price Premium (per-category fits)
    # =========================================================================
    # Set axis limits based on data range
    valid_lpr = log_price_ratio[np.isfinite(log_price_ratio)]
    x_min, x_max = 0, acc_arr.max() + 1
    y_min, y_max = valid_lpr.min() - 0.2, valid_lpr.max() + 0.2
    for cat, color in cat_colors.items():
        mask = categories == cat
        acc_cat = acc_arr[mask]
        lpr_cat = log_price_ratio[mask]

        # Compute Spearman correlation (robust to outliers)
        valid = np.isfinite(lpr_cat)
        rho, _ = spearmanr(acc_cat[valid], lpr_cat[valid])

        # Scatter
        ax1.scatter(
            acc_cat,
            lpr_cat,
            c=color,
            alpha=0.5,
            s=25,
            marker=cat_markers[cat],
            label=f"{cat} ($r_s$={rho:.2f})",
        )

        # Theil-Sen regression (robust to outliers)
        ts = TheilSenRegressor(random_state=42)
        ts.fit(acc_cat[valid].reshape(-1, 1), lpr_cat[valid])

        # Clip regression line to category's observed ACC range
        acc_min_cat = acc_cat[valid].min()
        acc_max_cat = acc_cat[valid].max()
        x_line_cat = np.linspace(acc_min_cat, acc_max_cat, 100)
        ax1.plot(
            x_line_cat,
            ts.predict(x_line_cat.reshape(-1, 1)),
            "--",
            color=color,
            linewidth=1.5,
        )

    # Mark saturated auctions (ties at the recording tick) distinctly
    if sat_arr.any():
        ax1.scatter(
            acc_arr[sat_arr],
            log_price_ratio[sat_arr],
            facecolors="none",
            edgecolors="black",
            s=80,
            linewidths=1.2,
            label=f"Saturated (n={int(sat_arr.sum())})",
        )

    setup_ax(
        ax1,
        xlabel=r"$\mathrm{ACC}_\delta$",
        ylabel=r"$\log_{10}$(Final Price / Opening Bid)",
        title=r"(a) $\mathrm{ACC}_\delta$ vs Price Premium",
    )
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)

    # =========================================================================
    # Panel (b): Mean Price Premium by ACC Quartile
    # =========================================================================
    # Compute ACC quartiles
    quartile_bounds = np.percentile(acc_arr, [25, 50, 75])
    quartile_labels = ["Q1", "Q2", "Q3", "Q4"]

    def get_quartile(x):
        if x < quartile_bounds[0]:
            return 0
        elif x < quartile_bounds[1]:
            return 1
        elif x < quartile_bounds[2]:
            return 2
        else:
            return 3

    quartiles = np.array([get_quartile(x) for x in acc_arr])

    # Compute mean and SE for each category × quartile
    bar_width = 0.22
    x_positions = np.arange(4)

    for i, (cat, color) in enumerate(cat_colors.items()):
        means = []
        sems = []
        for q in range(4):
            mask = (categories == cat) & (quartiles == q)
            if mask.sum() > 0:
                pr = price_ratio[mask]
                pr = pr[np.isfinite(pr) & (pr > 0)]
                means.append(pr.mean() if len(pr) > 0 else np.nan)
                sems.append(pr.std() / np.sqrt(len(pr)) if len(pr) > 1 else 0)
            else:
                means.append(np.nan)
                sems.append(0)

        offset = (i - 1) * bar_width
        ax2.bar(
            x_positions + offset,
            means,
            bar_width,
            yerr=sems,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            label=cat,
            capsize=2,
            alpha=0.85,
            hatch=cat_hatches[cat],
        )

    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(quartile_labels)
    ax2.set_yscale("log")
    ax2.set_ylim(bottom=1)
    setup_ax(
        ax2,
        xlabel=r"$\mathrm{ACC}_\delta$ Quartile",
        ylabel="Mean Price Premium\n(Final / Opening Bid)",
        title=r"(b) Price Premium by $\mathrm{ACC}_\delta$ Level",
    )
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()

    path = save_figure(fig, "fig_ebay_stability.png")
    print(f"Saved figure to {path}")

    # Print summary statistics
    print("\n=== Economic Validation Summary ===")
    valid_all = np.isfinite(log_price_ratio)
    rho_all, pval = spearmanr(acc_arr[valid_all], log_price_ratio[valid_all])
    print(
        f"Spearman correlation (ACC vs log price premium): "
        f"ρ = {rho_all:.3f}, p = {pval:.2e}"
    )

    print("\n=== Price Premium by ACC Quartile ===")
    for q, label in enumerate(quartile_labels):
        mask = quartiles == q
        pr = price_ratio[mask]
        print(f"{label:12s}: mean = {pr.mean():.2f}, n = {mask.sum()}")

    # Effect size (Q1 vs Q4)
    q1_mean = price_ratio[quartiles == 0].mean()
    q4_mean = price_ratio[quartiles == 3].mean()
    pct_diff = 100 * (q1_mean - q4_mean) / q1_mean
    print(f"\nEffect: Q4 (highest ACC) has {pct_diff:.1f}% lower price premium than Q1")


if __name__ == "__main__":
    generate_figure()
