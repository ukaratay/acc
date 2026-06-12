#!/usr/bin/env python3
"""Generate collusion detection figure for Operation Car Wash analysis.

Creates a 4-panel figure showing:
- (a) ROC curves comparing ACE vs simple screens
- (b) Component decomposition by collusion status
- (c) ACE distribution by status
- (d) Win/PTE ratio by status

Dataset: 95 Brazilian Oil Company procurement auctions (2002-2013)
         with confession-based ground truth labels.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from style import (
    COLOR_1,
    COLOR_3,
    COLOR_5,
    FIGSIZE_TALL,
    GRAY_MED,
    add_panel_label,
    save_figure,
    setup_ax,
)

# =============================================================================
# Data Loading
# =============================================================================

DATA_PATH = (
    Path(__file__).parent.parent.parent
    / "data"
    / "empirical"
    / "carwash"
    / "carwash_auctions.csv"
)


def load_data() -> pd.DataFrame:
    """Load and filter auction data for binary classification.

    Excludes 'Partly Collusive' auctions (ground_truth = -1).

    Returns:
        DataFrame with 95 auctions (27 collusive, 68 honest).
    """
    df = pd.read_csv(DATA_PATH)
    # Filter to binary classification (exclude partly collusive)
    df = df[df["ground_truth"] >= 0].copy()
    return df


# =============================================================================
# Analysis Functions
# =============================================================================


def compute_roc_data(df: pd.DataFrame) -> dict:
    """Compute ROC curves for different predictors.

    Returns dict with ROC data for each predictor.
    """
    y_true = df["ground_truth"].values

    results = {}

    # ACE: Higher = collusive (best ACC-based predictor)
    fpr, tpr, _ = roc_curve(y_true, df["ACE"])
    results["ACE"] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc_score(y_true, df["ACE"]),
        "color": COLOR_1,
        "linestyle": "-",
        "linewidth": 2.5,
    }

    # Win/PTE ratio: Higher = collusive (price manipulation signal)
    fpr, tpr, _ = roc_curve(y_true, df["win_pte_ratio"])
    results["Win/PTE"] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc_score(y_true, df["win_pte_ratio"]),
        "color": COLOR_3,
        "linestyle": "--",
        "linewidth": 2.0,
    }

    # G* (Gini): Lower = collusive
    fpr, tpr, _ = roc_curve(y_true, -df["G_star"])
    results["G*"] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc_score(y_true, -df["G_star"]),
        "color": COLOR_5,
        "linestyle": "-.",
        "linewidth": 2.0,
    }

    # n_bidders: Lower = collusive
    fpr, tpr, _ = roc_curve(y_true, -df["n_bidders"])
    results["n"] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": roc_auc_score(y_true, -df["n_bidders"]),
        "color": GRAY_MED,
        "linestyle": ":",
        "linewidth": 2.0,
    }

    return results


def compute_component_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean component values by collusion status."""
    stats_list = []
    for status in [1, 0]:
        subset = df[df["ground_truth"] == status]
        label = "Collusive" if status == 1 else "Not confirmed"

        stats_list.append(
            {
                "Status": label,
                "H": subset["H"].mean(),
                "H_se": subset["H"].sem(),
                "rho": subset["rho"].mean(),
                "rho_se": subset["rho"].sem(),
                "G_star": subset["G_star"].mean(),
                "G_star_se": subset["G_star"].sem(),
                "ACE": subset["ACE"].mean(),
                "ACE_se": subset["ACE"].sem(),
                "n": len(subset),
            }
        )

    return pd.DataFrame(stats_list)


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_roc_curves(ax: plt.Axes, roc_data: dict) -> None:
    """Plot ROC curves comparing ACE vs other predictors."""
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.4, label="_nolegend_")

    # Plot each predictor
    for name, data in roc_data.items():
        ax.plot(
            data["fpr"],
            data["tpr"],
            color=data["color"],
            linestyle=data["linestyle"],
            lw=data["linewidth"],
            label=f"{name} ({data['auc']:.2f})",
        )

    setup_ax(ax, "False Positive Rate", "True Positive Rate")
    ax.legend(loc="lower right", fontsize=10, title="AUC", title_fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)


def plot_component_bars(ax: plt.Axes, stats_df: pd.DataFrame) -> None:
    """Plot component values by collusion status (normalized to honest=1)."""
    x = np.arange(4)  # H, ρ, G*, ACE
    width = 0.35

    collusive = stats_df[stats_df["Status"] == "Collusive"].iloc[0]
    honest = stats_df[stats_df["Status"] == "Not confirmed"].iloc[0]

    components = ["H", "rho", "G_star", "ACE"]
    se_components = ["H_se", "rho_se", "G_star_se", "ACE_se"]
    coll_vals = [collusive[c] for c in components]
    honest_vals = [honest[c] for c in components]
    coll_ses = [collusive[c] for c in se_components]
    honest_ses = [honest[c] for c in se_components]

    # Normalize to honest baseline
    coll_norm = [
        c / h if h > 0 else 0 for c, h in zip(coll_vals, honest_vals, strict=True)
    ]
    honest_norm = [1.0] * 4

    # Compute error bars using delta method
    # For collusive bars: SE(c/h) ≈ (c/h) * sqrt((SE_c/c)² + (SE_h/h)²)
    coll_yerr = []
    for c, h, se_c, se_h in zip(
        coll_vals, honest_vals, coll_ses, honest_ses, strict=True
    ):
        if c > 0 and h > 0:
            ratio = c / h
            rel_se = np.sqrt((se_c / c) ** 2 + (se_h / h) ** 2)
            coll_yerr.append(ratio * rel_se)
        else:
            coll_yerr.append(0)

    # For honest baseline (normalized to 1.0): error = SE_h / h
    honest_yerr = [
        se_h / h if h > 0 else 0
        for se_h, h in zip(honest_ses, honest_vals, strict=True)
    ]

    # Bars
    ax.bar(
        x - width / 2,
        coll_norm,
        width,
        label="Collusive",
        color=COLOR_1,
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
        alpha=0.85,
        yerr=coll_yerr,
        capsize=2,
    )
    ax.bar(
        x + width / 2,
        honest_norm,
        width,
        label="Not confirmed",
        color=COLOR_5,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
        yerr=honest_yerr,
        capsize=2,
    )

    # Reference line
    ax.axhline(y=1.0, color="k", linestyle="--", lw=1.0, alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([r"$H$", r"$\rho$", r"$G^*$", "ACE"])
    setup_ax(ax, "", "Relative to Not-Confirmed")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 1.6)


def plot_ace_density(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Plot ACE distribution by collusion status."""
    collusive = df[df["ground_truth"] == 1]["ACE"]
    honest = df[df["ground_truth"] == 0]["ACE"]

    # Histogram bins
    all_ace = df["ACE"]
    bins = np.linspace(all_ace.min() - 0.1, all_ace.max() + 0.1, 15)
    bin_width = bins[1] - bins[0]
    bar_width = bin_width / 2.5

    # Collusive
    counts_c, _ = np.histogram(collusive, bins=bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.bar(
        bin_centers - bar_width / 2,
        counts_c,
        width=bar_width,
        color=COLOR_1,
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
        alpha=0.85,
        label=f"Collusive (μ={collusive.mean():.2f})",
    )

    # Honest
    counts_h, _ = np.histogram(honest, bins=bins, density=True)
    ax.bar(
        bin_centers + bar_width / 2,
        counts_h,
        width=bar_width,
        color=COLOR_5,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
        label=f"Not confirmed (μ={honest.mean():.2f})",
    )

    # KDE overlays
    if len(collusive) > 2:
        kde_x = np.linspace(collusive.min(), collusive.max(), 100)
        kde_c = stats.gaussian_kde(collusive)
        ax.plot(kde_x, kde_c(kde_x), color=COLOR_1, lw=2.0, linestyle="--")

    if len(honest) > 2:
        kde_x = np.linspace(honest.min(), honest.max(), 100)
        kde_h = stats.gaussian_kde(honest)
        ax.plot(kde_x, kde_h(kde_x), color=COLOR_5, lw=2.0, linestyle="--")

    setup_ax(ax, r"ACE = ACC / $\log n$", "Density")
    ax.legend(loc="upper right", fontsize=10)


def plot_win_pte_density(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Plot Win/PTE ratio distribution by collusion status."""
    collusive = df[df["ground_truth"] == 1]["win_pte_ratio"]
    honest = df[df["ground_truth"] == 0]["win_pte_ratio"]

    # Histogram bins
    all_ratio = df["win_pte_ratio"]
    bins = np.linspace(all_ratio.min() - 0.05, min(all_ratio.max(), 2.5), 15)
    bin_width = bins[1] - bins[0]
    bar_width = bin_width / 2.5

    # Collusive
    counts_c, _ = np.histogram(collusive, bins=bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.bar(
        bin_centers - bar_width / 2,
        counts_c,
        width=bar_width,
        color=COLOR_1,
        edgecolor="black",
        linewidth=0.5,
        hatch="///",
        alpha=0.85,
        label=f"Collusive (μ={collusive.mean():.2f})",
    )

    # Honest
    counts_h, _ = np.histogram(honest, bins=bins, density=True)
    ax.bar(
        bin_centers + bar_width / 2,
        counts_h,
        width=bar_width,
        color=COLOR_5,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.85,
        label=f"Not confirmed (μ={honest.mean():.2f})",
    )

    # Reference line at 1.0 (bid = estimate)
    ax.axvline(x=1.0, color="k", linestyle="--", lw=1.0, alpha=0.4)

    # KDE overlays
    if len(collusive) > 2:
        kde_x = np.linspace(collusive.min(), min(collusive.max(), 2.5), 100)
        kde_c = stats.gaussian_kde(collusive)
        ax.plot(kde_x, kde_c(kde_x), color=COLOR_1, lw=2.0, linestyle="--")

    if len(honest) > 2:
        kde_x = np.linspace(honest.min(), honest.max(), 100)
        kde_h = stats.gaussian_kde(honest)
        ax.plot(kde_x, kde_h(kde_x), color=COLOR_5, lw=2.0, linestyle="--")

    setup_ax(ax, "Winning Bid / PTE", "Density")
    ax.legend(loc="upper right", fontsize=10)


# =============================================================================
# Main Figure Generation
# =============================================================================


def generate_collusion_figure() -> Path:
    """Generate the main collusion detection figure (4 panels, 2x2 layout)."""
    print("Loading Operation Car Wash data...")
    df = load_data()
    n_coll = (df["ground_truth"] == 1).sum()
    n_hon = (df["ground_truth"] == 0).sum()
    print(f"  Sample: {len(df)} auctions ({n_coll} collusive, {n_hon} honest)")

    print("Computing ROC curves...")
    roc_data = compute_roc_data(df)

    print("Computing component statistics...")
    comp_stats = compute_component_stats(df)
    print(comp_stats.to_string())

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_TALL)

    # Panel (a): ROC curves
    plot_roc_curves(axes[0, 0], roc_data)
    add_panel_label(axes[0, 0], "(a)")

    # Panel (b): Component comparison
    plot_component_bars(axes[0, 1], comp_stats)
    add_panel_label(axes[0, 1], "(b)")

    # Panel (c): ACE density
    plot_ace_density(axes[1, 0], df)
    add_panel_label(axes[1, 0], "(c)")

    # Panel (d): Win/PTE density
    plot_win_pte_density(axes[1, 1], df)
    add_panel_label(axes[1, 1], "(d)")

    plt.tight_layout()

    # Save
    path = save_figure(fig, "fig_collusion.png")
    print(f"Saved to {path}")

    return path


def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print key statistics for the paper."""
    print("\n" + "=" * 70)
    print("COLLUSION DETECTION: SUMMARY STATISTICS")
    print("=" * 70)

    collusive = df[df["ground_truth"] == 1]
    honest = df[df["ground_truth"] == 0]
    y_true = df["ground_truth"].values

    print(f"\nSample: {len(df)} auctions")
    print(f"  Collusive (confession): n={len(collusive)}")
    print(f"  Honest: n={len(honest)}")

    # Hypothesis tests
    print("\n--- Hypothesis Tests (Collusive vs Honest) ---")
    for metric, _direction in [
        ("ACE", "higher"),
        ("H", "lower"),
        ("G_star", "lower"),
        ("rho", "?"),
        ("win_pte_ratio", "higher"),
        ("n_bidders", "lower"),
    ]:
        t_stat, p_val = stats.ttest_ind(collusive[metric], honest[metric])
        cohens_d = (collusive[metric].mean() - honest[metric].mean()) / np.sqrt(
            (collusive[metric].var() + honest[metric].var()) / 2
        )
        print(
            f"{metric:15s}: t={t_stat:7.3f}, p={p_val:.2e}, "
            f"d={cohens_d:6.3f}, "
            f"μ_coll={collusive[metric].mean():.3f}, "
            f"μ_hon={honest[metric].mean():.3f}"
        )

    # AUC comparison
    print("\n--- AUC Comparison ---")
    aucs = {
        "Win/PTE (↑)": roc_auc_score(y_true, df["win_pte_ratio"]),
        "ACE (↑)": roc_auc_score(y_true, df["ACE"]),
        "G* (↓)": roc_auc_score(y_true, -df["G_star"]),
        "CV (↓)": roc_auc_score(y_true, -df["cv"]),
        "n (↓)": roc_auc_score(y_true, -df["n_bidders"]),
        "H (↓)": roc_auc_score(y_true, -df["H"]),
        "ACC (↓)": roc_auc_score(y_true, -df["ACC"]),
    }

    for name, auc in sorted(aucs.items(), key=lambda x: -x[1]):
        print(f"  {name:15s}: AUC = {auc:.4f}")

    # Key findings
    print("\n--- KEY FINDINGS ---")
    print(f"1. ACE achieves AUC = {aucs['ACE (↑)']:.3f} for collusion detection")
    print(f"2. Win/PTE ratio is best single predictor: AUC = {aucs['Win/PTE (↑)']:.3f}")
    print(f"3. ACC alone has poor discrimination: AUC = {aucs['ACC (↓)']:.3f}")
    print("4. The log(n) normalization in ACE rescues the signal")


def main():
    """Generate figures and print statistics."""
    df = load_data()

    print_summary_statistics(df)
    generate_collusion_figure()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
