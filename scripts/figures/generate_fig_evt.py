#!/usr/bin/env python3
"""Generate Figure 4: EVT Three-Panel Figure.

Shows ACC component scaling across EVT domains:
- Panel (a): E[ρ_n] vs n by domain
- Panel (b): E[H(p)] vs n by domain (entropy only)
- Panel (c): E[ACC]/log(n) vs n (normalized ACC, shows asymptotic constants)

Requires: Run scripts/simulations/run_evt_domains.py first.

Key insight: ACC scaling differs by EVT domain:
- Weibull (Uniform): ACC ~ log(n), ρ → 1 at rate O(1/n)
- Gumbel (Exponential): ACC ~ log(n), ρ → 1 at rate O(1/log n)
- Frechet (Pareto): ACC ~ log(n), ρ → α/(α+1) < 1
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from style import (
    DIST_COLORS,
    DIST_MARKERS,
    DIST_ORDER_EXTENDED,
    FIGSIZE_WIDE,
    save_figure,
    setup_ax,
)


def load_evt_data() -> pl.DataFrame:
    """Load EVT simulation results."""
    data_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "simulations"
        / "evt_domains.parquet"
    )
    if not data_path.exists():
        raise FileNotFoundError(
            f"EVT data not found at {data_path}.\n"
            "Run: python scripts/simulations/run_evt_domains.py"
        )
    return pl.read_parquet(data_path)


def generate_figure() -> None:
    """Generate Figure 4: EVT three-panel figure."""
    df = load_evt_data()

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)
    ax1, ax2, ax3 = axes

    # Panel (a): E[ρ] vs n
    for dist in DIST_ORDER_EXTENDED:
        data = df.filter(pl.col("distribution") == dist).sort("n_bidders")
        n_vals = data["n_bidders"].to_numpy()
        rho_vals = data["mean_rho"].to_numpy()
        se_vals = data["se_rho"].to_numpy()

        ax1.errorbar(
            n_vals,
            rho_vals,
            yerr=1.96 * se_vals,
            color=DIST_COLORS[dist],
            marker=DIST_MARKERS[dist],
            markersize=5,
            linewidth=1.2,
            capsize=2,
            label=dist.capitalize(),
        )

    setup_ax(
        ax1,
        xlabel=r"$n$ (bidders)",
        ylabel=r"$E[\rho_n]$",
        title=r"(a) Competitive ratio",
    )
    ax1.set_xscale("log")
    ax1.set_ylim(0.5, 1.02)

    # Panel (b): E[H] vs n (entropy grows ~ log n for all)
    for dist in DIST_ORDER_EXTENDED:
        data = df.filter(pl.col("distribution") == dist).sort("n_bidders")
        n_vals = data["n_bidders"].to_numpy()
        h_vals = data["mean_entropy"].to_numpy()
        se_vals = data["se_entropy"].to_numpy()

        ax2.errorbar(
            n_vals,
            h_vals,
            yerr=1.96 * se_vals,
            color=DIST_COLORS[dist],
            marker=DIST_MARKERS[dist],
            markersize=5,
            linewidth=1.2,
            capsize=2,
            label=dist.capitalize(),
        )

    setup_ax(ax2, xlabel=r"$n$ (bidders)", ylabel=r"$E[H(p)]$", title=r"(b) Entropy")
    ax2.set_xscale("log")

    # Panel (c): E[ACC]/log(n) vs n - shows convergence to asymptotic constants
    for dist in DIST_ORDER_EXTENDED:
        data = df.filter(pl.col("distribution") == dist).sort("n_bidders")
        n_vals = data["n_bidders"].to_numpy()
        acc_vals = data["mean_acc"].to_numpy()
        se_vals = data["se_acc"].to_numpy()

        # Normalize by log(n) to show asymptotic constant
        log_n = np.log(n_vals)
        acc_normalized = acc_vals / log_n
        se_normalized = se_vals / log_n

        ax3.errorbar(
            n_vals,
            acc_normalized,
            yerr=1.96 * se_normalized,
            color=DIST_COLORS[dist],
            marker=DIST_MARKERS[dist],
            markersize=5,
            linewidth=1.2,
            capsize=2,
            label=dist.capitalize(),
        )

    setup_ax(
        ax3,
        xlabel=r"$n$ (bidders)",
        ylabel=r"$E[\mathrm{ACC}]/\log n$",
        title=r"(c) Normalized ACC",
    )
    ax3.set_xscale("log")
    ax3.legend(loc="upper right", fontsize=8)

    # Adjust layout
    fig.tight_layout()

    path = save_figure(fig, "fig_evt.png")
    print(f"Saved figure to {path}")


if __name__ == "__main__":
    generate_figure()
