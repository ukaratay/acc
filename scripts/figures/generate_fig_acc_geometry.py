#!/usr/bin/env python3
"""Generate Figure 1: 2D ACC Geometry with Continuous Colors.

Shows ACC in the (ρ, G*) plane for fixed entropy H.
Two panels show points A and B at their actual entropy levels.

All values computed from bid vectors as single source of truth.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from style import CMAP_SEQ, FIGSIZE_WIDE, save_figure

import acc_metrics as acc_lib


def compute_auction_metrics(bids: list[float]) -> dict:
    """Compute all ACC metrics from a bid vector using the library."""
    bids_arr = np.array(bids, dtype=float)
    h, rho, gstar, acc_val, _ = acc_lib.acc_components(bids_arr.reshape(1, -1))
    return {
        "bids": bids,
        "n": len(bids),
        "rho": float(rho[0]),
        "H": float(h[0]),
        "gstar": float(gstar[0]),
        "acc": float(acc_val[0]),
    }


def generate_figure() -> None:
    """Generate Figure 1: 2D ACC with continuous colors."""

    # =========================================================================
    # Compute metrics from bid vectors (SINGLE SOURCE OF TRUTH)
    # =========================================================================
    auction_A = compute_auction_metrics([10, 8, 7.5, 7, 6.5])
    auction_A["label"] = "A"
    auction_A["marker"] = "o"

    auction_B = compute_auction_metrics([10, 8, 2, 1, 0.5])
    auction_B["label"] = "B"
    auction_B["marker"] = "s"

    # Print values for paper (copy these to main.tex)
    print("=" * 60)
    print("VALUES FOR PAPER (copy to main.tex Table 1 and caption):")
    print("=" * 60)
    for name, a in [("A", auction_A), ("B", auction_B)]:
        print(f"Auction {name}: bids = {a['bids']}")
        print(f"  ρ = {a['rho']:.2f}")
        print(f"  H = {a['H']:.2f}")
        print(f"  G* = {a['gstar']:.2f}")
        print(f"  ACC = {a['acc']:.2f}")
        print()
    print(f"Ratio: ACC(A)/ACC(B) = {auction_A['acc'] / auction_B['acc']:.1f}×")
    print("=" * 60)

    # =========================================================================
    # Create figure
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Grid for (ρ, G*) plane
    rho_vals = np.linspace(0.01, 0.99, 300)
    gstar_vals = np.linspace(0.01, 0.99, 300)
    RHO, GSTAR = np.meshgrid(rho_vals, gstar_vals)

    # Each panel shows one auction at its actual H
    panels = [
        (auction_A, rf"(a) Auction A: $H = {auction_A['H']:.2f}$ nats"),
        (auction_B, rf"(b) Auction B: $H = {auction_B['H']:.2f}$ nats"),
    ]

    # Normalization (consistent across panels)
    vmin, vmax = 0, 5
    norm = Normalize(vmin=vmin, vmax=vmax)

    for idx, (ax, (auction, title)) in enumerate(zip(axes, panels, strict=True)):
        H = auction["H"]

        # Compute ACC surface for this H value: ACC = H * log(1 + ρ/G*)
        log_term = np.log(1 + RHO / GSTAR)
        ACC_surface = H * log_term

        # Continuous color plot using pcolormesh
        pcm = ax.pcolormesh(
            RHO,
            GSTAR,
            ACC_surface,
            cmap=CMAP_SEQ,
            norm=norm,
            shading="auto",
            rasterized=True,
        )

        # Sparse contour lines for reference (fewer, cleaner)
        cs = ax.contour(
            RHO,
            GSTAR,
            ACC_surface,
            levels=[1, 2, 3, 4],
            colors="white",
            linewidths=1.0,
            linestyles="-",
            alpha=0.7,
        )
        ax.clabel(cs, inline=True, fontsize=9, fmt="%.0f")

        # Plot the auction point at its actual location
        ax.scatter(
            [auction["rho"]],
            [auction["gstar"]],
            c="white",
            s=140,
            marker=auction["marker"],
            edgecolors="black",
            linewidths=2,
            zorder=10,
        )

        # Label with ACC value (using pre-computed exact value)
        ax.annotate(
            f"{auction['label']}\nACC = {auction['acc']:.2f}",
            xy=(auction["rho"], auction["gstar"]),
            xytext=(auction["rho"] - 0.30, auction["gstar"] + 0.22),
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="gray"
            ),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        )

        # Draw crosshairs at the auction location
        ax.axvline(
            x=auction["rho"], color="white", linestyle="--", linewidth=1, alpha=0.5
        )
        ax.axhline(
            y=auction["gstar"], color="white", linestyle="--", linewidth=1, alpha=0.5
        )

        # Axis labels and title
        ax.set_xlabel(r"$\rho$ (competitive ratio)", fontsize=11)
        if idx == 0:
            ax.set_ylabel(r"$G^*$ (normalized Gini)", fontsize=11)
        ax.set_title(title, fontsize=12)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

    # =========================================================================
    # Shared colorbar
    # =========================================================================
    fig.subplots_adjust(right=0.85, bottom=0.12, wspace=0.15)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.025, 0.72])
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    cbar.set_label("ACC", fontsize=12)
    cbar.set_ticks([0, 1, 2, 3, 4, 5])

    path = save_figure(fig, "fig_acc_geometry.png")
    print(f"\nSaved figure to {path}")


if __name__ == "__main__":
    generate_figure()
