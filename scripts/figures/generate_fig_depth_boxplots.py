#!/usr/bin/env python3
"""Generate Figure D.2: Depth curve boxplots across DGPs.

Shows boxplots of depth curve values D(k) = ACC(b^{(k)}) for different
truncation levels k, across distributions.

Key insight: Depth curves reveal competitive structure:
- "Deep" markets: Rising ACC as k increases (competition beyond top-2)
- "Shallow" markets: Flat ACC after k=2 (only top-2 matter)
"""

import matplotlib.pyplot as plt
import numpy as np
from style import (
    DIST_COLORS,
    DIST_MARKERS,
    DIST_ORDER_EXTENDED,
    FIGSIZE_WIDE,
    get_cubehelix_colors,
    save_figure,
    setup_ax,
)

from acc_metrics import acc


def compute_depth_curves_batch(values: np.ndarray) -> np.ndarray:
    """Compute depth curves D(k) = ACC(b^{(k)}) for k=2,...,n in batch.

    Pre-sorts all auctions once, then computes ACC per truncation level
    using the library's batch interface.

    Args:
        values: 2D array of shape (n_auctions, n_bidders).

    Returns:
        Array of shape (n_auctions, n_bidders-1) with depth curve values.
    """
    n_auctions, n_bidders = values.shape
    sorted_desc = np.sort(values, axis=1)[:, ::-1]

    depth_curves = np.zeros((n_auctions, n_bidders - 1))
    for k in range(2, n_bidders + 1):
        batch = sorted_desc[:, :k]
        depth_curves[:, k - 2] = acc(batch)

    return depth_curves


def simulate_depth_curves(
    distribution: str,
    n_bidders: int = 10,
    n_auctions: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Simulate depth curves for multiple auctions.

    Args:
        distribution: One of 'uniform', 'exponential', 'lognormal', 'pareto'.
        n_bidders: Number of bidders.
        n_auctions: Number of Monte Carlo samples.
        seed: Random seed.

    Returns:
        Array of shape (n_auctions, n_bidders-1) with depth curve values.
    """
    # Avoid Python's built-in hash() for seeding: it is randomized across runs unless
    # PYTHONHASHSEED is fixed, which breaks reproducibility of paper figures.
    seed_offsets = {
        "uniform": 0,
        "exponential": 1,
        "lognormal": 2,
        "pareto": 3,
    }
    if distribution not in seed_offsets:
        raise ValueError(f"Unknown distribution: {distribution}")
    rng = np.random.default_rng(seed + 10_000 * seed_offsets[distribution])

    # Generate values based on distribution
    if distribution == "uniform":
        values = rng.uniform(0, 1, (n_auctions, n_bidders))
    elif distribution == "exponential":
        values = rng.exponential(1, (n_auctions, n_bidders))
    elif distribution == "lognormal":
        values = rng.lognormal(0, 1, (n_auctions, n_bidders))
    elif distribution == "pareto":
        values = rng.pareto(4, (n_auctions, n_bidders)) + 1
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return compute_depth_curves_batch(values)


def generate_figure() -> None:
    """Generate Figure D.2: Depth curve boxplots."""
    n_bidders = 10
    n_auctions = 1000

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    ax1, ax2 = axes

    # Panel (a): Boxplots of D(k) for each k, one distribution (Uniform)
    depth_uniform = simulate_depth_curves("uniform", n_bidders, n_auctions)

    k_values = list(range(2, n_bidders + 1))
    colors = get_cubehelix_colors(len(k_values))

    bp1 = ax1.boxplot(
        [depth_uniform[:, k - 2] for k in k_values],
        labels=[str(k) for k in k_values],
        patch_artist=True,
    )

    for i, patch in enumerate(bp1["boxes"]):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.6)

    setup_ax(
        ax1,
        xlabel=r"$k$ (top-$k$ truncation)",
        ylabel=r"$D(k) = \mathrm{ACC}(b^{(k)})$",
        title=r"(a) Depth curve: Uniform",
    )

    # Panel (b): Mean depth curves across distributions
    # Cache to avoid re-simulating distributions already computed
    depth_cache = {"uniform": depth_uniform}

    for dist in DIST_ORDER_EXTENDED:
        if dist in depth_cache:
            depth_curves = depth_cache[dist]
        else:
            depth_curves = simulate_depth_curves(dist, n_bidders, n_auctions)
            depth_cache[dist] = depth_curves

        # Mean and SE for each k
        mean_depth = np.mean(depth_curves, axis=0)
        se_depth = np.std(depth_curves, axis=0) / np.sqrt(n_auctions)

        ax2.errorbar(
            k_values,
            mean_depth,
            yerr=1.96 * se_depth,
            color=DIST_COLORS[dist],
            marker=DIST_MARKERS[dist],
            markersize=4,
            linewidth=1.2,
            capsize=2,
            label=dist.capitalize(),
        )

    setup_ax(
        ax2,
        xlabel=r"$k$ (top-$k$ truncation)",
        ylabel=r"$E[D(k)]$",
        title=r"(b) Mean depth curves by distribution",
    )
    ax2.legend(loc="lower right", fontsize=8)

    fig.tight_layout()

    path = save_figure(fig, "fig_depth_boxplots.png")
    print(f"Saved figure to {path}")


if __name__ == "__main__":
    generate_figure()
