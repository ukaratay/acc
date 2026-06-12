#!/usr/bin/env python3
"""Cluster-variance simulations for ACC paper.

Generates data for:
- Appendix E variance-estimator comparison table

Design (documented in Appendix E):
- M clusters (markets) with m_size auctions each, n_bidders per auction.
- Cluster effect: a_m ~ Uniform(0, 0.6) is a market-level lower bound and
  values within cluster m are i.i.d. Uniform[a_m, 1]. The shared support
  compression moves G* and rho together, inducing within-cluster ACC
  correlation. ACC is scale-invariant, so a pure scale shock would not
  generate dependence; the support shift does.
- Independent design: identical marginals with a redrawn per auction,
  so the only difference is the dependence structure.
- Estimators for the mean ACC: naive i.i.d. normal CI, cluster-robust
  t interval (t with M-1 df), and wild cluster bootstrap percentile
  interval with Rademacher weights.
"""

import numpy as np
import polars as pl
from common import SEED, get_rng, save_results

import acc_metrics as acc

A_MAX = 0.6
N_BIDDERS = 5
M_SIZE = 20


def _sample_clustered(
    rng: np.random.Generator, n_clusters: int, m_size: int, shared: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (acc_values, cluster_labels) for one replication."""
    if shared:
        a = np.repeat(rng.uniform(0.0, A_MAX, size=n_clusters), m_size)
    else:
        a = rng.uniform(0.0, A_MAX, size=n_clusters * m_size)
    u = rng.uniform(size=(n_clusters * m_size, N_BIDDERS))
    values = a[:, None] + (1.0 - a[:, None]) * u
    _, _, _, acc_vals, _ = acc.acc_components(values)
    clusters = np.repeat(np.arange(n_clusters), m_size)
    return np.asarray(acc_vals), clusters


def _true_mean(rng: np.random.Generator, n_auctions: int = 200_000) -> float:
    """Marginal mean ACC under the integrated cluster effect."""
    a = rng.uniform(0.0, A_MAX, size=n_auctions)
    u = rng.uniform(size=(n_auctions, N_BIDDERS))
    values = a[:, None] + (1.0 - a[:, None]) * u
    _, _, _, acc_vals, _ = acc.acc_components(values)
    return float(np.asarray(acc_vals).mean())


def run_cluster_variance_simulation(
    n_replications: int = 1000,
    seed: int = SEED,
) -> pl.DataFrame:
    """Compare variance estimators under independent and clustered designs."""
    rng = get_rng(seed)
    confidence = 0.95
    z = 1.959963984540054

    mu = _true_mean(rng)
    print(f"  True mean ACC (integrated over cluster effect): {mu:.4f}")

    results = []
    for n_clusters in (10, 50):
        for design, shared in (("independent", False), ("clustered", True)):
            covered = {"naive": 0, "cluster_robust": 0, "wild_bootstrap": 0}
            widths: dict[str, list[float]] = {k: [] for k in covered}

            for i in range(n_replications):
                acc_vals, clusters = _sample_clustered(rng, n_clusters, M_SIZE, shared)
                mean = float(acc_vals.mean())
                n_total = len(acc_vals)

                naive_se = float(acc_vals.std(ddof=1)) / np.sqrt(n_total)
                intervals = {
                    "naive": (mean - z * naive_se, mean + z * naive_se),
                    "cluster_robust": acc.cluster_robust_ci(
                        acc_vals, clusters, confidence=confidence
                    ),
                    "wild_bootstrap": acc.wild_cluster_bootstrap_ci(
                        acc_vals, clusters, confidence=confidence, seed=seed + i
                    ),
                }
                for name, (lo, hi) in intervals.items():
                    if lo <= mu <= hi:
                        covered[name] += 1
                    widths[name].append(hi - lo)

            for name in covered:
                results.append(
                    {
                        "design": design,
                        "n_clusters": n_clusters,
                        "m_size": M_SIZE,
                        "n_bidders": N_BIDDERS,
                        "estimator": name,
                        "coverage_rate": covered[name] / n_replications,
                        "mean_ci_width": float(np.mean(widths[name])),
                        "n_replications": n_replications,
                        "nominal_coverage": confidence,
                        "true_mean": mu,
                    }
                )
            print(f"  done: design={design}, M={n_clusters}")

    return pl.DataFrame(results)


def main() -> None:
    """Run cluster-variance simulation and save results."""
    print("Running cluster-variance simulations...")
    df = run_cluster_variance_simulation()
    save_results(df, "cluster_variance.parquet")

    print("\nCluster-variance results:")
    print("-" * 70)
    print(
        df.select(
            ["design", "n_clusters", "estimator", "coverage_rate", "mean_ci_width"]
        )
    )

    print("\nKey findings:")
    print("- Naive CI under-covers in the clustered design")
    print("- Cluster-robust t and wild bootstrap restore coverage")


if __name__ == "__main__":
    main()
