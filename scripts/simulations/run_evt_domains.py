#!/usr/bin/env python3
"""EVT domain simulations for ACC paper.

Generates data for:
- Figure 4: Three-panel EVT figure (ρ_n, H/G*, ACC vs n by domain)
- Table 6a: Closed-form expressions verification

Verifies Theorems 16-17:
- Weibull domain (Uniform): E[ρ] = (n-1)/n, 1-E[ρ] = O(1/n)
- Gumbel domain (Exponential, Lognormal): 1-E[ρ] = O(1/log n)
- Frechet domain (Pareto): E[ρ] → α/(α+1) < 1 (bounded away from 1)

Key insight: ACC scales as ~ log(n) in all domains, but the coefficient differs.
In particular, Fréchet tails imply ρ converges to a limit < 1, lowering the
asymptotic log-slope relative to Weibull/Gumbel cases.
"""

import polars as pl
from common import (
    DISTRIBUTIONS,
    EVT_DOMAINS,
    N_TRIALS,
    SEED,
    get_rng,
    save_results,
)

import acc_metrics as acc


def run_evt_simulation(n_trials: int = N_TRIALS, seed: int = SEED) -> pl.DataFrame:
    """Run EVT simulation across distributions and n values.

    Returns DataFrame with columns:
        distribution, domain, n_bidders, mean_*, se_*
    for each ACC component (H, rho, gini, acc, evidence).
    """
    rng = get_rng(seed)

    results = []
    n_values = [3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    for dist_name in DISTRIBUTIONS:
        for n in n_values:
            # Sample auctions
            auctions = acc.sample_auctions(rng, n_trials, n, distribution=dist_name)

            # Compute ACC components using batch function
            h, rho, gini, acc_vals, evidence = acc.acc_components(auctions)

            # Also compute information rent for SP baseline
            rent = acc.information_rent(auctions)

            results.append(
                {
                    "distribution": dist_name,
                    "domain": EVT_DOMAINS[dist_name],
                    "n_bidders": n,
                    # Entropy
                    "mean_entropy": float(h.mean()),
                    "se_entropy": float(h.std(ddof=1) / (n_trials**0.5)),
                    # Competitive ratio
                    "mean_rho": float(rho.mean()),
                    "se_rho": float(rho.std(ddof=1) / (n_trials**0.5)),
                    # Gini
                    "mean_gini": float(gini.mean()),
                    "se_gini": float(gini.std(ddof=1) / (n_trials**0.5)),
                    # ACC
                    "mean_acc": float(acc_vals.mean()),
                    "se_acc": float(acc_vals.std(ddof=1) / (n_trials**0.5)),
                    "std_acc": float(acc_vals.std()),
                    # Evidence factor
                    "mean_evidence": float(evidence.mean()),
                    "se_evidence": float(evidence.std(ddof=1) / (n_trials**0.5)),
                    # Information rent
                    "mean_rent": float(rent.mean()),
                    "se_rent": float(rent.std(ddof=1) / (n_trials**0.5)),
                }
            )

    return pl.DataFrame(results)


def main() -> None:
    """Run EVT simulation and save results."""
    print("Running EVT domain simulations...")
    print(f"  {N_TRIALS:,} trials per configuration")
    print(f"  Distributions: {DISTRIBUTIONS}")
    print("  n values: [3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]")
    print()

    df = run_evt_simulation()
    save_results(df, "evt_domains.parquet")

    # Print summary
    print("\nSummary (n=20, second-price truthful):")
    print("-" * 60)
    summary = df.filter(pl.col("n_bidders") == 20).select(
        [
            "distribution",
            "mean_rho",
            "mean_entropy",
            "mean_gini",
            "mean_acc",
        ]
    )
    print(summary)

    # Verify EVT predictions
    print("\n\nEVT Verification:")
    print("-" * 60)

    # Weibull: E[rho] = (n-1)/n
    uniform_n100 = df.filter(
        (pl.col("distribution") == "uniform") & (pl.col("n_bidders") == 100)
    )
    rho_100 = uniform_n100["mean_rho"][0]
    print(f"Uniform n=100: E[ρ] = {rho_100:.4f}, theory = 0.99")

    # Frechet: E[rho] → α/(α+1) = 4/5 for α=4
    pareto_n100 = df.filter(
        (pl.col("distribution") == "pareto") & (pl.col("n_bidders") == 100)
    )
    rho_pareto = pareto_n100["mean_rho"][0]
    print(f"Pareto(4) n=100: E[ρ] = {rho_pareto:.4f}, theory = 0.80")


if __name__ == "__main__":
    main()
