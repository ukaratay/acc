#!/usr/bin/env python3
"""Information rent and welfare simulations for ACC paper.

Generates data for:
- Figure 5: ACC vs rent/welfare
- Table 8: Predictive regressions (ACC → Revenue, ACC → Rent, ACC → Welfare)

Verifies Props 10-11 (Second-price baseline):
- Normalized rent: IR = 1 - ρ = buyer's welfare share
- Seller's welfare share: ρ = v_(2) / v_(1)
- Revenue: Rev = ρ · v_(1)

Key insight: In second-price auctions, total welfare = v_(1) is fixed
by the DGP. ACC measures how surplus is SPLIT between seller (ρ share)
and buyer (1-ρ share). Higher ACC → higher seller welfare share.
"""

import numpy as np
import polars as pl
from common import (
    DISTRIBUTIONS,
    N_TRIALS,
    SEED,
    get_rng,
    save_results,
)

import acc_metrics as acc


def compute_robust_se(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Compute HC1 (heteroskedasticity-consistent) standard errors."""
    n, k = X.shape
    residuals = y - X @ beta
    # HC1: (X' diag(e²) X) * n/(n-k), avoiding O(n²) diagonal matrix
    meat = (X.T * residuals**2) @ X * (n / (n - k))
    bread = np.linalg.solve(X.T @ X, np.eye(k))
    vcov = bread @ meat @ bread
    return np.sqrt(np.diag(vcov))


def run_rent_simulation(n_trials: int = N_TRIALS, seed: int = SEED) -> pl.DataFrame:
    """Run simulation to validate ACC-rent relationship.

    For each distribution and n, compute:
    - ACC, rent, revenue
    - Correlation ACC ↔ rent
    - Regression R² for ACC → rent
    - Regression slopes with robust standard errors
    """
    rng = get_rng(seed)

    results = []
    n_values = [5, 10, 20]

    for dist_name in DISTRIBUTIONS:
        for n in n_values:
            # Sample auctions
            auctions = acc.sample_auctions(rng, n_trials, n, distribution=dist_name)

            # Compute ACC components
            h, rho, gini, acc_vals, _ = acc.acc_components(auctions)

            # Compute rent and revenue (under SP truthful: b = v)
            # Sort each auction to get v_(1), v_(2)
            sorted_auctions = np.sort(auctions, axis=1)
            v1 = sorted_auctions[:, -1]  # Highest value
            v2 = sorted_auctions[:, -2]  # Second-highest

            # Normalized rent = (v1 - v2) / v1 = 1 - ρ
            rent_normalized = (v1 - v2) / v1
            revenue = v2

            # Correlation between ACC and (1 - rent) = ρ
            corr_acc_rho = np.corrcoef(acc_vals, rho)[0, 1]

            # R² for ACC → rent regression with robust SE
            # rent = a + b * ACC + ε
            X_rent = np.column_stack([np.ones(n_trials), acc_vals])
            beta_rent = np.linalg.lstsq(X_rent, rent_normalized, rcond=None)[0]
            y_pred_rent = X_rent @ beta_rent
            ss_res_rent = np.sum((rent_normalized - y_pred_rent) ** 2)
            ss_tot_rent = np.sum((rent_normalized - rent_normalized.mean()) ** 2)
            r2_rent = 1 - ss_res_rent / ss_tot_rent
            slope_rent = beta_rent[1]
            se_rent = compute_robust_se(X_rent, rent_normalized, beta_rent)
            se_slope_rent = se_rent[1]

            # R² for ACC → revenue (controlling for v1) with robust SE
            # revenue = a + b * ACC + c * v1 + ε
            X_rev = np.column_stack([np.ones(n_trials), acc_vals, v1])
            y = revenue
            try:
                beta_rev = np.linalg.lstsq(X_rev, y, rcond=None)[0]
                y_pred = X_rev @ beta_rev
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2_revenue = 1 - ss_res / ss_tot
                slope_revenue = beta_rev[1]  # Coefficient on ACC
                se_rev = compute_robust_se(X_rev, y, beta_rev)
                se_slope_revenue = se_rev[1]
            except np.linalg.LinAlgError:
                r2_revenue = np.nan
                slope_revenue = np.nan
                se_slope_revenue = np.nan

            results.append(
                {
                    "distribution": dist_name,
                    "n_bidders": n,
                    "mean_acc": float(acc_vals.mean()),
                    "std_acc": float(acc_vals.std()),
                    "mean_rent": float(rent_normalized.mean()),
                    "mean_revenue": float(revenue.mean()),
                    "corr_acc_rho": float(corr_acc_rho),
                    "r2_acc_rent": float(r2_rent),
                    "r2_acc_revenue": float(r2_revenue),
                    "slope_rent": float(slope_rent),
                    "se_slope_rent": float(se_slope_rent),
                    "slope_revenue": float(slope_revenue),
                    "se_slope_revenue": float(se_slope_revenue),
                }
            )

    return pl.DataFrame(results)


def run_scatter_data(
    n_trials: int = 2000, n_bidders: int = 10, seed: int = SEED
) -> pl.DataFrame:
    """Generate scatter plot data for Figure 5.

    Returns individual auction data including welfare decomposition:
    - Total welfare = v1 (winner's value)
    - Seller welfare = v2 = revenue
    - Buyer welfare = v1 - v2 = absolute rent
    - Seller share = ρ = v2/v1
    - Buyer share = 1 - ρ = normalized rent
    """
    rng = get_rng(seed)

    all_data = []
    for dist_name in DISTRIBUTIONS:
        auctions = acc.sample_auctions(rng, n_trials, n_bidders, distribution=dist_name)

        _, rho, _, acc_vals, _ = acc.acc_components(auctions)

        # Welfare decomposition
        sorted_auctions = np.sort(auctions, axis=1)
        v1 = sorted_auctions[:, -1]  # Total welfare
        v2 = sorted_auctions[:, -2]  # Seller welfare (revenue)
        absolute_rent = v1 - v2  # Buyer welfare
        normalized_rent = absolute_rent / v1  # Buyer share = 1 - ρ

        for i in range(n_trials):
            all_data.append(
                {
                    "distribution": dist_name,
                    "acc": float(acc_vals[i]),
                    "rho": float(rho[i]),
                    # Welfare decomposition
                    "total_welfare": float(v1[i]),
                    "seller_welfare": float(v2[i]),
                    "buyer_welfare": float(absolute_rent[i]),
                    "seller_share": float(rho[i]),
                    "buyer_share": float(normalized_rent[i]),
                    # Legacy names for compatibility
                    "rent": float(normalized_rent[i]),
                }
            )

    return pl.DataFrame(all_data)


def main() -> None:
    """Run information rent simulations and save results."""
    print("Running information rent simulations...")
    print(f"  {N_TRIALS:,} trials per configuration")
    print()

    # Summary statistics
    df_summary = run_rent_simulation()
    save_results(df_summary, "information_rent_summary.parquet")

    # Scatter data for figure
    df_scatter = run_scatter_data()
    save_results(df_scatter, "information_rent_scatter.parquet")

    # Print summary
    print("\nPredictive Regression Results (n=10):")
    print("-" * 60)
    summary = df_summary.filter(pl.col("n_bidders") == 10).select(
        [
            "distribution",
            "corr_acc_rho",
            "r2_acc_rent",
            "r2_acc_revenue",
        ]
    )
    print(summary)

    print("\n\nKey finding:")
    print("ACC is highly correlated with ρ (and hence 1-rent) across distributions.")
    print("R² for ACC → rent varies by distribution due to H, G* effects.")


if __name__ == "__main__":
    main()
