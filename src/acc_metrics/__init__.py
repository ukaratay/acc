"""Auction Competitive Capacity (ACC) - Measuring competitive pressure in auctions.

The ACC formula is: ACC = H(p) · log(1 + r)  where r = ρ / G*

where:
    H(p) = Shannon entropy of bid shares p_i = b_i / Σb_j
    ρ = b_(2) / b_(1) (competitive ratio)
    G* = normalized Gini coefficient (dispersion measure)
    r = ρ / G* (adjusted ratio)

The evidence-factor index is: E(b) = (1 + r)^{H(p)}

Key properties:
    - ACC = log(E): Evidence factors multiply, so ACC values add
    - Concave in r: Diminishing marginal returns to competition
    - Under LP assumption: ACC governs sample complexity for discrimination

Higher ACC indicates stronger competitive pressure (lower information rent).

Example usage::

    import acc_metrics as acc
    from numpy.random import default_rng

    rng = default_rng(42)

    # Generate synthetic auction data
    bids = acc.sample_auctions(rng, n_auctions=10_000, n_bidders=10)

    # Compute all metrics at once (entropy, rho, gini, acc_vals, evidence)
    h, rho, gini, acc_vals, evidence = acc.acc_components(bids)

    # Effective number of bidders
    n_eff = acc.effective_bidders(bids)

    # Bootstrap confidence interval
    ci_low, ci_high = acc.bootstrap_bca_ci(acc_vals)
"""

# Core metrics
from acc_metrics.core import (
    acc,
    acc_components,
    acc_delta,
    competitive_ratio,
    dispersion_floor,
    effective_bidders,
    entropy,
    evidence_factor,
    information_rent,
    normalized_gini,
    saturation_flag,
)

# Dispersion measures
from acc_metrics.dispersion import (
    MEASURES,
    coefficient_of_variation,
    gini,
    median_absolute_deviation,
    normalized_range,
    quartile_variation,
)

# First-price auction equilibrium
from acc_metrics.equilibrium import (
    first_price_bids_exponential,
    first_price_bids_lognormal,
    first_price_bids_pareto,
    first_price_bids_uniform,
)

# Statistical inference
from acc_metrics.inference import (
    bootstrap_bca_ci,
    bootstrap_percentile_ci,
    cluster_robust_ci,
    cluster_robust_se,
    monte_carlo_ci,
    monte_carlo_se,
    normal_ci,
    wild_cluster_bootstrap_ci,
)

# Samplers (data generation)
from acc_metrics.samplers import (
    COPULAS,
    DISTRIBUTIONS,
    sample_auctions,
    sample_beta,
    sample_clayton_copula,
    sample_exponential,
    sample_frank_copula,
    sample_gamma,
    sample_gaussian_copula,
    sample_gumbel_copula,
    sample_lognormal,
    sample_pareto,
    sample_uniform,
    sample_weibull,
)

__all__ = [
    # Core metrics
    "acc",
    "acc_components",
    "acc_delta",
    "dispersion_floor",
    "saturation_flag",
    "evidence_factor",
    "competitive_ratio",
    "effective_bidders",
    "entropy",
    "information_rent",
    "normalized_gini",
    # Dispersion measures
    "gini",
    "median_absolute_deviation",
    "normalized_range",
    "coefficient_of_variation",
    "quartile_variation",
    "MEASURES",
    # Samplers
    "sample_uniform",
    "sample_exponential",
    "sample_lognormal",
    "sample_pareto",
    "sample_beta",
    "sample_weibull",
    "sample_gamma",
    "sample_auctions",
    "sample_gaussian_copula",
    "sample_clayton_copula",
    "sample_gumbel_copula",
    "sample_frank_copula",
    "DISTRIBUTIONS",
    "COPULAS",
    # Equilibrium
    "first_price_bids_uniform",
    "first_price_bids_exponential",
    "first_price_bids_lognormal",
    "first_price_bids_pareto",
    # Inference
    "bootstrap_percentile_ci",
    "bootstrap_bca_ci",
    "cluster_robust_se",
    "cluster_robust_ci",
    "wild_cluster_bootstrap_ci",
    "normal_ci",
    "monte_carlo_se",
    "monte_carlo_ci",
]

__version__ = "0.2.0"
