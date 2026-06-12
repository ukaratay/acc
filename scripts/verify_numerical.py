#!/usr/bin/env python3
"""
Numerical verification of ACC theorems using Monte Carlo simulation.

Each function returns True on success, raises AssertionError on failure.
Uses the acc_metrics package for Numba-optimized batch computations.

Theorem mapping to paper:
- Prop 12: ACC is concave in r (diminishing marginal returns)
- Prop 7: Evidence factors multiply, ACC values add
- Props 10-11: E[rho] increases with n (ordinal ranking)
- EVT: Convergence rates by domain (Weibull/Gumbel/Fréchet)
- Thm 18: Sample mean ACC is asymptotically normal (CLT)
- Coverage: 95% CI covers 95% of the time
"""

import numpy as np
from scipy import stats

import acc_metrics as acc
from acc_metrics.samplers import Sampler
from acc_metrics.samplers import sample_uniform as uniform_sampler

# =============================================================================
# Core ACC Verification Functions
# =============================================================================


def verify_acc_monotonic_in_n(
    seed: int = 42,
    n_trials: int = 10000,
) -> bool:
    """
    Verify: E[ACC] increases with n (more bidders -> more competition).

    As n increases:
    - E[ρ] → 1 (closer bids)
    - E[H] → log(n) (higher effective participation)
    - Combined effect: E[ACC] increases
    """
    rng = np.random.default_rng(seed)
    prev_mean_acc = 0.0

    for n in [3, 5, 10, 20]:
        # Use acc_metrics batch sampler
        auctions = acc.sample_auctions(rng, n_trials, n)
        _, _, _, acc_vals, _ = acc.acc_components(auctions)
        mean_acc = np.mean(acc_vals)

        assert mean_acc > prev_mean_acc, (
            f"E[ACC] should increase with n: E[ACC|n={n}]={mean_acc:.4f} "
            f"<= E[ACC|n=prev]={prev_mean_acc:.4f}"
        )
        prev_mean_acc = mean_acc

    return True


def verify_acc_concavity_numerical() -> bool:
    """
    Prop 12: ACC is concave in r (diminishing marginal returns).

    For fixed H, the marginal gain from increasing r decreases:
    d²[log(1+r)]/dr² = -1/(1+r)² < 0

    We verify by computing the ACC gain at different r levels.
    """
    # Test on a grid of r values (analytical, not Monte Carlo)
    r_vals = np.linspace(0.1, 10, 100)
    log_1_plus_r = np.log(1 + r_vals)

    # First derivative: should be positive and decreasing
    dr = r_vals[1] - r_vals[0]
    first_deriv = np.diff(log_1_plus_r) / dr  # ≈ 1/(1+r)

    # Second derivative: should be negative
    second_deriv = np.diff(first_deriv) / dr  # ≈ -1/(1+r)²

    # All second derivatives should be negative
    assert np.all(second_deriv < 0), (
        "log(1+r) should be concave: some second derivatives are positive"
    )

    # Also verify the analytical form
    r_mid = (r_vals[:-2] + r_vals[1:-1]) / 2
    expected_d2 = -1 / (1 + r_mid) ** 2

    # The numerical derivative should match the analytical one
    # (with finite difference error)
    rel_error = np.abs(second_deriv - expected_d2) / np.abs(expected_d2)
    mean_err = np.mean(rel_error)
    assert mean_err < 0.05, f"d²/dr² should match -1/(1+r)²: error = {mean_err:.4f}"

    return True


def verify_prop7_additivity(
    seed: int = 42,
    n_auctions: int = 100,
    n_replications: int = 500,
) -> bool:
    """
    Prop 7: Evidence factors multiply, so ACC values add.

    For T auctions: log(∏ E_t) = Σ ACC_t
    """
    rng = np.random.default_rng(seed)

    for _ in range(n_replications):
        # Generate batch of auctions
        auctions = acc.sample_auctions(rng, n_auctions, n_bidders=10)
        _, _, _, acc_vals, evidence = acc.acc_components(auctions)

        # Sum of ACC values
        sum_acc = np.sum(acc_vals)

        # Log of product of evidence factors
        log_prod_evidence = np.sum(np.log(evidence))

        # These should be equal (up to numerical precision)
        rel_error = (
            abs(sum_acc - log_prod_evidence) / abs(sum_acc) if sum_acc != 0 else 0
        )

        assert rel_error < 1e-10, (
            f"Additivity violated: {sum_acc:.6f} vs {log_prod_evidence:.6f}"
        )

    return True


def verify_rho_increases_with_n(
    sampler: Sampler = uniform_sampler,
    seed: int = 42,
    n_trials: int = 10000,
) -> bool:
    """
    Props 10-11: E[ρ] increases with n (closer bids with more competition).

    As n increases, the second-highest bid gets closer to the highest bid.
    """
    rng = np.random.default_rng(seed)
    prev_mean_rho = 0.0

    for n in [2, 5, 10, 20]:
        rhos = []
        for _ in range(n_trials):
            values = sampler(rng, n)
            sorted_vals = np.sort(values)
            rho = sorted_vals[-2] / sorted_vals[-1]
            rhos.append(rho)

        mean_rho = np.mean(rhos)
        assert mean_rho > prev_mean_rho, (
            f"E[ρ] should increase with n: E[ρ|n={n}]={mean_rho:.4f} "
            f"<= E[ρ|n=prev]={prev_mean_rho:.4f}"
        )
        prev_mean_rho = mean_rho

    return True


# =============================================================================
# EVT Domain Verification
# =============================================================================


def verify_evt_uniform(seed: int = 42, n_trials: int = 50000) -> bool:
    """
    EVT Uniform (Weibull domain): 1 - E[ρ] = O(1/n).

    For Uniform[0,1], E[ρ] = (n-1)/n, so convergence rate is 1/n.
    """
    rng = np.random.default_rng(seed)

    for n in [5, 10, 20, 50]:
        rhos = []
        for _ in range(n_trials):
            values = rng.uniform(0, 1, n)
            sorted_vals = np.sort(values)
            rhos.append(sorted_vals[-2] / sorted_vals[-1])

        mean_rho = np.mean(rhos)
        expected_rho = (n - 1) / n
        rel_error = abs(mean_rho - expected_rho) / expected_rho

        assert rel_error < 0.02, (
            f"Uniform EVT: E[ρ|n={n}]={mean_rho:.4f}, expected={(n - 1) / n:.4f}, "
            f"rel error={rel_error:.4f}"
        )

    return True


def verify_evt_exponential(seed: int = 42, n_trials: int = 50000) -> bool:
    """
    EVT Exponential (Gumbel domain): 1 - E[ρ] = O(1/log(n)).

    For Exponential distribution, convergence is logarithmic.
    """
    rng = np.random.default_rng(seed)

    prev_one_minus_rho = 1.0
    for n in [5, 10, 20, 50]:
        rhos = []
        for _ in range(n_trials):
            values = rng.exponential(1.0, n)
            sorted_vals = np.sort(values)
            rhos.append(sorted_vals[-2] / sorted_vals[-1])

        mean_rho = np.mean(rhos)
        one_minus_rho = 1 - mean_rho

        # Should decrease with n
        assert one_minus_rho < prev_one_minus_rho, (
            f"Exponential EVT: 1-E[ρ] should decrease with n: "
            f"n={n}, 1-ρ={one_minus_rho:.4f} >= prev={prev_one_minus_rho:.4f}"
        )
        prev_one_minus_rho = one_minus_rho

    return True


def verify_evt_pareto(seed: int = 42, n_trials: int = 50000) -> bool:
    """
    EVT Pareto (Fréchet domain): E[ρ] converges to a constant < 1.

    For heavy-tailed distributions, the ratio remains bounded away from 1.
    Unlike Weibull/Gumbel, it doesn't converge to 1.
    """
    rng = np.random.default_rng(seed)

    rho_by_n = {}
    for n in [5, 10, 20, 50, 100]:
        rhos = []
        for _ in range(n_trials):
            values = rng.pareto(4, n) + 1  # Pareto with α=4
            sorted_vals = np.sort(values)
            rhos.append(sorted_vals[-2] / sorted_vals[-1])

        rho_by_n[n] = np.mean(rhos)

    # For Pareto/Fréchet domain, E[ρ] should be bounded away from 1
    # Even at n=100, E[ρ] should be noticeably less than 1
    assert rho_by_n[100] < 0.95, (
        f"Pareto EVT: E[ρ] should stay away from 1: got {rho_by_n[100]:.4f} at n=100"
    )

    # Compare with uniform at same n: Pareto should have lower rho
    uniform_rho_100 = 0.99  # (100-1)/100 = 0.99
    assert rho_by_n[100] < uniform_rho_100 - 0.05, (
        f"Pareto EVT: E[ρ] should be much lower than uniform: "
        f"Pareto={rho_by_n[100]:.4f}, Uniform≈{uniform_rho_100:.2f}"
    )

    return True


def verify_pareto_beta_exact(seed: int = 42, n_trials: int = 50000) -> bool:
    """
    Theorem (EVT benchmarks, part iii): for Pareto(α), ρ_n ~ Beta(α, 1)
    EXACTLY at every n >= 2, so E[ρ_n] = α/(α+1) at all sample sizes,
    not only in the limit.
    """
    rng = np.random.default_rng(seed)
    alpha = 4.0
    expected_mean = alpha / (alpha + 1.0)

    for n in [2, 5, 50]:
        values = rng.pareto(alpha, size=(n_trials, n)) + 1.0
        values.sort(axis=1)
        rhos = values[:, -2] / values[:, -1]

        # Exact mean at every n (5-sigma Monte Carlo band)
        se = rhos.std() / np.sqrt(n_trials)
        assert abs(rhos.mean() - expected_mean) < 5 * se, (
            f"Pareto Beta exactness: E[ρ_{n}] = {rhos.mean():.4f}, "
            f"expected {expected_mean:.4f} ± {5 * se:.4f}"
        )

        # Distributional check against Beta(α, 1)
        ks = stats.kstest(rhos, "beta", args=(alpha, 1.0))
        assert ks.pvalue > 1e-4, (
            f"Pareto Beta exactness: KS test rejects Beta({alpha},1) at n={n} "
            f"(p = {ks.pvalue:.2e})"
        )

    return True


def verify_entropy_drift_constants(seed: int = 42) -> bool:
    """
    Lemma (entropy drift): H(p_n) - log n -> log μ - E[V log V]/μ a.s.

    Constants stated in the paper's Uniform-baseline example:
    U[0,1]: log(1/2) + 1/2 ≈ -0.1931; Exponential(1): γ - 1 ≈ -0.4228.
    """
    rng = np.random.default_rng(seed)
    n, reps = 2000, 300

    targets = {
        "uniform": (lambda size: rng.uniform(size=size), np.log(0.5) + 0.5),
        "exponential": (
            lambda size: rng.exponential(size=size),
            np.euler_gamma - 1.0,
        ),
    }
    for name, (draw, target) in targets.items():
        drifts = np.empty(reps)
        for i in range(reps):
            v = draw((n,))
            p = v / v.sum()
            drifts[i] = -(p * np.log(p)).sum() - np.log(n)
        assert abs(drifts.mean() - target) < 0.01, (
            f"Entropy drift ({name}): simulated {drifts.mean():.4f}, "
            f"analytic {target:.4f}"
        )

    return True


# =============================================================================
# CLT and Inference Verification
# =============================================================================


def verify_clt_numerical(
    seed: int = 42,
    n_bidders: int = 10,
    n_auctions: int = 1000,
    n_replications: int = 500,
) -> bool:
    """
    Thm 18: Sample mean ACC is asymptotically normal.

    For T >= 1000 auctions, the distribution of ACC_bar across replications
    should pass a normality test (D'Agostino-Pearson).
    """
    rng = np.random.default_rng(seed)

    acc_means = []
    for _ in range(n_replications):
        auctions = acc.sample_auctions(rng, n_auctions, n_bidders)
        _, _, _, acc_vals, _ = acc.acc_components(auctions)
        acc_means.append(np.mean(acc_vals))

    # D'Agostino-Pearson test for normality
    _, p_value = stats.normaltest(acc_means)

    # p > 0.005 means we don't reject normality at 0.5% level
    assert p_value > 0.005, (
        f"ACC sample means should be approximately normal: "
        f"D'Agostino-Pearson p-value={p_value:.4f} < 0.005"
    )

    return True


def verify_bootstrap_coverage(
    seed: int = 42,
    n_bidders: int = 10,
    n_auctions: int = 200,
    n_replications: int = 200,
    n_bootstrap: int = 500,
    confidence: float = 0.95,
) -> bool:
    """
    Coverage: 95% BCa CI should cover the true mean ~95% of the time.
    """
    rng = np.random.default_rng(seed)

    # First, estimate the "true" mean from a large sample
    large_sample = acc.sample_auctions(rng, 100000, n_bidders)
    _, _, _, acc_large, _ = acc.acc_components(large_sample)
    true_mean = np.mean(acc_large)

    covered = 0
    for i in range(n_replications):
        auctions = acc.sample_auctions(rng, n_auctions, n_bidders)
        _, _, _, acc_vals, _ = acc.acc_components(auctions)

        # Use seed parameter instead of rng
        ci_low, ci_high = acc.bootstrap_bca_ci(
            acc_vals, n_bootstrap=n_bootstrap, confidence=confidence, seed=seed + i
        )

        if ci_low <= true_mean <= ci_high:
            covered += 1

    coverage_rate = covered / n_replications

    # Coverage should be within a reasonable range
    tolerance = 0.08  # Allow some deviation due to finite samples
    assert abs(coverage_rate - confidence) < tolerance, (
        f"Coverage should be ~{confidence:.0%}: got {coverage_rate:.2%}"
    )

    return True


def verify_affiliation_increases_acc(
    seed: int = 42,
    n_bidders: int = 5,
    n_trials: int = 5000,
) -> bool:
    """
    Affiliated values should have higher E[ACC] than IPV.

    Affiliation increases E[v2|v1], so ρ is higher, and ACC is higher.
    """
    rng = np.random.default_rng(seed)

    # IPV case: independent uniform draws
    ipv_accs = []
    for _ in range(n_trials):
        values = rng.uniform(0, 1, n_bidders).reshape(1, -1)
        _, _, _, acc_val, _ = acc.acc_components(values)
        ipv_accs.append(acc_val[0])
    ipv_mean_acc = np.mean(ipv_accs)

    # Affiliated case: Gaussian copula with positive correlation
    corr = 0.5
    aff_accs = []
    for _ in range(n_trials):
        cov_matrix = np.full((n_bidders, n_bidders), corr)
        np.fill_diagonal(cov_matrix, 1.0)
        normals = rng.multivariate_normal(np.zeros(n_bidders), cov_matrix)
        values = stats.norm.cdf(normals).reshape(1, -1)
        _, _, _, acc_val, _ = acc.acc_components(values)
        aff_accs.append(acc_val[0])
    aff_mean_acc = np.mean(aff_accs)

    # Affiliation should increase ACC (allow small tolerance for MC noise)
    tolerance = -0.02
    assert aff_mean_acc >= ipv_mean_acc + tolerance, (
        f"Affiliated ACC should be >= IPV ACC: "
        f"affiliated={aff_mean_acc:.4f}, ipv={ipv_mean_acc:.4f}"
    )

    return True


# =============================================================================
# Rényi Representation (Appendix B, Lemma)
# =============================================================================


def verify_renyi_representation(seed: int = 42, n_trials: int = 50000) -> bool:
    """
    Line 1048: Order statistics of Uniform(0,1) equal cumulative exponential ratios.

    (U_(1),...,U_(n)) =^d (E_1/S_{n+1}, (E_1+E_2)/S_{n+1}, ..., S_n/S_{n+1})

    where E_i ~ Exp(1) i.i.d. and S_k = E_1+...+E_k.
    Verified via two-sample KS test on each order statistic.
    """
    rng = np.random.default_rng(seed)

    for n in [5, 10]:
        # Method 1: sort uniform draws
        uniform_order_stats = np.sort(rng.uniform(0, 1, (n_trials, n)), axis=1)

        # Method 2: exponential ratio construction
        exponentials = rng.exponential(1.0, (n_trials, n + 1))
        cumsum = np.cumsum(exponentials, axis=1)
        S_n1 = cumsum[:, -1:]  # S_{n+1}
        renyi_order_stats = cumsum[:, :n] / S_n1

        # KS test on each order statistic
        for k in range(n):
            ks_stat, p_value = stats.ks_2samp(
                uniform_order_stats[:, k], renyi_order_stats[:, k]
            )
            assert p_value > 0.001, (
                f"Rényi representation failed for n={n}, k={k + 1}: "
                f"KS stat={ks_stat:.4f}, p={p_value:.6f}"
            )

    return True


# =============================================================================
# Spacing Ratio ~ Beta(n-1, 1) (Appendix B)
# =============================================================================


def verify_spacing_ratio_beta(seed: int = 42, n_trials: int = 100000) -> bool:
    """
    Line 1067: U_(n-1)/U_(n) ~ Beta(n-1, 1) for i.i.d. Uniform(0,1).

    Equivalently, S_{n-1}/(S_{n-1} + E_n) ~ Beta(n-1, 1).
    Verified via KS test against Beta(n-1, 1) CDF.
    """
    rng = np.random.default_rng(seed)

    for n in [3, 5, 10, 20]:
        # Generate uniform order statistics and compute ratio
        samples = np.sort(rng.uniform(0, 1, (n_trials, n)), axis=1)
        ratios = samples[:, -2] / samples[:, -1]  # U_(n-1)/U_(n)

        # KS test against Beta(n-1, 1)
        ks_stat, p_value = stats.kstest(ratios, stats.beta(n - 1, 1).cdf)
        assert p_value > 0.001, (
            f"Spacing ratio Beta test failed for n={n}: "
            f"KS={ks_stat:.4f}, p={p_value:.6f}"
        )

        # Also verify the mean: E[Beta(n-1,1)] = (n-1)/n
        expected_mean = (n - 1) / n
        sample_mean = np.mean(ratios)
        rel_error = abs(sample_mean - expected_mean) / expected_mean
        assert rel_error < 0.01, (
            f"Spacing ratio mean for n={n}: {sample_mean:.4f} vs {expected_mean:.4f}"
        )

    return True


# =============================================================================
# Gumbel Domain: E[max] = H_n (Appendix B)
# =============================================================================


def verify_gumbel_harmonic(seed: int = 42, n_trials: int = 100000) -> bool:
    """
    Line 1095: E[V_(n)] = H_n ~ log(n) for i.i.d. Exp(1).

    The expected maximum of n i.i.d. standard exponential RVs is the
    n-th harmonic number H_n = 1 + 1/2 + ... + 1/n.
    """
    rng = np.random.default_rng(seed)

    for n in [5, 10, 20, 50]:
        maxima = np.max(rng.exponential(1.0, (n_trials, n)), axis=1)
        sample_mean = np.mean(maxima)
        H_n = sum(1.0 / k for k in range(1, n + 1))

        rel_error = abs(sample_mean - H_n) / H_n
        assert rel_error < 0.01, (
            f"E[max Exp(1)] for n={n}: {sample_mean:.4f} vs H_{n}={H_n:.4f}, "
            f"rel_error={rel_error:.4f}"
        )

    return True


# =============================================================================
# KL Link Under LP (Proposition, §4)
# =============================================================================


def verify_kl_link_under_lp(seed: int = 42, n_trials: int = 50000) -> bool:
    """
    Line 609: Under LP, D_KL(p_θ₀ || p_θ₁) = κ · E_θ₀[ACC] - Λ.

    LP is an ASSUMPTION, not a theorem. We verify two properties:
    1. Structural: log-LR and ACC are positively correlated (LP predicts this).
    2. Algebraic: when we FIT the affine model, predicted KL matches empirical KL
       (this tests E[κ·ACC - Λ] = κ·E[ACC] - Λ, i.e., linearity of expectation).

    LP is approximate for generic distribution pairs, so we use a moderate
    correlation threshold (r > 0.15, which is highly significant at n=50k).
    """
    rng = np.random.default_rng(seed)
    n_bidders = 5

    # Generate samples under θ₀ (uniform)
    auctions_0 = rng.uniform(0, 1, (n_trials, n_bidders))
    _, _, _, acc_vals_0, _ = acc.acc_components(auctions_0)

    # Compute log-likelihood ratio (vectorized)
    # p(b|θ₀) = 1 (uniform), p(b|θ₁) = ∏ Beta(2,5) pdf
    log_lr = np.sum(stats.beta.logpdf(auctions_0, 2, 5), axis=1)

    # Under LP: log(p₀/p₁) = κ·ACC - Λ, so -log_lr = κ·ACC - Λ
    neg_log_lr = -log_lr

    # Fit affine model: neg_log_lr = κ·ACC + intercept
    coeffs = np.polynomial.polynomial.polyfit(acc_vals_0, neg_log_lr, 1)
    intercept, kappa_hat = coeffs[0], coeffs[1]

    # Check 1: Positive correlation (structural LP prediction)
    correlation = np.corrcoef(acc_vals_0, neg_log_lr)[0, 1]
    assert correlation > 0.15, (
        f"ACC and log-LR should be positively correlated under LP: r={correlation:.3f}"
    )

    # Check 2: Regression-predicted KL matches empirical KL
    # This is algebraically guaranteed by OLS (passes through means),
    # but confirms the computation pipeline is correct.
    kl_empirical = np.mean(neg_log_lr)
    kl_predicted = kappa_hat * np.mean(acc_vals_0) + intercept
    rel_error = abs(kl_empirical - kl_predicted) / abs(kl_empirical)
    assert rel_error < 0.01, (
        f"KL link: empirical={kl_empirical:.4f} vs predicted={kl_predicted:.4f}, "
        f"rel_error={rel_error:.4f}"
    )

    return True


# =============================================================================
# Fano / Sample Complexity Bound (Appendix C)
# =============================================================================


def verify_fano_sample_complexity(
    seed: int = 42,
    n_bidders: int = 5,
    n_replications: int = 500,
) -> bool:
    """
    Lines 1244/1252: Classification error P_e >= (1/4)·exp(-T·D_KL).

    Sample complexity: T >= log(1/(4ε)) / D_KL for error rate ε.

    We verify: with T small relative to 1/D_KL, error rate is high;
    with T large, error rate drops below the Fano bound.
    """
    rng = np.random.default_rng(seed)

    # Two regimes: θ₀ = Uniform, θ₁ = Beta(3,3)
    # These have moderate KL divergence
    from scipy.stats import beta as beta_dist

    # Estimate D_KL(Uniform || Beta(3,3)) via Monte Carlo
    n_kl = 100000
    uniform_samples = rng.uniform(0, 1, n_kl)
    log_ratios = -beta_dist.logpdf(uniform_samples, 3, 3)  # log(1/p_β) since p_U=1
    d_kl = np.mean(log_ratios)

    # For each T, classify via mean log-LR and measure error
    error_rates = {}
    for T in [5, 50, 500]:
        errors = 0
        for _ in range(n_replications):
            # True regime chosen at random
            true_regime = rng.integers(0, 2)
            if true_regime == 0:
                bids = rng.uniform(0, 1, (T, n_bidders))
            else:
                bids = beta_dist.rvs(3, 3, size=(T, n_bidders), random_state=rng)

            # Classify via log-likelihood ratio
            llr = np.sum(beta_dist.logpdf(bids, 3, 3))  # log p(data|β) / p(data|U)
            predicted = 1 if llr > 0 else 0
            if predicted != true_regime:
                errors += 1

        error_rates[T] = errors / n_replications

    # Error should decrease with T
    assert error_rates[500] < error_rates[5], (
        f"Error should decrease with T: P_e(T=5)={error_rates[5]:.3f}, "
        f"P_e(T=500)={error_rates[500]:.3f}"
    )

    # Fano bound: P_e >= (1/4)·exp(-T·D_KL)
    # For T=500, the bound is very small, so empirical error should also be small
    fano_bound_500 = 0.25 * np.exp(-500 * d_kl)
    assert error_rates[500] >= fano_bound_500 - 0.01, (
        f"Error rate should respect Fano lower bound (with tolerance): "
        f"P_e={error_rates[500]:.4f}, bound={fano_bound_500:.6f}"
    )

    # With large T, error should be near zero
    assert error_rates[500] < 0.10, (
        f"With T=500, classification should be accurate: P_e={error_rates[500]:.3f}"
    )

    return True


# =============================================================================
# ACE Numerical Check (Appendix F)
# =============================================================================


def verify_ace_numerical(seed: int = 42) -> bool:
    """
    Line 1355: ACE = ACC / log(n_S). Verify on concrete bid vectors.

    ACE normalizes ACC by maximum possible entropy log(n_S),
    yielding competitive efficiency per unit participation capacity.
    """
    rng = np.random.default_rng(seed)

    for n in [3, 5, 10]:
        auctions = acc.sample_auctions(rng, 1000, n)
        _, _, _, acc_vals, _ = acc.acc_components(auctions)

        # ACE = ACC / log(n_S) where n_S = n (all bidders active for uniform)
        ace_vals = acc_vals / np.log(n)

        # ACE should be non-negative
        assert np.all(ace_vals >= -1e-10), f"ACE should be non-negative for n={n}"

        # For uniform draws, H(p) ≈ log(n), so ACE ≈ log(1+ρ/G*)
        # ACE mean should be positive and finite
        mean_ace = np.mean(ace_vals)
        assert 0 < mean_ace < 100, f"ACE mean should be finite: {mean_ace:.4f}"

    return True


# =============================================================================
# Runner
# =============================================================================


def run_all_numerical() -> dict[str, bool]:
    """Run all numerical verifications and return results."""
    checks = {
        # Core ACC properties
        "ACC_monotonic_in_n": verify_acc_monotonic_in_n,
        "ACC_concavity": verify_acc_concavity_numerical,
        "Prop_7_additivity": verify_prop7_additivity,
        "Rho_increases_with_n": verify_rho_increases_with_n,
        # EVT domains
        "EVT_uniform": verify_evt_uniform,
        "EVT_exponential": verify_evt_exponential,
        "EVT_pareto": verify_evt_pareto,
        "Pareto_Beta_exact": verify_pareto_beta_exact,
        "Entropy_drift_constants": verify_entropy_drift_constants,
        # Inference
        "CLT_numerical": verify_clt_numerical,
        "Bootstrap_coverage": verify_bootstrap_coverage,
        "Affiliation_increases_ACC": verify_affiliation_increases_acc,
        # New: gap-filling verifications
        "Renyi_representation": verify_renyi_representation,
        "Spacing_ratio_Beta": verify_spacing_ratio_beta,
        "Gumbel_harmonic": verify_gumbel_harmonic,
        "KL_link_under_LP": verify_kl_link_under_lp,
        "Fano_sample_complexity": verify_fano_sample_complexity,
        "ACE_numerical": verify_ace_numerical,
    }

    results = {}
    for name, func in checks.items():
        try:
            results[name] = func()
            print(f"  [PASS] {name}")
        except AssertionError as e:
            results[name] = False
            print(f"  [FAIL] {name}: {e}")
        except Exception as e:
            results[name] = False
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print(" ACC NUMERICAL VERIFICATION")
    print("=" * 60)
    print()
    results = run_all_numerical()
    passed = sum(results.values())
    total = len(results)
    print()
    print("=" * 60)
    print(f" {passed}/{total} numerical checks passed")
    print("=" * 60)
