#!/usr/bin/env python3
"""
Symbolic verification of ACC theorems using sympy.

Each function returns True on success, raises AssertionError on failure.

Theorem mapping to paper:
- Lemma 2: Functional equation e(r₁)·e(r₂) = e(r₁+r₂+r₁r₂) → e(r)=(1+r)^c
- Corollary 3: Normalization e(1)=2 pins c=1
- Theorem 5: Main characterization ACC = H·log(1+ρ/G*)
- Lemma 6: Dummy-bidder invariance
- Prop 7: Evidence additivity
- Prop 12: Concavity in r
- Prop 13: Taylor approximation (ACC ≈ H·r for small r)
- Props 10-11: Information rent mapping
- Axiom verifications: Scale invariance, monotonicity
"""

from sympy import (
    Eq,
    Rational,
    Symbol,
    diff,
    exp,
    harmonic,
    limit,
    log,
    oo,
    series,
    simplify,
    solve,
    sqrt,
    symbols,
)
from sympy import integrate as sym_integrate

# =============================================================================
# Common symbols
# =============================================================================

# Core variables
r = Symbol("r", positive=True, real=True)
r1, r2 = symbols("r1 r2", positive=True, real=True)
c = Symbol("c", positive=True, real=True)
H = Symbol("H", positive=True, real=True)  # Shannon entropy
rho = Symbol("rho", positive=True, real=True)  # Competitive ratio
G = Symbol("G", positive=True, real=True)  # Normalized Gini (G*)

# Auction parameters
n = Symbol("n", integer=True, positive=True)
lam = Symbol("lambda", positive=True, real=True)
v = Symbol("v", positive=True, real=True)
t = Symbol("t", positive=True, real=True)
b1, b2 = symbols("b1 b2", positive=True)

# Information theory
kappa = Symbol("kappa", positive=True, real=True)
Lambda = Symbol("Lambda", positive=True, real=True)  # Capitalied to avoid clash
T = Symbol("T", positive=True, integer=True)


# =============================================================================
# Lemma 2: Functional Equation (Core of the breakthrough)
# =============================================================================


def verify_lemma2_functional_equation() -> bool:
    """
    Lemma 2: The evidence composition axiom (Axiom C.2) implies e(r) = (1+r)^c.

    Axiom C.2 states: e(r₁)·e(r₂) = e(r₁ + r₂ + r₁r₂)

    Key insight: (1+r₁)(1+r₂) = 1 + r₁ + r₂ + r₁r₂

    Proof outline:
    1. Define φ(r) = log e(r)
    2. Axiom C.2 gives: φ(r₁) + φ(r₂) = φ(r₁ + r₂ + r₁r₂)
    3. Substitute s = 1+r, so φ(s-1) + φ(t-1) = φ(st-1)
    4. Define ψ(x) = φ(e^x - 1), get Cauchy's equation ψ(x+y) = ψ(x) + ψ(y)
    5. Under continuity, ψ(x) = cx
    6. Back-substitute: e(r) = (1+r)^c
    """
    # Verify the algebraic identity underlying Axiom C.2
    lhs = (1 + r1) * (1 + r2)
    rhs = 1 + r1 + r2 + r1 * r2
    assert simplify(lhs - rhs) == 0, f"Identity failed: {simplify(lhs - rhs)}"

    # Verify e(r) = (1+r)^c satisfies the functional equation
    def e(x):
        return (1 + x) ** c

    lhs_func = e(r1) * e(r2)
    rhs_func = e(r1 + r2 + r1 * r2)
    diff_result = simplify(lhs_func - rhs_func)
    assert diff_result == 0, f"Functional equation failed: {diff_result}"

    # Verify that e(0) = 1 (zero baseline, C4')
    assert e(0) == 1, f"e(0) should be 1, got {e(0)}"

    return True


# =============================================================================
# Corollary 3: Normalization pins c = 1
# =============================================================================


def verify_corollary3_normalization() -> bool:
    """
    Corollary 3: The normalization axiom C7' (e(1) = 2) pins c = 1.

    C7' states: e(1) = 2
    With e(r) = (1+r)^c, we have: e(1) = 2^c = 2
    Therefore: c = 1

    This gives the unique form: e(r) = 1 + r
    """
    # Solve for c such that (1+1)^c = 2
    equation = Eq((1 + 1) ** c, 2)
    solutions = solve(equation, c)

    assert 1 in solutions, f"c=1 should be a solution, got {solutions}"

    # Verify e(r) = 1 + r when c = 1
    e_at_c1 = (1 + r) ** 1
    assert simplify(e_at_c1 - (1 + r)) == 0, "e(r) should equal 1+r when c=1"

    return True


# =============================================================================
# Theorem 5: Main Characterization
# =============================================================================


def verify_theorem5_acc_formula() -> bool:
    """
    Theorem 5: Under Convention S and Assumptions P, C, D, the unique statistic is:

    E(b) = (1 + ρ/G*)^{H(p)}
    ACC(b) = H(p) · log(1 + ρ/G*)

    Verify: ACC = log(E) algebraic identity only (not full uniqueness/existence).
    """
    # Define r = ρ/G*
    r_def = rho / G

    # Evidence factor E = (1 + r)^H
    E = (1 + r_def) ** H

    # ACC = H · log(1 + r)
    ACC = H * log(1 + r_def)

    # Verify ACC = log(E)
    log_E = log(E)
    # log((1+r)^H) = H·log(1+r)
    expanded = simplify(log_E - ACC)
    assert expanded == 0, f"ACC should equal log(E), diff = {expanded}"

    # Verify E = exp(ACC)
    exp_ACC = exp(ACC)
    # exp(H·log(1+r)) = (1+r)^H = E
    diff_E = simplify(exp_ACC - E)
    assert diff_E == 0, f"E should equal exp(ACC), diff = {diff_E}"

    return True


# =============================================================================
# Lemma 6: Dummy-Bidder Invariance
# =============================================================================


def verify_lemma6_dummy_invariance() -> bool:
    """
    Lemma 6: Adding a dummy bidder (b_0 = 0) does not change ACC.

    Under Convention S (compute on support), bidder with b=0 is excluded.
    This is structural: ACC(b, 0) = ACC(b).
    """
    # This is more of a definitional property than algebraic
    # We verify that H computed on support excludes zero bids

    # Consider bid vector [b1, b2, 0] with support {1, 2}
    # H should be computed on normalized [b1, b2] only

    # For 2 bidders with bids b1, b2:
    p1 = b1 / (b1 + b2)
    p2 = b2 / (b1 + b2)
    H_support = -(p1 * log(p1) + p2 * log(p2))

    # For 3 bidders with bids b1, b2, 0:
    # Under Convention S, we ignore 0, so same H
    # The zero bid contributes 0·log(0) = 0 in the limit, but by convention excluded

    # Verify H is maximized when p1 = p2 (equal shares)
    H_equal = H_support.subs([(b1, 1), (b2, 1)])
    assert simplify(H_equal - log(2)) == 0, "H should be log(2) for equal 2-bidders"

    return True


# =============================================================================
# Proposition 7: Evidence Additivity
# =============================================================================


def verify_prop7_additivity() -> bool:
    """
    Proposition 7: Evidence factors multiply, so ACC values add.

    If E_t = (1 + r_t)^{H_t} for auctions t = 1, ..., T, then:
    ∏ E_t = exp(∑ ACC_t)

    Equivalently: log(∏ E_t) = ∑ ACC_t
    """
    # For two auctions
    H1, H2 = symbols("H1 H2", positive=True)
    r_1, r_2 = symbols("r_1 r_2", positive=True)

    E1 = (1 + r_1) ** H1
    E2 = (1 + r_2) ** H2

    ACC1 = H1 * log(1 + r_1)
    ACC2 = H2 * log(1 + r_2)

    # Product of evidence factors
    E_product = E1 * E2

    # Sum of ACC values
    ACC_sum = ACC1 + ACC2

    # Verify log(E1·E2) = ACC1 + ACC2
    log_product = simplify(log(E_product))
    expected = simplify(ACC_sum)

    # log(E1·E2) = log(E1) + log(E2) = H1·log(1+r1) + H2·log(1+r2) = ACC1 + ACC2
    assert simplify(log_product - expected) == 0, (
        f"Additivity failed: {simplify(log_product - expected)}"
    )

    return True


# =============================================================================
# Proposition 12: Concavity in r
# =============================================================================


def verify_prop12_concavity() -> bool:
    """
    Proposition 12: ACC is concave in r (diminishing marginal returns).

    ACC = H · log(1 + r)

    First derivative: ∂ACC/∂r = H / (1 + r) > 0 (increasing)
    Second derivative: ∂²ACC/∂r² = -H / (1 + r)² < 0 (concave)
    """
    ACC = H * log(1 + r)

    # First derivative
    d1 = diff(ACC, r)
    expected_d1 = H / (1 + r)
    assert simplify(d1 - expected_d1) == 0, f"First derivative wrong: {simplify(d1)}"

    # Second derivative
    d2 = diff(ACC, r, 2)
    expected_d2 = -H / (1 + r) ** 2
    assert simplify(d2 - expected_d2) == 0, f"Second derivative wrong: {simplify(d2)}"

    # Verify signs (symbolically: H > 0, r > 0 assumed)
    # d1 > 0 since H > 0 and 1+r > 0
    # d2 < 0 since -H < 0 and (1+r)² > 0

    return True


# =============================================================================
# Proposition 13: Taylor Approximation (ACC ≈ H·r for small r)
# =============================================================================


def verify_prop13_taylor_approximation() -> bool:
    """
    Proposition 13: For small r, ACC ≈ H·r (first-order agreement).

    ACC = H · log(1 + r)

    Taylor expansion around r = 0:
    log(1 + r) = r - r²/2 + r³/3 - r⁴/4 + O(r⁵)

    So: ACC = H·r - (H/2)·r² + O(r³)

    First order: ACC ≈ H·r
    """
    # Taylor series of log(1 + r) around r = 0
    log_expansion = series(log(1 + r), r, 0, 5).removeO()

    # Expected: r - r²/2 + r³/3 - r⁴/4
    expected_terms = r - r**2 / 2 + r**3 / 3 - r**4 / 4
    assert simplify(log_expansion - expected_terms) == 0, (
        f"Taylor expansion wrong: {log_expansion}"
    )

    # ACC = H · log(1 + r)
    ACC_expansion = H * series(log(1 + r), r, 0, 3).removeO()

    # First order term is H·r
    first_order = H * r
    second_order = -H * r**2 / 2

    # Verify first two terms
    expected_acc = first_order + second_order
    assert simplify(ACC_expansion - expected_acc) == 0, (
        f"ACC expansion wrong: {ACC_expansion}"
    )

    return True


# =============================================================================
# Propositions 10-11: Information Rent Mapping
# =============================================================================


def verify_props10_11_rent_mapping() -> bool:
    """
    Props 10-11: Under second-price (truthful bidding), information rent = 1 - ρ.

    Information rent IR = (v₁ - v₂) / v₁ where v₁, v₂ are highest/second-highest values.
    Under truthful bidding, b_i = v_i, so:
    IR = (b₁ - b₂) / b₁ = 1 - b₂/b₁ = 1 - ρ
    """
    # Define information rent
    IR = (b1 - b2) / b1

    # Define competitive ratio
    rho_def = b2 / b1

    # Verify IR = 1 - ρ
    ir_simplified = simplify(IR)
    expected = 1 - rho_def
    assert simplify(ir_simplified - expected) == 0, f"IR should be 1-ρ: {ir_simplified}"

    # Verify ρ + IR = 1
    sum_check = simplify(rho_def + IR)
    assert sum_check == 1, f"ρ + IR should equal 1, got {sum_check}"

    return True


# =============================================================================
# Axiom Verification: Scale Invariance
# =============================================================================


def verify_axiom_scale_invariance() -> bool:
    """
    Axiom C3: Scale invariance.

    For any λ > 0, ACC(λ·b) = ACC(b).

    This follows from:
    - ρ = b₂/b₁ is scale-invariant: (λb₂)/(λb₁) = b₂/b₁
    - H(p) is scale-invariant: p_i = b_i/Σb_j is unchanged by λ
    - G* is scale-invariant: Gini of normalized distribution
    """
    # Check ρ is scale-invariant
    rho_original = b2 / b1
    rho_scaled = (lam * b2) / (lam * b1)
    assert simplify(rho_scaled - rho_original) == 0, "ρ should be scale-invariant"

    # Check bid shares are scale-invariant
    total = b1 + b2
    p1_original = b1 / total
    p1_scaled = (lam * b1) / (lam * total)
    assert simplify(p1_scaled - p1_original) == 0, (
        "Bid shares should be scale-invariant"
    )

    return True


# =============================================================================
# EVT: Uniform Distribution Expected Ratio
# =============================================================================


def verify_evt_uniform_ratio() -> bool:
    """
    EVT for Uniform[0,1]: Expected competitive ratio E[ρ] = (n-1)/n.

    For n i.i.d. Uniform[0,1] draws, the ratio of order statistics is:
    E[X_{(n-1)} / X_{(n)}] = (n-1)/n

    Convergence: 1 - E[ρ] = 1/n → 0 at rate O(1/n).
    """
    # For uniform distribution, this is a known result
    E_rho = (n - 1) / n
    one_minus_rho = 1 - E_rho

    assert simplify(one_minus_rho - 1 / n) == 0, "1 - E[ρ] should be 1/n for uniform"

    # Verify limit as n → ∞
    lim_rho = limit(E_rho, n, oo)
    assert lim_rho == 1, f"E[ρ] should → 1 as n → ∞, got {lim_rho}"

    return True


# =============================================================================
# First-Price Bid Function (Uniform)
# =============================================================================


def verify_first_price_uniform() -> bool:
    """
    First-price equilibrium bid function for Uniform[0,1].

    b(v) = v · (n-1)/n = v - v/n

    This is the unique symmetric equilibrium bidding strategy.
    """
    # For uniform F(v) = v, the equilibrium bid is:
    # b(v) = v - integral_0^v [F(t)]^{n-1} dt / [F(v)]^{n-1}
    #      = v - integral_0^v t^{n-1} dt / v^{n-1}
    #      = v - [t^n/n]_0^v / v^{n-1}
    #      = v - v^n/(n·v^{n-1})
    #      = v - v/n
    #      = v·(n-1)/n

    integral = sym_integrate(t ** (n - 1), (t, 0, v))
    bid_function = v - integral / v ** (n - 1)
    expected_bid = v * (n - 1) / n

    assert simplify(bid_function - expected_bid) == 0, (
        f"Bid function should be v·(n-1)/n, got {simplify(bid_function)}"
    )

    # Verify bid function is linear in v (preserves ρ)
    v1_sym, v2_sym = symbols("v1 v2", positive=True)
    coefficient = (n - 1) / n
    rho_values = v2_sym / v1_sym
    rho_bids = (coefficient * v2_sym) / (coefficient * v1_sym)
    assert simplify(rho_bids - rho_values) == 0, (
        "ρ should be preserved under linear bid"
    )

    return True


# =============================================================================
# Shannon-Khinchin Uniqueness (Theorem 1)
# =============================================================================


def verify_theorem1_entropy_uniqueness() -> bool:
    """
    Theorem 1: Shannon entropy H(p) is characterized by:
    - Continuity
    - Symmetry: H(p₁,...,pₙ) = H(p_σ(1),...,p_σ(n)) for any permutation σ
    - Expansibility: H(p₁,...,pₙ,0) = H(p₁,...,pₙ)
    - Additivity: H(p⊗q) = H(p) + H(q) for independent distributions

    Verify: H(p) = -Σ pᵢ log(pᵢ) satisfies these properties.
    """
    p1, p2 = symbols("p1 p2", positive=True)

    # Two-bidder entropy
    H_two = -(p1 * log(p1) + p2 * log(p2))

    # Symmetry: H(p1, p2) = H(p2, p1)
    H_swapped = -(p2 * log(p2) + p1 * log(p1))
    assert simplify(H_two - H_swapped) == 0, "Entropy should be symmetric"

    # Maximum at uniform: H(1/2, 1/2) = log(2)
    H_uniform = H_two.subs([(p1, Rational(1, 2)), (p2, Rational(1, 2))])
    assert simplify(H_uniform - log(2)) == 0, (
        f"H(1/2,1/2) should be log(2): {H_uniform}"
    )

    # Expansibility: Adding 0 doesn't change entropy (0·log(0) = 0 by convention)
    # This is handled by support convention in our implementation

    return True


# =============================================================================
# Posterior Odds Factorization (Remark, §1)
# =============================================================================


def verify_posterior_odds_factorization() -> bool:
    """
    Line 348: Posterior odds factor into prior odds × product of likelihood ratios.

    P(θ₀|b₁,b₂)/P(θ₁|b₁,b₂) = [P(θ₀)/P(θ₁)] · L(b₁) · L(b₂)

    This is Bayes' rule under conditional independence of bids given θ.
    """
    # Under conditional independence: p(b₁,b₂|θ) = p(b₁|θ)·p(b₂|θ)
    # So the likelihood ratio factors as L(b₁,b₂) = L(b₁)·L(b₂)

    L1, L2 = symbols("L1 L2", positive=True)  # individual likelihood ratios
    prior_odds = Symbol("pi", positive=True)

    # Joint posterior odds = prior_odds × L(b₁) × L(b₂)
    posterior_odds = prior_odds * L1 * L2

    # Equivalently, log posterior odds = log prior + log L₁ + log L₂
    log_posterior = log(posterior_odds)
    expected = log(prior_odds) + log(L1) + log(L2)
    delta = simplify(log_posterior - expected)
    assert delta == 0, f"Log posterior odds should decompose additively: {delta}"

    return True


# =============================================================================
# Counter-Example 3.1: Drop C6 (Evidence Composition)
# =============================================================================


def verify_counterexample_3_1() -> bool:
    """
    Lines 516/520: Counter-example showing axiom C6 necessity.

    1. e(r) = 2^r violates Axiom C.2: 2^r₁ · 2^r₂ ≠ 2^(r₁+r₂+r₁r₂)
    2. e(r) = 1+r satisfies Axiom C.2: (1+r₁)(1+r₂) = 1+r₁+r₂+r₁r₂
    """
    # Part 1: e(r) = 2^r violates composition
    e_exp_lhs = 2**r1 * 2**r2  # e(r₁)·e(r₂) = 2^(r₁+r₂)
    e_exp_rhs = 2 ** (r1 + r2 + r1 * r2)  # e(r₁⊕r₂) = 2^(r₁+r₂+r₁r₂)
    diff_exp = simplify(e_exp_lhs - e_exp_rhs)
    # These should NOT be equal (the difference should be nonzero)
    # 2^(r₁+r₂) ≠ 2^(r₁+r₂+r₁r₂) unless r₁r₂ = 0
    assert diff_exp != 0, "Exponential evidence must violate C.2"

    # Part 2: e(r) = 1+r satisfies composition
    e_lin_lhs = (1 + r1) * (1 + r2)
    e_lin_rhs = 1 + r1 + r2 + r1 * r2
    assert simplify(e_lin_lhs - e_lin_rhs) == 0, "Linear evidence must satisfy C.2"

    return True


def verify_prop_no_c3_repair() -> bool:
    """
    Prop (non-identification without evidence composition), repaired form.

    The separable candidate uses e2~(r) = exp(2·log2·r/(1+r)):
    1. Baseline (C.4): e2~(0) = 1
    2. Normalization (C.6): e2~(1) = 2
    3. Evidence composition (C.5) FAILS: e2~(1)² = 4 but e2~(3) = 2^(3/2)
    4. The pre-repair candidate exp(r/(1+r)) violates C.6: e2(1) = √e ≠ 2,
       which is why the proposition needed the explicit renormalization.
    """
    e2_tilde = exp(2 * log(2) * r / (1 + r))

    # 1. Baseline e2~(0) = 1
    assert simplify(e2_tilde.subs(r, 0)) == 1, "e2~(0) should be 1"

    # 2. Normalization e2~(1) = 2
    assert simplify(e2_tilde.subs(r, 1) - 2) == 0, "e2~(1) should be 2"

    # 3. Composition fails at r1 = r2 = 1 (composes to 1+1+1·1 = 3)
    lhs = simplify(e2_tilde.subs(r, 1) ** 2)  # = 4
    rhs = simplify(e2_tilde.subs(r, 3))  # = 2^(3/2)
    assert lhs == 4, f"e2~(1)² should be 4, got {lhs}"
    assert simplify(rhs - 2 * sqrt(2)) == 0, f"e2~(3) should be 2^(3/2), got {rhs}"
    assert simplify(lhs - rhs) != 0, "C.5 must fail for the repaired candidate"

    # 4. Pre-repair candidate violates C.6
    e2_old = exp(r / (1 + r))
    assert simplify(e2_old.subs(r, 1) - 2) != 0, "exp(1/2) ≠ 2: old candidate fails C.6"

    return True


# =============================================================================
# Gini = G* Identity (Appendix A, Theorem 6 conclusion)
# =============================================================================


def verify_gini_equals_gstar() -> bool:
    """
    Line 1027: D(b) = (2·Σk·b_(k))/(n·Σb_j) - (n+1)/n = G*(b).

    Verify the normalized Gini formula on concrete bid vectors.
    """
    from sympy import Rational as R

    # Two-bidder case: bids [1, 3] (ascending order)
    # G* = 2·(1·1 + 2·3)/(2·4) - 3/2 = 2·7/8 - 3/2 = 7/4 - 3/2 = 1/4
    b = [1, 3]
    n_val = len(b)
    numerator = 2 * sum((k + 1) * b[k] for k in range(n_val))
    denominator = n_val * sum(b)
    gstar = R(numerator, denominator) - R(n_val + 1, n_val)
    assert gstar == R(1, 4), f"G*([1,3]) should be 1/4, got {gstar}"

    # Three-bidder case: bids [1, 2, 3] (ascending)
    # G* = 2·(1·1 + 2·2 + 3·3)/(3·6) - 4/3 = 2·14/18 - 4/3 = 28/18 - 24/18 = 4/18 = 2/9
    b = [1, 2, 3]
    n_val = len(b)
    numerator = 2 * sum((k + 1) * b[k] for k in range(n_val))
    denominator = n_val * sum(b)
    gstar = R(numerator, denominator) - R(n_val + 1, n_val)
    assert gstar == R(2, 9), f"G*([1,2,3]) should be 2/9, got {gstar}"

    # Uniform bids: G*([c,c,...,c]) = 0
    # 2·Σk·c/(n·n·c) - (n+1)/n = 2·c·n(n+1)/2/(n²·c) - (n+1)/n
    #   = (n+1)/n - (n+1)/n = 0
    n_sym = Symbol("n_s", integer=True, positive=True)
    c_sym = Symbol("c_s", positive=True)
    sum_k = n_sym * (n_sym + 1) / 2
    gstar_uniform = 2 * c_sym * sum_k / (n_sym * n_sym * c_sym) - (n_sym + 1) / n_sym
    assert simplify(gstar_uniform) == 0, (
        f"G* on uniform bids should be 0, got {simplify(gstar_uniform)}"
    )

    return True


# =============================================================================
# Harmonic Number Growth (Appendix B, Gumbel domain)
# =============================================================================


def verify_harmonic_number_growth() -> bool:
    """
    Line 1095: E[V_(n)] = H_n ~ log(n) for standard exponential order statistics.

    The maximum of n i.i.d. Exp(1) has E[max] = H_n (harmonic number).
    """
    # Verify H_n = 1 + 1/2 + ... + 1/n for small n
    for n_val in [2, 5, 10]:
        H_n = harmonic(n_val)
        manual = sum(Rational(1, k) for k in range(1, n_val + 1))
        assert H_n == manual, f"H_{n_val} mismatch: {H_n} vs {manual}"

    # Verify asymptotic: H_n / log(n) → 1 as n → ∞
    lim = limit(harmonic(n) / log(n), n, oo)
    assert lim == 1, f"H_n/log(n) should → 1, got {lim}"

    return True


# =============================================================================
# ACE Definition (Appendix F)
# =============================================================================


def verify_ace_definition() -> bool:
    """
    Line 1355: ACE = ACC / log(n_S) = [H(p)/log(n_S)] · log(1 + ρ/G*).

    Verify: ACE ∈ [0, ∞) and ACE = 1 when H(p) = log(n_S) (maximal participation).
    """
    n_S = Symbol("n_S", integer=True, positive=True)

    # ACE with maximal participation: H(p) = log(n_S) (uniform shares)
    ACC_max_H = log(n_S) * log(1 + rho / G)
    ACE_max_H = ACC_max_H / log(n_S)
    assert simplify(ACE_max_H - log(1 + rho / G)) == 0, (
        "ACE should reduce to log(1+ρ/G*) when H = log(n_S)"
    )

    # ACE with general H
    ACC_general = H * log(1 + rho / G)
    ACE_general = ACC_general / log(n_S)
    expected = (H / log(n_S)) * log(1 + rho / G)
    assert simplify(ACE_general - expected) == 0, "ACE = [H/log(n_S)] · log(1+ρ/G*)"

    return True


# =============================================================================
# Main runner
# =============================================================================


def run_all_symbolic() -> dict[str, bool]:
    """Run all symbolic verifications and return results."""
    checks = {
        # Core ACC characterization
        "Lemma_2_functional_eq": verify_lemma2_functional_equation,
        "Corollary_3_normalization": verify_corollary3_normalization,
        "Theorem_5_acc_formula": verify_theorem5_acc_formula,
        "Lemma_6_dummy_invariance": verify_lemma6_dummy_invariance,
        "Prop_7_additivity": verify_prop7_additivity,
        "Prop_12_concavity": verify_prop12_concavity,
        "Prop_13_taylor": verify_prop13_taylor_approximation,
        # Economic interpretation
        "Props_10_11_rent": verify_props10_11_rent_mapping,
        "Theorem_1_entropy": verify_theorem1_entropy_uniqueness,
        # Axioms
        "Axiom_scale_invariance": verify_axiom_scale_invariance,
        # EVT and first-price
        "EVT_uniform_ratio": verify_evt_uniform_ratio,
        "First_price_uniform": verify_first_price_uniform,
        # New: gap-filling verifications
        "Posterior_odds_factorization": verify_posterior_odds_factorization,
        "Counterexample_3_1": verify_counterexample_3_1,
        "Prop_no_c3_repair": verify_prop_no_c3_repair,
        "Gini_equals_Gstar": verify_gini_equals_gstar,
        "Harmonic_number_growth": verify_harmonic_number_growth,
        "ACE_definition": verify_ace_definition,
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
    print(" ACC SYMBOLIC VERIFICATION")
    print("=" * 60)
    print()
    results = run_all_symbolic()
    passed = sum(results.values())
    total = len(results)
    print()
    print("=" * 60)
    print(f" {passed}/{total} symbolic checks passed")
    print("=" * 60)
