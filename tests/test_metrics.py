"""Tests for ACC (Auction Competitive Capacity) metrics.

ACC = H(p) · log(1 + r) where r = ρ / G*

Key properties:
- ACC = log(E) where E is the evidence factor
- Concave in r (diminishing marginal returns)
- Evidence factors multiply, ACC values add
"""

import math

import numpy as np
import pytest

from acc_metrics import (
    acc,
    acc_delta,
    bootstrap_bca_ci,
    dispersion_floor,
    effective_bidders,
    entropy,
    evidence_factor,
    first_price_bids_uniform,
    information_rent,
    normalized_gini,
    saturation_flag,
)
from acc_metrics.core import competitive_ratio
from acc_metrics.equilibrium import (
    first_price_bids_exponential,
    first_price_bids_lognormal,
    first_price_bids_pareto,
)


class TestNormalizedGini:
    """Tests for Gini coefficient G* calculation."""

    def test_single_bid_returns_zero(self):
        assert normalized_gini([100]) == 0.0

    def test_identical_bids(self):
        assert normalized_gini([100, 100, 100]) == 0.0

    def test_two_different_bids(self):
        # For [a, b] with a < b: G = (b-a)/(a+b)
        # Sorted: [50, 100]. G = (2*(1*50 + 2*100))/(2*150) - 3/2
        # = 500/300 - 1.5 = 5/3 - 1.5 = 1/6
        result = normalized_gini([100, 50])
        assert pytest.approx(result, rel=1e-6) == 1 / 6

    def test_scale_invariance(self):
        """Gini coefficient should be scale-invariant (D1)."""
        bids = [100, 90, 80, 70, 60]
        scaled_bids = [b * 1000 for b in bids]
        assert pytest.approx(normalized_gini(bids), rel=1e-6) == normalized_gini(
            scaled_bids
        )

    def test_permutation_invariance(self):
        """Gini should be unchanged by reordering (D2)."""
        bids = [100, 90, 80, 70, 60]
        shuffled = [80, 100, 60, 90, 70]
        assert pytest.approx(normalized_gini(bids), rel=1e-6) == normalized_gini(
            shuffled
        )

    def test_dummy_bidder_invariance(self):
        """Dummy bidders (exactly 0) should be ignored by the G* computation."""
        bids = [100, 50]
        with_dummy = [100, 50, 0]
        assert pytest.approx(normalized_gini(bids), rel=1e-12) == normalized_gini(
            with_dummy
        )

    def test_boundedness(self):
        """Gini should be in [0, 1) for positive bids (D4)."""
        bids = [100, 90, 80, 70, 60]
        result = normalized_gini(bids)
        assert 0 <= result < 1

    def test_schur_convexity(self):
        """Gini should decrease under doubly stochastic averaging (D3).

        This is the key property that G* satisfies.
        """
        # Original bids
        bids = np.array([100, 50, 75])
        gini_orig = normalized_gini(bids)

        # Average two elements (doubly stochastic operation)
        averaged = np.array([100, 62.5, 62.5])  # Average of 50 and 75
        gini_avg = normalized_gini(averaged)

        # Gini should decrease (strict inequality for non-permutation DS matrix)
        assert gini_avg < gini_orig

    def test_d4_equal_transfer_sensitivity(self):
        """New D.4: Equal adjacent-rank transfer sensitivity.

        A Pigou-Dalton transfer of epsilon from rank k to rank k+1
        (descending order) should change G* by the same amount for all k.
        This is the NEW axiom that correctly characterizes G*.
        """
        bids = np.array([100.0, 80.0, 60.0, 40.0, 20.0])
        eps = 0.5

        # Transfer from rank 1 to rank 2: (100-eps, 80+eps, 60, 40, 20)
        bids_12 = np.array([100 - eps, 80 + eps, 60, 40, 20])
        delta_12 = normalized_gini(bids_12) - normalized_gini(bids)

        # Transfer from rank 2 to rank 3: (100, 80-eps, 60+eps, 40, 20)
        bids_23 = np.array([100, 80 - eps, 60 + eps, 40, 20])
        delta_23 = normalized_gini(bids_23) - normalized_gini(bids)

        # Transfer from rank 3 to rank 4: (100, 80, 60-eps, 40+eps, 20)
        bids_34 = np.array([100, 80, 60 - eps, 40 + eps, 20])
        delta_34 = normalized_gini(bids_34) - normalized_gini(bids)

        # Transfer from rank 4 to rank 5: (100, 80, 60, 40-eps, 20+eps)
        bids_45 = np.array([100, 80, 60, 40 - eps, 20 + eps])
        delta_45 = normalized_gini(bids_45) - normalized_gini(bids)

        # All deltas should be equal (this is the axiom)
        assert pytest.approx(delta_12, rel=1e-6) == delta_23
        assert pytest.approx(delta_23, rel=1e-6) == delta_34
        assert pytest.approx(delta_34, rel=1e-6) == delta_45

        # All deltas should be negative (equalizing transfer reduces dispersion)
        assert delta_12 < 0

    def test_d4_old_uniform_sensitivity_violated(self):
        """Regression test: OLD D.4 (uniform sensitivity) is NOT satisfied by G*.

        The old axiom required dD/db_(k) constant across k, which G* violates.
        This test documents the fix.
        """
        bids = np.array([100.0, 80.0, 60.0, 40.0, 20.0])
        eps = 0.01

        # Compute partial derivatives numerically
        partials = []
        # Bids are in descending order: (100, 80, 60, 40, 20)
        # Perturb each rank position
        for i in range(5):
            bids_up = bids.copy()
            bids_up[i] += eps
            partial = (normalized_gini(bids_up) - normalized_gini(bids)) / eps
            partials.append(partial)

        # Under OLD D.4, all partials would be equal.
        # For G*, they are NOT equal (they depend on the bid values).
        # The partial derivatives for G* are: (w_k * S - W) / S^2
        # which depends on S (total), W (weighted sum), and w_k (rank weight).
        assert pytest.approx(partials[0], rel=0.01) != partials[2], (
            "Old uniform sensitivity should NOT hold for G*"
        )

    def test_d1_calibration(self):
        """D.1 calibration: G*(2, 1) = 1/6."""
        assert pytest.approx(normalized_gini([2, 1]), rel=1e-6) == 1 / 6

    def test_d5_replication_invariance(self):
        """D.5: G*(b^(m)) = G*(b) for any integer replication m."""
        bids = [100.0, 80.0, 60.0, 40.0, 20.0]
        g_original = normalized_gini(bids)
        for m in [2, 3, 5]:
            replicated = sorted(bids * m)  # each bid repeated m times
            g_replicated = normalized_gini(replicated)
            assert pytest.approx(g_replicated, rel=1e-6) == g_original, (
                f"Replication m={m}: G*={g_replicated} != {g_original}"
            )


class TestCompetitiveRatio:
    """Tests for ρ = b_(2) / b_(1) calculation."""

    def test_single_bid_returns_one(self):
        assert competitive_ratio([100]) == 1.0

    def test_identical_bids(self):
        assert competitive_ratio([100, 100, 100]) == 1.0

    def test_distinct_bids(self):
        # ρ = b2/b1 = 90/100 = 0.9
        result = competitive_ratio([100, 90, 50])
        assert pytest.approx(result, rel=1e-6) == 0.9

    def test_dummy_bidder_invariance(self):
        """Dummy bidders (exactly 0) should not affect ρ."""
        bids = [100, 90]
        with_dummy = [100, 90, 0]
        assert pytest.approx(competitive_ratio(bids), rel=1e-12) == competitive_ratio(
            with_dummy
        )


class TestACC:
    """Tests for Auction Competitive Capacity.

    ACC = H(p) · log(1 + r) where r = ρ / G*

    where:
        H(p) = Shannon entropy of bid shares (effective participation)
        ρ = b_(2) / b_(1) (competitive ratio)
        G* = normalized Gini coefficient (dispersion measure)
        r = ρ / G* (adjusted ratio)
    """

    def test_single_bid_returns_zero(self):
        assert acc([100]) == 0.0

    def test_single_active_bidder_with_dummy_returns_zero(self):
        """If only one bidder is active (others are zero), ACC is 0 by convention."""
        assert acc([100, 0]) == 0.0

    def test_identical_bids_high_acc(self):
        # G* = 0 for identical bids → r = ∞ → log(1+r) = ∞
        # But H is still finite, so ACC is finite * inf
        # Implementation should handle this gracefully
        result = acc([100, 100])
        assert result > 0  # Should be very high or inf

    def test_competitive_auction(self):
        # Highly competitive: close bids, many bidders
        bids = [100, 99, 98, 97, 96]
        result = acc(bids)
        assert result > 0
        assert math.isfinite(result)

    def test_scale_invariance(self):
        """ACC should be scale-invariant (Axiom C3)."""
        bids = [100, 90, 80, 70, 60]
        scaled_bids = [b * 1000 for b in bids]
        assert pytest.approx(acc(bids), rel=1e-6) == acc(scaled_bids)

    def test_dummy_bidder_invariance(self):
        """Dummy bidders (exactly 0) should not affect ACC."""
        bids = [100, 90, 80, 70, 60]
        with_dummy = [100, 90, 80, 70, 60, 0]
        assert pytest.approx(acc(bids), rel=1e-12) == acc(with_dummy)

    def test_monotonic_in_rho(self):
        """ACC should increase with competitive ratio (Axiom C2)."""
        # Higher ρ (closer bids) should give higher ACC
        bids_low_rho = [100, 50, 75, 60, 80]  # ρ = 0.8
        bids_high_rho = [100, 95, 75, 60, 80]  # ρ = 0.95

        assert acc(bids_high_rho) > acc(bids_low_rho)

    def test_inverse_monotonic_in_dispersion(self):
        """ACC should decrease with dispersion (Axiom C5')."""
        # Higher dispersion should give lower ACC (for same ρ)
        bids_low_disp = [100, 95, 90, 85, 80]  # Low dispersion
        bids_high_disp = [100, 95, 50, 45, 40]  # High dispersion

        # With similar ρ (~0.95), higher dispersion should give lower ACC
        assert acc(bids_low_disp) > acc(bids_high_disp)

    def test_acc_is_log_of_evidence(self):
        """ACC should equal log(E) where E is the evidence factor."""
        bids = [100, 90, 80, 70, 60]
        acc_val = acc(bids)
        e_val = evidence_factor(bids)

        # ACC = log(E)
        assert pytest.approx(acc_val, rel=1e-6) == math.log(e_val)

    def test_batch_processing(self):
        """ACC handles 2D batch input correctly."""
        auctions = np.array(
            [
                [100, 90, 80, 70, 60],
                [100, 95, 90, 85, 80],
                [100, 99, 98, 97, 96],
            ]
        )
        results = acc(auctions)
        assert isinstance(results, np.ndarray) and results.shape == (3,)
        values = results.tolist()
        # All should be positive and finite
        assert all(r > 0 and math.isfinite(r) for r in values)
        # More competitive (3rd) should have higher ACC
        assert values[2] > values[0]


class TestEvidenceFactor:
    """Tests for evidence factor E = (1 + r)^H.

    Key property: Evidence factors multiply across auctions.
    """

    def test_single_bid_returns_one(self):
        """E = 1 for single bidder (no evidence)."""
        assert evidence_factor([100]) == 1.0

    def test_evidence_greater_than_one(self):
        """E > 1 for competitive auctions."""
        bids = [100, 90, 80, 70, 60]
        result = evidence_factor(bids)
        assert result > 1.0

    def test_evidence_is_exp_of_acc(self):
        """E should equal exp(ACC)."""
        bids = [100, 90, 80, 70, 60]
        e_val = evidence_factor(bids)
        acc_val = acc(bids)

        assert pytest.approx(e_val, rel=1e-6) == math.exp(acc_val)

    def test_multiplicative_property(self):
        """E(combined) = E(a) * E(b) for independent evidence."""
        # This tests the additivity of ACC: log(E1*E2) = log(E1) + log(E2)
        bids1 = [100, 90, 80, 70, 60]
        bids2 = [100, 95, 90, 85, 80]

        e1 = evidence_factor(bids1)
        e2 = evidence_factor(bids2)
        acc1 = acc(bids1)
        acc2 = acc(bids2)

        # log(E1*E2) = ACC1 + ACC2
        assert pytest.approx(math.log(e1 * e2), rel=1e-6) == acc1 + acc2

    def test_batch_processing(self):
        """Evidence factor handles 2D batch input correctly."""
        auctions = np.array(
            [
                [100, 90, 80, 70, 60],
                [100, 95, 90, 85, 80],
            ]
        )
        results = evidence_factor(auctions)
        assert isinstance(results, np.ndarray) and results.shape == (2,)
        assert all(r > 1.0 for r in results.tolist())


class TestEntropy:
    """Tests for Shannon entropy of bid shares H(p).

    H(p) = -Σ p_i log(p_i) where p_i = b_i / Σb_j

    Key property: For equal bids, H(p) = log(n).
    """

    def test_single_bid_returns_zero(self):
        """H(p) = 0 for single bidder (no uncertainty)."""
        assert entropy([100]) == 0.0

    def test_equal_bids_gives_log_n(self):
        """H(p) = log(n) when all bids are equal."""
        # Equal bids → uniform distribution → maximum entropy
        for n in [2, 3, 5, 10, 20]:
            bids = [100] * n
            result = entropy(bids)
            expected = math.log(n)
            assert pytest.approx(result, rel=1e-6) == expected

    def test_two_bids_unequal(self):
        """H(p) for binary distribution."""
        # p = [0.5, 0.5] → H = log(2)
        assert pytest.approx(entropy([50, 50]), rel=1e-6) == math.log(2)

        # p = [0.75, 0.25] → H = -0.75*log(0.75) - 0.25*log(0.25)
        bids = [75, 25]
        p1, p2 = 0.75, 0.25
        expected = -(p1 * math.log(p1) + p2 * math.log(p2))
        assert pytest.approx(entropy(bids), rel=1e-6) == expected

    def test_single_dominant_approaches_zero(self):
        """H(p) → 0 when one bidder dominates."""
        bids = [1000, 1, 1, 1]
        result = entropy(bids)
        assert result < 0.1  # Should be very small

    def test_scale_invariance(self):
        """Entropy of shares is scale-invariant."""
        bids = [100, 90, 80, 70, 60]
        scaled_bids = [b * 1000 for b in bids]
        assert pytest.approx(entropy(bids), rel=1e-6) == entropy(scaled_bids)

    def test_permutation_invariance(self):
        """Entropy is unchanged by reordering bids."""
        bids = [100, 90, 80, 70, 60]
        shuffled = [80, 100, 60, 90, 70]
        assert pytest.approx(entropy(bids), rel=1e-6) == entropy(shuffled)

    def test_entropy_bounded_by_log_n(self):
        """H(p) ≤ log(n) for any bid distribution."""
        bids = [100, 80, 60, 40, 20]
        n = len(bids)
        result = entropy(bids)
        assert 0 <= result <= math.log(n)

    def test_batch_processing(self):
        """Entropy handles 2D batch input correctly."""
        auctions = np.array([[100, 100, 100], [100, 50, 25], [100, 100, 100]])
        results = entropy(auctions)
        assert hasattr(results, "shape") and results.shape == (3,)
        # Equal bids should give log(3)
        assert pytest.approx(results[0], rel=1e-6) == math.log(3)  # type: ignore[index]
        assert pytest.approx(results[2], rel=1e-6) == math.log(3)  # type: ignore[index]
        # Unequal bids should give less
        assert results[1] < math.log(3)  # type: ignore[index]


class TestEffectiveBidders:
    """Tests for effective number of bidders n_eff = exp(H(p)).

    This is the perplexity of the bid share distribution.
    """

    def test_single_bid_returns_one(self):
        """n_eff = 1 for single bidder."""
        assert effective_bidders([100]) == 1.0

    def test_equal_bids_gives_n(self):
        """n_eff = n when all bids are equal."""
        for n in [2, 3, 5, 10, 20]:
            bids = [100] * n
            result = effective_bidders(bids)
            assert pytest.approx(result, rel=1e-6) == n

    def test_dominant_bidder_approaches_one(self):
        """n_eff → 1 when one bidder dominates."""
        bids = [1000, 1, 1, 1]
        result = effective_bidders(bids)
        assert result < 1.5  # Should be close to 1

    def test_two_equal_bidders_rest_negligible(self):
        """n_eff ≈ 2 when only two bidders have significant share."""
        bids = [100, 100, 1, 1, 1]
        result = effective_bidders(bids)
        # Should be close to 2 since only two bidders matter
        assert 1.9 < result < 2.5

    def test_effective_bidders_bounded(self):
        """1 ≤ n_eff ≤ n for any bid distribution."""
        bids = [100, 80, 60, 40, 20]
        n = len(bids)
        result = effective_bidders(bids)
        assert 1 <= result <= n

    def test_batch_processing(self):
        """Effective bidders handles 2D batch input correctly."""
        auctions = np.array([[100, 100, 100], [100, 50, 25], [100, 100, 100]])
        results = effective_bidders(auctions)
        assert hasattr(results, "shape") and results.shape == (3,)
        # Equal bids should give n=3
        assert pytest.approx(results[0], rel=1e-6) == 3.0  # type: ignore[index]
        assert pytest.approx(results[2], rel=1e-6) == 3.0  # type: ignore[index]
        # Unequal bids should give less
        assert results[1] < 3.0  # type: ignore[index]


class TestSortingNetworks:
    """Tests for internal sorting network correctness.

    Sorting networks are used for small arrays (n <= 9) to speed up
    computation of order statistics. A bug here corrupts rho and Gini.
    """

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    def _test_sort_small(self, n: int, rng, n_trials: int = 10000):
        """Test _sort_small for a specific array size."""
        from acc_metrics._sorting import _sort_small

        for _ in range(n_trials):
            arr = rng.random(n)
            original = arr.copy()
            _sort_small(arr)
            expected = np.sort(original)
            if not np.allclose(arr, expected):
                return False, original, arr, expected
        return True, None, None, None

    def test_sort_n2(self, rng):
        """Sorting network for n=2 is correct."""
        success, orig, got, expected = self._test_sort_small(2, rng)
        assert success, f"n=2 failed: {orig} -> {got}, expected {expected}"

    def test_sort_n3(self, rng):
        """Sorting network for n=3 is correct."""
        success, orig, got, expected = self._test_sort_small(3, rng)
        assert success, f"n=3 failed: {orig} -> {got}, expected {expected}"

    def test_sort_n4(self, rng):
        """Sorting network for n=4 is correct."""
        success, orig, got, expected = self._test_sort_small(4, rng)
        assert success, f"n=4 failed: {orig} -> {got}, expected {expected}"

    def test_sort_n5(self, rng):
        """Sorting network for n=5 is correct."""
        success, orig, got, expected = self._test_sort_small(5, rng)
        assert success, f"n=5 failed: {orig} -> {got}, expected {expected}"

    def test_sort_n6(self, rng):
        """Sorting network for n=6 is correct."""
        success, orig, got, expected = self._test_sort_small(6, rng)
        assert success, f"n=6 failed: {orig} -> {got}, expected {expected}"

    def test_sort_n7(self, rng):
        """Sorting network for n=7 is correct."""
        success, orig, got, expected = self._test_sort_small(7, rng)
        assert success, f"n=7 failed: {orig} -> {got}, expected {expected}"

    def test_sort_n8(self, rng):
        """Sorting network for n=8 is correct."""
        success, orig, got, expected = self._test_sort_small(8, rng)
        assert success, f"n=8 failed: {orig} -> {got}, expected {expected}"

    def test_sort_n9(self, rng):
        """Sorting network for n=9 is correct."""
        success, orig, got, expected = self._test_sort_small(9, rng)
        assert success, f"n=9 failed: {orig} -> {got}, expected {expected}"

    def test_sort_n10_insertion(self, rng):
        """n=10 uses insertion sort (not buggy network)."""
        # n=10 should fall through to insertion sort since threshold is 9
        success, orig, got, expected = self._test_sort_small(10, rng)
        assert success, f"n=10 failed: {orig} -> {got}, expected {expected}"

    def test_sort_n11_insertion(self, rng):
        """n=11 uses insertion sort."""
        success, orig, got, expected = self._test_sort_small(11, rng)
        assert success, f"n=11 failed: {orig} -> {got}, expected {expected}"

    def test_sort_preserves_values(self, rng):
        """Sorting should preserve all values (no corruption)."""
        from acc_metrics._sorting import _sort_small

        for n in range(2, 12):
            for _ in range(1000):
                arr = rng.random(n)
                original = arr.copy()
                _sort_small(arr)
                # Same multiset of values
                assert np.allclose(np.sort(arr), np.sort(original))

    def test_competitive_ratio_uses_correct_sort(self, rng):
        """competitive_ratio should return correct values for all n."""
        for n in [3, 5, 7, 9, 10, 15, 20]:
            for _ in range(1000):
                bids = rng.uniform(1, 100, n)
                result = competitive_ratio(bids)
                # Manual calculation
                sorted_bids = np.sort(bids)[::-1]
                expected = sorted_bids[1] / sorted_bids[0]
                assert pytest.approx(result, rel=1e-10) == expected, (
                    f"n={n}: got {result}, expected {expected}"
                )

    def test_normalized_gini_uses_correct_sort(self, rng):
        """normalized_gini should return correct values for all n."""
        for n in [3, 5, 7, 9, 10, 15, 20]:
            for _ in range(1000):
                bids = rng.uniform(1, 100, n)
                result = normalized_gini(bids)
                # Gini should be in valid range
                assert 0 <= result < 1, f"n={n}: Gini {result} out of range"


class TestReverseAuctions:
    """Tests for reverse=True path (procurement auctions)."""

    def test_competitive_ratio_reverse(self):
        """In reverse auctions, ρ = b_(1)/b_(2) (lowest / second-lowest)."""
        bids = [50, 60, 70, 80, 100]
        rho_fwd = competitive_ratio(bids, reverse=False)
        rho_rev = competitive_ratio(bids, reverse=True)
        # Forward: 80/100 = 0.8, Reverse: 50/60 ≈ 0.833
        assert rho_fwd != rho_rev
        assert rho_fwd == pytest.approx(80 / 100, rel=1e-6)
        assert rho_rev == pytest.approx(50 / 60, rel=1e-6)

    def test_acc_reverse(self):
        """ACC with reverse=True should be positive and finite."""
        bids = [50, 60, 70, 80, 100]
        result = acc(bids, reverse=True)
        assert result > 0
        assert math.isfinite(result)

    def test_information_rent_reverse(self):
        """Information rent in reverse = 1 - ρ_reverse."""
        bids = [50, 60, 70, 80, 100]
        ir = information_rent(bids, reverse=True)
        rho = competitive_ratio(bids, reverse=True)
        assert pytest.approx(ir, rel=1e-6) == 1 - rho

    def test_acc_reverse_batch(self):
        """Batch reverse ACC should work on 2D input."""
        auctions = np.array(
            [
                [50, 60, 70, 80, 100],
                [10, 20, 30, 40, 50],
            ]
        )
        results = acc(auctions, reverse=True)
        assert isinstance(results, np.ndarray) and results.shape == (2,)
        assert all(r > 0 and math.isfinite(r) for r in results.tolist())


class TestBCaEdgeCases:
    """Tests for BCa bootstrap edge cases (M1, M2 guards)."""

    def test_n1_degenerate(self):
        """Single observation should return point interval."""
        lo, hi = bootstrap_bca_ci(np.array([5.0]))
        assert lo == 5.0
        assert hi == 5.0

    def test_n0_degenerate(self):
        """Empty array should return (0, 0)."""
        lo, hi = bootstrap_bca_ci(np.array([]))
        assert lo == 0.0
        assert hi == 0.0

    def test_identical_values(self):
        """All identical values should not produce NaN."""
        data = np.full(100, 3.14)
        lo, hi = bootstrap_bca_ci(data)
        assert math.isfinite(lo)
        assert math.isfinite(hi)
        assert lo == pytest.approx(3.14, rel=1e-6)
        assert hi == pytest.approx(3.14, rel=1e-6)

    def test_n2(self):
        """n=2 should work without error."""
        data = np.array([1.0, 2.0])
        lo, hi = bootstrap_bca_ci(data, seed=42)
        assert math.isfinite(lo)
        assert math.isfinite(hi)
        assert lo <= hi

    def test_n_bootstrap_zero_raises(self):
        """n_bootstrap=0 should raise ValueError, not ZeroDivisionError."""
        with pytest.raises(ValueError, match="n_bootstrap must be >= 1"):
            bootstrap_bca_ci(np.array([1.0, 2.0, 3.0]), n_bootstrap=0)

    def test_large_skew(self):
        """Highly skewed data should not produce NaN."""
        rng = np.random.default_rng(42)
        data = rng.exponential(1.0, 50)
        lo, hi = bootstrap_bca_ci(data, seed=42)
        assert math.isfinite(lo)
        assert math.isfinite(hi)
        assert lo < hi


class TestEquilibriumValidation:
    """Tests for equilibrium function parameter validation (m5 guards)."""

    def test_uniform_n_lt_2_raises(self):
        """first_price_bids_uniform should reject n < 2."""
        with pytest.raises(ValueError, match="n >= 2"):
            first_price_bids_uniform(np.array([0.5]), n=1)

    def test_exponential_n_lt_2_raises(self):
        """first_price_bids_exponential should reject n < 2."""
        with pytest.raises(ValueError, match="n >= 2"):
            first_price_bids_exponential(np.array([1.0]), n=1)

    def test_lognormal_n_lt_2_raises(self):
        """first_price_bids_lognormal should reject n < 2."""
        with pytest.raises(ValueError, match="n >= 2"):
            first_price_bids_lognormal(np.array([1.0]), n=1)

    def test_lognormal_negative_sigma_raises(self):
        """first_price_bids_lognormal should reject sigma <= 0."""
        with pytest.raises(ValueError, match="sigma > 0"):
            first_price_bids_lognormal(np.array([1.0]), n=5, sigma=-1.0)

    def test_pareto_n_lt_2_raises(self):
        """first_price_bids_pareto should reject n < 2."""
        with pytest.raises(ValueError, match="n >= 2"):
            first_price_bids_pareto(np.array([2.0]), n=1)

    def test_pareto_negative_alpha_raises(self):
        """first_price_bids_pareto should reject alpha <= 0."""
        with pytest.raises(ValueError, match="alpha > 0"):
            first_price_bids_pareto(np.array([2.0]), n=5, alpha=-1.0)

    def test_pareto_negative_xm_raises(self):
        """first_price_bids_pareto should reject x_m <= 0."""
        with pytest.raises(ValueError, match="x_m > 0"):
            first_price_bids_pareto(np.array([2.0]), n=5, x_m=-1.0)

    def test_uniform_bids_correct(self):
        """Uniform equilibrium bids: b(v) = (n-1)/n * v."""
        n = 5
        values = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        bids = first_price_bids_uniform(values, n=n)
        expected = values * (n - 1) / n
        np.testing.assert_allclose(bids, expected, rtol=1e-10)


class TestAccDelta:
    """Tests for the finite-resolution estimand ACC_delta and SAT_delta."""

    def test_floor_formula(self):
        """G_delta = (n-1)·delta/(n²·b_bar) on the active support."""
        n, c, d = 5, 100.0, 0.01
        expected = (n - 1) * d / (n**2 * c)
        assert dispersion_floor([c] * n, delta=d) == pytest.approx(expected)

    def test_floor_ignores_zero_bids(self):
        """Dummy (zero) bids are excluded from n and b_bar."""
        assert dispersion_floor([100.0, 100.0, 0.0], delta=0.01) == pytest.approx(
            dispersion_floor([100.0, 100.0], delta=0.01)
        )

    def test_tied_bids_saturate_and_stay_finite(self):
        """Equal bids: SAT=1, ACC_delta finite and equal to H·log(1+rho/floor)."""
        bids = [100.0] * 5
        d = 0.01
        assert saturation_flag(bids, delta=d) is True
        val = acc_delta(bids, delta=d)
        assert math.isfinite(val)
        floor = dispersion_floor(bids, delta=d)
        expected = math.log(5) * math.log1p(1.0 / floor)
        assert val == pytest.approx(expected, rel=1e-9)

    def test_unsaturated_equals_acc(self):
        """When G* exceeds the floor, ACC_delta coincides with ACC exactly."""
        bids = [100.0, 80.0, 60.0, 40.0, 20.0]
        assert saturation_flag(bids, delta=0.01) is False
        assert acc_delta(bids, delta=0.01) == pytest.approx(acc(bids), rel=1e-12)

    def test_not_scale_invariant_when_saturated(self):
        """Scaling tied bids up shrinks the relative floor and raises ACC_delta."""
        tied = [100.0] * 4
        scaled = [1000.0] * 4
        assert acc_delta(scaled, delta=0.01) > acc_delta(tied, delta=0.01)

    def test_scale_invariant_when_unsaturated(self):
        """Unsaturated observations inherit the scale invariance of ACC."""
        bids = np.array([100.0, 80.0, 60.0, 40.0])
        assert acc_delta(bids * 10, delta=0.01) == pytest.approx(
            acc_delta(bids, delta=0.01), rel=1e-12
        )

    def test_single_bidder_convention(self):
        """Fewer than two active bidders: ACC_delta = 0, not saturated."""
        assert acc_delta([100.0], delta=0.01) == 0.0
        assert acc_delta([100.0, 0.0], delta=0.01) == 0.0
        assert saturation_flag([100.0, 0.0], delta=0.01) is False

    def test_batch_shapes_and_consistency(self):
        """2D input returns per-row values matching the 1D path."""
        auctions = np.array(
            [
                [100.0, 100.0, 100.0],
                [100.0, 80.0, 60.0],
            ]
        )
        vals = np.asarray(acc_delta(auctions, delta=0.01))
        flags = np.asarray(saturation_flag(auctions, delta=0.01))
        assert vals.shape == (2,)
        assert flags.tolist() == [True, False]
        assert vals[0] == pytest.approx(acc_delta(auctions[0], delta=0.01))
        assert vals[1] == pytest.approx(acc_delta(auctions[1], delta=0.01))

    def test_reverse_auction_path(self):
        """Procurement mode: tied costs saturate; dispersed costs match acc()."""
        tied_costs = [50.0, 50.0, 50.0]
        assert saturation_flag(tied_costs, delta=0.01) is True
        assert math.isfinite(acc_delta(tied_costs, delta=0.01, reverse=True))

        costs = [50.0, 60.0, 80.0]
        assert acc_delta(costs, delta=0.01, reverse=True) == pytest.approx(
            acc(costs, reverse=True), rel=1e-12
        )

    def test_rejects_nonpositive_delta(self):
        """delta must be positive for all three functions."""
        with pytest.raises(ValueError, match="delta"):
            acc_delta([1.0, 2.0], delta=0.0)
        with pytest.raises(ValueError, match="delta"):
            dispersion_floor([1.0, 2.0], delta=-1.0)
        with pytest.raises(ValueError, match="delta"):
            saturation_flag([1.0, 2.0], delta=0.0)
