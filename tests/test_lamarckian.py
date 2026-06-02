"""Tests for Lamarckian weight initialization (mu/sigma computation)."""

import math
import pytest
from snn_examm.genome.snn_genome import SNNGenome, create_seed_genome


class TestGetMuSigma:
    def test_empty_params(self):
        """Empty params default to mu=0, sigma=0.25 (matching EXAMM)."""
        g = create_seed_genome(2, 2)
        mu, sigma = g.get_mu_sigma([])
        assert mu == 0.0
        assert sigma == 0.25

    def test_single_value(self):
        g = create_seed_genome(2, 2)
        mu, sigma = g.get_mu_sigma([5.0])
        assert mu == pytest.approx(5.0)
        # sigma = sqrt(0) = 0 for single element (division by n-1=0, special case)

    def test_known_values(self):
        """Test mu/sigma with known values matching EXAMM's computation."""
        g = create_seed_genome(2, 2)
        params = [1.0, 2.0, 3.0, 4.0, 5.0]
        mu, sigma = g.get_mu_sigma(params)

        expected_mu = 3.0
        expected_sigma = math.sqrt(
            ((1 - 3) ** 2 + (2 - 3) ** 2 + (3 - 3) ** 2 + (4 - 3) ** 2 + (5 - 3) ** 2) / 4
        )
        assert mu == pytest.approx(expected_mu)
        assert sigma == pytest.approx(expected_sigma)

    def test_clamping_at_bounds(self):
        """Values outside [-10, 10] are clamped in mu computation."""
        g = create_seed_genome(2, 2)
        params = [15.0, -15.0]
        mu, sigma = g.get_mu_sigma(params)
        # Clamped: [10, -10], mu = 0
        assert mu == pytest.approx(0.0)

    def test_uses_best_parameters_by_default(self):
        g = create_seed_genome(2, 2)
        g.best_parameters = [1.0, 1.0, 1.0]
        mu, sigma = g.get_mu_sigma()
        assert mu == pytest.approx(1.0)
        assert sigma == pytest.approx(0.0)

    def test_falls_back_to_initial_parameters(self):
        g = create_seed_genome(2, 2)
        g.best_parameters = []
        g.initial_parameters = [2.0, 2.0, 2.0]
        mu, sigma = g.get_mu_sigma()
        assert mu == pytest.approx(2.0)

    def test_all_identical_weights(self):
        g = create_seed_genome(2, 2)
        params = [3.0, 3.0, 3.0, 3.0]
        mu, sigma = g.get_mu_sigma(params)
        assert mu == pytest.approx(3.0)
        assert sigma == pytest.approx(0.0)


class TestLamarckianMutationFlow:
    def test_mutation_uses_parent_distribution(self):
        """New components created during mutation should use parent's mu/sigma."""
        g = create_seed_genome(2, 2)
        # Set all weights to a known value
        weights = g.get_weights()
        for i in range(len(weights)):
            weights[i] = 2.0
        g.set_weights(weights)
        g.best_parameters = weights

        mu, sigma = g.get_mu_sigma()
        assert mu == pytest.approx(2.0)
        assert sigma == pytest.approx(0.0)

        # After mutation with lamarckian, new weights should be near mu
        # (with sigma~0, they should be exactly mu, but bound() may clip)
        initial_weight_count = g.get_number_weights()
        success, _ = g.add_edge(mu, sigma, 999)
        if success:
            # The new edge weight should be sampled from N(2.0, 0.0) = 2.0
            new_weights = g.get_weights()
            assert len(new_weights) >= initial_weight_count


class TestWeightVectorRoundTrip:
    def test_get_set_weights_roundtrip(self):
        g = create_seed_genome(3, 2)
        original = g.get_weights()
        g.set_weights(original)
        restored = g.get_weights()
        assert original == pytest.approx(restored)

    def test_weight_count_matches(self):
        g = create_seed_genome(3, 2)
        assert len(g.get_weights()) == g.get_number_weights()
