"""Tests for LIF neuron dynamics."""

import math
import pytest
from snn_examm.genome.lif_node import LIFNode, INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER, bound


class TestBound:
    def test_within_range(self):
        assert bound(5.0) == 5.0
        assert bound(-5.0) == -5.0

    def test_clamp_high(self):
        assert bound(15.0) == 10.0

    def test_clamp_low(self):
        assert bound(-15.0) == -10.0


class TestLIFNodeBasics:
    def test_hidden_node_has_3_weights(self):
        node = LIFNode(1, HIDDEN_LAYER, 0.5)
        assert node.get_number_weights() == 3

    def test_input_node_has_0_weights(self):
        node = LIFNode(1, INPUT_LAYER, 0.0)
        assert node.get_number_weights() == 0

    def test_output_node_has_0_weights(self):
        node = LIFNode(1, OUTPUT_LAYER, 1.0)
        assert node.get_number_weights() == 0

    def test_get_set_weights(self):
        node = LIFNode(1, HIDDEN_LAYER, 0.5)
        node.set_weights([1.5, 0.8, 0.3])
        w = node.get_weights()
        assert w == [1.5, 0.8, 0.3]

    def test_copy(self):
        node = LIFNode(1, HIDDEN_LAYER, 0.5)
        node.v_thresh = 2.0
        node.beta = 0.85
        node.bias = 0.1
        copy = node.copy()
        assert copy.innovation_number == 1
        assert copy.v_thresh == 2.0
        assert copy.beta == 0.85
        assert copy.bias == 0.1
        # Ensure independence
        copy.v_thresh = 3.0
        assert node.v_thresh == 2.0


class TestLIFDynamics:
    def test_input_node_passthrough(self):
        node = LIFNode(1, INPUT_LAYER, 0.0)
        node.input_fired(0.7)
        output = node.step()
        assert output == 0.7

    def test_hidden_node_subthreshold(self):
        """Below threshold: no spike, membrane decays."""
        node = LIFNode(1, HIDDEN_LAYER, 0.5)
        node.v_thresh = 1.0
        node.beta = 0.9
        node.bias = 0.0
        node.reset()

        node.input_fired(0.5)
        output = node.step()
        assert output == 0.0  # No spike
        assert node.membrane_potential == pytest.approx(0.5)

    def test_hidden_node_spike(self):
        """Above threshold: spike and reset."""
        node = LIFNode(1, HIDDEN_LAYER, 0.5)
        node.v_thresh = 1.0
        node.beta = 0.9
        node.bias = 0.0
        node.reset()

        node.input_fired(1.5)
        output = node.step()
        assert output == 1.0  # Spike
        assert node.membrane_potential == 0.0  # Reset
        assert node.spike_count == 1

    def test_hidden_node_membrane_decay(self):
        """Membrane potential decays by beta each step."""
        node = LIFNode(1, HIDDEN_LAYER, 0.5)
        node.v_thresh = 10.0  # High threshold so no spike
        node.beta = 0.8
        node.bias = 0.0
        node.reset()

        node.input_fired(1.0)
        node.step()
        # V = 0.8 * 0 + 1.0 = 1.0
        assert node.membrane_potential == pytest.approx(1.0)

        node.input_fired(0.0)
        node.step()
        # V = 0.8 * 1.0 + 0.0 = 0.8
        assert node.membrane_potential == pytest.approx(0.8)

        node.input_fired(0.0)
        node.step()
        # V = 0.8 * 0.8 + 0.0 = 0.64
        assert node.membrane_potential == pytest.approx(0.64)

    def test_hidden_node_spike_count(self):
        """Multiple spikes are counted."""
        node = LIFNode(1, HIDDEN_LAYER, 0.5)
        node.v_thresh = 0.5
        node.beta = 0.0  # No memory
        node.bias = 0.0
        node.reset()

        for _ in range(5):
            node.input_fired(1.0)
            node.step()

        assert node.spike_count == 5

    def test_output_node_accumulates(self):
        """Output nodes accumulate input values."""
        node = LIFNode(1, OUTPUT_LAYER, 1.0)
        node.reset()

        node.input_fired(1.0)
        node.step()
        assert node.current_output == pytest.approx(1.0)

        node.input_fired(2.0)
        node.step()
        assert node.current_output == pytest.approx(3.0)

    def test_reset_clears_state(self):
        node = LIFNode(1, HIDDEN_LAYER, 0.5)
        node.membrane_potential = 5.0
        node.spike_count = 10
        node.reset()
        assert node.membrane_potential == 0.0
        assert node.spike_count == 0


class TestLIFInitialization:
    def test_initialize_random(self):
        node = LIFNode(1, HIDDEN_LAYER, 0.5)
        node.initialize_random()
        assert 0.5 <= node.v_thresh <= 2.0
        assert 0.5 <= node.beta <= 0.99
        assert -0.5 <= node.bias <= 0.5

    def test_initialize_lamarckian(self):
        node = LIFNode(1, HIDDEN_LAYER, 0.5)
        node.initialize_lamarckian(0.0, 1.0)
        # Just check bounds are respected
        assert -10.0 <= node.v_thresh <= 10.0
        assert -10.0 <= node.beta <= 10.0
        assert -10.0 <= node.bias <= 10.0

    def test_input_node_init_noop(self):
        node = LIFNode(1, INPUT_LAYER, 0.0)
        original_thresh = node.v_thresh
        node.initialize_random()
        # Input nodes don't change internal weights
        assert node.v_thresh == original_thresh
