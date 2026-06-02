"""Tests for SNN forward pass and network construction."""

import numpy as np
import pytest

from snn_examm.genome.lif_node import LIFNode, INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER
from snn_examm.genome.synapse import Synapse, RecurrentSynapse
from snn_examm.genome.snn import SNN


def make_simple_network():
    """Create a minimal SNN: 2 inputs -> 1 hidden -> 2 outputs."""
    nodes = [
        LIFNode(1, INPUT_LAYER, 0.0),
        LIFNode(2, INPUT_LAYER, 0.0),
        LIFNode(3, HIDDEN_LAYER, 0.5),
        LIFNode(4, OUTPUT_LAYER, 1.0),
        LIFNode(5, OUTPUT_LAYER, 1.0),
    ]
    # Set hidden node params
    nodes[2].v_thresh = 1.0
    nodes[2].beta = 0.9
    nodes[2].bias = 0.0

    synapses = [
        Synapse(1, 1, 3, weight=1.0),   # input1 -> hidden
        Synapse(2, 2, 3, weight=1.0),   # input2 -> hidden
        Synapse(3, 3, 4, weight=1.0),   # hidden -> output1
        Synapse(4, 3, 5, weight=0.5),   # hidden -> output2
    ]
    return nodes, synapses


class TestSNNConstruction:
    def test_builds_from_lists(self):
        nodes, synapses = make_simple_network()
        snn = SNN(nodes, synapses, [])
        assert len(snn.input_nodes) == 2
        assert len(snn.hidden_nodes) == 1
        assert len(snn.output_nodes) == 2

    def test_nodes_sorted_by_depth(self):
        nodes, synapses = make_simple_network()
        snn = SNN(nodes, synapses, [])
        depths = [n.depth for n in snn.nodes]
        assert depths == sorted(depths)


class TestSNNForwardPass:
    def test_single_step(self):
        nodes, synapses = make_simple_network()
        snn = SNN(nodes, synapses, [])
        snn.reset()

        obs = np.array([0.5, 0.5])
        snn.forward_step(obs, 0)

        # Input nodes output 0.5 each
        # Hidden: V = 0.9*0 + 0.5*1.0 + 0.5*1.0 + 0 = 1.0 >= 1.0 -> spike
        # Output1 receives 1.0 * 1.0 = 1.0
        # Output2 receives 1.0 * 0.5 = 0.5
        assert snn.output_nodes[0].current_output == pytest.approx(1.0)
        assert snn.output_nodes[1].current_output == pytest.approx(0.5)

    def test_run_returns_values(self):
        nodes, synapses = make_simple_network()
        snn = SNN(nodes, synapses, [])
        snn.reset()

        obs = np.array([0.5, 0.5])
        outputs = snn.run(obs, t_sim=5)
        assert len(outputs) == 2
        # Outputs should be non-zero since hidden node spikes on first step
        assert any(v != 0 for v in outputs)

    def test_reset_clears_state(self):
        nodes, synapses = make_simple_network()
        snn = SNN(nodes, synapses, [])

        obs = np.array([1.0, 1.0])
        snn.run(obs, t_sim=5)
        snn.reset()

        for n in snn.nodes:
            assert n.membrane_potential == 0.0
            assert n.spike_count == 0


class TestSNNRecurrent:
    def test_recurrent_synapse_delays(self):
        """Recurrent synapses inject values from past timesteps."""
        nodes = [
            LIFNode(1, INPUT_LAYER, 0.0),
            LIFNode(2, HIDDEN_LAYER, 0.5),
            LIFNode(3, OUTPUT_LAYER, 1.0),
        ]
        nodes[1].v_thresh = 0.5
        nodes[1].beta = 0.0
        nodes[1].bias = 0.0

        synapses = [
            Synapse(1, 1, 2, weight=1.0),
            Synapse(2, 2, 3, weight=1.0),
        ]
        # Self-recurrent on hidden node with depth 1
        rec_synapses = [
            RecurrentSynapse(3, 1, 2, 2, weight=0.5),
        ]

        snn = SNN(nodes, synapses, rec_synapses)
        snn.reset()

        # Step 1: input=1.0, hidden gets 1.0 >= 0.5 -> spike
        snn.forward_step(np.array([1.0]), 0)
        assert snn.hidden_nodes[0].spike_count == 1

        # Step 2: input=0.0, hidden gets 0 + recurrent(1.0*0.5) = 0.5 >= 0.5 -> spike
        snn.forward_step(np.array([0.0]), 1)
        assert snn.hidden_nodes[0].spike_count == 2
