"""Tests for genome structural mutations."""

import pytest
from snn_examm.genome.snn_genome import SNNGenome, create_seed_genome
from snn_examm.genome.lif_node import HIDDEN_LAYER


class TestAddEdge:
    def test_adds_new_edge(self):
        g = create_seed_genome(2, 2)
        initial_edges = len(g.synapses)
        success, _ = g.add_edge(0.0, 0.5, 100)
        if success:
            assert len(g.synapses) > initial_edges

    def test_respects_depth_constraint(self):
        """All feedforward edges go from lower depth to higher depth."""
        g = create_seed_genome(3, 3)
        for _ in range(10):
            g.add_edge(0.0, 0.5, 200 + _)
        for s in g.synapses:
            src = g._find_node(s.input_node_id)
            dst = g._find_node(s.output_node_id)
            if src and dst:
                assert src.depth <= dst.depth


class TestAddRecurrentEdge:
    def test_adds_recurrent_edge(self):
        g = create_seed_genome(2, 2)
        initial = len(g.recurrent_synapses)
        success, _ = g.add_recurrent_edge(0.0, 0.5, 100, 1, 5)
        if success:
            assert len(g.recurrent_synapses) > initial

    def test_recurrent_depth_in_range(self):
        g = create_seed_genome(2, 2)
        for i in range(10):
            g.add_recurrent_edge(0.0, 0.5, 200 + i, 1, 5)
        for rs in g.recurrent_synapses:
            assert 1 <= rs.recurrent_depth <= 5


class TestAddNode:
    def test_adds_node_with_edges(self):
        g = create_seed_genome(2, 2)
        initial_nodes = len(g.nodes)
        initial_edges = len(g.synapses)
        success, _, _ = g.add_node(0.0, 0.5, 100, 100, 1, 5)
        if success:
            assert len(g.nodes) == initial_nodes + 1
            assert len(g.synapses) >= initial_edges + 2  # 2 new edges minimum

    def test_new_node_is_hidden(self):
        g = create_seed_genome(2, 2)
        g.add_node(0.0, 0.5, 100, 100)
        hidden = [n for n in g.nodes if n.layer_type == HIDDEN_LAYER]
        assert len(hidden) >= 2  # seed has 1, we added 1


class TestSplitEdge:
    def test_split_edge(self):
        g = create_seed_genome(2, 2)
        initial_nodes = len(g.nodes)
        success, _, _ = g.split_edge(0.0, 0.5, 100, 100)
        if success:
            assert len(g.nodes) == initial_nodes + 1
            # Original edge should be disabled
            disabled = [s for s in g.synapses if not s.enabled]
            assert len(disabled) >= 1


class TestEnableDisable:
    def test_disable_edge(self):
        g = create_seed_genome(2, 2)
        initial_enabled = sum(1 for s in g.synapses if s.enabled)
        success = g.disable_edge()
        if success:
            new_enabled = sum(1 for s in g.synapses if s.enabled)
            assert new_enabled == initial_enabled - 1

    def test_enable_edge(self):
        g = create_seed_genome(2, 2)
        g.disable_edge()
        disabled_count = sum(1 for s in g.synapses if not s.enabled)
        if disabled_count > 0:
            success = g.enable_edge()
            assert success

    def test_disable_node(self):
        g = create_seed_genome(2, 2)
        # Add extra hidden nodes first
        g.add_node(0.0, 0.5, 100, 100)
        success = g.disable_node()
        if success:
            disabled = [n for n in g.nodes if not n.enabled and n.layer_type == HIDDEN_LAYER]
            assert len(disabled) >= 1


class TestReachability:
    def test_all_seed_nodes_reachable(self):
        g = create_seed_genome(2, 2)
        g.assign_reachability()
        for n in g.nodes:
            assert n.is_reachable(), f"Node {n.innovation_number} should be reachable"

    def test_disconnected_node_unreachable(self):
        g = create_seed_genome(2, 2)
        # Add an isolated node
        from snn_examm.genome.lif_node import LIFNode
        isolated = LIFNode(999, HIDDEN_LAYER, 0.5)
        g.nodes.append(isolated)
        g.assign_reachability()
        assert not isolated.is_reachable()
