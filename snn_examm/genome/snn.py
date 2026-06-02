"""Runnable Spiking Neural Network built from a genome.

Mirrors rnn/rnn.hxx — the executable network constructed from an SNNGenome.
Handles forward propagation through LIF neurons with feedforward and recurrent synapses.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .lif_node import LIFNode, INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER
from .synapse import Synapse, RecurrentSynapse


class SNN:
    """Spiking Neural Network for inference.

    Nodes are sorted by depth. At each simulation timestep:
    1. Input nodes receive observation values
    2. Feedforward synapses propagate weighted outputs
    3. Recurrent synapses deliver delayed outputs
    4. Each node computes its LIF step
    """

    def __init__(
        self,
        nodes: list[LIFNode],
        synapses: list[Synapse],
        recurrent_synapses: list[RecurrentSynapse],
    ):
        # Deep copy so the SNN owns its own state
        self.nodes = [n.copy() for n in nodes]
        self.synapses = [s.copy() for s in synapses]
        self.recurrent_synapses = [rs.copy() for rs in recurrent_synapses]

        # Build lookup maps
        self.node_map: dict[int, LIFNode] = {n.innovation_number: n for n in self.nodes}

        # Sort nodes by depth for forward propagation order
        self.nodes.sort(key=lambda n: (n.depth, n.innovation_number))

        # Separate node lists by layer
        self.input_nodes = [n for n in self.nodes if n.layer_type == INPUT_LAYER and n.enabled]
        self.hidden_nodes = [n for n in self.nodes if n.layer_type == HIDDEN_LAYER and n.enabled]
        self.output_nodes = [n for n in self.nodes if n.layer_type == OUTPUT_LAYER and n.enabled]

        # Build synapse maps: output_node_id -> list of (input_node_id, weight)
        self.forward_synapses: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for s in self.synapses:
            if s.enabled and s.is_reachable():
                self.forward_synapses[s.output_node_id].append(
                    (s.input_node_id, s.weight)
                )

        # Recurrent synapse history buffer: stores past outputs per node
        self.max_recurrent_depth = 0
        self.recurrent_map: list[tuple[int, int, int, float]] = []  # (input_id, output_id, depth, weight)
        for rs in self.recurrent_synapses:
            if rs.enabled and rs.is_reachable():
                self.recurrent_map.append(
                    (rs.input_node_id, rs.output_node_id, rs.recurrent_depth, rs.weight)
                )
                self.max_recurrent_depth = max(self.max_recurrent_depth, rs.recurrent_depth)

        # History buffer for recurrent connections: node_id -> deque of past outputs
        self.output_history: dict[int, list[float]] = {}
        self._init_history()

    def _init_history(self) -> None:
        """Initialize output history buffer for recurrent synapses."""
        self.output_history = {}
        if self.max_recurrent_depth > 0:
            for node in self.nodes:
                self.output_history[node.innovation_number] = [0.0] * (self.max_recurrent_depth + 1)
        self.history_index = 0

    def reset(self) -> None:
        """Reset all node states and history for a new episode."""
        for node in self.nodes:
            node.reset()
        self._init_history()

    def forward_step(self, observations: np.ndarray, sim_step: int) -> None:
        """Run one simulation timestep.

        Args:
            observations: Input observation values (one per input node).
            sim_step: Current simulation timestep index (for recurrent delay tracking).
        """
        # 1. Inject observations into input nodes
        for i, node in enumerate(self.input_nodes):
            if i < len(observations):
                node.input_fired(float(observations[i]))
            node.step()

        # 2. Process hidden and output nodes in depth order
        processing_nodes = [n for n in self.nodes
                           if n.layer_type != INPUT_LAYER and n.enabled]

        for node in processing_nodes:
            nid = node.innovation_number

            # Feedforward synapses: accumulate weighted inputs from source nodes
            if nid in self.forward_synapses:
                for src_id, weight in self.forward_synapses[nid]:
                    src_node = self.node_map.get(src_id)
                    if src_node:
                        node.input_fired(src_node.current_output * weight)

            # Recurrent synapses: inject delayed outputs
            if self.max_recurrent_depth > 0:
                for inp_id, out_id, depth, weight in self.recurrent_map:
                    if out_id == nid and sim_step >= depth:
                        past_idx = (self.history_index - depth) % (self.max_recurrent_depth + 1)
                        past_output = self.output_history.get(inp_id, [0.0] * (self.max_recurrent_depth + 1))[past_idx]
                        node.input_fired(past_output * weight)

            node.step()

        # 3. Update history buffer
        if self.max_recurrent_depth > 0:
            for node in self.nodes:
                if node.innovation_number in self.output_history:
                    self.output_history[node.innovation_number][self.history_index % (self.max_recurrent_depth + 1)] = node.current_output
            self.history_index += 1

    def get_output_spike_counts(self) -> list[int]:
        """Return spike counts for each output node."""
        return [n.spike_count for n in self.output_nodes]

    def get_output_potentials(self) -> list[float]:
        """Return membrane potentials for output nodes (used for action readout)."""
        return [n.membrane_potential for n in self.output_nodes]

    def get_output_values(self) -> list[float]:
        """Return current output values for output nodes."""
        return [n.current_output for n in self.output_nodes]

    def run(self, observations: np.ndarray, t_sim: int) -> list[float]:
        """Run the SNN for t_sim steps on given observations.

        Returns accumulated output values for output nodes.
        """
        # Reset spike counts for this decision step (but keep membrane state)
        for node in self.output_nodes:
            node.spike_count = 0
            node.membrane_potential = 0.0

        for t in range(t_sim):
            self.forward_step(observations, self.history_index)

        return self.get_output_values()
