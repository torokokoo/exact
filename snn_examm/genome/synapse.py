"""Synapse (edge) classes for SNN genomes.

Mirrors RNN_Edge (rnn/rnn_edge.hxx) and RNN_Recurrent_Edge (rnn/rnn_recurrent_edge.hxx).
"""

from __future__ import annotations

import random

from .lif_node import LIFNode, bound


class Synapse:
    """Feedforward synapse connecting two nodes.

    Mirrors RNN_Edge: carries a single weight, connects input_node -> output_node.
    """

    def __init__(
        self,
        innovation_number: int,
        input_node_id: int,
        output_node_id: int,
        weight: float = 0.0,
    ):
        self.innovation_number = innovation_number
        self.input_node_id = input_node_id
        self.output_node_id = output_node_id
        self.weight = weight
        self.enabled = True
        self.forward_reachable = True
        self.backward_reachable = True

    def initialize_random(self) -> None:
        self.weight = random.uniform(-1.0, 1.0)

    def initialize_xavier(self, fan_in: int, fan_out: int) -> None:
        import math
        r = math.sqrt(6.0 / max(fan_in + fan_out, 1))
        self.weight = random.uniform(-r, r)

    def initialize_lamarckian(self, mu: float, sigma: float) -> None:
        self.weight = bound(random.gauss(mu, sigma))

    def is_reachable(self) -> bool:
        return self.forward_reachable and self.backward_reachable

    def copy(self) -> Synapse:
        s = Synapse(
            self.innovation_number,
            self.input_node_id,
            self.output_node_id,
            self.weight,
        )
        s.enabled = self.enabled
        s.forward_reachable = self.forward_reachable
        s.backward_reachable = self.backward_reachable
        return s

    def __repr__(self) -> str:
        return (
            f"Synapse(inn={self.innovation_number}, "
            f"{self.input_node_id}->{self.output_node_id}, "
            f"w={self.weight:.4f}, en={self.enabled})"
        )


class RecurrentSynapse:
    """Recurrent synapse with a time-delay (recurrent_depth).

    Mirrors RNN_Recurrent_Edge: delivers input_node's output at time t
    to output_node at time t + recurrent_depth.
    """

    def __init__(
        self,
        innovation_number: int,
        recurrent_depth: int,
        input_node_id: int,
        output_node_id: int,
        weight: float = 0.0,
    ):
        self.innovation_number = innovation_number
        self.recurrent_depth = recurrent_depth
        self.input_node_id = input_node_id
        self.output_node_id = output_node_id
        self.weight = weight
        self.enabled = True
        self.forward_reachable = True
        self.backward_reachable = True

    def initialize_random(self) -> None:
        self.weight = random.uniform(-1.0, 1.0)

    def initialize_xavier(self, fan_in: int, fan_out: int) -> None:
        import math
        r = math.sqrt(6.0 / max(fan_in + fan_out, 1))
        self.weight = random.uniform(-r, r)

    def initialize_lamarckian(self, mu: float, sigma: float) -> None:
        self.weight = bound(random.gauss(mu, sigma))

    def is_reachable(self) -> bool:
        return self.forward_reachable and self.backward_reachable

    def copy(self) -> RecurrentSynapse:
        rs = RecurrentSynapse(
            self.innovation_number,
            self.recurrent_depth,
            self.input_node_id,
            self.output_node_id,
            self.weight,
        )
        rs.enabled = self.enabled
        rs.forward_reachable = self.forward_reachable
        rs.backward_reachable = self.backward_reachable
        return rs

    def __repr__(self) -> str:
        return (
            f"RecurrentSynapse(inn={self.innovation_number}, "
            f"{self.input_node_id}->{self.output_node_id}, "
            f"depth={self.recurrent_depth}, w={self.weight:.4f})"
        )
