"""Leaky Integrate-and-Fire neuron node for SNN genomes.

Mirrors RNN_Node_Interface from rnn/rnn_node_interface.hxx.
"""

from __future__ import annotations

import copy
import math
import random

INPUT_LAYER = 0
HIDDEN_LAYER = 1
OUTPUT_LAYER = 2


def bound(value: float) -> float:
    """Clamp value to [-10, 10], matching EXAMM's bound()."""
    return max(-10.0, min(10.0, value))


class LIFNode:
    """Leaky Integrate-and-Fire neuron.

    Dynamics (hidden layer only):
        V(t+1) = beta * V(t) + input_current + bias
        spike(t) = 1 if V(t) >= V_thresh else 0
        V(t) = 0 after spike (hard reset)

    Input nodes: pass-through (output = input current directly).
    Output nodes: accumulate input, output = accumulated value (no spike).

    Evolvable parameters: V_thresh, beta, bias (3 weights for hidden nodes).
    Input/output nodes have 0 evolvable parameters.
    """

    def __init__(
        self,
        innovation_number: int,
        layer_type: int,
        depth: float,
        parameter_name: str = "",
    ):
        self.innovation_number = innovation_number
        self.layer_type = layer_type
        self.depth = depth
        self.parameter_name = parameter_name

        self.enabled = True
        self.forward_reachable = True
        self.backward_reachable = True

        # LIF parameters (only meaningful for hidden nodes)
        self.v_thresh = 1.0
        self.beta = 0.9
        self.bias = 0.0

        # Runtime state
        self.membrane_potential = 0.0
        self.spike_count = 0
        self.current_output = 0.0

        # Per-timestep accumulators for the forward pass
        self.input_accumulator = 0.0
        self.inputs_fired_count = 0
        self.total_inputs = 0

    def get_number_weights(self) -> int:
        if self.layer_type == HIDDEN_LAYER:
            return 3  # v_thresh, beta, bias
        return 0

    def get_weights(self) -> list[float]:
        if self.layer_type == HIDDEN_LAYER:
            return [self.v_thresh, self.beta, self.bias]
        return []

    def set_weights(self, params: list[float]) -> None:
        if self.layer_type == HIDDEN_LAYER and len(params) >= 3:
            self.v_thresh = params[0]
            self.beta = params[1]
            self.bias = params[2]

    def get_weights_offset(self, offset: int, parameters: list[float]) -> int:
        if self.layer_type == HIDDEN_LAYER:
            self.v_thresh = parameters[offset]
            self.beta = parameters[offset + 1]
            self.bias = parameters[offset + 2]
            return offset + 3
        return offset

    def set_weights_to_vector(self, offset: int, parameters: list[float]) -> int:
        if self.layer_type == HIDDEN_LAYER:
            parameters[offset] = self.v_thresh
            parameters[offset + 1] = self.beta
            parameters[offset + 2] = self.bias
            return offset + 3
        return offset

    def initialize_random(self) -> None:
        if self.layer_type == HIDDEN_LAYER:
            self.v_thresh = random.uniform(0.5, 2.0)
            self.beta = random.uniform(0.5, 0.99)
            self.bias = random.uniform(-0.5, 0.5)

    def initialize_xavier(self, fan_in: int, fan_out: int) -> None:
        if self.layer_type == HIDDEN_LAYER:
            r = math.sqrt(6.0 / max(fan_in + fan_out, 1))
            self.v_thresh = random.uniform(0.5, 2.0)
            self.beta = random.uniform(0.5, 0.99)
            self.bias = bound(random.uniform(-r, r))

    def initialize_lamarckian(self, mu: float, sigma: float) -> None:
        """Initialize weights from N(mu, sigma), mirroring EXAMM's Lamarckian init."""
        if self.layer_type == HIDDEN_LAYER:
            self.v_thresh = bound(random.gauss(mu, sigma))
            self.beta = bound(random.gauss(mu, sigma))
            self.bias = bound(random.gauss(mu, sigma))

    def reset(self) -> None:
        """Reset runtime state for a new evaluation."""
        self.membrane_potential = 0.0
        self.spike_count = 0
        self.current_output = 0.0
        self.input_accumulator = 0.0
        self.inputs_fired_count = 0

    def input_fired(self, value: float) -> None:
        """Receive input from an incoming synapse."""
        self.input_accumulator += value
        self.inputs_fired_count += 1

    def step(self) -> float:
        """Execute one simulation timestep and return output.

        For input nodes: output = accumulated input (pass-through).
        For hidden nodes: LIF dynamics with spike output.
        For output nodes: accumulate input (no spike), output = accumulated value.
        """
        if self.layer_type == INPUT_LAYER:
            self.current_output = self.input_accumulator
        elif self.layer_type == HIDDEN_LAYER:
            self.membrane_potential = (
                self.beta * self.membrane_potential
                + self.input_accumulator
                + self.bias
            )
            if self.membrane_potential >= self.v_thresh:
                self.current_output = 1.0
                self.spike_count += 1
                self.membrane_potential = 0.0  # hard reset
            else:
                self.current_output = 0.0
        elif self.layer_type == OUTPUT_LAYER:
            self.membrane_potential += self.input_accumulator
            self.current_output = self.membrane_potential

        self.input_accumulator = 0.0
        self.inputs_fired_count = 0
        return self.current_output

    def is_reachable(self) -> bool:
        return self.forward_reachable and self.backward_reachable

    def copy(self) -> LIFNode:
        """Deep copy this node."""
        new_node = LIFNode(
            self.innovation_number,
            self.layer_type,
            self.depth,
            self.parameter_name,
        )
        new_node.enabled = self.enabled
        new_node.forward_reachable = self.forward_reachable
        new_node.backward_reachable = self.backward_reachable
        new_node.v_thresh = self.v_thresh
        new_node.beta = self.beta
        new_node.bias = self.bias
        new_node.total_inputs = self.total_inputs
        return new_node

    def __repr__(self) -> str:
        layer_names = {INPUT_LAYER: "IN", HIDDEN_LAYER: "HID", OUTPUT_LAYER: "OUT"}
        return (
            f"LIFNode(inn={self.innovation_number}, "
            f"layer={layer_names.get(self.layer_type, '?')}, "
            f"depth={self.depth:.2f})"
        )
