"""SNN Genome — the evolvable network representation.

Mirrors RNN_Genome from rnn/rnn_genome.hxx.
Contains nodes (LIF neurons), synapses, recurrent synapses, and weight vectors.
Supports structural mutations, Lamarckian weight init, and building runnable SNNs.
"""

from __future__ import annotations

import math
import random
from bisect import insort
from typing import Optional

from .lif_node import LIFNode, INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER, bound
from .synapse import Synapse, RecurrentSynapse
from .snn import SNN


class SNNGenome:
    """Evolvable SNN genome with structural mutations and Lamarckian weight init.

    Faithfully ports EXAMM's RNN_Genome mutation/crossover/weight logic.
    """

    def __init__(
        self,
        nodes: list[LIFNode],
        synapses: list[Synapse],
        recurrent_synapses: list[RecurrentSynapse],
        weight_initialize: str = "xavier",
        mutated_component_weight: str = "lamarckian",
        weight_inheritance: str = "lamarckian",
    ):
        self.nodes = nodes
        self.synapses = synapses
        self.recurrent_synapses = recurrent_synapses

        self.weight_initialize = weight_initialize
        self.mutated_component_weight = mutated_component_weight
        self.weight_inheritance = weight_inheritance

        self.generation_id = -1
        self.group_id = -1  # island assignment
        self.parent_ids: list[int] = []
        self.generated_by = ""

        self.fitness: float = float("-inf")
        self.initial_parameters: list[float] = []
        self.best_parameters: list[float] = []

        # Harada frequency-based parent selection
        self.search_frequency: float = 0.0

    # ─── Weight Vector Operations ────────────────────────────────────────

    def get_number_weights(self) -> int:
        count = 0
        for s in self.synapses:
            count += 1
        for rs in self.recurrent_synapses:
            count += 1
        for n in self.nodes:
            count += n.get_number_weights()
        return count

    def get_weights(self) -> list[float]:
        """Extract flat weight vector. Order: edges, recurrent edges, node internals.

        Same order as EXAMM's RNN_Genome::get_weights (rnn_genome.cxx).
        """
        params: list[float] = []
        for s in self.synapses:
            params.append(s.weight)
        for rs in self.recurrent_synapses:
            params.append(rs.weight)
        for n in self.nodes:
            params.extend(n.get_weights())
        return params

    def set_weights(self, parameters: list[float]) -> None:
        """Set weights from flat vector. Must match get_weights() order."""
        offset = 0
        for s in self.synapses:
            s.weight = parameters[offset]
            offset += 1
        for rs in self.recurrent_synapses:
            rs.weight = parameters[offset]
            offset += 1
        for n in self.nodes:
            nw = n.get_number_weights()
            if nw > 0:
                n.set_weights(parameters[offset : offset + nw])
                offset += nw

    def get_mu_sigma(self, params: Optional[list[float]] = None) -> tuple[float, float]:
        """Compute mu and sigma from weight vector.

        Exact port of RNN_Genome::get_mu_sigma (rnn_genome.cxx:1714-1777).
        """
        if params is None:
            params = self.best_parameters if self.best_parameters else self.initial_parameters

        if not params:
            return 0.0, 0.25

        mu = 0.0
        for p in params:
            if p < -10.0:
                mu += -10.0
            elif p > 10.0:
                mu += 10.0
            else:
                mu += p
        mu /= len(params)

        sigma = 0.0
        for p in params:
            sigma += (mu - p) ** 2

        if len(params) > 1:
            sigma /= len(params) - 1
        sigma = math.sqrt(sigma)

        if math.isnan(mu) or math.isinf(mu) or math.isnan(sigma) or math.isinf(sigma):
            raise ValueError(f"mu or sigma is not a number: mu={mu}, sigma={sigma}")

        if mu < -11.0 or mu > 11.0 or sigma < 0.0 or sigma > 30.0:
            mu = max(-11.0, min(11.0, mu))
            sigma = max(0.0, min(30.0, sigma))

        return mu, sigma

    def snapshot_parameters(self) -> None:
        """Save current weights as initial_parameters."""
        self.initial_parameters = self.get_weights()

    def snapshot_best_parameters(self) -> None:
        """Save current weights as best_parameters (called after fitness evaluation)."""
        self.best_parameters = self.get_weights()

    # ─── Weight Initialization ───────────────────────────────────────────

    def initialize_randomly(self) -> None:
        """Initialize all weights using the configured weight_initialize method."""
        if self.weight_initialize == "random":
            for s in self.synapses:
                s.initialize_random()
            for rs in self.recurrent_synapses:
                rs.initialize_random()
            for n in self.nodes:
                n.initialize_random()
        elif self.weight_initialize == "xavier":
            for n in self.nodes:
                fan_in = self._get_fan_in(n.innovation_number)
                fan_out = self._get_fan_out(n.innovation_number)
                n.initialize_xavier(fan_in, fan_out)
            for s in self.synapses:
                out_node = self._find_node(s.output_node_id)
                if out_node:
                    fan_in = self._get_fan_in(out_node.innovation_number)
                    fan_out = self._get_fan_out(out_node.innovation_number)
                    s.initialize_xavier(fan_in, fan_out)
            for rs in self.recurrent_synapses:
                out_node = self._find_node(rs.output_node_id)
                if out_node:
                    fan_in = self._get_fan_in(out_node.innovation_number)
                    fan_out = self._get_fan_out(out_node.innovation_number)
                    rs.initialize_xavier(fan_in, fan_out)
        else:
            # Default to random
            for s in self.synapses:
                s.initialize_random()
            for rs in self.recurrent_synapses:
                rs.initialize_random()
            for n in self.nodes:
                n.initialize_random()

        self.snapshot_parameters()

    def _get_fan_in(self, node_id: int) -> int:
        count = 0
        for s in self.synapses:
            if s.output_node_id == node_id and s.enabled:
                count += 1
        for rs in self.recurrent_synapses:
            if rs.output_node_id == node_id and rs.enabled:
                count += 1
        return max(count, 1)

    def _get_fan_out(self, node_id: int) -> int:
        count = 0
        for s in self.synapses:
            if s.input_node_id == node_id and s.enabled:
                count += 1
        for rs in self.recurrent_synapses:
            if rs.input_node_id == node_id and rs.enabled:
                count += 1
        return max(count, 1)

    def _find_node(self, innovation_number: int) -> Optional[LIFNode]:
        for n in self.nodes:
            if n.innovation_number == innovation_number:
                return n
        return None

    # ─── Structural Mutations ────────────────────────────────────────────
    # Ported from rnn_genome.cxx:1978-2300

    def _init_new_edge_weight(self, synapse, mu: float, sigma: float) -> None:
        """Initialize a new edge weight based on mutated_component_weight setting."""
        if self.mutated_component_weight == "lamarckian":
            synapse.initialize_lamarckian(mu, sigma)
        elif self.mutated_component_weight == "xavier":
            out_node = self._find_node(synapse.output_node_id)
            if out_node:
                fan_in = self._get_fan_in(out_node.innovation_number)
                fan_out = self._get_fan_out(out_node.innovation_number)
                synapse.initialize_xavier(fan_in, fan_out)
            else:
                synapse.initialize_random()
        else:
            synapse.initialize_random()

    def _init_new_node_weight(self, node: LIFNode, mu: float, sigma: float) -> None:
        """Initialize a new node's weights based on mutated_component_weight setting."""
        if self.mutated_component_weight == "lamarckian":
            node.initialize_lamarckian(mu, sigma)
        elif self.mutated_component_weight == "xavier":
            fan_in = self._get_fan_in(node.innovation_number)
            fan_out = self._get_fan_out(node.innovation_number)
            node.initialize_xavier(fan_in, fan_out)
        else:
            node.initialize_random()

    def add_edge(
        self, mu: float, sigma: float, edge_innovation_count: int
    ) -> tuple[bool, int]:
        """Add a feedforward edge between two random reachable nodes at different depths.

        Port of RNN_Genome::add_edge (rnn_genome.cxx:1978-2026).
        Returns (success, updated_edge_innovation_count).
        """
        reachable = [n for n in self.nodes if n.is_reachable()]
        if len(reachable) < 2:
            return False, edge_innovation_count

        n1 = random.choice(reachable)
        candidates = [n for n in reachable if n.depth != n1.depth]
        if not candidates:
            return False, edge_innovation_count
        n2 = random.choice(candidates)

        # Ensure n1.depth < n2.depth (feedforward direction)
        if n2.depth < n1.depth:
            n1, n2 = n2, n1

        # Check if edge already exists
        for s in self.synapses:
            if s.input_node_id == n1.innovation_number and s.output_node_id == n2.innovation_number:
                if not s.enabled:
                    s.enabled = True
                    return True, edge_innovation_count
                return False, edge_innovation_count

        edge_innovation_count += 1
        new_syn = Synapse(edge_innovation_count, n1.innovation_number, n2.innovation_number)
        self._init_new_edge_weight(new_syn, mu, sigma)
        self.synapses.append(new_syn)
        self.synapses.sort(key=lambda s: s.innovation_number)
        return True, edge_innovation_count

    def add_recurrent_edge(
        self,
        mu: float,
        sigma: float,
        edge_innovation_count: int,
        min_recurrent_depth: int = 1,
        max_recurrent_depth: int = 10,
    ) -> tuple[bool, int]:
        """Add a recurrent edge between two random reachable nodes.

        Port of RNN_Genome::add_recurrent_edge (rnn_genome.cxx:2028-2066).
        """
        possible_input = [n for n in self.nodes if n.is_reachable()]
        possible_output = [n for n in self.nodes if n.is_reachable() and n.layer_type != INPUT_LAYER]

        if not possible_input or not possible_output:
            return False, edge_innovation_count

        n1 = random.choice(possible_input)
        n2 = random.choice(possible_output)
        rec_depth = random.randint(min_recurrent_depth, max_recurrent_depth)

        # Check if this exact recurrent edge already exists
        for rs in self.recurrent_synapses:
            if (rs.input_node_id == n1.innovation_number
                    and rs.output_node_id == n2.innovation_number
                    and rs.recurrent_depth == rec_depth):
                if not rs.enabled:
                    rs.enabled = True
                    return True, edge_innovation_count
                return False, edge_innovation_count

        edge_innovation_count += 1
        new_rs = RecurrentSynapse(
            edge_innovation_count, rec_depth,
            n1.innovation_number, n2.innovation_number,
        )
        self._init_new_edge_weight(new_rs, mu, sigma)
        self.recurrent_synapses.append(new_rs)
        self.recurrent_synapses.sort(key=lambda rs: rs.innovation_number)
        return True, edge_innovation_count

    def disable_edge(self) -> bool:
        """Disable a random enabled edge. Port of rnn_genome.cxx:2069-2103."""
        enabled = [s for s in self.synapses if s.enabled]
        enabled_rec = [rs for rs in self.recurrent_synapses if rs.enabled]
        all_enabled = enabled + enabled_rec

        if not all_enabled:
            return False

        choice = random.choice(all_enabled)
        choice.enabled = False
        return True

    def enable_edge(self) -> bool:
        """Enable a random disabled edge. Port of rnn_genome.cxx:2105-2143."""
        disabled = [s for s in self.synapses if not s.enabled]
        disabled_rec = [rs for rs in self.recurrent_synapses if not rs.enabled]
        all_disabled = disabled + disabled_rec

        if not all_disabled:
            return False

        choice = random.choice(all_disabled)
        choice.enabled = True
        return True

    def add_node(
        self,
        mu: float,
        sigma: float,
        edge_innovation_count: int,
        node_innovation_count: int,
        min_recurrent_depth: int = 1,
        max_recurrent_depth: int = 10,
    ) -> tuple[bool, int, int]:
        """Add a new hidden LIF node connected to random input and output nodes.

        Port of RNN_Genome::add_node (rnn_genome.cxx).
        Returns (success, edge_inn_count, node_inn_count).
        """
        input_nodes = [n for n in self.nodes if n.is_reachable() and n.layer_type != OUTPUT_LAYER]
        output_nodes = [n for n in self.nodes if n.is_reachable() and n.layer_type != INPUT_LAYER]

        if not input_nodes or not output_nodes:
            return False, edge_innovation_count, node_innovation_count

        src = random.choice(input_nodes)
        dst = random.choice(output_nodes)

        # New node depth is between src and dst
        if src.depth >= dst.depth:
            new_depth = (src.depth + dst.depth) / 2.0
        else:
            new_depth = (src.depth + dst.depth) / 2.0

        node_innovation_count += 1
        new_node = LIFNode(node_innovation_count, HIDDEN_LAYER, new_depth)
        self._init_new_node_weight(new_node, mu, sigma)
        self.nodes.append(new_node)
        self.nodes.sort(key=lambda n: (n.depth, n.innovation_number))

        # Connect src -> new_node
        edge_innovation_count += 1
        e1 = Synapse(edge_innovation_count, src.innovation_number, new_node.innovation_number)
        self._init_new_edge_weight(e1, mu, sigma)
        self.synapses.append(e1)

        # Connect new_node -> dst
        edge_innovation_count += 1
        e2 = Synapse(edge_innovation_count, new_node.innovation_number, dst.innovation_number)
        self._init_new_edge_weight(e2, mu, sigma)
        self.synapses.append(e2)

        self.synapses.sort(key=lambda s: s.innovation_number)
        return True, edge_innovation_count, node_innovation_count

    def split_edge(
        self,
        mu: float,
        sigma: float,
        edge_innovation_count: int,
        node_innovation_count: int,
    ) -> tuple[bool, int, int]:
        """Split a random enabled edge by inserting a new node.

        Port of RNN_Genome::split_edge (rnn_genome.cxx:2145+).
        Disables the original edge, creates a new node at midpoint depth,
        and connects with two new edges.
        """
        enabled_edges = [s for s in self.synapses if s.enabled]
        enabled_rec = [rs for rs in self.recurrent_synapses if rs.enabled]

        all_enabled = enabled_edges + enabled_rec
        if not all_enabled:
            return False, edge_innovation_count, node_innovation_count

        edge = random.choice(all_enabled)
        edge.enabled = False

        n1 = self._find_node(edge.input_node_id)
        n2 = self._find_node(edge.output_node_id)
        if not n1 or not n2:
            return False, edge_innovation_count, node_innovation_count

        new_depth = (n1.depth + n2.depth) / 2.0

        node_innovation_count += 1
        new_node = LIFNode(node_innovation_count, HIDDEN_LAYER, new_depth)
        self._init_new_node_weight(new_node, mu, sigma)
        self.nodes.append(new_node)
        self.nodes.sort(key=lambda n: (n.depth, n.innovation_number))

        # Connect n1 -> new_node -> n2
        edge_innovation_count += 1
        e1 = Synapse(edge_innovation_count, n1.innovation_number, new_node.innovation_number)
        self._init_new_edge_weight(e1, mu, sigma)
        self.synapses.append(e1)

        edge_innovation_count += 1
        e2 = Synapse(edge_innovation_count, new_node.innovation_number, n2.innovation_number)
        self._init_new_edge_weight(e2, mu, sigma)
        self.synapses.append(e2)

        self.synapses.sort(key=lambda s: s.innovation_number)
        return True, edge_innovation_count, node_innovation_count

    def enable_node(self) -> bool:
        """Enable a random disabled hidden node."""
        disabled = [n for n in self.nodes if not n.enabled and n.layer_type == HIDDEN_LAYER]
        if not disabled:
            return False
        random.choice(disabled).enabled = True
        return True

    def disable_node(self) -> bool:
        """Disable a random enabled hidden node."""
        enabled = [n for n in self.nodes if n.enabled and n.layer_type == HIDDEN_LAYER]
        if not enabled:
            return False
        random.choice(enabled).enabled = False
        return True

    # ─── Reachability Analysis ───────────────────────────────────────────

    def assign_reachability(self) -> None:
        """Mark nodes/edges as forward/backward reachable.

        Port of RNN_Genome::assign_reachability.
        """
        # Reset all
        for n in self.nodes:
            n.forward_reachable = n.layer_type == INPUT_LAYER
            n.backward_reachable = n.layer_type == OUTPUT_LAYER

        # Forward pass: from inputs forward through edges
        changed = True
        while changed:
            changed = False
            for s in self.synapses:
                if not s.enabled:
                    continue
                src = self._find_node(s.input_node_id)
                dst = self._find_node(s.output_node_id)
                if src and dst and src.forward_reachable and not dst.forward_reachable:
                    dst.forward_reachable = True
                    changed = True
            for rs in self.recurrent_synapses:
                if not rs.enabled:
                    continue
                src = self._find_node(rs.input_node_id)
                dst = self._find_node(rs.output_node_id)
                if src and dst and src.forward_reachable and not dst.forward_reachable:
                    dst.forward_reachable = True
                    changed = True

        # Backward pass: from outputs backward through edges
        changed = True
        while changed:
            changed = False
            for s in self.synapses:
                if not s.enabled:
                    continue
                src = self._find_node(s.input_node_id)
                dst = self._find_node(s.output_node_id)
                if src and dst and dst.backward_reachable and not src.backward_reachable:
                    src.backward_reachable = True
                    changed = True
            for rs in self.recurrent_synapses:
                if not rs.enabled:
                    continue
                src = self._find_node(rs.input_node_id)
                dst = self._find_node(rs.output_node_id)
                if src and dst and dst.backward_reachable and not src.backward_reachable:
                    src.backward_reachable = True
                    changed = True

        # Mark edges reachable based on their endpoints
        for s in self.synapses:
            src = self._find_node(s.input_node_id)
            dst = self._find_node(s.output_node_id)
            if src and dst:
                s.forward_reachable = src.forward_reachable and dst.forward_reachable
                s.backward_reachable = src.backward_reachable and dst.backward_reachable
        for rs in self.recurrent_synapses:
            src = self._find_node(rs.input_node_id)
            dst = self._find_node(rs.output_node_id)
            if src and dst:
                rs.forward_reachable = src.forward_reachable and dst.forward_reachable
                rs.backward_reachable = src.backward_reachable and dst.backward_reachable

    # ─── Build Runnable SNN ──────────────────────────────────────────────

    def build_snn(self) -> SNN:
        """Construct a runnable SNN from this genome."""
        enabled_nodes = [n for n in self.nodes if n.enabled and n.is_reachable()]
        enabled_synapses = [s for s in self.synapses if s.enabled and s.is_reachable()]
        enabled_rec = [rs for rs in self.recurrent_synapses if rs.enabled and rs.is_reachable()]
        return SNN(enabled_nodes, enabled_synapses, enabled_rec)

    # ─── Copy ────────────────────────────────────────────────────────────

    def copy(self) -> SNNGenome:
        """Deep copy. Port of RNN_Genome::copy."""
        new_nodes = [n.copy() for n in self.nodes]
        new_synapses = [s.copy() for s in self.synapses]
        new_rec = [rs.copy() for rs in self.recurrent_synapses]

        g = SNNGenome(
            new_nodes, new_synapses, new_rec,
            weight_initialize=self.weight_initialize,
            mutated_component_weight=self.mutated_component_weight,
            weight_inheritance=self.weight_inheritance,
        )
        g.generation_id = self.generation_id
        g.group_id = self.group_id
        g.parent_ids = list(self.parent_ids)
        g.generated_by = self.generated_by
        g.fitness = self.fitness
        g.initial_parameters = list(self.initial_parameters)
        g.best_parameters = list(self.best_parameters)
        g.search_frequency = self.search_frequency
        return g

    # ─── Utility ─────────────────────────────────────────────────────────

    def get_enabled_edge_count(self) -> int:
        return sum(1 for s in self.synapses if s.enabled)

    def get_enabled_recurrent_edge_count(self) -> int:
        return sum(1 for rs in self.recurrent_synapses if rs.enabled)

    def get_enabled_node_count(self) -> int:
        return sum(1 for n in self.nodes if n.enabled and n.layer_type == HIDDEN_LAYER)

    def get_input_count(self) -> int:
        return sum(1 for n in self.nodes if n.layer_type == INPUT_LAYER)

    def get_output_count(self) -> int:
        return sum(1 for n in self.nodes if n.layer_type == OUTPUT_LAYER)

    def get_structural_hash(self) -> str:
        """Compute a structural hash for duplicate detection."""
        self.assign_reachability()
        parts = []
        for n in sorted(self.nodes, key=lambda x: x.innovation_number):
            if n.enabled and n.is_reachable():
                parts.append(f"n{n.innovation_number}")
        for s in sorted(self.synapses, key=lambda x: x.innovation_number):
            if s.enabled and s.is_reachable():
                parts.append(f"e{s.input_node_id}_{s.output_node_id}")
        for rs in sorted(self.recurrent_synapses, key=lambda x: x.innovation_number):
            if rs.enabled and rs.is_reachable():
                parts.append(f"r{rs.input_node_id}_{rs.output_node_id}_{rs.recurrent_depth}")
        return "|".join(parts)

    def __repr__(self) -> str:
        return (
            f"SNNGenome(gen={self.generation_id}, "
            f"nodes={len(self.nodes)}, "
            f"edges={len(self.synapses)}, "
            f"rec={len(self.recurrent_synapses)}, "
            f"fitness={self.fitness:.4f})"
        )


# ─── Factory Functions ───────────────────────────────────────────────────

def create_seed_genome(
    n_inputs: int,
    n_outputs: int,
    weight_initialize: str = "xavier",
    mutated_component_weight: str = "lamarckian",
    weight_inheritance: str = "lamarckian",
) -> SNNGenome:
    """Create a minimal seed genome: input nodes -> 1 hidden node -> output nodes.

    This mirrors EXAMM's initial seed genome creation.
    """
    nodes: list[LIFNode] = []
    synapses: list[Synapse] = []

    innovation = 0

    # Input nodes at depth 0.0
    input_nodes = []
    for i in range(n_inputs):
        innovation += 1
        node = LIFNode(innovation, INPUT_LAYER, 0.0, parameter_name=f"input_{i}")
        nodes.append(node)
        input_nodes.append(node)

    # One hidden node at depth 0.5
    innovation += 1
    hidden_node = LIFNode(innovation, HIDDEN_LAYER, 0.5)
    nodes.append(hidden_node)

    # Output nodes at depth 1.0
    output_nodes = []
    for i in range(n_outputs):
        innovation += 1
        node = LIFNode(innovation, OUTPUT_LAYER, 1.0, parameter_name=f"output_{i}")
        nodes.append(node)
        output_nodes.append(node)

    edge_inn = innovation

    # Connect all inputs -> hidden
    for inp in input_nodes:
        edge_inn += 1
        synapses.append(Synapse(edge_inn, inp.innovation_number, hidden_node.innovation_number))

    # Connect hidden -> all outputs
    for out in output_nodes:
        edge_inn += 1
        synapses.append(Synapse(edge_inn, hidden_node.innovation_number, out.innovation_number))

    genome = SNNGenome(
        nodes, synapses, [],
        weight_initialize=weight_initialize,
        mutated_component_weight=mutated_component_weight,
        weight_inheritance=weight_inheritance,
    )
    genome.initialize_randomly()
    genome.assign_reachability()
    return genome
