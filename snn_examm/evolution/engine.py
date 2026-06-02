"""Evolution Engine — main controller for mutation, crossover, and genome lifecycle.

Port of examm/examm.{hxx,cxx}.
Orchestrates the island evolver, provides mutation and crossover operations,
and manages innovation number counters.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Optional

from ..genome.lif_node import LIFNode, HIDDEN_LAYER, bound
from ..genome.synapse import Synapse, RecurrentSynapse
from ..genome.snn_genome import SNNGenome, create_seed_genome
from .island_evolver import IslandEvolver

logger = logging.getLogger(__name__)

# Mutation operation probabilities (mirroring EXAMM's mutation distribution)
MUTATION_OPS = [
    ("add_edge", 0.20),
    ("add_recurrent_edge", 0.15),
    ("add_node", 0.15),
    ("split_edge", 0.15),
    ("disable_edge", 0.10),
    ("enable_edge", 0.10),
    ("enable_node", 0.05),
    ("disable_node", 0.05),
    ("clone", 0.05),
]


class EvolverEngine:
    """Main evolution controller.

    Owns the island evolver and provides mutation/crossover as callables.
    Manages global innovation number counters.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        number_islands: int = 10,
        island_size: int = 10,
        max_genomes: int = 2000,
        mutation_rate: float = 0.3,
        intra_island_crossover_rate: float = 0.7,
        more_fit_crossover_rate: float = 0.5,
        less_fit_crossover_rate: float = 0.5,
        extinction_event_generation_number: int = 0,
        islands_to_exterminate: int = 1,
        num_mutations: int = 1,
        repopulation_method: str = "bestGenome",
        weight_initialize: str = "xavier",
        mutated_component_weight: str = "lamarckian",
        weight_inheritance: str = "lamarckian",
        min_recurrent_depth: int = 1,
        max_recurrent_depth: int = 10,
    ):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.more_fit_crossover_rate = more_fit_crossover_rate
        self.less_fit_crossover_rate = less_fit_crossover_rate
        self.weight_initialize = weight_initialize
        self.mutated_component_weight = mutated_component_weight
        self.weight_inheritance = weight_inheritance
        self.min_recurrent_depth = min_recurrent_depth
        self.max_recurrent_depth = max_recurrent_depth

        # Create seed genome
        seed = create_seed_genome(
            n_inputs, n_outputs,
            weight_initialize=weight_initialize,
            mutated_component_weight=mutated_component_weight,
            weight_inheritance=weight_inheritance,
        )

        # Initialize innovation counters from seed
        self.node_innovation_count = max(n.innovation_number for n in seed.nodes)
        self.edge_innovation_count = max(
            max((s.innovation_number for s in seed.synapses), default=0),
            max((rs.innovation_number for rs in seed.recurrent_synapses), default=0),
        )

        # Create island evolver
        self.island_evolver = IslandEvolver(
            number_islands=number_islands,
            island_size=island_size,
            seed_genome=seed,
            mutation_rate=mutation_rate,
            intra_island_crossover_rate=intra_island_crossover_rate,
            extinction_event_generation_number=extinction_event_generation_number,
            islands_to_exterminate=islands_to_exterminate,
            num_mutations=num_mutations,
            repopulation_method=repopulation_method,
            max_genomes=max_genomes,
        )

    def generate_genome(self) -> Optional[SNNGenome]:
        """Generate the next genome for evaluation."""
        return self.island_evolver.generate_genome(
            mutate_fn=self.mutate,
            crossover_fn=self.crossover,
        )

    def insert_genome(self, genome: SNNGenome) -> int:
        """Insert an evaluated genome back into its island."""
        return self.island_evolver.insert_genome(genome)

    def get_best_genome(self) -> Optional[SNNGenome]:
        return self.island_evolver.get_best_genome()

    def get_stats(self) -> dict:
        return self.island_evolver.get_stats()

    # ─── Mutation ────────────────────────────────────────────────────────
    # Port of EXAMM::mutate (examm.cxx:581-740)

    def mutate(self, num_mutations: int, genome: SNNGenome) -> None:
        """Apply num_mutations random structural mutations to a genome.

        Computes Lamarckian mu/sigma from parent's weight vector,
        then applies random operations.
        """
        mu, sigma = genome.get_mu_sigma()

        for _ in range(num_mutations):
            op = self._select_mutation_op()

            if op == "add_edge":
                success, self.edge_innovation_count = genome.add_edge(
                    mu, sigma, self.edge_innovation_count
                )
            elif op == "add_recurrent_edge":
                success, self.edge_innovation_count = genome.add_recurrent_edge(
                    mu, sigma, self.edge_innovation_count,
                    self.min_recurrent_depth, self.max_recurrent_depth,
                )
            elif op == "add_node":
                success, self.edge_innovation_count, self.node_innovation_count = genome.add_node(
                    mu, sigma, self.edge_innovation_count, self.node_innovation_count,
                    self.min_recurrent_depth, self.max_recurrent_depth,
                )
            elif op == "split_edge":
                success, self.edge_innovation_count, self.node_innovation_count = genome.split_edge(
                    mu, sigma, self.edge_innovation_count, self.node_innovation_count,
                )
            elif op == "disable_edge":
                genome.disable_edge()
            elif op == "enable_edge":
                genome.enable_edge()
            elif op == "enable_node":
                genome.enable_node()
            elif op == "disable_node":
                genome.disable_node()
            elif op == "clone":
                pass  # Clone = copy without structural change

        genome.assign_reachability()
        genome.snapshot_parameters()

    def _select_mutation_op(self) -> str:
        """Select a mutation operation using weighted random choice."""
        ops, weights = zip(*MUTATION_OPS)
        return random.choices(ops, weights=weights, k=1)[0]

    # ─── Crossover ───────────────────────────────────────────────────────
    # Port of EXAMM::crossover (examm.cxx:995-1264)

    def crossover(self, p1: SNNGenome, p2: SNNGenome) -> SNNGenome:
        """Create a child genome from two parents using innovation-number alignment.

        p1 is treated as the fitter parent. If p2 is actually fitter, swap them.
        Matching edges: blend weights with t = Uniform[-0.5, 1.5].
        Disjoint/excess from p1: include based on more_fit_crossover_rate.
        Disjoint/excess from p2: include based on less_fit_crossover_rate.
        """
        # Ensure p1 is fitter
        if p2.fitness > p1.fitness:
            p1, p2 = p2, p1

        # Set weights from best parameters for mu/sigma computation
        if p1.best_parameters:
            p1.set_weights(p1.best_parameters)
        elif p1.initial_parameters:
            p1.set_weights(p1.initial_parameters)

        if p2.best_parameters:
            p2.set_weights(p2.best_parameters)
        elif p2.initial_parameters:
            p2.set_weights(p2.initial_parameters)

        # Build child nodes, edges, and recurrent edges
        child_node_map: dict[int, LIFNode] = {}  # innovation -> node
        child_synapses: list[Synapse] = []
        child_recurrent_synapses: list[RecurrentSynapse] = []

        # Sort parent edges by innovation for alignment
        p1_edges = sorted(p1.synapses, key=lambda s: s.innovation_number)
        p2_edges = sorted(p2.synapses, key=lambda s: s.innovation_number)

        # Merge feedforward edges
        self._merge_edges(
            p1_edges, p2_edges, p1, p2,
            child_synapses, child_node_map,
        )

        # Sort parent recurrent edges by innovation for alignment
        p1_rec = sorted(p1.recurrent_synapses, key=lambda rs: rs.innovation_number)
        p2_rec = sorted(p2.recurrent_synapses, key=lambda rs: rs.innovation_number)

        # Merge recurrent edges
        self._merge_recurrent_edges(
            p1_rec, p2_rec, p1, p2,
            child_recurrent_synapses, child_node_map,
        )

        # Ensure all input/output nodes are present
        for n in p1.nodes:
            if n.innovation_number not in child_node_map:
                if n.layer_type != HIDDEN_LAYER or n.is_reachable():
                    child_node_map[n.innovation_number] = n.copy()

        child_nodes = sorted(child_node_map.values(), key=lambda n: (n.depth, n.innovation_number))

        child = SNNGenome(
            child_nodes, child_synapses, child_recurrent_synapses,
            weight_initialize=p1.weight_initialize,
            mutated_component_weight=p1.mutated_component_weight,
            weight_inheritance=p1.weight_inheritance,
        )

        child.parent_ids = [p1.generation_id, p2.generation_id]
        if p1.group_id == p2.group_id:
            child.generated_by = "crossover"
        else:
            child.generated_by = "island_crossover"

        # If weight inheritance is NOT lamarckian, re-initialize all weights
        if self.weight_inheritance == self.weight_initialize:
            child.initialize_randomly()

        child.assign_reachability()
        child.fitness = float("-inf")
        child.initial_parameters = child.get_weights()
        child.best_parameters = []

        return child

    def _merge_edges(
        self,
        p1_edges: list[Synapse],
        p2_edges: list[Synapse],
        p1: SNNGenome,
        p2: SNNGenome,
        child_synapses: list[Synapse],
        child_node_map: dict[int, LIFNode],
    ) -> None:
        """Merge feedforward edges using innovation-number alignment.

        Port of the edge merge loop in EXAMM::crossover (examm.cxx:1065-1130).
        """
        i, j = 0, 0
        while i < len(p1_edges) and j < len(p2_edges):
            e1 = p1_edges[i]
            e2 = p2_edges[j]

            if e1.innovation_number == e2.innovation_number:
                # Matching edge: blend weights
                t = random.uniform(-0.5, 1.5)
                new_weight = t * (e2.weight - e1.weight) + e1.weight
                new_edge = e1.copy()
                new_edge.weight = new_weight
                new_edge.enabled = True
                child_synapses.append(new_edge)
                self._ensure_edge_nodes(e1, e2, p1, p2, child_node_map, t)
                i += 1
                j += 1
            elif e1.innovation_number < e2.innovation_number:
                # Disjoint from fitter parent
                enabled = e1.is_reachable()
                new_edge = e1.copy()
                new_edge.enabled = enabled
                child_synapses.append(new_edge)
                self._ensure_nodes_from_edge(e1, p1, child_node_map)
                i += 1
            else:
                # Disjoint from less fit parent
                enabled = e2.is_reachable() and random.random() < self.less_fit_crossover_rate
                new_edge = e2.copy()
                new_edge.enabled = enabled
                child_synapses.append(new_edge)
                self._ensure_nodes_from_edge(e2, p2, child_node_map)
                j += 1

        # Remaining from fitter parent
        while i < len(p1_edges):
            e1 = p1_edges[i]
            enabled = e1.is_reachable()
            new_edge = e1.copy()
            new_edge.enabled = enabled
            child_synapses.append(new_edge)
            self._ensure_nodes_from_edge(e1, p1, child_node_map)
            i += 1

        # Remaining from less fit parent
        while j < len(p2_edges):
            e2 = p2_edges[j]
            enabled = e2.is_reachable() and random.random() < self.less_fit_crossover_rate
            new_edge = e2.copy()
            new_edge.enabled = enabled
            child_synapses.append(new_edge)
            self._ensure_nodes_from_edge(e2, p2, child_node_map)
            j += 1

    def _merge_recurrent_edges(
        self,
        p1_rec: list[RecurrentSynapse],
        p2_rec: list[RecurrentSynapse],
        p1: SNNGenome,
        p2: SNNGenome,
        child_rec: list[RecurrentSynapse],
        child_node_map: dict[int, LIFNode],
    ) -> None:
        """Merge recurrent edges. Same logic as _merge_edges."""
        i, j = 0, 0
        while i < len(p1_rec) and j < len(p2_rec):
            r1 = p1_rec[i]
            r2 = p2_rec[j]

            if r1.innovation_number == r2.innovation_number:
                t = random.uniform(-0.5, 1.5)
                new_weight = t * (r2.weight - r1.weight) + r1.weight
                new_re = r1.copy()
                new_re.weight = new_weight
                new_re.enabled = True
                child_rec.append(new_re)
                self._ensure_rec_edge_nodes(r1, r2, p1, p2, child_node_map, t)
                i += 1
                j += 1
            elif r1.innovation_number < r2.innovation_number:
                enabled = r1.is_reachable()
                new_re = r1.copy()
                new_re.enabled = enabled
                child_rec.append(new_re)
                self._ensure_nodes_from_rec_edge(r1, p1, child_node_map)
                i += 1
            else:
                enabled = r2.is_reachable() and random.random() < self.less_fit_crossover_rate
                new_re = r2.copy()
                new_re.enabled = enabled
                child_rec.append(new_re)
                self._ensure_nodes_from_rec_edge(r2, p2, child_node_map)
                j += 1

        while i < len(p1_rec):
            r1 = p1_rec[i]
            enabled = r1.is_reachable()
            new_re = r1.copy()
            new_re.enabled = enabled
            child_rec.append(new_re)
            self._ensure_nodes_from_rec_edge(r1, p1, child_node_map)
            i += 1

        while j < len(p2_rec):
            r2 = p2_rec[j]
            enabled = r2.is_reachable() and random.random() < self.less_fit_crossover_rate
            new_re = r2.copy()
            new_re.enabled = enabled
            child_rec.append(new_re)
            self._ensure_nodes_from_rec_edge(r2, p2, child_node_map)
            j += 1

    def _ensure_edge_nodes(
        self,
        e1: Synapse, e2: Synapse,
        p1: SNNGenome, p2: SNNGenome,
        child_node_map: dict[int, LIFNode],
        t: float,
    ) -> None:
        """Ensure both endpoint nodes exist in child, with blended weights for matching edges."""
        for node_id in (e1.input_node_id, e1.output_node_id):
            if node_id not in child_node_map:
                n1 = p1._find_node(node_id)
                n2 = p2._find_node(node_id)
                if n1:
                    new_node = n1.copy()
                    if n2 and n1.layer_type == HIDDEN_LAYER:
                        # Blend node weights
                        w1 = n1.get_weights()
                        w2 = n2.get_weights()
                        if w1 and w2 and len(w1) == len(w2):
                            blended = [t * (b - a) + a for a, b in zip(w1, w2)]
                            new_node.set_weights(blended)
                    child_node_map[node_id] = new_node
                elif n2:
                    child_node_map[node_id] = n2.copy()

    def _ensure_nodes_from_edge(
        self,
        edge: Synapse,
        parent: SNNGenome,
        child_node_map: dict[int, LIFNode],
    ) -> None:
        """Ensure both endpoint nodes of an edge exist in child."""
        for node_id in (edge.input_node_id, edge.output_node_id):
            if node_id not in child_node_map:
                n = parent._find_node(node_id)
                if n:
                    child_node_map[node_id] = n.copy()

    def _ensure_rec_edge_nodes(
        self,
        r1: RecurrentSynapse, r2: RecurrentSynapse,
        p1: SNNGenome, p2: SNNGenome,
        child_node_map: dict[int, LIFNode],
        t: float,
    ) -> None:
        """Same as _ensure_edge_nodes for recurrent edges."""
        for node_id in (r1.input_node_id, r1.output_node_id):
            if node_id not in child_node_map:
                n1 = p1._find_node(node_id)
                n2 = p2._find_node(node_id)
                if n1:
                    new_node = n1.copy()
                    if n2 and n1.layer_type == HIDDEN_LAYER:
                        w1 = n1.get_weights()
                        w2 = n2.get_weights()
                        if w1 and w2 and len(w1) == len(w2):
                            blended = [t * (b - a) + a for a, b in zip(w1, w2)]
                            new_node.set_weights(blended)
                    child_node_map[node_id] = new_node
                elif n2:
                    child_node_map[node_id] = n2.copy()

    def _ensure_nodes_from_rec_edge(
        self,
        edge: RecurrentSynapse,
        parent: SNNGenome,
        child_node_map: dict[int, LIFNode],
    ) -> None:
        for node_id in (edge.input_node_id, edge.output_node_id):
            if node_id not in child_node_map:
                n = parent._find_node(node_id)
                if n:
                    child_node_map[node_id] = n.copy()
