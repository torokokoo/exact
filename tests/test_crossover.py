"""Tests for crossover with innovation number alignment."""

import pytest
from snn_examm.genome.snn_genome import SNNGenome, create_seed_genome
from snn_examm.genome.lif_node import INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER
from snn_examm.evolution.engine import EvolverEngine


def make_engine():
    return EvolverEngine(
        n_inputs=2, n_outputs=2,
        number_islands=3, island_size=5, max_genomes=100,
    )


class TestCrossoverBasics:
    def test_crossover_produces_child(self):
        engine = make_engine()
        p1 = create_seed_genome(2, 2)
        p2 = create_seed_genome(2, 2)
        p1.fitness = 10.0
        p2.fitness = 5.0
        p1.initial_parameters = p1.get_weights()
        p2.initial_parameters = p2.get_weights()

        child = engine.crossover(p1, p2)
        assert child is not None
        assert child.parent_ids == [p1.generation_id, p2.generation_id]

    def test_child_has_input_output_nodes(self):
        engine = make_engine()
        p1 = create_seed_genome(2, 2)
        p2 = create_seed_genome(2, 2)
        p1.fitness = 10.0
        p2.fitness = 5.0
        p1.initial_parameters = p1.get_weights()
        p2.initial_parameters = p2.get_weights()

        child = engine.crossover(p1, p2)
        assert child.get_input_count() == 2
        assert child.get_output_count() == 2

    def test_child_fitness_reset(self):
        engine = make_engine()
        p1 = create_seed_genome(2, 2)
        p2 = create_seed_genome(2, 2)
        p1.fitness = 10.0
        p2.fitness = 5.0
        p1.initial_parameters = p1.get_weights()
        p2.initial_parameters = p2.get_weights()

        child = engine.crossover(p1, p2)
        assert child.fitness == float("-inf")

    def test_generated_by_crossover(self):
        engine = make_engine()
        p1 = create_seed_genome(2, 2)
        p2 = create_seed_genome(2, 2)
        p1.fitness = 10.0
        p2.fitness = 5.0
        p1.group_id = 0
        p2.group_id = 0
        p1.initial_parameters = p1.get_weights()
        p2.initial_parameters = p2.get_weights()

        child = engine.crossover(p1, p2)
        assert child.generated_by == "crossover"

    def test_island_crossover_different_islands(self):
        engine = make_engine()
        p1 = create_seed_genome(2, 2)
        p2 = create_seed_genome(2, 2)
        p1.fitness = 10.0
        p2.fitness = 5.0
        p1.group_id = 0
        p2.group_id = 1
        p1.initial_parameters = p1.get_weights()
        p2.initial_parameters = p2.get_weights()

        child = engine.crossover(p1, p2)
        assert child.generated_by == "island_crossover"


class TestCrossoverSwapsFitterParent:
    def test_fitter_parent_is_p1(self):
        """If p2 is fitter, crossover swaps them."""
        engine = make_engine()
        p1 = create_seed_genome(2, 2)
        p2 = create_seed_genome(2, 2)
        p1.fitness = 5.0
        p2.fitness = 10.0
        p1.generation_id = 1
        p2.generation_id = 2
        p1.initial_parameters = p1.get_weights()
        p2.initial_parameters = p2.get_weights()

        child = engine.crossover(p1, p2)
        # Parent IDs should be [fitter, less_fit] = [2, 1]
        assert child.parent_ids[0] == 2


class TestCrossoverWithMutatedParents:
    def test_crossover_after_mutations(self):
        engine = make_engine()
        p1 = create_seed_genome(2, 2)
        p2 = create_seed_genome(2, 2)

        # Mutate parents differently
        engine.mutate(3, p1)
        engine.mutate(3, p2)
        p1.fitness = 10.0
        p2.fitness = 5.0

        child = engine.crossover(p1, p2)
        assert child is not None
        assert child.get_input_count() == 2
        assert child.get_output_count() == 2
        # Child may have more edges than either parent
        assert len(child.synapses) > 0
