"""Tests for Island class."""

import pytest

from snn_examm.genome.snn_genome import SNNGenome, create_seed_genome
from snn_examm.evolution.island import Island, INITIALIZING, FILLED, REPOPULATING


def make_genome(fitness: float) -> SNNGenome:
    g = create_seed_genome(2, 2)
    g.fitness = fitness
    g.generation_id = int(abs(fitness) * 100)
    return g


class TestIslandBasics:
    def test_create_empty(self):
        island = Island(0, 5)
        assert island.size() == 0
        assert island.is_initializing()
        assert not island.is_full()

    def test_insert_genome(self):
        island = Island(0, 5)
        g = make_genome(10.0)
        pos = island.insert_genome(g)
        assert pos == 0
        assert island.size() == 1

    def test_sorted_descending(self):
        island = Island(0, 5)
        for f in [5.0, 10.0, 3.0, 8.0, 1.0]:
            island.insert_genome(make_genome(f))

        fitnesses = [g.fitness for g in island.genomes]
        assert fitnesses == sorted(fitnesses, reverse=True)

    def test_fills_and_transitions(self):
        island = Island(0, 3)
        for f in [1.0, 2.0, 3.0]:
            island.insert_genome(make_genome(f))
        assert island.is_full()
        assert island.status == FILLED

    def test_rejects_worse_when_full(self):
        island = Island(0, 3)
        for f in [5.0, 10.0, 8.0]:
            island.insert_genome(make_genome(f))

        pos = island.insert_genome(make_genome(3.0))
        assert pos == -1
        assert island.size() == 3

    def test_evicts_worst_when_inserting_better(self):
        island = Island(0, 3)
        for f in [5.0, 8.0, 10.0]:
            island.insert_genome(make_genome(f))

        pos = island.insert_genome(make_genome(7.0))
        assert pos >= 0
        assert island.size() == 3
        # Worst (5.0) should be evicted
        assert island.get_worst_fitness() >= 7.0


class TestIslandBestWorst:
    def test_best_fitness(self):
        island = Island(0, 5)
        for f in [3.0, 7.0, 5.0]:
            island.insert_genome(make_genome(f))
        assert island.get_best_fitness() == 7.0

    def test_worst_fitness(self):
        island = Island(0, 5)
        for f in [3.0, 7.0, 5.0]:
            island.insert_genome(make_genome(f))
        assert island.get_worst_fitness() == 3.0


class TestIslandExtinction:
    def test_erase_island(self):
        island = Island(0, 5)
        for f in [3.0, 7.0, 5.0]:
            island.insert_genome(make_genome(f))

        island.erase_island()
        assert island.size() == 0
        assert island.is_repopulating()
        assert island.been_erased()

    def test_rejects_stale_genomes_after_erase(self):
        island = Island(0, 5)
        g1 = make_genome(10.0)
        g1.generation_id = 5
        island.insert_genome(g1)

        island.erase_island()

        stale = make_genome(20.0)
        stale.generation_id = 3  # older than erase
        pos = island.insert_genome(stale)
        assert pos == -1


class TestIslandCopy:
    def test_copy_random_genome(self):
        island = Island(0, 5)
        for f in [3.0, 7.0, 5.0]:
            island.insert_genome(make_genome(f))

        copy = island.copy_random_genome()
        assert copy is not None
        assert copy.fitness in [3.0, 5.0, 7.0]

    def test_copy_two_random_genomes(self):
        island = Island(0, 5)
        for f in [3.0, 7.0, 5.0]:
            island.insert_genome(make_genome(f))

        g1, g2 = island.copy_two_random_genomes()
        assert g1 is not None and g2 is not None
        assert g1.fitness != g2.fitness or g1 is not g2
